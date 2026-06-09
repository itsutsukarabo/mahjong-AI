"""
手牌類推モデル v12: v11 に学習時制約付きソフトマックス + MSE合計ペナルティ復活

入力 (374次元): v10 と同一

アーキテクチャ:
  v10 バックボーン + block heads:
    triplet_head: Linear(d_model, 1) → sigmoid → P(刻子|聴牌) [34次元]
    seq_head:     Linear(d_model, 1) → sigmoid → P(順子|聴牌) [21次元; m1-7/p1-7/s1-7]
    pair_head:    Linear(d_model, 1) → sigmoid → P(対子|聴牌) [34次元]

損失関数:
  loss = CE + λ_gs*MSE(sum) + λ_red_ce*CE_red + λ_red_cons*relu_cons
       + λ_block * (BCE_triplet + BCE_seq + BCE_pair) / 3
  block 損失は tenpai サンプルのみで計算 (tenpai_mask)

label_block フォーマット (89次元 flat):
  [0:34]  triplet ソフトラベル
  [34:55] seq ソフトラベル (21次元)
  [55:89] pair ソフトラベル (34次元)

ONNX 出力: logits, red_logits, triplet_logits, seq_logits, pair_logits

評価 (TICKET-034): remaining バケット別 accuracy を eval_result.json に追記
"""

import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- 設定 ----

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v12"

CONFIG = {
    "input_dim":    374,
    "d_model":      256,
    "nhead":        4,
    "num_layers":   3,
    "n_pai":        34,
    "n_count_cls":  5,
    "dropout":      0.1,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   512,
    "epochs":       200,
    "early_stop_patience": 7,
    "lambda_block": 0.3,
}

VISIBLE_OFFSET     = 185
REMAINING_OFFSET   = 94
GPU_TEMP_THRESHOLD = 65
GPU_COOL_INTERVAL  = 20

LAMBDA_RED_CE      = 0.3
LAMBDA_RED_CONS    = 0.1
LAMBDA_SUM         = 0.5   # 合計枚数MSEペナルティ（v10と同値）

BLOCK_DIM = 89  # triplet(34) + seq(21) + pair(34)

REMAINING_BUCKETS = {
    "early":   (51, 9999),
    "mid":     (31,   50),
    "late":    (11,   30),
    "endgame": ( 0,   10),
}

# 順子トークンのタイルインデックス (m1-7: 0-6, p1-7: 9-15, s1-7: 18-24)
SEQ_TILE_INDICES = list(range(0, 7)) + list(range(9, 16)) + list(range(18, 25))


# ---- 制約付きソフトマックス ----

def find_lambda(logits, k_float, n_iter=50):
    """Binary search for λ s.t. Σ_i E_λ[c_i] = k. Returns λ detached from grad."""
    count_vals = torch.arange(5, device=logits.device, dtype=torch.float32)
    B = logits.shape[0]
    with torch.no_grad():
        lo = torch.full((B,), -20.0, device=logits.device)
        hi = torch.full((B,),  20.0, device=logits.device)
        for _ in range(n_iter):
            mid   = (lo + hi) / 2
            adj   = logits - mid.view(B, 1, 1) * count_vals
            probs = F.softmax(adj, dim=-1)
            e_sum = (probs * count_vals).sum(-1).sum(-1)   # (B,)
            lo    = torch.where(e_sum > k_float, mid, lo)
            hi    = torch.where(e_sum > k_float, hi,  mid)
    return (lo + hi) / 2


def constrained_softmax_probs(logits, k):
    """
    logits: (B, 34, 5)  masked logits from model
    k:      (B,)        int tensor, total tile count per sample
    Returns probs (B, 34, 5) with E[Σ c_i] = k exactly.
    λ is computed detached; gradients flow through logits → probs only.
    """
    count_vals = torch.arange(5, device=logits.device, dtype=torch.float32)
    lam  = find_lambda(logits, k.float())          # (B,), no grad
    adj  = logits - lam.view(-1, 1, 1) * count_vals
    return F.softmax(adj, dim=-1)


def get_stage_weights(features):
    """残り牌枚数に基づくゲーム終盤重み (B,)。終盤ほど高い。"""
    remaining = (features[:, REMAINING_OFFSET] * 70).round().long()
    w = torch.ones(len(features), device=features.device)
    w[remaining <= 10] = 4.0
    w[(remaining > 10) & (remaining <= 30)] = 3.0
    w[(remaining > 30) & (remaining <= 50)] = 2.0
    return w


def weighted_eae(probs, labels, weights):
    """
    Stage-weighted Expected Absolute Error (枚数ズレの期待値):
      EAE = Σ_tiles Σ_c |c - true| × P(c=c)  per sample
    probs:   (B, 34, 5)  constrained softmax 出力
    labels:  (B, 34)     int 正解枚数
    weights: (B,)        ゲーム終盤重み
    Returns: スカラー（重み付き平均 EAE）
    """
    count_vals = torch.arange(probs.shape[-1], device=probs.device, dtype=torch.float32)
    abs_diff       = (count_vals - labels.unsqueeze(-1).float()).abs()  # (B, 34, 5)
    eae_per_sample = (abs_diff * probs).sum(-1).sum(-1)                 # (B,)
    return (eae_per_sample * weights).sum() / weights.sum()


def gpu_temp():
    """GPU温度を取得。WSL・Windows ネイティブ・Linux いずれでも動作する。"""
    candidates = ["nvidia-smi"]
    if sys.platform == "win32":
        # Windows ネイティブ: CUDA インストール先や System32 を追加
        candidates += [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
    else:
        # WSL / Linux: wsl ライブラリパスと Windows マウント経由を追加
        candidates += [
            "/usr/lib/wsl/lib/nvidia-smi",
            "/mnt/c/Windows/System32/nvidia-smi.exe",
        ]
    for cmd in candidates:
        try:
            r = subprocess.run(
                [cmd, "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                return int(r.stdout.strip())
        except Exception:
            continue
    return 0


def wait_for_cool():
    temp = gpu_temp()
    if temp <= GPU_TEMP_THRESHOLD:
        return
    print(f"  [thermal] GPU {temp}C > {GPU_TEMP_THRESHOLD}C - cooling...", flush=True)
    while True:
        time.sleep(GPU_COOL_INTERVAL)
        temp = gpu_temp()
        print(f"  [thermal] GPU {temp}C", flush=True)
        if temp <= GPU_TEMP_THRESHOLD:
            print("  [thermal] cool enough, resuming", flush=True)
            break


# ---- データセット ----

class HandInferenceDataset(Dataset):
    """numpy 配列を受け取り torch.Tensor に変換する（Python dict の巨大なオーバーヘッドを回避）"""
    def __init__(self, features, labels, labels_red, labels_block, tenpai_mask):
        self.features     = torch.as_tensor(features,     dtype=torch.float32)
        self.labels       = torch.as_tensor(labels,       dtype=torch.long).clamp(0, 4)
        self.labels_red   = torch.as_tensor(labels_red,   dtype=torch.long)
        self.labels_block = torch.as_tensor(labels_block, dtype=torch.float32)
        self.tenpai_mask  = torch.as_tensor(tenpai_mask,  dtype=torch.bool)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.labels_red[idx],
            self.labels_block[idx],
            self.tenpai_mask[idx],
        )


# ---- モデル ----

class HandInferenceV11(nn.Module):

    SEQ_TILE_IDX = torch.tensor(SEQ_TILE_INDICES)

    def __init__(self, input_dim, d_model, nhead, num_layers, n_pai, n_count_cls, dropout):
        super().__init__()
        self.n_pai       = n_pai
        self.n_count_cls = n_count_cls

        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

        self.tile_embed   = nn.Embedding(n_pai, d_model)
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head         = nn.Linear(d_model, n_count_cls)
        self.red_head     = nn.Linear(d_model, 2)

        # block heads
        self.triplet_head = nn.Linear(d_model, 1)
        self.pair_head    = nn.Linear(d_model, 1)
        self.seq_head     = nn.Linear(d_model, 1)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))  # 5m/5p/5s
        self.register_buffer("seq_tile_idx", self.SEQ_TILE_IDX)

    def forward(self, x):
        g = self.global_encoder(x)                                            # [B, d_model]

        tile_emb = self.tile_embed(self.tile_ids)                             # [34, d_model]
        vis = x[:, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai].unsqueeze(-1)  # [B, 34, 1]
        vis_emb = self.visible_proj(vis)                                      # [B, 34, d_model]

        tokens = g.unsqueeze(1) + tile_emb.unsqueeze(0) + vis_emb             # [B, 34, d_model]
        out    = self.transformer(tokens)                                      # [B, 34, d_model]
        logits_raw = self.head(out)                                            # [B, 34, 5]

        vis_raw = (x[:, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai] * 4).round().long().clamp(0, 4)
        hidden_remaining = (4 - vis_raw).clamp(min=0)
        mask = self.count_range > hidden_remaining.unsqueeze(-1)
        logits = logits_raw.masked_fill(mask, float('-inf'))

        red_tokens = out[:, self.red_tile_idx, :]                             # [B, 3, d_model]
        red_logits = self.red_head(red_tokens)                                # [B, 3, 2]

        # block heads
        triplet_logits = self.triplet_head(out).squeeze(-1)                   # [B, 34]
        pair_logits    = self.pair_head(out).squeeze(-1)                      # [B, 34]
        seq_tokens     = out[:, self.seq_tile_idx, :]                         # [B, 21, d_model]
        seq_logits     = self.seq_head(seq_tokens).squeeze(-1)                # [B, 21]

        return logits, logits_raw, red_logits, triplet_logits, seq_logits, pair_logits


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_nll = 0.0
    five_idx  = model.red_tile_idx

    for features, labels, labels_red, labels_block, tenpai_mask in loader:
        features     = features.to(device)
        labels       = labels.to(device)
        labels_red   = labels_red.to(device)
        labels_block = labels_block.to(device)
        tenpai_mask  = tenpai_mask.to(device)
        optimizer.zero_grad()

        logits, _, red_logits, tri_logits, seq_logits, pair_logits = model(features)

        count_vals = torch.arange(5, device=device, dtype=torch.float32)
        k          = labels.float().sum(dim=-1).long()

        # ① 制約付きソフトマックスで NLL（訓練・評価を揃える）
        lam        = find_lambda(logits, k.float())                        # detached
        adj        = logits - lam.view(-1, 1, 1) * count_vals
        loss_nll   = F.cross_entropy(adj.reshape(-1, 5), labels.reshape(-1))

        # ② MSE合計ペナルティ（raw softmax で勾配を安定させる）
        probs_raw  = F.softmax(logits, dim=-1)
        pred_sum   = (probs_raw * count_vals).sum(-1).sum(-1)              # (B,)
        loss_sum   = F.mse_loss(pred_sum, k.float())

        loss_red_ce = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))

        probs_constrained = constrained_softmax_probs(logits, k)
        prob_has_red      = F.softmax(red_logits, dim=-1)[:, :, 1]
        prob_cnt_ge1      = 1 - probs_constrained[:, five_idx, 0]
        loss_red_cons     = F.relu(prob_has_red - prob_cnt_ge1).mean()

        # block 損失（tenpai サンプルのみ）
        block_loss = 0.0
        n_tenpai = tenpai_mask.sum().item()
        if n_tenpai > 0:
            tri = tri_logits[tenpai_mask]
            seq = seq_logits[tenpai_mask]
            pai = pair_logits[tenpai_mask]
            t_tri = labels_block[tenpai_mask, :34]
            t_seq = labels_block[tenpai_mask, 34:55]
            t_pai = labels_block[tenpai_mask, 55:89]
            block_loss = (F.binary_cross_entropy_with_logits(tri, t_tri) +
                          F.binary_cross_entropy_with_logits(seq, t_seq) +
                          F.binary_cross_entropy_with_logits(pai, t_pai)) / 3

        loss = (loss_nll
                + LAMBDA_SUM      * loss_sum
                + LAMBDA_RED_CE   * loss_red_ce
                + LAMBDA_RED_CONS * loss_red_cons
                + CONFIG["lambda_block"] * block_loss)
        loss.backward()
        optimizer.step()

        total_nll += loss_nll.item() * len(features)

    return total_nll / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    """Returns (val_eae, val_acc): stage-weighted EAE がメイン指標 (early stopping 用)"""
    model.eval()
    total_wsum = 0.0
    total_w    = 0.0
    correct    = 0
    total      = 0
    for features, labels, labels_red, labels_block, tenpai_mask in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _, _, _, _ = model(features)
        k     = labels.float().sum(dim=-1).long()
        probs = constrained_softmax_probs(logits, k)

        stage_w        = get_stage_weights(features)
        eae_val        = weighted_eae(probs, labels, stage_w)
        total_wsum    += eae_val.item() * stage_w.sum().item()
        total_w       += stage_w.sum().item()

        preds   = probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.numel()
    val_eae = total_wsum / total_w if total_w > 0 else float('inf')
    return val_eae, correct / total


@torch.no_grad()
def eval_by_remaining(model, loader, device):
    model.eval()
    count_vals = torch.arange(5, device=device, dtype=torch.float32)
    buckets = {k: {"correct": 0, "total": 0, "eae_sum": 0.0} for k in REMAINING_BUCKETS}

    for features, labels, labels_red, labels_block, tenpai_mask in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _, _, _, _ = model(features)

        k_vals = labels.float().sum(dim=-1).long()
        probs  = constrained_softmax_probs(logits, k_vals)
        preds  = probs.argmax(dim=-1)
        correct_mask = (preds == labels)

        abs_diff       = (count_vals - labels.unsqueeze(-1).float()).abs()
        eae_per_sample = (abs_diff * probs).sum(-1).sum(-1).cpu()  # (B,)

        remaining_vals = (features[:, REMAINING_OFFSET] * 70).round().long().cpu()

        for i, rem in enumerate(remaining_vals):
            rem = rem.item()
            for name, (lo, hi) in REMAINING_BUCKETS.items():
                if lo <= rem <= hi:
                    b = buckets[name]
                    b["correct"]  += correct_mask[i].sum().item()
                    b["total"]    += correct_mask[i].numel()
                    b["eae_sum"]  += eae_per_sample[i].item()
                    break

    result = {}
    for name, b in buckets.items():
        n_samples = b["total"] // model.n_pai if model.n_pai > 0 else 1
        result[name] = {
            "acc": b["correct"] / b["total"] if b["total"] > 0 else 0.0,
            "eae": b["eae_sum"] / n_samples if n_samples > 0 else 0.0,
            "n":   n_samples,
        }
    return result


@torch.no_grad()
def eval_metrics_detailed(model, loader, device):
    """非ゼロ Recall/Precision/F1, EAE, MAE, 手牌完全一致率, 合計枚数 MAE を算出"""
    model.eval()
    all_preds  = []
    all_labels = []
    count_vals = torch.arange(5, device=device, dtype=torch.float32)
    total_wsum = 0.0
    total_w    = 0.0

    for features, labels, labels_red, labels_block, tenpai_mask in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _, _, _, _ = model(features)
        k     = labels.float().sum(dim=-1).long()
        probs = constrained_softmax_probs(logits, k)
        preds = probs.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        stage_w    = get_stage_weights(features)
        abs_diff   = (count_vals - labels.unsqueeze(-1).float()).abs()
        eae_s      = (abs_diff * probs).sum(-1).sum(-1)
        total_wsum += (eae_s * stage_w).sum().item()
        total_w    += stage_w.sum().item()

    preds  = torch.cat(all_preds,  dim=0)   # (N, 34)
    labels = torch.cat(all_labels, dim=0)   # (N, 34)

    true_nz = (labels >= 1)
    pred_nz = (preds  >= 1)
    tp = (true_nz & pred_nz).float().sum()
    fn = (true_nz & ~pred_nz).float().sum()
    fp = (~true_nz & pred_nz).float().sum()

    recall    = (tp / (tp + fn + 1e-9)).item()
    precision = (tp / (tp + fp + 1e-9)).item()
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    mae_nz         = (preds[true_nz].float() - labels[true_nz].float()).abs().mean().item()
    exact_acc      = (preds == labels).all(dim=-1).float().mean().item()
    pred_total_mae = (preds.float().sum(dim=-1) - labels.float().sum(dim=-1)).abs().mean().item()
    weighted_eae_v = total_wsum / total_w if total_w > 0 else 0.0

    return {
        "weighted_eae":      round(weighted_eae_v, 4),  # ★ メイン指標
        "recall_nonzero":    round(recall,    4),
        "precision_nonzero": round(precision, 4),
        "f1_nonzero":        round(f1,        4),
        "mae_nonzero":       round(mae_nz,    4),
        "hand_exact_acc":    round(exact_acc, 4),
        "pred_total_mae":    round(pred_total_mae, 4),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ndjson_path = DATA_DIR / "hand_inference.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}")
        sys.exit(1)
    print(f"読み込み中: {ndjson_path}", flush=True)

    # Python dict は float オブジェクトのオーバーヘッドが大きいため
    # numpy 配列へ直接変換しながら読み込む（メモリ使用量を約 7 分の 1 に削減）
    _feat, _lab, _lred, _lblk, _tmask = [], [], [], [], []
    lb_found = False
    with open(ndjson_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            _feat.append(s["features"])
            _lab.append(s["label_hand"])
            _lred.append(s.get("label_red") or [0, 0, 0])
            lb = s.get("label_block")
            if lb is not None:
                _lblk.append(lb["triplet"] + lb["seq"] + lb["pair"])
                _tmask.append(True)
                lb_found = True
            else:
                _lblk.append([0.0] * BLOCK_DIM)
                _tmask.append(False)

    if not lb_found:
        print("label_block フィールドがありません。extract_features.js を再実行してください")
        sys.exit(1)

    feat_np  = np.array(_feat,  dtype=np.float32); del _feat
    lab_np   = np.array(_lab,   dtype=np.int64);   del _lab
    lred_np  = np.array(_lred,  dtype=np.int64);   del _lred
    lblk_np  = np.array(_lblk,  dtype=np.float32); del _lblk
    tmask_np = np.array(_tmask, dtype=bool);        del _tmask
    gc.collect()
    print(f"総サンプル数: {len(feat_np)}")

    if feat_np.shape[1] != CONFIG["input_dim"]:
        print(f"次元数不一致: expected {CONFIG['input_dim']}, got {feat_np.shape[1]}")
        print("extract_features.js → add_yaku_features.py → add_tenpai_features.py の順に実行してください")
        sys.exit(1)

    # ノイズ除外: 手牌合計0枚 かつ 副露なし = ゲーム終了後の残骸レコード
    before   = len(feat_np)
    keep     = (lab_np.sum(axis=1) > 0) | (feat_np[:, 78] + feat_np[:, 79] + feat_np[:, 80] > 0)
    feat_np  = feat_np[keep];  lab_np  = lab_np[keep]
    lred_np  = lred_np[keep];  lblk_np = lblk_np[keep]; tmask_np = tmask_np[keep]
    print(f"ノイズ除外: {before - len(feat_np)} samples → {len(feat_np)}")
    print(f"tenpai サンプル数: {tmask_np.sum()} ({tmask_np.sum()/len(feat_np)*100:.1f}%)")

    rng  = np.random.default_rng(42)
    idx  = rng.permutation(len(feat_np))
    feat_np  = feat_np[idx];  lab_np  = lab_np[idx]
    lred_np  = lred_np[idx];  lblk_np = lblk_np[idx]; tmask_np = tmask_np[idx]

    n       = len(feat_np)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    sl_tr, sl_va, sl_te = slice(None, n_train), slice(n_train, n_train+n_val), slice(n_train+n_val, None)

    train_dataset = HandInferenceDataset(feat_np[sl_tr], lab_np[sl_tr], lred_np[sl_tr], lblk_np[sl_tr], tmask_np[sl_tr])
    val_dataset   = HandInferenceDataset(feat_np[sl_va], lab_np[sl_va], lred_np[sl_va], lblk_np[sl_va], tmask_np[sl_va])
    test_dataset  = HandInferenceDataset(feat_np[sl_te], lab_np[sl_te], lred_np[sl_te], lblk_np[sl_te], tmask_np[sl_te])
    del feat_np, lab_np, lred_np, lblk_np, tmask_np; gc.collect()
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    use_gpu = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)

    model = HandInferenceV11(
        input_dim   = CONFIG["input_dim"],
        d_model     = CONFIG["d_model"],
        nhead       = CONFIG["nhead"],
        num_layers  = CONFIG["num_layers"],
        n_pai       = CONFIG["n_pai"],
        n_count_cls = CONFIG["n_count_cls"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_eae = math.inf
    patience_cnt = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_nll         = train_epoch(model, train_loader, optimizer, device)
        val_eae, val_acc  = eval_epoch(model, val_loader, device)
        scheduler.step(val_eae)
        print(f"epoch {epoch:3d}  train_nll={train_nll:.4f}  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}", flush=True)
        log_entry = {"epoch": epoch, "train_nll": train_nll,
                     "val_eae": val_eae, "val_acc": val_acc}
        with open(MODEL_DIR / "train_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        wait_for_cool()

        if val_eae < best_val_eae:
            best_val_eae = val_eae
            patience_cnt = 0
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["early_stop_patience"]:
                print("early stopping")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    test_eae, test_acc = eval_epoch(model, test_loader, device)
    print(f"test_eae={test_eae:.4f}  test_acc={test_acc:.4f}")

    bucket_result = eval_by_remaining(model, test_loader, device)
    print("バケット別精度:")
    for name, b in bucket_result.items():
        print(f"  {name:8s}: acc={b['acc']:.4f}  eae={b['eae']:.4f}  n={b['n']}")

    metrics = eval_metrics_detailed(model, test_loader, device)
    print("\n詳細評価指標:")
    for k_name, v in metrics.items():
        print(f"  {k_name}: {v}")

    eval_result = {
        "test_eae": test_eae,
        "test_acc": test_acc,
        "by_remaining_bucket": bucket_result,
        **metrics,
        "config": CONFIG,
    }
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(eval_result, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    model.eval().cpu()
    dummy = torch.zeros(1, CONFIG["input_dim"])

    class _OnnxWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _, red_logits, tri_logits, seq_logits, pair_logits = self.m(x)
            return logits, red_logits, tri_logits, seq_logits, pair_logits

    torch.onnx.export(
        _OnnxWrapper(model), dummy,
        str(MODEL_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["logits", "red_logits", "triplet_logits", "seq_logits", "pair_logits"],
        dynamic_axes={
            "features":        {0: "batch_size"},
            "logits":          {0: "batch_size"},
            "red_logits":      {0: "batch_size"},
            "triplet_logits":  {0: "batch_size"},
            "seq_logits":      {0: "batch_size"},
            "pair_logits":     {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
