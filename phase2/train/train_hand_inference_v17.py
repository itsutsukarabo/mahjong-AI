"""
手牌類推モデル v17

v16からの変更点:
  - 字牌×聴牌制約 loss を削除（LAMBDA_HONOR=1.0 が NLL 最適化を阻害したため）
  - 自己回帰デコード: 並列予測ヘッド → Causal Transformer Decoder (2層)
      学習時: teacher forcing（シフト済みラベルを入力）
      推論時: greedy autoregressive（34ステップ逐次生成）
  - 牌間 attention・DP デコードは v16 から継承
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
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v17"

CONFIG = {
    "input_dim":           374,
    "d_model":             256,
    "nhead":               4,
    "num_layers":          3,
    "tile_attn_nhead":     4,
    "dec_layers":          2,
    "n_pai":               34,
    "n_count_cls":         5,
    "n_players":           3,
    "dropout":             0.1,
    "lr":                  1e-3,
    "weight_decay":        1e-4,
    "batch_size":          256,
    "epochs":              200,
    "early_stop_patience": 7,
}

VISIBLE_OFFSET     = 185
REMAINING_OFFSET   = 94
GPU_TEMP_THRESHOLD = 65
GPU_COOL_INTERVAL  = 20

LAMBDA_SUM         = 0.05
LAMBDA_CROSS       = 0.1
LAMBDA_RED_CE      = 0.3
LAMBDA_RED_CONS    = 0.1
GRAD_CLIP_NORM     = 1.0

REMAINING_BUCKETS = {
    "early":   (51, 9999),
    "mid":     (31,   50),
    "late":    (11,   30),
    "endgame": ( 0,   10),
}


# ---- 制約付きソフトマックス (EAE計算用) ----

def find_lambda(logits, k_float, n_iter=50):
    count_vals = torch.arange(5, device=logits.device, dtype=torch.float32)
    B = logits.shape[0]
    with torch.no_grad():
        lo = torch.full((B,), -20.0, device=logits.device)
        hi = torch.full((B,),  20.0, device=logits.device)
        for _ in range(n_iter):
            mid   = (lo + hi) / 2
            adj   = logits - mid.view(B, 1, 1) * count_vals
            probs = F.softmax(adj, dim=-1)
            e_sum = (probs * count_vals).sum(-1).sum(-1)
            lo    = torch.where(e_sum > k_float, mid, lo)
            hi    = torch.where(e_sum > k_float, hi,  mid)
    return (lo + hi) / 2


def constrained_softmax_probs(logits, k):
    count_vals = torch.arange(5, device=logits.device, dtype=torch.float32)
    lam = find_lambda(logits, k.float())
    adj = logits - lam.view(-1, 1, 1) * count_vals
    return F.softmax(adj, dim=-1)


# ---- DP デコード ----

@torch.no_grad()
def dp_decode(logits, k):
    """
    合計枚数制約付き最適手牌割り当て。
    logits: (B, 34, 5)
    k:      (B,) — 手牌合計枚数
    戻り値: (B, 34) int64  sum(dim=-1) == k が保証される
    """
    B, N, C = logits.shape
    K_max = int(k.max().item())
    NEG_INF = -1e9

    log_probs = F.log_softmax(logits, dim=-1).cpu()
    k_cpu = k.cpu()

    dp      = torch.full((B, K_max + 1), NEG_INF)
    dp[:, 0] = 0.0
    choices  = torch.zeros(N, B, K_max + 1, dtype=torch.long)

    for i in range(N):
        new_dp = torch.full_like(dp, NEG_INF)
        for c in range(C):
            jf_max = K_max - c
            if jf_max < 0:
                continue
            val = dp[:, :jf_max + 1] + log_probs[:, i, c].unsqueeze(-1)
            sl  = slice(c, c + jf_max + 1)
            better = val > new_dp[:, sl]
            new_dp[:, sl]     = torch.where(better, val,                       new_dp[:, sl])
            choices[i, :, sl] = torch.where(better, torch.tensor(c, dtype=torch.long), choices[i, :, sl])
        dp = new_dp

    preds     = torch.zeros(B, N, dtype=torch.long)
    remaining = k_cpu.clone()
    b_idx     = torch.arange(B)
    for i in range(N - 1, -1, -1):
        c = choices[i, b_idx, remaining]
        preds[:, i] = c
        remaining  -= c

    return preds


# ---- ユーティリティ ----

def get_stage_weights(features):
    remaining = (features[:, REMAINING_OFFSET] * 70).round().long()
    w = torch.ones(len(features), device=features.device)
    w[remaining <= 10] = 4.0
    w[(remaining > 10) & (remaining <= 30)] = 3.0
    w[(remaining > 30) & (remaining <= 50)] = 2.0
    return w


def weighted_eae(probs, labels, weights):
    count_vals     = torch.arange(probs.shape[-1], device=probs.device, dtype=torch.float32)
    abs_diff       = (count_vals - labels.unsqueeze(-1).float()).abs()
    eae_per_sample = (abs_diff * probs).sum(-1).sum(-1)
    return (eae_per_sample * weights).sum() / weights.sum()


# ---- GPU温度 ----

def gpu_temp():
    candidates = ["nvidia-smi"]
    if sys.platform == "win32":
        candidates += [r"C:\Windows\System32\nvidia-smi.exe",
                       r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
    else:
        candidates += ["/usr/lib/wsl/lib/nvidia-smi",
                       "/mnt/c/Windows/System32/nvidia-smi.exe"]
    for cmd in candidates:
        try:
            r = subprocess.run([cmd, "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
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
    def __init__(self, features, labels, labels_red):
        self.features   = torch.as_tensor(features,   dtype=torch.float32)
        self.labels     = torch.as_tensor(labels,     dtype=torch.long).clamp(0, 4)
        self.labels_red = torch.as_tensor(labels_red, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.labels_red[idx]


# ---- モデル ----

class HandInferenceV17(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, tile_attn_nhead,
                 dec_layers, n_pai=34, n_count_cls=5, n_players=3, dropout=0.1):
        super().__init__()
        self.n_pai       = n_pai
        self.n_count_cls = n_count_cls
        self.n_players   = n_players

        # グローバルエンコーダ（プレイヤー間で重み共有）
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

        # 埋め込み
        self.tile_embed   = nn.Embedding(n_pai,     d_model)
        self.player_embed = nn.Embedding(n_players, d_model)
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        # Transformer（102トークン: 34 × 3プレイヤー）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 牌間 self-attention（v16 から継承: Encoder memory 強化）
        self.tile_attn = nn.MultiheadAttention(
            d_model, tile_attn_nhead, dropout=dropout, batch_first=True
        )
        self.tile_norm = nn.LayerNorm(d_model)

        # 自己回帰デコーダ
        self.count_embed = nn.Embedding(n_count_cls, d_model)
        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.causal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # 出力ヘッド（プレイヤー間で重み共有）
        self.head     = nn.Linear(d_model, n_count_cls)
        self.red_head = nn.Linear(d_model, 2)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("player_ids",   torch.arange(n_players))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))

    def _encode(self, x):
        """Encoder → memory (B*P, 34, d_model)"""
        B, P, F = x.shape
        g = self.global_encoder(x.reshape(B * P, F)).reshape(B, P, -1)

        tile_emb   = self.tile_embed(self.tile_ids)
        player_emb = self.player_embed(self.player_ids)
        vis        = x[:, :, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai]
        vis_emb    = self.visible_proj(vis.unsqueeze(-1))

        tokens = (g.unsqueeze(2)
                + tile_emb
                + player_emb.view(1, P, 1, -1)
                + vis_emb)
        tokens = tokens.reshape(B, P * self.n_pai, -1)

        out      = self.transformer(tokens).reshape(B, P, self.n_pai, -1)
        out_flat = out.reshape(B * P, self.n_pai, -1)
        attn_out, _ = self.tile_attn(out_flat, out_flat, out_flat)
        memory   = self.tile_norm(out_flat + attn_out)  # (B*P, 34, d_model)

        return memory, out, vis

    def _vis_mask(self, vis):
        """可視枚数から count 上限マスクを生成 → (B, P, 34, 5) bool"""
        vis_raw          = (vis * 4).round().long().clamp(0, 4)
        hidden_remaining = (4 - vis_raw).clamp(min=0)
        return self.count_range > hidden_remaining.unsqueeze(-1)

    def forward(self, x, labels=None):
        """
        labels: (B, P, 34) int — Noneのとき greedy 自己回帰推論
        戻り値: (logits_masked, logits_raw, red_logits)
          logits_masked: 可視マスク適用済み (B, P, 34, 5)
          logits_raw:    マスクなし        (B, P, 34, 5)
        """
        B, P, _ = x.shape
        memory, out_4d, vis = self._encode(x)
        mask = self._vis_mask(vis)  # (B, P, 34, 5) bool

        if labels is not None:
            # 学習: teacher forcing
            lbl = labels.reshape(B * P, self.n_pai)  # (B*P, 34)
            bos = self.bos.expand(B * P, 1, -1)
            tgt = torch.cat([bos, self.count_embed(lbl[:, :-1])], dim=1)  # (B*P, 34, d)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                self.n_pai, device=x.device
            )
            dec_out    = self.causal_decoder(tgt, memory, tgt_mask=causal_mask)
            logits_raw = self.head(dec_out).reshape(B, P, self.n_pai, -1)
        else:
            # 推論: greedy 自己回帰
            logits_raw = self._greedy_decode(memory, mask, B, P, x.device)

        logits = logits_raw.masked_fill(mask, float('-inf'))

        red_tokens = out_4d[:, :, self.red_tile_idx, :]
        red_logits = self.red_head(red_tokens)

        return logits, logits_raw, red_logits

    def _greedy_decode(self, memory, mask, B, P, device):
        """Greedy 自己回帰デコード。mask: (B, P, 34, 5)"""
        BP        = B * P
        mask_flat = mask.reshape(BP, self.n_pai, self.n_count_cls)  # (B*P, 34, 5)
        cur_seq   = self.bos.expand(BP, 1, -1).clone()
        all_logits = []

        for step in range(self.n_pai):
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                step + 1, device=device
            )
            dec_out = self.causal_decoder(cur_seq, memory, tgt_mask=causal_mask)
            logit   = self.head(dec_out[:, -1:, :])  # (B*P, 1, 5)

            # 可視マスクを適用して greedy 選択（不可能な枚数を除外）
            logit_masked = logit.masked_fill(mask_flat[:, step:step + 1, :], float('-inf'))
            all_logits.append(logit_masked)

            pred     = logit_masked.argmax(dim=-1)       # (B*P, 1)
            next_emb = self.count_embed(pred)             # (B*P, 1, d)
            cur_seq  = torch.cat([cur_seq, next_emb], dim=1)

        return torch.cat(all_logits, dim=1).reshape(B, P, self.n_pai, self.n_count_cls)


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_ce   = 0.0
    count_vals = torch.arange(model.n_count_cls, device=device, dtype=torch.float32)
    five_idx   = model.red_tile_idx

    for features, labels, labels_red in loader:
        features   = features.to(device)
        labels     = labels.to(device)
        labels_red = labels_red.to(device)
        optimizer.zero_grad()

        # teacher forcing で forward
        _, logits_raw, red_logits = model(features, labels=labels)

        # ① NLL
        loss_nll = F.cross_entropy(logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1))

        # ② MSE 合計制約
        probs    = F.softmax(logits_raw, dim=-1)
        pred_sum = (probs * count_vals).sum(-1)
        loss_sum = F.mse_loss(pred_sum.sum(-1), labels.float().sum(-1))

        # ③ タイル横断制約（3人合計 ≤ 残り枚数）
        pred_tile_total = pred_sum.sum(dim=1)
        visible_counts  = features[:, 0, VISIBLE_OFFSET:VISIBLE_OFFSET + model.n_pai] * 4
        max_hidden      = (4 - visible_counts).clamp(min=0)
        loss_cross      = F.relu(pred_tile_total - max_hidden).mean()

        # ④ 赤牌損失
        loss_red_ce = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))

        # ⑤ 赤牌整合性制約
        prob_has_red  = F.softmax(red_logits, dim=-1)[:, :, :, 1]
        prob_cnt_ge1  = 1 - probs[:, :, five_idx, 0]
        loss_red_cons = F.relu(prob_has_red - prob_cnt_ge1).mean()

        loss = (loss_nll
                + LAMBDA_SUM      * loss_sum
                + LAMBDA_CROSS    * loss_cross
                + LAMBDA_RED_CE   * loss_red_ce
                + LAMBDA_RED_CONS * loss_red_cons)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_ce += loss_nll.item() * len(features)

    return total_ce / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    """val_eae (constrained softmax) と val_acc (DP decode) を返す。"""
    model.eval()
    total_wsum = 0.0
    total_w    = 0.0
    correct    = 0
    total      = 0

    for features, labels, labels_red in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _ = model(features)  # greedy 自己回帰

        for p in range(3):
            k_p     = labels[:, p].float().sum(dim=-1).long()
            probs_p = constrained_softmax_probs(logits[:, p], k_p)
            stage_w = get_stage_weights(features[:, p])

            eae_val    = weighted_eae(probs_p, labels[:, p], stage_w)
            total_wsum += eae_val.item() * stage_w.sum().item()
            total_w    += stage_w.sum().item()

            preds_p = dp_decode(logits[:, p], k_p).to(device)
            correct += (preds_p == labels[:, p]).sum().item()
            total   += labels[:, p].numel()

    val_eae = total_wsum / total_w if total_w > 0 else float('inf')
    return val_eae, correct / total


@torch.no_grad()
def eval_by_remaining(model, loader, device):
    model.eval()
    count_vals = torch.arange(5, device=device, dtype=torch.float32)
    buckets = {k: {"correct": 0, "total": 0, "eae_sum": 0.0, "n_samples": 0}
               for k in REMAINING_BUCKETS}

    for features, labels, labels_red in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _ = model(features)
        remaining_vals = (features[:, 0, REMAINING_OFFSET] * 70).round().long().cpu()

        for p in range(3):
            k_vals = labels[:, p].float().sum(dim=-1).long()
            probs  = constrained_softmax_probs(logits[:, p], k_vals)
            preds  = dp_decode(logits[:, p], k_vals)
            correct_mask   = (preds == labels[:, p].cpu())
            abs_diff       = (count_vals.cpu() - labels[:, p].cpu().unsqueeze(-1).float()).abs()
            eae_per_sample = (abs_diff * probs.cpu()).sum(-1).sum(-1)

            for i, rem in enumerate(remaining_vals):
                rem = rem.item()
                for name, (lo, hi) in REMAINING_BUCKETS.items():
                    if lo <= rem <= hi:
                        b = buckets[name]
                        b["correct"]   += correct_mask[i].sum().item()
                        b["total"]     += correct_mask[i].numel()
                        b["eae_sum"]   += eae_per_sample[i].item()
                        b["n_samples"] += 1
                        break

    result = {}
    for name, b in buckets.items():
        n = b["n_samples"]
        result[name] = {
            "acc": b["correct"] / b["total"] if b["total"] > 0 else 0.0,
            "eae": b["eae_sum"] / n          if n > 0 else 0.0,
            "n":   n,
        }
    return result


@torch.no_grad()
def eval_metrics_detailed(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    count_vals = torch.arange(5, device=device, dtype=torch.float32)
    total_wsum = 0.0
    total_w    = 0.0

    for features, labels, labels_red in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _ = model(features)

        for p in range(3):
            k      = labels[:, p].float().sum(dim=-1).long()
            probs  = constrained_softmax_probs(logits[:, p], k)
            preds  = dp_decode(logits[:, p], k)
            all_preds.append(preds)
            all_labels.append(labels[:, p].cpu())

            stage_w    = get_stage_weights(features[:, p])
            abs_diff   = (count_vals - labels[:, p].unsqueeze(-1).float()).abs()
            eae_s      = (abs_diff * probs).sum(-1).sum(-1)
            total_wsum += (eae_s * stage_w).sum().item()
            total_w    += stage_w.sum().item()

    preds  = torch.cat(all_preds,  dim=0)
    labels = torch.cat(all_labels, dim=0)

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
        "weighted_eae":      round(weighted_eae_v, 4),
        "recall_nonzero":    round(recall,    4),
        "precision_nonzero": round(precision, 4),
        "f1_nonzero":        round(f1,        4),
        "mae_nonzero":       round(mae_nz,    4),
        "hand_exact_acc":    round(exact_acc, 4),
        "pred_total_mae":    round(pred_total_mae, 4),
    }


def main():
    resume = "--resume" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  resume: {resume}")

    ndjson_path = DATA_DIR / "hand_inference_v15.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}")
        sys.exit(1)
    print(f"読み込み中: {ndjson_path}", flush=True)

    _feat, _lab, _lred = [], [], []
    with open(ndjson_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            _feat.append(s["features"])
            _lab.append(s["label_hand"])
            _lred.append(s["label_red"])

    feat_np = np.array(_feat, dtype=np.float32); del _feat
    lab_np  = np.array(_lab,  dtype=np.int64);   del _lab
    lred_np = np.array(_lred, dtype=np.int64);   del _lred
    gc.collect()
    print(f"総サンプル数: {len(feat_np)}, shape: {feat_np.shape}")

    if feat_np.shape[1:] != (3, CONFIG["input_dim"]):
        print(f"次元数不一致: expected (3, {CONFIG['input_dim']}), got {feat_np.shape[1:]}")
        sys.exit(1)

    before   = len(feat_np)
    lab_sum  = lab_np.sum(axis=-1)
    meld_sum = (feat_np[:, :, 78] + feat_np[:, :, 79] + feat_np[:, :, 80])
    keep     = (lab_sum.sum(axis=1) > 0) | (meld_sum.sum(axis=1) > 0)
    feat_np  = feat_np[keep]; lab_np = lab_np[keep]; lred_np = lred_np[keep]
    print(f"ノイズ除外: {before - len(feat_np)} samples → {len(feat_np)}")

    rng     = np.random.default_rng(42)
    idx     = rng.permutation(len(feat_np))
    feat_np = feat_np[idx]; lab_np = lab_np[idx]; lred_np = lred_np[idx]

    n       = len(feat_np)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    sl_tr = slice(None, n_train)
    sl_va = slice(n_train, n_train + n_val)
    sl_te = slice(n_train + n_val, None)

    train_dataset = HandInferenceDataset(feat_np[sl_tr], lab_np[sl_tr], lred_np[sl_tr])
    val_dataset   = HandInferenceDataset(feat_np[sl_va], lab_np[sl_va], lred_np[sl_va])
    test_dataset  = HandInferenceDataset(feat_np[sl_te], lab_np[sl_te], lred_np[sl_te])
    del feat_np, lab_np, lred_np; gc.collect()
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    use_gpu      = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=use_gpu, persistent_workers=True)

    model = HandInferenceV17(
        input_dim      = CONFIG["input_dim"],
        d_model        = CONFIG["d_model"],
        nhead          = CONFIG["nhead"],
        num_layers     = CONFIG["num_layers"],
        tile_attn_nhead= CONFIG["tile_attn_nhead"],
        dec_layers     = CONFIG["dec_layers"],
        n_pai          = CONFIG["n_pai"],
        n_count_cls    = CONFIG["n_count_cls"],
        n_players      = CONFIG["n_players"],
        dropout        = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_eae = math.inf
    patience_cnt = 0
    start_epoch  = 1
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if resume and (MODEL_DIR / "model.pt").exists() and (MODEL_DIR / "train_log.json").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
        logs = [json.loads(l) for l in (MODEL_DIR / "train_log.json").read_text().splitlines() if l.strip()]
        best_val_eae = min(e["val_eae"] for e in logs)
        last_epoch   = max(e["epoch"]   for e in logs)
        patience_cnt = 0
        for e in reversed(logs):
            if e["val_eae"] <= best_val_eae:
                break
            patience_cnt += 1
        start_epoch = last_epoch + 1
        print(f"resume: epoch {start_epoch} から再開  best_val_eae={best_val_eae:.4f}  patience={patience_cnt}",
              flush=True)

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_nll        = train_epoch(model, train_loader, optimizer, device)
        val_eae, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step(val_eae)
        print(f"epoch {epoch:3d}  train_nll={train_nll:.4f}  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}",
              flush=True)
        log_entry = {"epoch": epoch, "train_nll": train_nll, "val_eae": val_eae, "val_acc": val_acc}
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
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
