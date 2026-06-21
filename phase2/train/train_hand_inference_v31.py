"""
手牌類推モデル v31: v29 + wait_logits(34) 直接出力ヘッド追加

v29からの変更点:
  - wait_head 追加: per-tile で P(待ち牌) を直接出力 (34次元)
  - 損失 loss_wait: テンパイサンプル(shanten=0)のみ BCE で学習
      真の待ち牌ラベルは label_hand + compute_shanten でオンザフライ計算
  - 評価指標 (毎epoch):
      wait_f1        : F1 スコア (threshold=0.3)
      wait_hit_rate  : ≥1 待ち牌正解率 (直感的な "何割当てた" 指標)
      wait_top1_acc  : top-1 予測が真の待ち牌である割合
  - best_wait モデルを model_best_wait.pt に別途保存
  - ONNX 出力に wait_logits(1,3,34) を追加
  - v29 チェックポイントから学習開始
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

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from enumerate_decompositions import (
    build_block_selection_matrix, N_BLOCKS_WITH_TATSU as N_BLOCKS,
    compute_shanten,
)

# ---- 設定 ----

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v31"
V29_DIR   = Path(__file__).parent.parent / "models" / "hand_inference" / "v29"

CONFIG = {
    "input_dim":    442,
    "d_model":      256,
    "nhead":        4,
    "num_layers":   3,
    "n_pai":        34,
    "n_count_cls":  5,
    "n_players":    3,
    "n_blocks":     N_BLOCKS,  # 134
    "dropout":      0.1,
    "lr":           5e-4,
    "weight_decay": 1e-4,
    "batch_size":   256,
    "epochs":       200,
    "early_stop_patience": 10,
}

VISIBLE_OFFSET     = 185
REMAINING_OFFSET   = 94
GPU_TEMP_THRESHOLD = 78
GPU_COOL_INTERVAL  = 20
N_PAI              = 34

LAMBDA_SUM         = 0.05
LAMBDA_CROSS       = 0.1
LAMBDA_RED_CE      = 0.3
LAMBDA_RED_CONS    = 0.1
LAMBDA_BLOCK       = 0.3
LAMBDA_WAIT        = 0.5   # wait_logits 損失の重み
POS_WEIGHT_MAX     = 10.0
GRAD_CLIP_NORM     = 1.0

WAIT_EVAL_THRESH   = 0.3   # wait_f1 / hit_rate 計算時の閾値

REMAINING_BUCKETS = {
    "early":   (51, 9999),
    "mid":     (31,   50),
    "late":    (11,   30),
    "endgame": ( 0,   10),
}


# ---- 待ち牌ラベル生成 (オンザフライ) ----

def compute_true_waits(counts34: list, n_melds: int) -> list:
    waits = []
    for t in range(N_PAI):
        c = list(counts34)
        if c[t] >= 4:
            continue
        c[t] += 1
        if compute_shanten(c, n_melds) == -1:
            waits.append(t)
    return waits


def make_wait_labels(labels_hand_np: np.ndarray, labels_shanten_np: np.ndarray) -> np.ndarray:
    """
    labels_hand_np:    (B, P, 34)  int  各牌の枚数
    labels_shanten_np: (B, P)      int
    Returns: (B, P, 34) float  真の待ち牌フラグ (shanten!=0 は全ゼロ)
    """
    B, P, _ = labels_hand_np.shape
    wait_labels = np.zeros((B, P, N_PAI), dtype=np.float32)
    for b in range(B):
        for p in range(P):
            if labels_shanten_np[b, p] != 0:
                continue
            counts34 = labels_hand_np[b, p].tolist()
            total    = int(sum(counts34))
            if total == 0 or total % 3 != 1:
                continue
            n_melds = (13 - total) // 3
            for t in compute_true_waits(counts34, n_melds):
                wait_labels[b, p, t] = 1.0
    return wait_labels


# ---- 評価指標 ----

@torch.no_grad()
def eval_wait_metrics(model, loader, device, threshold=WAIT_EVAL_THRESH):
    """wait_logits から直接待ち牌指標を計算 (毎 epoch 実行)"""
    model.eval()
    prec_list = []; recall_list = []; f1_list = []
    hit_list  = []; top1_list   = []

    for features, labels, _, _, labels_shanten in loader:
        features = features.to(device)
        _, _, _, _, wait_logits = model(features)
        wait_probs = torch.sigmoid(wait_logits).cpu().numpy()  # (B, P, 34)
        hand_np    = labels.numpy()                             # (B, P, 34)
        sh_np      = labels_shanten.numpy()                     # (B, P)

        for b in range(len(features)):
            for p in range(3):
                if sh_np[b, p] != 0:
                    continue
                counts34 = hand_np[b, p].tolist()
                total    = int(sum(counts34))
                if total == 0 or total % 3 != 1:
                    continue
                n_melds    = (13 - total) // 3
                true_waits = set(compute_true_waits(counts34, n_melds))
                if not true_waits:
                    continue

                probs      = wait_probs[b, p]
                pred_waits = {t for t in range(N_PAI) if probs[t] >= threshold}
                top1       = int(np.argmax(probs))

                tp = len(pred_waits & true_waits)
                fp = len(pred_waits - true_waits)
                fn = len(true_waits - pred_waits)
                prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1     = 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0.0

                prec_list.append(prec);  recall_list.append(recall)
                f1_list.append(f1)
                hit_list.append(1.0 if tp > 0 else 0.0)
                top1_list.append(1.0 if top1 in true_waits else 0.0)

    if not f1_list:
        return {"wait_f1": 0.0, "wait_prec": 0.0, "wait_recall": 0.0,
                "wait_hit_rate": 0.0, "wait_top1_acc": 0.0, "n_tenpai": 0}
    return {
        "wait_f1":       round(float(np.mean(f1_list)),      4),
        "wait_prec":     round(float(np.mean(prec_list)),    4),
        "wait_recall":   round(float(np.mean(recall_list)),  4),
        "wait_hit_rate": round(float(np.mean(hit_list)),     4),
        "wait_top1_acc": round(float(np.mean(top1_list)),    4),
        "n_tenpai":      len(f1_list),
    }


# ---- 制約付きソフトマックス ----

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


def get_stage_weights(features):
    remaining = (features[:, REMAINING_OFFSET] * 70).round()
    meld_tiles = (features[:, 78] * 3 + features[:, 79] * 3 + features[:, 80] * 4)
    effective_remaining = (remaining - meld_tiles * 2).clamp(min=0).long()
    w = torch.ones(len(features), device=features.device)
    w[effective_remaining <= 10] = 4.0
    w[(effective_remaining > 10) & (effective_remaining <= 30)] = 3.0
    w[(effective_remaining > 30) & (effective_remaining <= 50)] = 2.0
    return w


def weighted_eae(probs, labels, weights):
    count_vals     = torch.arange(probs.shape[-1], device=probs.device, dtype=torch.float32)
    abs_diff       = (count_vals - labels.unsqueeze(-1).float()).abs()
    eae_per_sample = (abs_diff * probs).sum(-1).sum(-1)
    return (eae_per_sample * weights).sum() / weights.sum()


# ---- GPU温度 ----

def gpu_temp():
    candidates = ["nvidia-smi",
                  "/usr/lib/wsl/lib/nvidia-smi",
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
    def __init__(self, features, labels, labels_red, labels_block, labels_shanten):
        self.features       = torch.as_tensor(features,       dtype=torch.float32)
        self.labels         = torch.as_tensor(labels,         dtype=torch.long).clamp(0, 4)
        self.labels_red     = torch.as_tensor(labels_red,     dtype=torch.long)
        self.labels_block   = torch.as_tensor(labels_block,   dtype=torch.float32)
        self.labels_shanten = torch.as_tensor(labels_shanten, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx], self.labels_red[idx],
                self.labels_block[idx], self.labels_shanten[idx])


# ---- モデル ----

class HandInferenceV31(nn.Module):
    """v29 + wait_head: P(牌t が待ち牌) を直接出力"""

    def __init__(self, input_dim, d_model, nhead, num_layers,
                 n_pai=34, n_count_cls=5, n_players=3, n_blocks=134, dropout=0.1):
        super().__init__()
        self.n_pai       = n_pai
        self.n_count_cls = n_count_cls
        self.n_players   = n_players
        self.n_blocks    = n_blocks

        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )
        self.tile_embed   = nn.Embedding(n_pai,     d_model)
        self.player_embed = nn.Embedding(n_players, d_model)
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head       = nn.Linear(d_model, n_count_cls)
        self.red_head   = nn.Linear(d_model, 2)
        self.block_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        # v31 新規: 待ち牌直接予測ヘッド
        self.wait_head  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("player_ids",   torch.arange(n_players))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))
        block_sel = torch.tensor(
            build_block_selection_matrix(include_tatsu=True), dtype=torch.float32
        )
        self.register_buffer("block_sel", block_sel)

    def forward(self, x):
        B, P, F = x.shape
        g = self.global_encoder(x.reshape(B * P, F)).reshape(B, P, -1)

        tile_emb   = self.tile_embed(self.tile_ids)
        player_emb = self.player_embed(self.player_ids)
        vis        = x[:, :, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai]
        vis_emb    = self.visible_proj(vis.unsqueeze(-1))

        tokens = (g.unsqueeze(2) + tile_emb + player_emb.view(1, P, 1, -1) + vis_emb)
        tokens = tokens.reshape(B, P * self.n_pai, -1)
        out    = self.transformer(tokens).reshape(B, P, self.n_pai, -1)  # (B,P,34,d)

        logits_raw = self.head(out)
        vis_raw    = (vis * 4).round().long().clamp(0, 4)
        mask       = self.count_range > (4 - vis_raw).clamp(min=0).unsqueeze(-1)
        logits     = logits_raw.masked_fill(mask, float('-inf'))

        red_tokens   = out[:, :, self.red_tile_idx, :]
        red_logits   = self.red_head(red_tokens)

        out_flat     = out.reshape(B * P, self.n_pai, -1)
        block_feats  = torch.bmm(
            self.block_sel.unsqueeze(0).expand(B * P, -1, -1), out_flat
        ).reshape(B, P, self.n_blocks, -1)
        block_logits = self.block_head(block_feats).squeeze(-1)  # (B, P, 134)

        wait_logits  = self.wait_head(out).squeeze(-1)           # (B, P, 34)

        return logits, logits_raw, red_logits, block_logits, wait_logits


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device, pos_weight_block, pos_weight_wait):
    model.train()
    total_ce   = 0.0
    count_vals = torch.arange(model.n_count_cls, device=device, dtype=torch.float32)
    five_idx   = model.red_tile_idx

    for features, labels, labels_red, labels_block, labels_shanten in loader:
        features       = features.to(device)
        labels         = labels.to(device)
        labels_red     = labels_red.to(device)
        labels_block   = labels_block.to(device)
        labels_shanten = labels_shanten.to(device)
        optimizer.zero_grad()

        _, logits_raw, red_logits, block_logits, wait_logits = model(features)
        B, P = features.shape[:2]

        # ---- NLL (tile count) ----
        loss_nll_per = F.cross_entropy(
            logits_raw.reshape(-1, model.n_count_cls),
            labels.reshape(-1), reduction='none',
        ).reshape(B, P, -1).mean(-1)
        stage_w  = torch.stack(
            [get_stage_weights(features[:, p]) for p in range(model.n_players)], dim=1
        )
        loss_nll = (loss_nll_per * stage_w).sum() / stage_w.sum()

        # ---- sum / cross constraints ----
        probs    = F.softmax(logits_raw, dim=-1)
        pred_sum = (probs * count_vals).sum(-1)
        loss_sum = F.mse_loss(pred_sum.sum(-1), labels.float().sum(-1))

        pred_tile_total = pred_sum.sum(dim=1)
        visible_counts  = features[:, 0, VISIBLE_OFFSET:VISIBLE_OFFSET + model.n_pai] * 4
        max_hidden      = (4 - visible_counts).clamp(min=0)
        loss_cross      = F.relu(pred_tile_total - max_hidden).mean()

        # ---- red tile ----
        loss_red_ce   = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))
        prob_has_red  = F.softmax(red_logits, dim=-1)[:, :, :, 1]
        prob_cnt_ge1  = 1 - probs[:, :, five_idx, 0]
        loss_red_cons = F.relu(prob_has_red - prob_cnt_ge1).mean()

        # ---- block (shanten<=1 のみ) ----
        block_w = torch.where(
            labels_shanten <= 1,
            (0.5 ** labels_shanten.float()),
            torch.zeros_like(labels_shanten.float()),
        )
        loss_block_raw = F.binary_cross_entropy_with_logits(
            block_logits.reshape(-1, model.n_blocks),
            labels_block.reshape(-1, model.n_blocks),
            pos_weight=pos_weight_block, reduction='none',
        ).mean(dim=-1)
        loss_block = (loss_block_raw * block_w.reshape(-1)).mean()

        # ---- wait (shanten==0 のみ) ----
        wait_mask = (labels_shanten == 0).float()  # (B, P)
        if wait_mask.sum() > 0:
            # 真の待ち牌ラベルをオンザフライ計算 (CPU)
            wait_labels_np = make_wait_labels(
                labels.cpu().numpy(), labels_shanten.cpu().numpy()
            )
            wait_labels = torch.tensor(wait_labels_np, device=device)  # (B, P, 34)
            loss_wait_raw = F.binary_cross_entropy_with_logits(
                wait_logits.reshape(-1, model.n_pai),
                wait_labels.reshape(-1, model.n_pai),
                pos_weight=pos_weight_wait, reduction='none',
            ).mean(dim=-1)  # (B*P,)
            loss_wait = (loss_wait_raw * wait_mask.reshape(-1)).sum() / (wait_mask.sum() * model.n_pai)
        else:
            loss_wait = torch.tensor(0.0, device=device)

        loss = (loss_nll
                + LAMBDA_SUM      * loss_sum
                + LAMBDA_CROSS    * loss_cross
                + LAMBDA_RED_CE   * loss_red_ce
                + LAMBDA_RED_CONS * loss_red_cons
                + LAMBDA_BLOCK    * loss_block
                + LAMBDA_WAIT     * loss_wait)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_ce += loss_nll.item() * len(features)

    return total_ce / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_wsum = 0.0
    total_w    = 0.0
    correct    = 0
    total      = 0

    for features, labels, _, _, _ in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _, _, _ = model(features)

        for p in range(3):
            k_p     = labels[:, p].float().sum(dim=-1).long()
            probs_p = constrained_softmax_probs(logits[:, p], k_p)
            stage_w = get_stage_weights(features[:, p])
            eae_val = weighted_eae(probs_p, labels[:, p], stage_w)
            total_wsum += eae_val.item() * stage_w.sum().item()
            total_w    += stage_w.sum().item()
            preds_p = probs_p.argmax(dim=-1)
            correct += (preds_p == labels[:, p]).sum().item()
            total   += labels[:, p].numel()

    val_eae = total_wsum / total_w if total_w > 0 else float('inf')
    return val_eae, correct / total


def main():
    resume = "--resume" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  resume: {resume}", flush=True)

    ndjson_path = DATA_DIR / "hand_inference_v26.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}"); sys.exit(1)
    print(f"読み込み中: {ndjson_path}", flush=True)

    n_lines   = sum(1 for ln in open(ndjson_path, encoding="utf-8") if ln.strip())
    INPUT_DIM = CONFIG["input_dim"]
    feat_np   = np.empty((n_lines, 3, INPUT_DIM), dtype=np.float32)
    lab_np    = np.empty((n_lines, 3, 34),         dtype=np.int64)
    lred_np   = np.empty((n_lines, 3, 3),           dtype=np.int64)
    lblk_np   = np.empty((n_lines, 3, N_BLOCKS),    dtype=np.float32)
    lsh_np    = np.empty((n_lines, 3),              dtype=np.int64)
    print(f"総サンプル数: {n_lines}", flush=True)

    with open(ndjson_path, encoding="utf-8") as f:
        i = 0
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            feat_np[i] = s["features"]
            lab_np[i]  = s["label_hand"]
            lred_np[i] = s["label_red"]
            lblk_np[i] = s["label_block"]
            lsh_np[i]  = s["label_shanten"]
            i += 1
            if i % 50000 == 0: print(f"  {i}/{n_lines}", flush=True)

    pos_rate_block = float(lblk_np.mean())
    pw_block = min((1 - pos_rate_block) / (pos_rate_block + 1e-9), POS_WEIGHT_MAX)
    pos_weight_block = torch.full((N_BLOCKS,), pw_block, device=device)
    print(f"block pos_rate={pos_rate_block:.4f}  pos_weight={pw_block:.2f}")

    # wait の pos_weight: テンパイサンプルの待ち牌比率から推定
    # 典型的に 1-8 待ち / 34 牌 ≈ 10% → pos_weight ≈ 9
    pos_rate_wait = 0.10
    pw_wait = min((1 - pos_rate_wait) / pos_rate_wait, POS_WEIGHT_MAX)
    pos_weight_wait = torch.full((N_PAI,), pw_wait, device=device)
    print(f"wait  pos_rate≈{pos_rate_wait:.2f}  pos_weight={pw_wait:.2f}")

    before   = len(feat_np)
    lab_sum  = lab_np.sum(axis=-1)
    meld_sum = feat_np[:, :, 78] + feat_np[:, :, 79] + feat_np[:, :, 80]
    keep     = (lab_sum.sum(axis=1) > 0) | (meld_sum.sum(axis=1) > 0)
    feat_np  = feat_np[keep]; lab_np  = lab_np[keep]
    lred_np  = lred_np[keep]; lblk_np = lblk_np[keep]; lsh_np = lsh_np[keep]
    print(f"ノイズ除外: {before - len(feat_np)} → {len(feat_np)}")

    rng      = np.random.default_rng(42)
    idx      = rng.permutation(len(feat_np))
    feat_np  = feat_np[idx]; lab_np  = lab_np[idx]
    lred_np  = lred_np[idx]; lblk_np = lblk_np[idx]; lsh_np = lsh_np[idx]

    n       = len(feat_np)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    sl_tr   = slice(None, n_train)
    sl_va   = slice(n_train, n_train + n_val)
    sl_te   = slice(n_train + n_val, None)

    def make_ds(sl):
        return HandInferenceDataset(
            feat_np[sl], lab_np[sl], lred_np[sl], lblk_np[sl], lsh_np[sl]
        )

    train_dataset = make_ds(sl_tr)
    val_dataset   = make_ds(sl_va)
    test_dataset  = make_ds(sl_te)
    del feat_np, lab_np, lred_np, lblk_np, lsh_np; gc.collect()
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    use_gpu      = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)

    model = HandInferenceV31(
        input_dim   = CONFIG["input_dim"],
        d_model     = CONFIG["d_model"],
        nhead       = CONFIG["nhead"],
        num_layers  = CONFIG["num_layers"],
        n_pai       = CONFIG["n_pai"],
        n_count_cls = CONFIG["n_count_cls"],
        n_players   = CONFIG["n_players"],
        n_blocks    = CONFIG["n_blocks"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
    best_val_eae = math.inf
    best_wait_f1 = 0.0
    patience_cnt = 0
    start_epoch  = 1

    # v29 チェックポイントから開始 (wait_head のみランダム初期化)
    if not resume:
        v29_ckpt = V29_DIR / "model.pt"
        if v29_ckpt.exists():
            state_dict = torch.load(v29_ckpt, map_location=device, weights_only=True)
            state_dict = state_dict.get("model_state_dict", state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"v29 ロード完了  missing={missing}  unexpected={unexpected}", flush=True)
        else:
            print("v29 チェックポイントなし → ランダム初期化", flush=True)

    if resume and (MODEL_DIR / "model.pt").exists() and (MODEL_DIR / "train_log.json").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=True))
        logs         = [json.loads(l) for l in (MODEL_DIR / "train_log.json").read_text().splitlines() if l.strip()]
        best_val_eae = min(e["val_eae"] for e in logs)
        best_wait_f1 = max((e.get("wait_f1", 0.0) for e in logs), default=0.0)
        last_epoch   = max(e["epoch"] for e in logs)
        patience_cnt = 0
        for e in reversed(logs):
            if e["val_eae"] <= best_val_eae: break
            patience_cnt += 1
        start_epoch = last_epoch + 1
        print(f"resume: epoch {start_epoch}  best_val_eae={best_val_eae:.4f}  best_wait_f1={best_wait_f1:.4f}", flush=True)

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_nll        = train_epoch(model, train_loader, optimizer, device, pos_weight_block, pos_weight_wait)
        val_eae, val_acc = eval_epoch(model, val_loader, device)
        wait_m           = eval_wait_metrics(model, val_loader, device)
        scheduler.step(val_eae)

        log_entry = {
            "epoch":     epoch,
            "train_nll": round(train_nll, 4),
            "val_eae":   round(val_eae,   4),
            "val_acc":   round(val_acc,   4),
            **wait_m,
        }
        print(
            f"epoch {epoch:3d}  nll={train_nll:.4f}  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}"
            f"  wait_f1={wait_m['wait_f1']:.4f}  hit={wait_m['wait_hit_rate']:.4f}"
            f"  top1={wait_m['wait_top1_acc']:.4f}",
            flush=True,
        )
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
                print("early stopping"); break

        if wait_m["wait_f1"] > best_wait_f1:
            best_wait_f1 = wait_m["wait_f1"]
            torch.save(model.state_dict(), MODEL_DIR / "model_best_wait.pt")
            print(f"  [saved best_wait: f1={best_wait_f1:.4f}  hit={wait_m['wait_hit_rate']:.4f}  top1={wait_m['wait_top1_acc']:.4f}]", flush=True)

    # ---- テスト評価 ----
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=True))
    test_eae, test_acc = eval_epoch(model, test_loader, device)
    test_wait_m        = eval_wait_metrics(model, test_loader, device)
    print(f"\n[model.pt (best val_eae)]")
    print(f"  test_eae={test_eae:.4f}  test_acc={test_acc:.4f}")
    for k, v in test_wait_m.items():
        print(f"  {k}: {v}")

    if (MODEL_DIR / "model_best_wait.pt").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "model_best_wait.pt", map_location=device, weights_only=True))
        best_wait_test = eval_wait_metrics(model, test_loader, device)
        print(f"\n[model_best_wait.pt (best wait_f1)]")
        for k, v in best_wait_test.items():
            print(f"  {k}: {v}")
        test_wait_m = best_wait_test

    eval_result = {
        "test_eae":    test_eae,
        "test_acc":    test_acc,
        **test_wait_m,
        "config":      CONFIG,
    }
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(eval_result, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    # ---- ONNX エクスポート (model_best_wait.pt を使用) ----
    model.eval().cpu()
    dummy = torch.zeros(1, 3, CONFIG["input_dim"])

    class _OnnxWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _, red_logits, block_logits, wait_logits = self.m(x)
            return logits, red_logits, block_logits, wait_logits

    torch.onnx.export(
        _OnnxWrapper(model), dummy,
        str(MODEL_DIR / "model.onnx"),
        input_names  = ["features"],
        output_names = ["logits", "red_logits", "block_logits", "wait_logits"],
        dynamic_axes = {
            "features":     {0: "batch"},
            "logits":       {0: "batch"},
            "red_logits":   {0: "batch"},
            "block_logits": {0: "batch"},
            "wait_logits":  {0: "batch"},
        },
        opset_version = 17,
    )
    print(f"\nONNX保存: {MODEL_DIR / 'model.onnx'}")


if __name__ == "__main__":
    main()
