"""
手牌類推モデル v26: v23失敗修正版 (N_BLOCKS=134, 全シャンテン数対応, 排他的バックトラッキング)

v23からの変更点:
  - ラベル生成: テンパイ時のみ → 全シャンテン数で compute_soft_labels_v2 を使用
  - 排他的バックトラッキング: 445s系の対子/ターツ競合を正確に列挙
  - シャンテン重み付き block_loss: テンパイに近いほど損失を強く反映
  - eval_by_remaining 追加: 残り牌数バケット別の精度分析 (v15と同じ)
  - データ: hand_inference_v26.ndjson (label_shanten フィールドを含む)
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
    build_block_selection_matrix, N_BLOCKS_WITH_TATSU as N_BLOCKS
)

# ---- 設定 ----

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v26"

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
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   256,
    "epochs":       200,
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
LAMBDA_BLOCK       = 0.3
POS_WEIGHT_MAX     = 10.0
GRAD_CLIP_NORM     = 1.0

REMAINING_BUCKETS = {
    "early":   (51, 9999),
    "mid":     (31,   50),
    "late":    (11,   30),
    "endgame": ( 0,   10),
}


# ---- 制約付きソフトマックス (eval用) ----

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


def dp_decode(logits, k):
    B, T, C = logits.shape
    NEG_INF = float('-inf')
    dp   = torch.full((B, T + 1, k.max().item() + 1), NEG_INF, device=logits.device)
    back = torch.zeros((B, T + 1, k.max().item() + 1), dtype=torch.long, device=logits.device)
    dp[:, 0, 0] = 0.0
    for t in range(T):
        for c in range(C):
            prev_max = k.max().item() - c
            if prev_max < 0:
                continue
            valid   = dp[:, t, :prev_max + 1]
            updated = valid + logits[:, t, c].unsqueeze(1)
            mask    = updated > dp[:, t + 1, c:prev_max + c + 1]
            dp[:, t + 1, c:prev_max + c + 1]   = torch.where(mask, updated, dp[:, t + 1, c:prev_max + c + 1])
            back[:, t + 1, c:prev_max + c + 1] = torch.where(mask, c, back[:, t + 1, c:prev_max + c + 1])
    preds = torch.zeros(B, T, dtype=torch.long, device=logits.device)
    cur   = k.clone()
    for t in range(T, 0, -1):
        chosen = back[torch.arange(B), t, cur]
        preds[:, t - 1] = chosen
        cur = cur - chosen
    return preds


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
    def __init__(self, features, labels, labels_red, labels_block, labels_shanten):
        self.features        = torch.as_tensor(features,        dtype=torch.float32)
        self.labels          = torch.as_tensor(labels,          dtype=torch.long).clamp(0, 4)
        self.labels_red      = torch.as_tensor(labels_red,      dtype=torch.long)
        self.labels_block    = torch.as_tensor(labels_block,    dtype=torch.float32)
        self.labels_shanten  = torch.as_tensor(labels_shanten,  dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx], self.labels_red[idx],
                self.labels_block[idx], self.labels_shanten[idx])


# ---- モデル ----

class HandInferenceV26(nn.Module):
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

        self.head     = nn.Linear(d_model, n_count_cls)
        self.red_head = nn.Linear(d_model, 2)

        self.block_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("player_ids",   torch.arange(n_players))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))
        block_sel = torch.tensor(
            build_block_selection_matrix(include_tatsu=True), dtype=torch.float32
        )
        self.register_buffer("block_sel", block_sel)  # (134, 34)

    def forward(self, x):
        B, P, F = x.shape

        g = self.global_encoder(x.reshape(B * P, F)).reshape(B, P, -1)

        tile_emb   = self.tile_embed(self.tile_ids)
        player_emb = self.player_embed(self.player_ids)
        vis        = x[:, :, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai]
        vis_emb    = self.visible_proj(vis.unsqueeze(-1))

        tokens = (g.unsqueeze(2) + tile_emb + player_emb.view(1, P, 1, -1) + vis_emb)
        tokens = tokens.reshape(B, P * self.n_pai, -1)

        out = self.transformer(tokens)
        out = out.reshape(B, P, self.n_pai, -1)

        logits_raw = self.head(out)
        vis_raw          = (vis * 4).round().long().clamp(0, 4)
        hidden_remaining = (4 - vis_raw).clamp(min=0)
        mask = self.count_range > hidden_remaining.unsqueeze(-1)
        logits = logits_raw.masked_fill(mask, float('-inf'))

        red_tokens = out[:, :, self.red_tile_idx, :]
        red_logits = self.red_head(red_tokens)

        out_flat    = out.reshape(B * P, self.n_pai, -1)
        block_feats = torch.bmm(
            self.block_sel.unsqueeze(0).expand(B * P, -1, -1),  # (B*P, 134, 34)
            out_flat,                                            # (B*P, 34, d)
        ).reshape(B, P, self.n_blocks, -1)                      # (B, 3, 134, d)
        block_logits = self.block_head(block_feats).squeeze(-1) # (B, 3, 134)

        return logits, logits_raw, red_logits, block_logits


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_ce = 0.0
    count_vals = torch.arange(model.n_count_cls, device=device, dtype=torch.float32)
    five_idx   = model.red_tile_idx

    for features, labels, labels_red, labels_block, labels_shanten in loader:
        features        = features.to(device)
        labels          = labels.to(device)
        labels_red      = labels_red.to(device)
        labels_block    = labels_block.to(device)
        labels_shanten  = labels_shanten.to(device)
        optimizer.zero_grad()

        _, logits_raw, red_logits, block_logits = model(features)

        B, P = features.shape[:2]

        loss_nll_per = F.cross_entropy(
            logits_raw.reshape(-1, model.n_count_cls),
            labels.reshape(-1),
            reduction='none',
        ).reshape(B, P, -1).mean(-1)

        stage_w = torch.stack(
            [get_stage_weights(features[:, p]) for p in range(model.n_players)],
            dim=1,
        )
        loss_nll = (loss_nll_per * stage_w).sum() / stage_w.sum()

        probs    = F.softmax(logits_raw, dim=-1)
        pred_sum = (probs * count_vals).sum(-1)
        loss_sum = F.mse_loss(pred_sum.sum(-1), labels.float().sum(-1))

        pred_tile_total = pred_sum.sum(dim=1)
        visible_counts  = features[:, 0, VISIBLE_OFFSET:VISIBLE_OFFSET + model.n_pai] * 4
        max_hidden      = (4 - visible_counts).clamp(min=0)
        loss_cross      = F.relu(pred_tile_total - max_hidden).mean()

        loss_red_ce = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))

        prob_has_red  = F.softmax(red_logits, dim=-1)[:, :, :, 1]
        prob_cnt_ge1  = 1 - probs[:, :, five_idx, 0]
        loss_red_cons = F.relu(prob_has_red - prob_cnt_ge1).mean()

        # シャンテン重み付き block_loss
        # shanten=0(テンパイ): 重み1.0, shanten=1: 0.5, shanten=2: 0.25 ...
        block_w = (0.5 ** labels_shanten.float()).clamp(min=0.0625)  # (B, 3)
        loss_block_raw = F.binary_cross_entropy_with_logits(
            block_logits.reshape(-1, model.n_blocks),
            labels_block.reshape(-1, model.n_blocks),
            pos_weight=pos_weight,
            reduction='none',
        ).mean(dim=-1)  # (B*3,)
        loss_block = (loss_block_raw * block_w.reshape(-1)).mean()

        loss = (loss_nll
                + LAMBDA_SUM      * loss_sum
                + LAMBDA_CROSS    * loss_cross
                + LAMBDA_RED_CE   * loss_red_ce
                + LAMBDA_RED_CONS * loss_red_cons
                + LAMBDA_BLOCK    * loss_block)
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

    for features, labels, labels_red, labels_block, labels_shanten in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _, _ = model(features)

        for p in range(3):
            k_p     = labels[:, p].float().sum(dim=-1).long()
            probs_p = constrained_softmax_probs(logits[:, p], k_p)
            stage_w = get_stage_weights(features[:, p])

            eae_val    = weighted_eae(probs_p, labels[:, p], stage_w)
            total_wsum += eae_val.item() * stage_w.sum().item()
            total_w    += stage_w.sum().item()

            preds_p = probs_p.argmax(dim=-1)
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

    for features, labels, labels_red, labels_block, labels_shanten in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _, _ = model(features)

        remaining_vals = (features[:, 0, REMAINING_OFFSET] * 70).round().long().cpu()

        for p in range(3):
            k_vals = labels[:, p].float().sum(dim=-1).long()
            probs  = constrained_softmax_probs(logits[:, p], k_vals)
            preds  = probs.argmax(dim=-1)
            correct_mask   = (preds == labels[:, p])
            abs_diff       = (count_vals - labels[:, p].unsqueeze(-1).float()).abs()
            eae_per_sample = (abs_diff * probs).sum(-1).sum(-1).cpu()

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
    all_preds     = []
    all_labels    = []
    all_blk_pred  = []
    all_blk_true  = []
    count_vals    = torch.arange(5, device=device, dtype=torch.float32)
    total_wsum    = 0.0
    total_w       = 0.0

    for features, labels, labels_red, labels_block, labels_shanten in loader:
        features, labels = features.to(device), labels.to(device)
        labels_block = labels_block.to(device)
        logits, _, _, block_logits = model(features)

        for p in range(3):
            k     = labels[:, p].float().sum(dim=-1).long()
            probs = constrained_softmax_probs(logits[:, p], k)
            preds = dp_decode(logits[:, p], k)
            all_preds.append(preds.cpu())
            all_labels.append(labels[:, p].cpu())

            stage_w    = get_stage_weights(features[:, p])
            abs_diff   = (count_vals - labels[:, p].unsqueeze(-1).float()).abs()
            eae_s      = (abs_diff * probs).sum(-1).sum(-1)
            total_wsum += (eae_s * stage_w).sum().item()
            total_w    += stage_w.sum().item()

        blk_pred = (torch.sigmoid(block_logits) > 0.5).cpu()
        blk_true = (labels_block > 0.5).cpu()
        all_blk_pred.append(blk_pred.reshape(-1, model.n_blocks))
        all_blk_true.append(blk_true.reshape(-1, model.n_blocks))

    preds      = torch.cat(all_preds,  dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    true_nz = (labels_cat >= 1)
    pred_nz = (preds  >= 1)
    tp = (true_nz & pred_nz).float().sum()
    fn = (true_nz & ~pred_nz).float().sum()
    fp = (~true_nz & pred_nz).float().sum()

    recall    = (tp / (tp + fn + 1e-9)).item()
    precision = (tp / (tp + fp + 1e-9)).item()
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    exact_acc      = (preds == labels_cat).all(dim=-1).float().mean().item()
    pred_total_mae = (preds.float().sum(dim=-1) - labels_cat.float().sum(dim=-1)).abs().mean().item()
    weighted_eae_v = total_wsum / total_w if total_w > 0 else 0.0

    blk_pred_all = torch.cat(all_blk_pred, dim=0)
    blk_true_all = torch.cat(all_blk_true, dim=0)
    blk_tp = (blk_pred_all & blk_true_all).float().sum()
    blk_fp = (blk_pred_all & ~blk_true_all).float().sum()
    blk_fn = (~blk_pred_all & blk_true_all).float().sum()
    blk_prec   = (blk_tp / (blk_tp + blk_fp + 1e-9)).item()
    blk_recall = (blk_tp / (blk_tp + blk_fn + 1e-9)).item()
    blk_f1     = 2 * blk_prec * blk_recall / (blk_prec + blk_recall + 1e-9)

    return {
        "weighted_eae":      round(weighted_eae_v, 4),
        "recall_nonzero":    round(recall,    4),
        "precision_nonzero": round(precision, 4),
        "f1_nonzero":        round(f1,        4),
        "hand_exact_acc":    round(exact_acc, 4),
        "pred_total_mae":    round(pred_total_mae, 4),
        "block_f1":          round(blk_f1,    4),
        "block_precision":   round(blk_prec,  4),
        "block_recall":      round(blk_recall,4),
    }


def main():
    resume = "--resume" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  resume: {resume}")

    ndjson_path = DATA_DIR / "hand_inference_v26.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}")
        print("python phase2/scripts/add_block_labels.py --include-tatsu --v2 を先に実行してください")
        sys.exit(1)
    print(f"読み込み中: {ndjson_path}", flush=True)

    n_lines = sum(1 for ln in open(ndjson_path, encoding="utf-8") if ln.strip())
    print(f"総サンプル数: {n_lines}", flush=True)

    INPUT_DIM = CONFIG["input_dim"]
    feat_np    = np.empty((n_lines, 3, INPUT_DIM), dtype=np.float32)
    lab_np     = np.empty((n_lines, 3, 34),        dtype=np.int64)
    lred_np    = np.empty((n_lines, 3, 3),          dtype=np.int64)
    lblk_np    = np.empty((n_lines, 3, N_BLOCKS),   dtype=np.float32)
    lsh_np     = np.empty((n_lines, 3),             dtype=np.int64)

    with open(ndjson_path, encoding="utf-8") as f:
        i = 0
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if feat_np.shape[1:] != (3, INPUT_DIM):
                raise ValueError(f"特徴量次元不一致: {np.array(s['features']).shape} != (3, {INPUT_DIM})")
            feat_np[i] = s["features"]
            lab_np[i]  = s["label_hand"]
            lred_np[i] = s["label_red"]
            lblk_np[i] = s["label_block"]
            lsh_np[i]  = s["label_shanten"]
            i += 1
            if i % 50000 == 0:
                print(f"  {i}/{n_lines}", flush=True)

    print(f"features: {feat_np.shape}, label_block: {lblk_np.shape}, label_shanten: {lsh_np.shape}")

    pos_rate = float(lblk_np.mean())
    pw_val = min((1.0 - pos_rate) / (pos_rate + 1e-9), POS_WEIGHT_MAX)
    print(f"pos_rate={pos_rate:.4f}  pos_weight={pw_val:.2f} (max={POS_WEIGHT_MAX})")
    pos_weight = torch.full((N_BLOCKS,), pw_val, device=device)

    before   = len(feat_np)
    lab_sum  = lab_np.sum(axis=-1)
    meld_sum = (feat_np[:, :, 78] + feat_np[:, :, 79] + feat_np[:, :, 80])
    keep     = (lab_sum.sum(axis=1) > 0) | (meld_sum.sum(axis=1) > 0)
    feat_np  = feat_np[keep]; lab_np  = lab_np[keep]
    lred_np  = lred_np[keep]; lblk_np = lblk_np[keep]; lsh_np = lsh_np[keep]
    print(f"ノイズ除外: {before - len(feat_np)} → {len(feat_np)}")

    rng     = np.random.default_rng(42)
    idx     = rng.permutation(len(feat_np))
    feat_np = feat_np[idx]; lab_np  = lab_np[idx]
    lred_np = lred_np[idx]; lblk_np = lblk_np[idx]; lsh_np = lsh_np[idx]

    n       = len(feat_np)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    sl_tr = slice(None, n_train)
    sl_va = slice(n_train, n_train + n_val)
    sl_te = slice(n_train + n_val, None)

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

    model = HandInferenceV26(
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_eae = math.inf
    patience_cnt = 0
    start_epoch  = 1
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if resume and (MODEL_DIR / "model.pt").exists() and (MODEL_DIR / "train_log.json").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=True))
        logs = [json.loads(l) for l in (MODEL_DIR / "train_log.json").read_text().splitlines() if l.strip()]
        best_val_eae = min(e["val_eae"] for e in logs)
        last_epoch   = max(e["epoch"]   for e in logs)
        patience_cnt = 0
        for e in reversed(logs):
            if e["val_eae"] <= best_val_eae:
                break
            patience_cnt += 1
        start_epoch = last_epoch + 1
        print(f"resume: epoch {start_epoch}  best_val_eae={best_val_eae:.4f}  patience={patience_cnt}", flush=True)

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_nll        = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_eae, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step(val_eae)
        print(f"epoch {epoch:3d}  train_nll={train_nll:.4f}  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}", flush=True)
        with open(MODEL_DIR / "train_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train_nll": train_nll, "val_eae": val_eae, "val_acc": val_acc}) + "\n")
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

    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=True))
    test_eae, test_acc = eval_epoch(model, test_loader, device)
    print(f"test_eae={test_eae:.4f}  test_acc={test_acc:.4f}")

    metrics        = eval_metrics_detailed(model, test_loader, device)
    bucket_result  = eval_by_remaining(model, test_loader, device)
    print("\n詳細評価指標:")
    for k_name, v in metrics.items():
        print(f"  {k_name}: {v}")
    print("\n残り牌バケット別評価:")
    for bname, bv in bucket_result.items():
        print(f"  {bname}: acc={bv['acc']:.4f}  eae={bv['eae']:.4f}  n={bv['n']}")

    eval_result = {
        "test_eae": test_eae,
        "test_acc": test_acc,
        **metrics,
        "by_remaining_bucket": bucket_result,
        "config": CONFIG,
    }
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(eval_result, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    model.eval().cpu()
    dummy = torch.zeros(1, 3, CONFIG["input_dim"])

    class _OnnxWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _, red_logits, block_logits = self.m(x)
            return logits, red_logits, block_logits

    torch.onnx.export(
        _OnnxWrapper(model), dummy,
        str(MODEL_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["logits", "red_logits", "block_logits"],
        dynamic_axes={
            "features":     {0: "batch_size"},
            "logits":       {0: "batch_size"},
            "red_logits":   {0: "batch_size"},
            "block_logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
