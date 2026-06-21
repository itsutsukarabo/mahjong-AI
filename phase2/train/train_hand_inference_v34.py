"""
手牌類推モデル v34: v32チェックポイントから低LRでfine-tune + 全意図ラベル統合

v32 からの変更点:
  - 初期チェックポイント: v32 (model_best_wait.pt or model.pt)
  - lr: 5e-4 → 1e-4 (fine-tune安定化)
  - ReduceLROnPlateau patience: 4 → 6
  - early_stop_patience: 10 → 15
  - label_noise (①): loss_weight=0 でノイズサンプル除外
  - retreat_head: Linear(d_model,1) binary BCE, LAMBDA=0.2
  - push_head:    Linear(d_model,1) binary BCE, LAMBDA=0.3
  - ONNX出力: v32と同じ4出力 (retreat/pushは学習専用)

v33との違い:
  - v33はv32 + lr=5e-4 → 表現崩壊。v34はlr=1e-4で安定fine-tune
  - v34はlabel_push(③)も追加
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
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v34"
V32_DIR   = Path(__file__).parent.parent / "models" / "hand_inference" / "v32"

CONFIG = {
    "input_dim":    442,
    "d_model":      256,
    "nhead":        4,
    "num_layers":   3,
    "n_pai":        34,
    "n_count_cls":  5,
    "n_players":    3,
    "n_blocks":     N_BLOCKS,
    "dropout":      0.1,
    "lr":           1e-4,       # v32の5e-4から低下 (fine-tune安定化)
    "weight_decay": 1e-4,
    "batch_size":   256,
    "epochs":       200,
    "early_stop_patience": 15,  # v32の10から増加
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
LAMBDA_WAIT        = 0.5
LAMBDA_RETREAT     = 0.2
LAMBDA_PUSH        = 0.3
POS_WEIGHT_MAX     = 10.0
GRAD_CLIP_NORM     = 1.0

WAIT_EVAL_THRESH   = 0.3

# ---- ターツ形ラベル空間 (113クラス) ----

N_TATSU      = 113
RYANMEN_BASE = 0
PENCHAN_BASE = 18
KANCHAN_BASE = 24
TANKI_BASE   = 45
SHANPON_BASE = 79

def _build_tatsu_wait_table():
    table = [None] * N_TATSU
    for suit in range(3):
        for n in range(1, 7):
            t = suit * 9 + n
            table[RYANMEN_BASE + suit * 6 + (n - 1)] = frozenset({t - 1, t + 2})
    for suit in range(3):
        table[PENCHAN_BASE + suit * 2]     = frozenset({suit * 9 + 2})
        table[PENCHAN_BASE + suit * 2 + 1] = frozenset({suit * 9 + 6})
    for suit in range(3):
        for n in range(7):
            table[KANCHAN_BASE + suit * 7 + n] = frozenset({suit * 9 + n + 1})
    for t in range(N_PAI):
        table[TANKI_BASE + t] = frozenset({t})
    for t in range(N_PAI):
        table[SHANPON_BASE + t] = frozenset({t})
    return table

_TATSU_WAIT_TABLE = _build_tatsu_wait_table()


# ---- 分解補助関数 ----

def _can_form_sets_only(counts, n_sets):
    if n_sets == 0:
        return all(c == 0 for c in counts)
    for first in range(34):
        if counts[first] > 0: break
    else:
        return False
    if counts[first] >= 3:
        counts[first] -= 3
        if _can_form_sets_only(counts, n_sets - 1):
            counts[first] += 3; return True
        counts[first] += 3
    if first < 27 and first % 9 <= 6:
        if counts[first+1] > 0 and counts[first+2] > 0:
            counts[first] -= 1; counts[first+1] -= 1; counts[first+2] -= 1
            if _can_form_sets_only(counts, n_sets - 1):
                counts[first] += 1; counts[first+1] += 1; counts[first+2] += 1; return True
            counts[first] += 1; counts[first+1] += 1; counts[first+2] += 1
    return False


def _can_form_sets_and_pair(counts, n_sets):
    for p in range(34):
        if counts[p] >= 2:
            counts[p] -= 2
            if _can_form_sets_only(counts, n_sets):
                counts[p] += 2; return True
            counts[p] += 2
    return False


# ---- ターツ形ラベル生成 ----

def make_tatsu_labels_for_hand(counts34, n_melds):
    n_complete = 4 - n_melds
    counts = list(counts34)
    candidates = []

    for t in range(27):
        suit = t // 9; n = t % 9
        if n > 7: continue
        t1, t2 = t, t + 1
        if counts[t1] < 1 or counts[t2] < 1: continue
        counts[t1] -= 1; counts[t2] -= 1
        valid = _can_form_sets_and_pair(counts, n_complete - 1)
        counts[t1] += 1; counts[t2] += 1
        if not valid: continue
        if n == 0:
            candidates.append(([PENCHAN_BASE + suit * 2], frozenset({t + 2})))
        elif n == 7:
            candidates.append(([PENCHAN_BASE + suit * 2 + 1], frozenset({t - 1})))
        else:
            candidates.append(([RYANMEN_BASE + suit * 6 + (n - 1)], frozenset({t - 1, t + 2})))

    for t in range(27):
        suit = t // 9; n = t % 9
        if n > 6: continue
        t1, t2 = t, t + 2
        if counts[t1] < 1 or counts[t2] < 1: continue
        counts[t1] -= 1; counts[t2] -= 1
        valid = _can_form_sets_and_pair(counts, n_complete - 1)
        counts[t1] += 1; counts[t2] += 1
        if valid:
            candidates.append(([KANCHAN_BASE + suit * 7 + n], frozenset({t + 1})))

    for t in range(34):
        if counts[t] < 1: continue
        counts[t] -= 1
        valid = _can_form_sets_only(counts, n_complete)
        counts[t] += 1
        if valid:
            candidates.append(([TANKI_BASE + t], frozenset({t})))

    for t1 in range(34):
        if counts[t1] < 2: continue
        for t2 in range(t1 + 1, 34):
            if counts[t2] < 2: continue
            counts[t1] -= 2; counts[t2] -= 2
            valid = _can_form_sets_only(counts, n_complete - 1)
            counts[t1] += 2; counts[t2] += 2
            if valid:
                candidates.append(([SHANPON_BASE + t1, SHANPON_BASE + t2], frozenset({t1, t2})))

    if not candidates:
        return np.zeros(N_TATSU, dtype=np.float32)

    active = set()
    for i, (idx_list, waits_i) in enumerate(candidates):
        if not any(waits_i < waits_j for j, (_, waits_j) in enumerate(candidates) if i != j):
            active.update(idx_list)

    labels = np.zeros(N_TATSU, dtype=np.float32)
    for idx in active:
        labels[idx] = 1.0
    return labels


def make_tatsu_label_batch(labels_hand_np, labels_shanten_np):
    B, P, _ = labels_hand_np.shape
    out = np.zeros((B, P, N_TATSU), dtype=np.float32)
    for b in range(B):
        for p in range(P):
            if labels_shanten_np[b, p] != 0: continue
            counts34 = labels_hand_np[b, p].tolist()
            total = int(sum(counts34))
            if total == 0 or total % 3 != 1: continue
            out[b, p] = make_tatsu_labels_for_hand(counts34, (13 - total) // 3)
    return out


# ---- 評価補助 ----

def compute_true_waits(counts34, n_melds):
    waits = []
    for t in range(N_PAI):
        c = list(counts34)
        if c[t] >= 4: continue
        c[t] += 1
        if compute_shanten(c, n_melds) == -1:
            waits.append(t)
    return waits


def tatsu_probs_to_tile_probs(tatsu_probs):
    tp = np.zeros(N_PAI, dtype=np.float32)
    for suit in range(3):
        for n in range(1, 7):
            t = suit * 9 + n; idx = RYANMEN_BASE + suit * 6 + (n - 1); p = tatsu_probs[idx]
            tp[t-1] = max(tp[t-1], p); tp[t+2] = max(tp[t+2], p)
    for suit in range(3):
        tp[suit*9+2] = max(tp[suit*9+2], tatsu_probs[PENCHAN_BASE + suit*2])
        tp[suit*9+6] = max(tp[suit*9+6], tatsu_probs[PENCHAN_BASE + suit*2 + 1])
    for suit in range(3):
        for n in range(7):
            tp[suit*9+n+1] = max(tp[suit*9+n+1], tatsu_probs[KANCHAN_BASE + suit*7 + n])
    for t in range(N_PAI):
        tp[t] = max(tp[t], tatsu_probs[TANKI_BASE + t])
        tp[t] = max(tp[t], tatsu_probs[SHANPON_BASE + t])
    return tp


@torch.no_grad()
def eval_wait_metrics(model, loader, device, threshold=WAIT_EVAL_THRESH):
    model.eval()
    prec_list = []; recall_list = []; f1_list = []; hit_list = []; top1_list = []
    for batch in loader:
        features = batch[0].to(device); labels = batch[1]; labels_shanten = batch[4]
        out = model(features)
        wait_tatsu_probs = torch.sigmoid(out[4]).cpu().numpy()
        hand_np = labels.numpy(); sh_np = labels_shanten.numpy()
        for b in range(len(features)):
            for p in range(3):
                if sh_np[b, p] != 0: continue
                counts34 = hand_np[b, p].tolist(); total = int(sum(counts34))
                if total == 0 or total % 3 != 1: continue
                true_waits = set(compute_true_waits(counts34, (13 - total) // 3))
                if not true_waits: continue
                tile_probs = tatsu_probs_to_tile_probs(wait_tatsu_probs[b, p])
                pred_waits = {t for t in range(N_PAI) if tile_probs[t] >= threshold}
                top1 = int(np.argmax(tile_probs))
                tp = len(pred_waits & true_waits); fp = len(pred_waits - true_waits); fn = len(true_waits - pred_waits)
                prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
                recall = tp/(tp+fn) if (tp+fn) > 0 else 0.0
                f1 = 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0.0
                prec_list.append(prec); recall_list.append(recall); f1_list.append(f1)
                hit_list.append(1.0 if tp > 0 else 0.0)
                top1_list.append(1.0 if top1 in true_waits else 0.0)
    if not f1_list:
        return {"wait_f1": 0.0, "wait_prec": 0.0, "wait_recall": 0.0, "wait_hit_rate": 0.0, "wait_top1_acc": 0.0, "n_tenpai": 0}
    return {
        "wait_f1":       round(float(np.mean(f1_list)), 4),
        "wait_prec":     round(float(np.mean(prec_list)), 4),
        "wait_recall":   round(float(np.mean(recall_list)), 4),
        "wait_hit_rate": round(float(np.mean(hit_list)), 4),
        "wait_top1_acc": round(float(np.mean(top1_list)), 4),
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
            mid = (lo + hi) / 2
            adj = logits - mid.view(B, 1, 1) * count_vals
            e_sum = (F.softmax(adj, dim=-1) * count_vals).sum(-1).sum(-1)
            lo = torch.where(e_sum > k_float, mid, lo)
            hi = torch.where(e_sum > k_float, hi, mid)
    return (lo + hi) / 2


def constrained_softmax_probs(logits, k):
    count_vals = torch.arange(5, device=logits.device, dtype=torch.float32)
    lam = find_lambda(logits, k.float())
    return F.softmax(logits - lam.view(-1, 1, 1) * count_vals, dim=-1)


def get_stage_weights(features):
    remaining = (features[:, REMAINING_OFFSET] * 70).round()
    meld_tiles = (features[:, 78] * 3 + features[:, 79] * 3 + features[:, 80] * 4)
    eff = (remaining - meld_tiles * 2).clamp(min=0).long()
    w = torch.ones(len(features), device=features.device)
    w[eff <= 10] = 4.0
    w[(eff > 10) & (eff <= 30)] = 3.0
    w[(eff > 30) & (eff <= 50)] = 2.0
    return w


def weighted_eae(probs, labels, weights):
    count_vals = torch.arange(probs.shape[-1], device=probs.device, dtype=torch.float32)
    eae = (((count_vals - labels.unsqueeze(-1).float()).abs() * probs).sum(-1).sum(-1))
    return (eae * weights).sum() / weights.sum()


# ---- GPU温度 ----

def gpu_temp():
    for cmd in ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "/mnt/c/Windows/System32/nvidia-smi.exe"]:
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
    if temp <= GPU_TEMP_THRESHOLD: return
    print(f"  [thermal] GPU {temp}C > {GPU_TEMP_THRESHOLD}C - cooling...", flush=True)
    while True:
        time.sleep(GPU_COOL_INTERVAL)
        temp = gpu_temp()
        print(f"  [thermal] GPU {temp}C", flush=True)
        if temp <= GPU_TEMP_THRESHOLD:
            print("  [thermal] cool enough, resuming", flush=True); break


# ---- データセット ----

class HandInferenceDataset(Dataset):
    def __init__(self, features, labels, labels_red, labels_block, labels_shanten,
                 noise_mask, retreat_label, push_label):
        self.features       = torch.as_tensor(features,       dtype=torch.float32)
        self.labels         = torch.as_tensor(labels,         dtype=torch.long).clamp(0, 4)
        self.labels_red     = torch.as_tensor(labels_red,     dtype=torch.long)
        self.labels_block   = torch.as_tensor(labels_block,   dtype=torch.float32)
        self.labels_shanten = torch.as_tensor(labels_shanten, dtype=torch.long)
        self.noise_mask     = torch.as_tensor(noise_mask,     dtype=torch.bool)
        self.retreat_label  = torch.as_tensor(retreat_label,  dtype=torch.float32)
        self.push_label     = torch.as_tensor(push_label,     dtype=torch.float32)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx], self.labels_red[idx],
                self.labels_block[idx], self.labels_shanten[idx],
                self.noise_mask[idx], self.retreat_label[idx], self.push_label[idx])


# ---- モデル ----

class HandInferenceV34(nn.Module):
    """v32アーキテクチャ + retreat_head(④) + push_head(③)"""

    def __init__(self, input_dim, d_model, nhead, num_layers,
                 n_pai=34, n_count_cls=5, n_players=3, n_blocks=134,
                 n_tatsu=N_TATSU, dropout=0.1):
        super().__init__()
        self.n_pai = n_pai; self.n_count_cls = n_count_cls
        self.n_players = n_players; self.n_blocks = n_blocks; self.n_tatsu = n_tatsu

        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, d_model),
        )
        self.tile_embed   = nn.Embedding(n_pai,     d_model)
        self.player_embed = nn.Embedding(n_players, d_model)
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head       = nn.Linear(d_model, n_count_cls)
        self.red_head   = nn.Linear(d_model, 2)
        self.block_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.wait_pool  = nn.Linear(d_model, 1, bias=False)
        self.wait_head  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_tatsu))
        # 補助ヘッド: retreat(④) と push(③) は global_pool を共有
        self.retreat_head = nn.Linear(d_model, 1)
        self.push_head    = nn.Linear(d_model, 1)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("player_ids",   torch.arange(n_players))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))
        self.register_buffer("block_sel",
            torch.tensor(build_block_selection_matrix(include_tatsu=True), dtype=torch.float32))

    def forward(self, x):
        B, P, F = x.shape
        g = self.global_encoder(x.reshape(B*P, F)).reshape(B, P, -1)

        vis     = x[:, :, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai]
        tokens  = (g.unsqueeze(2)
                   + self.tile_embed(self.tile_ids)
                   + self.player_embed(self.player_ids).view(1, P, 1, -1)
                   + self.visible_proj(vis.unsqueeze(-1)))
        out = self.transformer(tokens.reshape(B, P*self.n_pai, -1)).reshape(B, P, self.n_pai, -1)

        vis_raw  = (vis * 4).round().long().clamp(0, 4)
        logits_raw = self.head(out)
        mask       = self.count_range > (4 - vis_raw).clamp(min=0).unsqueeze(-1)
        logits     = logits_raw.masked_fill(mask, float('-inf'))

        red_logits   = self.red_head(out[:, :, self.red_tile_idx, :])
        out_flat     = out.reshape(B*P, self.n_pai, -1)
        block_logits = self.block_head(
            torch.bmm(self.block_sel.unsqueeze(0).expand(B*P, -1, -1), out_flat)
        ).squeeze(-1).reshape(B, P, self.n_blocks)

        attn_w      = torch.softmax(self.wait_pool(out).squeeze(-1), dim=-1)
        pooled      = (out * attn_w.unsqueeze(-1)).sum(dim=2)       # (B, P, d)
        wait_logits = self.wait_head(pooled)                         # (B, P, 113)

        global_pool   = pooled.mean(dim=1)                           # (B, d)
        retreat_logit = self.retreat_head(global_pool).squeeze(-1)   # (B,)
        push_logit    = self.push_head(global_pool).squeeze(-1)      # (B,)

        return logits, logits_raw, red_logits, block_logits, wait_logits, retreat_logit, push_logit


# ---- 学習 ----

def train_epoch(model, loader, optimizer, device,
                pos_weight_block, pos_weight_wait, pos_weight_retreat, pos_weight_push):
    model.train()
    total_ce = 0.0
    count_vals = torch.arange(model.n_count_cls, device=device, dtype=torch.float32)
    five_idx   = model.red_tile_idx

    for batch in loader:
        (features, labels, labels_red, labels_block, labels_shanten,
         noise_mask, retreat_label, push_label) = [x.to(device) for x in batch]
        optimizer.zero_grad()

        logits, logits_raw, red_logits, block_logits, wait_logits, retreat_logit, push_logit = model(features)
        B, P = features.shape[:2]

        # ノイズマスク: True → weight=0
        sample_w    = (~noise_mask).float()                      # (B,)
        stage_w     = torch.stack([get_stage_weights(features[:, p]) for p in range(P)], dim=1)
        effective_w = stage_w * sample_w.unsqueeze(1)            # (B, P)

        # NLL
        loss_nll_per = F.cross_entropy(
            logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1), reduction='none'
        ).reshape(B, P, -1).mean(-1)
        denom = effective_w.sum()
        loss_nll = (loss_nll_per * effective_w).sum() / denom if denom > 0 else torch.tensor(0., device=device)

        # sum / cross
        probs = F.softmax(logits_raw, dim=-1)
        pred_sum = (probs * count_vals).sum(-1)
        loss_sum   = F.mse_loss(pred_sum.sum(-1), labels.float().sum(-1))
        visible_counts = features[:, 0, VISIBLE_OFFSET:VISIBLE_OFFSET + model.n_pai] * 4
        loss_cross = F.relu(pred_sum.sum(dim=1) - (4 - visible_counts).clamp(min=0)).mean()

        # red
        loss_red_ce   = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))
        prob_has_red  = F.softmax(red_logits, dim=-1)[:, :, :, 1]
        prob_cnt_ge1  = 1 - probs[:, :, five_idx, 0]
        loss_red_cons = F.relu(prob_has_red - prob_cnt_ge1).mean()

        # block
        block_w   = torch.where(labels_shanten <= 1, 0.5 ** labels_shanten.float(),
                                 torch.zeros_like(labels_shanten.float())) * sample_w.unsqueeze(1)
        denom_blk = block_w.reshape(-1).sum()
        loss_block = (F.binary_cross_entropy_with_logits(
            block_logits.reshape(-1, model.n_blocks),
            labels_block.reshape(-1, model.n_blocks),
            pos_weight=pos_weight_block, reduction='none'
        ).mean(dim=-1) * block_w.reshape(-1)).sum() / denom_blk if denom_blk > 0 else torch.tensor(0., device=device)

        # wait/tatsu
        wait_mask  = (labels_shanten == 0).float() * sample_w.unsqueeze(1)
        denom_wait = wait_mask.sum()
        if denom_wait > 0:
            tatsu_labels = torch.tensor(
                make_tatsu_label_batch(labels.cpu().numpy(), labels_shanten.cpu().numpy()),
                device=device
            )
            loss_wait = (F.binary_cross_entropy_with_logits(
                wait_logits.reshape(-1, model.n_tatsu),
                tatsu_labels.reshape(-1, model.n_tatsu),
                pos_weight=pos_weight_wait, reduction='none'
            ).mean(dim=-1) * wait_mask.reshape(-1)).sum() / (denom_wait * model.n_tatsu)
        else:
            loss_wait = torch.tensor(0.0, device=device)

        # retreat (④)
        denom_ret = sample_w.sum()
        loss_retreat = (F.binary_cross_entropy_with_logits(
            retreat_logit, retreat_label, pos_weight=pos_weight_retreat, reduction='none'
        ) * sample_w).sum() / denom_ret if denom_ret > 0 else torch.tensor(0., device=device)

        # push (③)
        loss_push = (F.binary_cross_entropy_with_logits(
            push_logit, push_label, pos_weight=pos_weight_push, reduction='none'
        ) * sample_w).sum() / denom_ret if denom_ret > 0 else torch.tensor(0., device=device)

        loss = (loss_nll
                + LAMBDA_SUM      * loss_sum
                + LAMBDA_CROSS    * loss_cross
                + LAMBDA_RED_CE   * loss_red_ce
                + LAMBDA_RED_CONS * loss_red_cons
                + LAMBDA_BLOCK    * loss_block
                + LAMBDA_WAIT     * loss_wait
                + LAMBDA_RETREAT  * loss_retreat
                + LAMBDA_PUSH     * loss_push)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_ce += loss_nll.item() * len(features)

    return total_ce / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_wsum = 0.0; total_w = 0.0; correct = 0; total = 0
    for batch in loader:
        features = batch[0].to(device); labels = batch[1].to(device)
        logits = model(features)[0]
        for p in range(3):
            k_p = labels[:, p].float().sum(dim=-1).long()
            probs_p = constrained_softmax_probs(logits[:, p], k_p)
            stage_w = get_stage_weights(features[:, p])
            total_wsum += weighted_eae(probs_p, labels[:, p], stage_w).item() * stage_w.sum().item()
            total_w    += stage_w.sum().item()
            correct    += (probs_p.argmax(dim=-1) == labels[:, p]).sum().item()
            total      += labels[:, p].numel()
    return total_wsum / total_w if total_w > 0 else float('inf'), correct / total


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
    feat_np     = np.empty((n_lines, 3, INPUT_DIM), dtype=np.float32)
    lab_np      = np.empty((n_lines, 3, 34),         dtype=np.int64)
    lred_np     = np.empty((n_lines, 3, 3),           dtype=np.int64)
    lblk_np     = np.empty((n_lines, 3, N_BLOCKS),    dtype=np.float32)
    lsh_np      = np.empty((n_lines, 3),              dtype=np.int64)
    lnoise_np   = np.empty((n_lines,),                dtype=bool)
    lretreat_np = np.empty((n_lines,),                dtype=np.float32)
    lpush_np    = np.empty((n_lines,),                dtype=np.float32)
    print(f"総サンプル数: {n_lines}", flush=True)

    noise_cnt = retreat_cnt = push_cnt = 0
    with open(ndjson_path, encoding="utf-8") as f:
        i = 0
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            feat_np[i]     = s["features"]
            lab_np[i]      = s["label_hand"]
            lred_np[i]     = s["label_red"]
            lblk_np[i]     = s["label_block"]
            lsh_np[i]      = s["label_shanten"]
            lnoise_np[i]   = bool(s.get("label_noise", False))
            lretreat_np[i] = 1.0 if s.get("label_retreat", {}).get("is_retreat", False) else 0.0
            lpush_np[i]    = 1.0 if s.get("label_push",    {}).get("is_push",    False) else 0.0
            if lnoise_np[i]:   noise_cnt   += 1
            if lretreat_np[i]: retreat_cnt += 1
            if lpush_np[i]:    push_cnt    += 1
            i += 1
            if i % 50000 == 0: print(f"  {i}/{n_lines}", flush=True)

    print(f"label_noise={noise_cnt} ({noise_cnt/n_lines*100:.1f}%)")
    print(f"label_retreat={retreat_cnt} ({retreat_cnt/n_lines*100:.2f}%)")
    print(f"label_push={push_cnt} ({push_cnt/n_lines*100:.2f}%)")

    pw_block   = min((1 - float(lblk_np.mean())) / (float(lblk_np.mean()) + 1e-9), POS_WEIGHT_MAX)
    pw_retreat = min((1 - retreat_cnt/n_lines) / (retreat_cnt/n_lines + 1e-9), POS_WEIGHT_MAX)
    pw_push    = min((1 - push_cnt/n_lines)    / (push_cnt/n_lines    + 1e-9), POS_WEIGHT_MAX)
    pw_wait    = min(0.98 / 0.02, POS_WEIGHT_MAX)
    pos_weight_block   = torch.full((N_BLOCKS,), pw_block,   device=device)
    pos_weight_wait    = torch.full((N_TATSU,),  pw_wait,    device=device)
    pos_weight_retreat = torch.tensor(pw_retreat, device=device)
    pos_weight_push    = torch.tensor(pw_push,    device=device)
    print(f"block pw={pw_block:.2f}  wait pw={pw_wait:.2f}  retreat pw={pw_retreat:.2f}  push pw={pw_push:.2f}")

    # 空サンプル除外
    lab_sum  = lab_np.sum(axis=-1)
    meld_sum = feat_np[:, :, 78] + feat_np[:, :, 79] + feat_np[:, :, 80]
    keep = (lab_sum.sum(axis=1) > 0) | (meld_sum.sum(axis=1) > 0)
    feat_np=feat_np[keep]; lab_np=lab_np[keep]; lred_np=lred_np[keep]
    lblk_np=lblk_np[keep]; lsh_np=lsh_np[keep]; lnoise_np=lnoise_np[keep]
    lretreat_np=lretreat_np[keep]; lpush_np=lpush_np[keep]
    print(f"有効サンプル: {len(feat_np)}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(feat_np))
    feat_np=feat_np[idx]; lab_np=lab_np[idx]; lred_np=lred_np[idx]
    lblk_np=lblk_np[idx]; lsh_np=lsh_np[idx]; lnoise_np=lnoise_np[idx]
    lretreat_np=lretreat_np[idx]; lpush_np=lpush_np[idx]

    n = len(feat_np); n_train = int(n*0.8); n_val = int(n*0.1)
    sl_tr = slice(None, n_train); sl_va = slice(n_train, n_train+n_val); sl_te = slice(n_train+n_val, None)

    def make_ds(sl):
        return HandInferenceDataset(
            feat_np[sl], lab_np[sl], lred_np[sl], lblk_np[sl], lsh_np[sl],
            lnoise_np[sl], lretreat_np[sl], lpush_np[sl]
        )

    train_ds = make_ds(sl_tr); val_ds = make_ds(sl_va); test_ds = make_ds(sl_te)
    del feat_np, lab_np, lred_np, lblk_np, lsh_np, lnoise_np, lretreat_np, lpush_np; gc.collect()
    print(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_gpu = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)

    model = HandInferenceV34(
        input_dim=CONFIG["input_dim"], d_model=CONFIG["d_model"], nhead=CONFIG["nhead"],
        num_layers=CONFIG["num_layers"], n_pai=CONFIG["n_pai"], n_count_cls=CONFIG["n_count_cls"],
        n_players=CONFIG["n_players"], n_blocks=CONFIG["n_blocks"], n_tatsu=N_TATSU,
        dropout=CONFIG["dropout"],
    ).to(device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6)
    best_val_eae = math.inf; best_wait_f1 = 0.0; patience_cnt = 0; start_epoch = 1

    if not resume:
        ckpt = V32_DIR / "model_best_wait.pt"
        if not ckpt.exists(): ckpt = V32_DIR / "model.pt"
        if ckpt.exists():
            sd = torch.load(ckpt, map_location=device, weights_only=True)
            sd = sd.get("model_state_dict", sd)
            # retreat_head, push_head は v34 新規 → 除外してランダム初期化
            sd = {k: v for k, v in sd.items()
                  if not k.startswith("retreat_head") and not k.startswith("push_head")}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"v32 ロード  missing={len(missing)}  unexpected={len(unexpected)}", flush=True)
            print(f"  missing(新規): {missing}", flush=True)
        else:
            print("v32 チェックポイントなし → ランダム初期化", flush=True)

    if resume and (MODEL_DIR / "model.pt").exists() and (MODEL_DIR / "train_log.json").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=True))
        logs         = [json.loads(l) for l in (MODEL_DIR / "train_log.json").read_text().splitlines() if l.strip()]
        best_val_eae = min(e["val_eae"] for e in logs)
        best_wait_f1 = max((e.get("wait_f1", 0.0) for e in logs), default=0.0)
        last_epoch   = max(e["epoch"] for e in logs)
        patience_cnt = 0   # optimizer state未保存のためLRリセット → patience_cntもリセット
        start_epoch  = last_epoch + 1
        print(f"resume: epoch {start_epoch}  best_val_eae={best_val_eae:.4f}  patience_cnt=0(LR reset)", flush=True)

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_nll        = train_epoch(model, train_loader, optimizer, device,
                                       pos_weight_block, pos_weight_wait, pos_weight_retreat, pos_weight_push)
        val_eae, val_acc = eval_epoch(model, val_loader, device)
        wait_m           = eval_wait_metrics(model, val_loader, device)
        scheduler.step(val_eae)

        log_entry = {"epoch": epoch, "train_nll": round(train_nll, 4),
                     "val_eae": round(val_eae, 4), "val_acc": round(val_acc, 4), **wait_m}
        print(f"epoch {epoch:3d}  nll={train_nll:.4f}  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}"
              f"  wait_f1={wait_m['wait_f1']:.4f}  hit={wait_m['wait_hit_rate']:.4f}"
              f"  top1={wait_m['wait_top1_acc']:.4f}", flush=True)
        with open(MODEL_DIR / "train_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        wait_for_cool()

        if val_eae < best_val_eae:
            best_val_eae = val_eae; patience_cnt = 0
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["early_stop_patience"]:
                print("early stopping"); break

        if wait_m["wait_f1"] > best_wait_f1:
            best_wait_f1 = wait_m["wait_f1"]
            torch.save(model.state_dict(), MODEL_DIR / "model_best_wait.pt")
            print(f"  [saved best_wait f1={best_wait_f1:.4f}]", flush=True)

    # テスト評価
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=True))
    test_eae, test_acc = eval_epoch(model, test_loader, device)
    test_wait_m        = eval_wait_metrics(model, test_loader, device)
    print(f"\n[model.pt] test_eae={test_eae:.4f}  test_acc={test_acc:.4f}")
    for k, v in test_wait_m.items(): print(f"  {k}: {v}")

    if (MODEL_DIR / "model_best_wait.pt").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "model_best_wait.pt", map_location=device, weights_only=True))
        bw = eval_wait_metrics(model, test_loader, device)
        print(f"\n[model_best_wait.pt]")
        for k, v in bw.items(): print(f"  {k}: {v}")
        test_wait_m = bw

    (MODEL_DIR / "eval_result.json").write_text(json.dumps({"test_eae": test_eae, "test_acc": test_acc, **test_wait_m, "config": CONFIG}, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps({**CONFIG, "n_tatsu": N_TATSU}, indent=2))

    # ONNX エクスポート (v32 と同一インターフェース)
    model.eval().cpu()
    dummy = torch.zeros(1, 3, CONFIG["input_dim"])

    class _Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _, red_logits, block_logits, wait_logits, _, _ = self.m(x)
            return logits, red_logits, block_logits, wait_logits

    torch.onnx.export(
        _Wrap(model), dummy, str(MODEL_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["logits", "red_logits", "block_logits", "wait_logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"},
                      "red_logits": {0: "batch"}, "block_logits": {0: "batch"}, "wait_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"\nONNX保存: {MODEL_DIR / 'model.onnx'}")


if __name__ == "__main__":
    main()
