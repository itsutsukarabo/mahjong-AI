"""
手牌類推モデル v18

v17からの変更点:
  - KV-cache 付き CachedDecoderLayer で greedy decode を高速化
      cross-attn K/V: memory から事前計算し全 step で再利用 (固定)
      self-attn K/V: step ごとに 1 トークン追加のみ (O(N²) → O(N))
  - early stopping を greedy val_eae → teacher forcing val_nll に変更
      greedy val_eae は序盤の露出バイアスで単調減少せず early stopping が誤作動するため
  - val_eae は VAL_EAE_INTERVAL epoch ごとに補助ログとして記録
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
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v18"

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
    "early_stop_patience": 10,
    "val_eae_interval":    5,   # greedy val_eae を計算する間隔 (epoch)
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


# ---- DP デコード ----

@torch.no_grad()
def dp_decode(logits, k):
    B, N, C = logits.shape
    K_max   = int(k.max().item())
    NEG_INF = -1e9

    log_probs = F.log_softmax(logits, dim=-1).cpu()
    k_cpu     = k.cpu()

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
            new_dp[:, sl]     = torch.where(better, val, new_dp[:, sl])
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


# ---- KV-cache 付きデコーダ層 ----

def _project_kv(mha: nn.MultiheadAttention, source: torch.Tensor):
    """source から K/V の線形射影のみ抽出する。"""
    d = mha.embed_dim
    W = mha.in_proj_weight
    b = mha.in_proj_bias
    k = F.linear(source, W[d:2*d], b[d:2*d] if b is not None else None)
    v = F.linear(source, W[2*d:],  b[2*d:]  if b is not None else None)
    return k, v


def _attend(mha: nn.MultiheadAttention,
            q_input: torch.Tensor,
            k_proj:  torch.Tensor,
            v_proj:  torch.Tensor) -> torch.Tensor:
    """
    pre-projected K/V に対して Q を射影しアテンションを計算する。
    q_input: (B, T_q, d)
    k_proj:  (B, T_k, d)  — _project_kv の出力
    v_proj:  (B, T_k, d)
    戻り値:  (B, T_q, d)
    """
    d     = mha.embed_dim
    nhead = mha.num_heads
    dh    = d // nhead
    B, T_q, _ = q_input.shape

    # Q を射影
    w_q = mha.in_proj_weight[:d]
    b_q = mha.in_proj_bias[:d] if mha.in_proj_bias is not None else None
    q = F.linear(q_input, w_q, b_q)  # (B, T_q, d)

    # multi-head に分割
    q = q.reshape(B, T_q, nhead, dh).transpose(1, 2)        # (B, H, T_q, dh)
    k = k_proj.reshape(B, -1, nhead, dh).transpose(1, 2)    # (B, H, T_k, dh)
    v = v_proj.reshape(B, -1, nhead, dh).transpose(1, 2)    # (B, H, T_k, dh)

    scale  = dh ** -0.5
    attn_w = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # (B, H, T_q, T_k)
    out    = (attn_w @ v).transpose(1, 2).reshape(B, T_q, d)     # (B, T_q, d)
    return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)


class CachedDecoderLayer(nn.Module):
    """
    Pre-LN Transformer Decoder 層。
    学習: forward_train (フル系列, teacher forcing)
    推論: precompute_cross_kv + forward_step (1トークン, KV-cache)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_train(self, x: torch.Tensor, memory: torch.Tensor,
                      causal_mask: torch.Tensor) -> torch.Tensor:
        """フル系列 forward (teacher forcing)。causal_mask は (T, T) の上三角 -inf マスク。"""
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal_mask)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, memory, memory)[0]
        x = x + self.ff(self.norm3(x))
        return x

    def precompute_cross_kv(self, memory: torch.Tensor):
        """cross-attn の K/V を memory から事前計算して返す (推論時に固定)。"""
        return _project_kv(self.cross_attn, memory)  # (k, v) 各 (B, 34, d)

    def forward_step(self, x: torch.Tensor,
                     cross_kv,
                     self_kv_cache):
        """
        1 トークン分の推論ステップ。KV-cache を更新して返す。
        x:            (B, 1, d) — 現在のトークン
        cross_kv:     (k, v) 各 (B, 34, d) — precompute_cross_kv の結果
        self_kv_cache:(k, v) 各 (B, step, d) or None
        戻り値: (out (B,1,d), new_self_kv_cache)
        """
        # ---- self-attention ----
        h     = self.norm1(x)
        k_new, v_new = _project_kv(self.self_attn, h)  # (B, 1, d) 各

        if self_kv_cache is not None:
            k_self = torch.cat([self_kv_cache[0], k_new], dim=1)
            v_self = torch.cat([self_kv_cache[1], v_new], dim=1)
        else:
            k_self, v_self = k_new, v_new

        x = x + _attend(self.self_attn, h, k_self, v_self)
        new_self_kv = (k_self, v_self)

        # ---- cross-attention (cached K/V を再利用) ----
        h = self.norm2(x)
        x = x + _attend(self.cross_attn, h, cross_kv[0], cross_kv[1])

        # ---- FFN ----
        x = x + self.ff(self.norm3(x))
        return x, new_self_kv


# ---- モデル ----

class HandInferenceV18(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, tile_attn_nhead,
                 dec_layers, n_pai=34, n_count_cls=5, n_players=3, dropout=0.1):
        super().__init__()
        self.n_pai       = n_pai
        self.n_count_cls = n_count_cls
        self.n_players   = n_players

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

        self.tile_attn = nn.MultiheadAttention(
            d_model, tile_attn_nhead, dropout=dropout, batch_first=True
        )
        self.tile_norm = nn.LayerNorm(d_model)

        # 自己回帰デコーダ (KV-cache 対応)
        self.count_embed   = nn.Embedding(n_count_cls, d_model)
        self.bos           = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.decoder_layers = nn.ModuleList([
            CachedDecoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(dec_layers)
        ])

        self.head     = nn.Linear(d_model, n_count_cls)
        self.red_head = nn.Linear(d_model, 2)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("player_ids",   torch.arange(n_players))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))

    def _encode(self, x):
        B, P, F = x.shape
        g = self.global_encoder(x.reshape(B * P, F)).reshape(B, P, -1)

        tile_emb   = self.tile_embed(self.tile_ids)
        player_emb = self.player_embed(self.player_ids)
        vis        = x[:, :, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai]
        vis_emb    = self.visible_proj(vis.unsqueeze(-1))

        tokens = (g.unsqueeze(2) + tile_emb + player_emb.view(1, P, 1, -1) + vis_emb)
        tokens = tokens.reshape(B, P * self.n_pai, -1)

        out      = self.transformer(tokens).reshape(B, P, self.n_pai, -1)
        out_flat = out.reshape(B * P, self.n_pai, -1)
        attn_out, _ = self.tile_attn(out_flat, out_flat, out_flat)
        memory   = self.tile_norm(out_flat + attn_out)  # (B*P, 34, d)

        return memory, out, vis

    def _vis_mask(self, vis):
        vis_raw          = (vis * 4).round().long().clamp(0, 4)
        hidden_remaining = (4 - vis_raw).clamp(min=0)
        return self.count_range > hidden_remaining.unsqueeze(-1)  # (B, P, 34, 5)

    def forward(self, x, labels=None):
        """
        labels: (B, P, 34) — None のとき greedy KV-cache 推論
        """
        B, P, _ = x.shape
        memory, out_4d, vis = self._encode(x)
        mask = self._vis_mask(vis)

        if labels is not None:
            # teacher forcing
            lbl = labels.reshape(B * P, self.n_pai)
            bos = self.bos.expand(B * P, 1, -1)
            tgt = torch.cat([bos, self.count_embed(lbl[:, :-1])], dim=1)  # (B*P, 34, d)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                self.n_pai, device=x.device
            )
            dec_x = tgt
            for layer in self.decoder_layers:
                dec_x = layer.forward_train(dec_x, memory, causal_mask)
            logits_raw = self.head(dec_x).reshape(B, P, self.n_pai, -1)
        else:
            # KV-cache greedy 推論
            logits_raw = self._greedy_decode_cached(memory, mask, B, P, x.device)

        logits     = logits_raw.masked_fill(mask, float('-inf'))
        red_tokens = out_4d[:, :, self.red_tile_idx, :]
        red_logits = self.red_head(red_tokens)
        return logits, logits_raw, red_logits

    def _greedy_decode_cached(self, memory, mask, B, P, device):
        """KV-cache を使った greedy 自己回帰デコード。"""
        BP        = B * P
        mask_flat = mask.reshape(BP, self.n_pai, self.n_count_cls)

        # cross-attn K/V を事前計算 (各層 × 1回)
        cross_kv_caches = [layer.precompute_cross_kv(memory)
                           for layer in self.decoder_layers]
        # self-attn KV-cache (各層 × ステップごとに成長)
        self_kv_caches = [None] * len(self.decoder_layers)

        cur_token  = self.bos.expand(BP, 1, -1).clone()
        all_logits = []

        for step in range(self.n_pai):
            x = cur_token
            new_self_kv_caches = []

            for idx, layer in enumerate(self.decoder_layers):
                x, new_kv = layer.forward_step(x, cross_kv_caches[idx],
                                                self_kv_caches[idx])
                new_self_kv_caches.append(new_kv)

            self_kv_caches = new_self_kv_caches

            logit        = self.head(x)  # (BP, 1, 5)
            logit_masked = logit.masked_fill(mask_flat[:, step:step + 1, :], float('-inf'))
            all_logits.append(logit_masked)

            pred      = logit_masked.argmax(dim=-1)  # (BP, 1)
            cur_token = self.count_embed(pred)        # (BP, 1, d)

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

        _, logits_raw, red_logits = model(features, labels=labels)

        loss_nll  = F.cross_entropy(logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1))
        probs     = F.softmax(logits_raw, dim=-1)
        pred_sum  = (probs * count_vals).sum(-1)
        loss_sum  = F.mse_loss(pred_sum.sum(-1), labels.float().sum(-1))

        pred_tile_total = pred_sum.sum(dim=1)
        visible_counts  = features[:, 0, VISIBLE_OFFSET:VISIBLE_OFFSET + model.n_pai] * 4
        max_hidden      = (4 - visible_counts).clamp(min=0)
        loss_cross      = F.relu(pred_tile_total - max_hidden).mean()

        loss_red_ce   = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))
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
def eval_nll(model, loader, device):
    """teacher forcing ベースの val_nll (early stopping 用)。"""
    model.eval()
    total_ce = 0.0
    for features, labels, labels_red in loader:
        features = features.to(device)
        labels   = labels.to(device)
        _, logits_raw, _ = model(features, labels=labels)
        loss = F.cross_entropy(logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1))
        total_ce += loss.item() * len(features)
    return total_ce / len(loader.dataset)


@torch.no_grad()
def eval_eae(model, loader, device):
    """KV-cache greedy decode による val_eae と val_acc。"""
    model.eval()
    total_wsum = 0.0
    total_w    = 0.0
    correct    = 0
    total      = 0

    for features, labels, labels_red in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _ = model(features)  # KV-cache greedy

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

    return total_wsum / total_w if total_w > 0 else float('inf'), correct / total


@torch.no_grad()
def eval_by_remaining(model, loader, device):
    model.eval()
    count_vals = torch.arange(5, device=device, dtype=torch.float32)
    buckets    = {k: {"correct": 0, "total": 0, "eae_sum": 0.0, "n_samples": 0}
                  for k in REMAINING_BUCKETS}

    for features, labels, labels_red in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _     = model(features)
        remaining_vals   = (features[:, 0, REMAINING_OFFSET] * 70).round().long().cpu()

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
            k     = labels[:, p].float().sum(dim=-1).long()
            probs = constrained_softmax_probs(logits[:, p], k)
            preds = dp_decode(logits[:, p], k)
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
    mae_nz    = (preds[true_nz].float() - labels[true_nz].float()).abs().mean().item()
    exact_acc = (preds == labels).all(dim=-1).float().mean().item()
    pred_total_mae = (preds.float().sum(-1) - labels.float().sum(-1)).abs().mean().item()

    return {
        "weighted_eae":      round(total_wsum / total_w if total_w > 0 else 0.0, 4),
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
        print(f"ファイルなし: {ndjson_path}"); sys.exit(1)
    print(f"読み込み中: {ndjson_path}", flush=True)

    _feat, _lab, _lred = [], [], []
    with open(ndjson_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            _feat.append(s["features"])
            _lab.append(s["label_hand"])
            _lred.append(s["label_red"])

    feat_np = np.array(_feat, dtype=np.float32); del _feat
    lab_np  = np.array(_lab,  dtype=np.int64);   del _lab
    lred_np = np.array(_lred, dtype=np.int64);   del _lred
    gc.collect()
    print(f"総サンプル数: {len(feat_np)}, shape: {feat_np.shape}")

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
    sl_tr   = slice(None, n_train)
    sl_va   = slice(n_train, n_train + n_val)
    sl_te   = slice(n_train + n_val, None)

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

    model = HandInferenceV18(
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

    best_val_nll = math.inf
    patience_cnt = 0
    start_epoch  = 1
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if resume and (MODEL_DIR / "model.pt").exists() and (MODEL_DIR / "train_log.json").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
        logs         = [json.loads(l) for l in (MODEL_DIR / "train_log.json").read_text().splitlines() if l.strip()]
        best_val_nll = min(e["val_nll"] for e in logs)
        last_epoch   = max(e["epoch"]   for e in logs)
        patience_cnt = 0
        for e in reversed(logs):
            if e["val_nll"] <= best_val_nll:
                break
            patience_cnt += 1
        start_epoch = last_epoch + 1
        print(f"resume: epoch {start_epoch} から再開  best_val_nll={best_val_nll:.4f}  patience={patience_cnt}",
              flush=True)

    val_eae_interval = CONFIG["val_eae_interval"]

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_nll = train_epoch(model, train_loader, optimizer, device)
        val_nll   = eval_nll(model, val_loader, device)
        scheduler.step(val_nll)

        # greedy val_eae は間引いて計算
        if epoch % val_eae_interval == 0:
            val_eae, val_acc = eval_eae(model, val_loader, device)
            print(f"epoch {epoch:3d}  train_nll={train_nll:.4f}  val_nll={val_nll:.4f}"
                  f"  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}", flush=True)
        else:
            val_eae, val_acc = None, None
            print(f"epoch {epoch:3d}  train_nll={train_nll:.4f}  val_nll={val_nll:.4f}",
                  flush=True)

        log_entry = {"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll,
                     "val_eae": val_eae, "val_acc": val_acc}
        with open(MODEL_DIR / "train_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        wait_for_cool()

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            patience_cnt = 0
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["early_stop_patience"]:
                print("early stopping")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    test_eae, test_acc = eval_eae(model, test_loader, device)
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
