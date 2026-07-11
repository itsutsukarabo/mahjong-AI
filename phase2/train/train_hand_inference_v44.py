"""
手牌類推モデル v44: 固定λ復帰 + yaku統合維持 + batch256

v43 からの変更:
  1. UncertaintyWeights廃止 → v39式固定λに全面復帰
     nll×1.0, SUM×0.05, CROSS×0.1, RED_CE×0.3, RED_CONS×0.1
     BLOCK×0.3, WAIT×0.5×(n_meld+1), SHANTEN×0.2, RETREAT×0.1
     PUSH×0.1, FURITEN×0.4, YAKU×0.1
  2. batch_size: 128 → 256
  3. scheduler: mode="max"(composite) → mode="min"(val_eae), patience=4
  4. early stopping: composite基準 → val_eae基準, patience=15
  5. eval_epoch: val_eae(inference_w)に加えval_eae_stage(stage_w)を参考値として併記
  6. yaku_head・label_yaku・label_won はv43から引き継ぎ維持
"""

import gc
import json
import math
import subprocess
import sys

# ---- fix_furiten_penalty: フリテン制約テーブルと損失関数 ----

def _build_furiten_mask():
    """tile t を保有する必要がある tatsu の (34, 113) 対応行列を構築する。"""
    import numpy as _np
    N_PAI = 34
    RYANMEN_BASE = 0; PENCHAN_BASE = 18; KANCHAN_BASE = 24
    TANKI_BASE = 45; SHANPON_BASE = 79
    mat = _np.zeros((N_PAI, 113), dtype=_np.float32)
    # 両面: index = suit*6+(n-1), n in 1-6, holds {suit*9+n, suit*9+n+1}
    for suit in range(3):
        for n in range(1, 7):
            t   = suit * 9 + n
            idx = RYANMEN_BASE + suit * 6 + (n - 1)
            mat[t,     idx] = 1.0
            mat[t + 1, idx] = 1.0
    # 辺張: [1,2]→PENCHAN+suit*2, [8,9]→PENCHAN+suit*2+1  (0-indexed: [0,1] [7,8])
    for suit in range(3):
        lo = PENCHAN_BASE + suit * 2
        mat[suit*9 + 0, lo] = 1.0;  mat[suit*9 + 1, lo] = 1.0
        hi = PENCHAN_BASE + suit * 2 + 1
        mat[suit*9 + 7, hi] = 1.0;  mat[suit*9 + 8, hi] = 1.0
    # 嵌張: index = suit*7+n, n in 0-6, holds {suit*9+n, suit*9+n+2}
    for suit in range(3):
        for n in range(7):
            idx = KANCHAN_BASE + suit * 7 + n
            mat[suit*9 + n,     idx] = 1.0
            mat[suit*9 + n + 2, idx] = 1.0
    # 単騎/双碰: tile t → TANKI_BASE+t / SHANPON_BASE+t
    for t in range(N_PAI):
        mat[t, TANKI_BASE  + t] = 1.0
        mat[t, SHANPON_BASE + t] = 1.0
    return mat

_FURITEN_MASK_NP = _build_furiten_mask()


def compute_furiten_penalties(wait_logits, disc_tok, disc_mask, labels_shanten):
    """
    フリテン系ペナルティ。テンパイサンプルのみ対象。

    wait_logits:    (B, P, 113)
    disc_tok:       (B, P, T, 44)
      dim34=turn_norm  dim35=tsumogiri  dim36=riichi_decl  dim38=self_role
    disc_mask:      (B, P, T)  True=padding
    labels_shanten: (B, P)     0=tenpai
    """
    import torch
    B, P, T, TD = disc_tok.shape
    device = wait_logits.device

    fm     = torch.tensor(_FURITEN_MASK_NP, dtype=torch.float32, device=device)  # (34, 113)
    valid  = (~disc_mask).float()                        # (B, P, T)
    tenpai = (labels_shanten == 0).float().to(device)   # (B, P)

    self_role  = disc_tok[..., 38]  # 1 = target player の自捨て
    turn_norm  = disc_tok[..., 34]  # 捨て牌通し番号 / 70
    riichi_bit = disc_tok[..., 36]  # 1 = リーチ宣言トークン

    # ---- (1) 自捨て牌フリテン ----
    self_tiles = (
        disc_tok[..., :34]
        * self_role.unsqueeze(-1)
        * valid.unsqueeze(-1)
    ).sum(dim=2).clamp(0, 1)                             # (B, P, 34)

    mask_self = (
        torch.matmul(self_tiles, fm).clamp(0, 1)
        * tenpai.unsqueeze(-1)
    )                                                    # (B, P, 113)

    # ---- (2) スルーフリテン: リーチ後他家捨て牌スルー ----
    riichi_self = self_role * riichi_bit * valid         # (B, P, T)
    in_riichi   = (riichi_self.sum(dim=2) > 0).float()  # (B, P)

    # リーチ宣言の turn_norm (未リーチは 999 で "全て対象外")
    riichi_turn = torch.where(
        in_riichi > 0,
        (turn_norm * riichi_self).max(dim=2).values,
        torch.full((B, P), 999.0, device=device),
    )                                                    # (B, P)

    after_riichi = (turn_norm > riichi_turn.unsqueeze(2)).float()  # (B, P, T)
    other_role   = 1.0 - self_role                                 # (B, P, T)

    passed_tiles = (
        disc_tok[..., :34]
        * other_role.unsqueeze(-1)
        * after_riichi.unsqueeze(-1)
        * valid.unsqueeze(-1)
    ).sum(dim=2).clamp(0, 1)                             # (B, P, 34)

    mask_passed = (
        torch.matmul(passed_tiles, fm).clamp(0, 1)
        * tenpai.unsqueeze(-1)
        * in_riichi.unsqueeze(-1)
    )                                                    # (B, P, 113)

    # ---- 合算してペナルティ計算 ----
    furiten_mask = (mask_self + mask_passed).clamp(0, 1)
    denom = furiten_mask.sum()
    if denom < 1.0:
        return torch.tensor(0.0, device=device)

    return (torch.sigmoid(wait_logits) * furiten_mask).sum() / denom

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
from feature_offsets import (
    HI_TOTAL,
    HI_GAME_START, HI_GAME_DIM,
    HI_SCORE_START, HI_SCORE_DIM,
    HI_SELF_MELD, HI_MELD_DIM,
    HI_VISIBLE,
    HI_RED_VIS,
    HI_WIND_START, HI_WIND_DIM,
    HI_TOKEN_DIM, HI_TOKEN_MAX,
)

# ---- 設定 ----

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v44"

CONFIG = {
    "fixed_dim":    HI_TOTAL,   # 674
    "token_dim":    HI_TOKEN_DIM,  # 44
    "disc_d":       128,   # DiscardEncoder 出力次元
    "disc_nhead":   4,
    "disc_layers":  3,
    "d_model":      256,
    "nhead":        4,
    "num_layers":   3,
    "n_pai":        34,
    "n_count_cls":  5,
    "n_players":    3,
    "n_blocks":     N_BLOCKS,
    "dropout":      0.1,
    "lr":           5e-4,
    "weight_decay": 1e-4,
    "batch_size":   256,
    "epochs":       200,
    "early_stop_patience": 15,
}

# v41 レイアウトに対応したオフセット（yaku_prob廃止・673+tenpai=674次元）
VISIBLE_OFFSET   = HI_VISIBLE        # 97
REMAINING_OFFSET = HI_GAME_START     # 50
MELD_CHI_OFFSET  = 34               # target_meld[34] = n_chi
MELD_PON_OFFSET  = 35               # target_meld[35] = n_pon
MELD_KAN_OFFSET  = 36               # target_meld[36] = n_kan

# self_feat: 視点プレイヤーの文脈特徴量 (retreat/push ヘッド用)
# score(11) + game_state(9) + self_meld(38) + visible_counts(34) + red_visible(3) = 95
SELF_SCORE_START   = HI_SCORE_START   # 39
SELF_GAME_START    = HI_GAME_START    # 50
SELF_SELF_MELD     = HI_SELF_MELD     # 59
SELF_VISIBLE       = HI_VISIBLE       # 97
SELF_RED_VIS       = HI_RED_VIS       # 134
SELF_FEAT_DIM = HI_SCORE_DIM + HI_GAME_DIM + HI_MELD_DIM + 34 + 3  # 95

GPU_TEMP_THRESHOLD = 76
GPU_COOL_INTERVAL  = 20
GPU_COOL_TARGET    = 70
N_PAI              = 34

LAMBDA_SUM      = 0.05   # 手牌総枚数 MSE
LAMBDA_CROSS    = 0.1    # 牌種上限制約
LAMBDA_RED_CE   = 0.3
LAMBDA_RED_CONS = 0.1
LAMBDA_BLOCK    = 0.3
LAMBDA_WAIT     = 0.5    # wait mask に (n_meld+1) スケーリング付き
LAMBDA_SHANTEN  = 0.2
LAMBDA_RETREAT  = 0.1
LAMBDA_PUSH     = 0.1
LAMBDA_FURITEN  = 0.4
LAMBDA_YAKU     = 0.1

N_YAKU             = 21
POS_WEIGHT_MAX     = 10.0
GRAD_CLIP_NORM     = 1.0

WAIT_EVAL_THRESH   = 0.3
N_SHANTEN_CLASSES  = 3   # 0=テンパイ, 1=1向聴, 2=2向聴以上

# ---- ターツ形 (v36 と同一) ----

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


def _can_form_sets_only(counts, n_sets):
    if n_sets == 0: return all(c == 0 for c in counts)
    for first in range(34):
        if counts[first] > 0: break
    else: return False
    if counts[first] >= 3:
        counts[first] -= 3
        if _can_form_sets_only(counts, n_sets - 1): counts[first] += 3; return True
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
            if _can_form_sets_only(counts, n_sets): counts[p] += 2; return True
            counts[p] += 2
    return False

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
        if n == 0:     candidates.append(([PENCHAN_BASE + suit*2],         frozenset({t+2})))
        elif n == 7:   candidates.append(([PENCHAN_BASE + suit*2 + 1],     frozenset({t-1})))
        else:          candidates.append(([RYANMEN_BASE + suit*6 + (n-1)], frozenset({t-1, t+2})))
    for t in range(27):
        suit = t // 9; n = t % 9
        if n > 6: continue
        t1, t2 = t, t + 2
        if counts[t1] < 1 or counts[t2] < 1: continue
        counts[t1] -= 1; counts[t2] -= 1
        valid = _can_form_sets_and_pair(counts, n_complete - 1)
        counts[t1] += 1; counts[t2] += 1
        if valid: candidates.append(([KANCHAN_BASE + suit*7 + n], frozenset({t+1})))
    for t in range(34):
        if counts[t] < 1: continue
        counts[t] -= 1
        if _can_form_sets_only(counts, n_complete): candidates.append(([TANKI_BASE + t], frozenset({t})))
        counts[t] += 1
    for t1 in range(34):
        if counts[t1] < 2: continue
        for t2 in range(t1+1, 34):
            if counts[t2] < 2: continue
            counts[t1] -= 2; counts[t2] -= 2
            if _can_form_sets_only(counts, n_complete-1):
                candidates.append(([SHANPON_BASE+t1, SHANPON_BASE+t2], frozenset({t1, t2})))
            counts[t1] += 2; counts[t2] += 2
    if not candidates: return np.zeros(N_TATSU, dtype=np.float32)
    active = set()
    for i, (il, wi) in enumerate(candidates):
        if not any(wi < wj for j, (_, wj) in enumerate(candidates) if i != j):
            active.update(il)
    labels = np.zeros(N_TATSU, dtype=np.float32)
    for idx in active: labels[idx] = 1.0
    return labels

def make_tatsu_label_batch(labels_hand_np, labels_shanten_np):
    B, P, _ = labels_hand_np.shape
    out = np.zeros((B, P, N_TATSU), dtype=np.float32)
    for b in range(B):
        for p in range(P):
            if labels_shanten_np[b, p] != 0: continue
            counts34 = labels_hand_np[b, p].tolist(); total = int(sum(counts34))
            if total == 0 or total % 3 != 1: continue
            out[b, p] = make_tatsu_labels_for_hand(counts34, (13 - total) // 3)
    return out


def compute_true_waits(counts34, n_melds):
    waits = []
    for t in range(N_PAI):
        c = list(counts34)
        if c[t] >= 4: continue
        c[t] += 1
        if compute_shanten(c, n_melds) == -1: waits.append(t)
    return waits

def tatsu_probs_to_tile_probs(tatsu_probs):
    tp = np.zeros(N_PAI, dtype=np.float32)
    for suit in range(3):
        for n in range(1, 7):
            t = suit*9+n; p = tatsu_probs[RYANMEN_BASE+suit*6+(n-1)]
            tp[t-1] = max(tp[t-1], p); tp[t+2] = max(tp[t+2], p)
    for suit in range(3):
        tp[suit*9+2] = max(tp[suit*9+2], tatsu_probs[PENCHAN_BASE+suit*2])
        tp[suit*9+6] = max(tp[suit*9+6], tatsu_probs[PENCHAN_BASE+suit*2+1])
    for suit in range(3):
        for n in range(7): tp[suit*9+n+1] = max(tp[suit*9+n+1], tatsu_probs[KANCHAN_BASE+suit*7+n])
    for t in range(N_PAI):
        tp[t] = max(tp[t], tatsu_probs[TANKI_BASE+t])
        tp[t] = max(tp[t], tatsu_probs[SHANPON_BASE+t])
    return tp

@torch.no_grad()
def eval_wait_metrics(model, loader, device, threshold=WAIT_EVAL_THRESH):
    """wait_f1 等を inference_w 重みづき加重平均で返す。"""
    model.eval()
    soft_f1_wsum=0.0; f1_wsum=0.0; hit_wsum=0.0; top1_wsum=0.0; wsum=0.0; n_tenpai=0
    for batch in loader:
        features = batch[0].to(device)
        disc_tok  = batch[1].to(device)
        disc_mask = batch[2].to(device)
        labels = batch[3]; labels_shanten = batch[6]
        out = model(features, disc_tok, disc_mask)
        wait_tatsu_probs = torch.sigmoid(out[4]).cpu().numpy()
        hand_np=labels.numpy(); sh_np=labels_shanten.numpy()
        iw_per_player = [get_inference_weights(features[:, p]).cpu().numpy() for p in range(3)]
        for b in range(len(features)):
            for p in range(3):
                if sh_np[b,p] != 0: continue
                counts34=hand_np[b,p].tolist(); total=int(sum(counts34))
                if total==0 or total%3!=1: continue
                true_waits=set(compute_true_waits(counts34,(13-total)//3))
                if not true_waits: continue
                tile_probs=tatsu_probs_to_tile_probs(wait_tatsu_probs[b,p])
                top1=int(np.argmax(tile_probs))
                w = float(iw_per_player[p][b])

                # ---- soft F1 ----
                soft_tp = sum(tile_probs[t] for t in true_waits)
                soft_fp = sum(tile_probs[t] for t in range(N_PAI) if t not in true_waits)
                soft_fn = len(true_waits) - soft_tp
                denom   = 2*soft_tp + soft_fp + soft_fn
                soft_f1 = (2*soft_tp / denom) if denom > 0 else 0.0

                # ---- threshold ベース wait_f1 ----
                pred_waits={t for t in range(N_PAI) if tile_probs[t]>=threshold}
                tp=len(pred_waits&true_waits); fp=len(pred_waits-true_waits); fn=len(true_waits-pred_waits)
                prec=tp/(tp+fp) if (tp+fp)>0 else 0.0
                recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1=2*prec*recall/(prec+recall) if (prec+recall)>0 else 0.0

                soft_f1_wsum += soft_f1 * w
                f1_wsum      += f1 * w
                hit_wsum     += (1.0 if tp>0 else 0.0) * w
                top1_wsum    += (1.0 if top1 in true_waits else 0.0) * w
                wsum         += w
                n_tenpai     += 1
    if wsum == 0:
        return {"soft_f1":0.0,"wait_f1":0.0,"wait_hit_rate":0.0,"wait_top1_acc":0.0,"n_tenpai":0}
    return {
        "soft_f1":       round(float(soft_f1_wsum) / float(wsum), 4),
        "wait_f1":       round(float(f1_wsum) / float(wsum), 4),
        "wait_hit_rate": round(float(hit_wsum) / float(wsum), 4),
        "wait_top1_acc": round(float(top1_wsum) / float(wsum), 4),
        "n_tenpai":      int(n_tenpai),
    }


# ---- 制約付きソフトマックス ----

def find_lambda(logits, k_float, n_iter=50):
    count_vals=torch.arange(5,device=logits.device,dtype=torch.float32); B=logits.shape[0]
    with torch.no_grad():
        lo=torch.full((B,),-20.0,device=logits.device); hi=torch.full((B,),20.0,device=logits.device)
        for _ in range(n_iter):
            mid=(lo+hi)/2; adj=logits-mid.view(B,1,1)*count_vals
            e_sum=(F.softmax(adj,dim=-1)*count_vals).sum(-1).sum(-1)
            lo=torch.where(e_sum>k_float,mid,lo); hi=torch.where(e_sum>k_float,hi,mid)
    return (lo+hi)/2

def constrained_softmax_probs(logits, k):
    count_vals=torch.arange(5,device=logits.device,dtype=torch.float32)
    lam=find_lambda(logits,k.float())
    return F.softmax(logits-lam.view(-1,1,1)*count_vals,dim=-1)

def get_stage_weights(features):
    """残り牌数ベースの4段階重み(v39互換)。val_eae_stage 参考値計算用。"""
    remaining = (features[:, REMAINING_OFFSET] * 70).round()
    meld_tiles = (features[:, MELD_CHI_OFFSET] * 3
                  + features[:, MELD_PON_OFFSET] * 3
                  + features[:, MELD_KAN_OFFSET] * 4)
    eff = (remaining - meld_tiles * 2).clamp(min=0).long()
    w = torch.ones(len(features), device=features.device)
    w[eff <= 10] = 4.0
    w[(eff > 10) & (eff <= 30)] = 3.0
    w[(eff > 30) & (eff <= 50)] = 2.0
    return w

def get_inference_weights(features, n_pai=34):
    """非公開牌数（wall + 相手手牌）の逆数で重みづけ。
    visible_counts (VISIBLE_OFFSET, 34次元, /4正規化) には捨て牌・副露自手タイル・
    自分の手牌が含まれ、副露タイルは公開情報として扱われる。
    非公開牌数が少ない（推定が容易）ほど重みを大きくする。
    """
    visible_total = (features[:, VISIBLE_OFFSET:VISIBLE_OFFSET + n_pai] * 4).sum(-1)
    n_hidden = (4 * n_pai - visible_total).clamp(min=1.0)
    w = 1.0 / n_hidden
    mean_w = w.mean()
    return w / mean_w if mean_w > 0 else w



def weighted_eae(probs, labels, weights):
    count_vals=torch.arange(probs.shape[-1],device=probs.device,dtype=torch.float32)
    eae=((count_vals-labels.unsqueeze(-1).float()).abs()*probs).sum(-1).sum(-1)
    return (eae*weights).sum()/weights.sum()


# ---- GPU温度 ----

def gpu_temp():
    for cmd in ["nvidia-smi","/usr/lib/wsl/lib/nvidia-smi","/mnt/c/Windows/System32/nvidia-smi.exe"]:
        try:
            r=subprocess.run([cmd,"--query-gpu=temperature.gpu","--format=csv,noheader"],capture_output=True,text=True,timeout=5)
            if r.returncode==0 and r.stdout.strip(): return int(r.stdout.strip())
        except Exception: continue
    return 0

def wait_for_cool():
    temp=gpu_temp()
    if temp<=GPU_TEMP_THRESHOLD: return
    print(f"  [thermal] GPU {temp}C > {GPU_TEMP_THRESHOLD}C - cooling...",flush=True)
    while True:
        time.sleep(GPU_COOL_INTERVAL); temp=gpu_temp()
        print(f"  [thermal] GPU {temp}C",flush=True)
        if temp<=GPU_TEMP_THRESHOLD: print("  [thermal] cool enough, resuming",flush=True); break

def cool_after_epoch():
    temp=gpu_temp()
    if temp<=GPU_COOL_TARGET: return
    print(f"  [epoch-cool] GPU {temp}C → waiting for {GPU_COOL_TARGET}C...",flush=True)
    while True:
        time.sleep(GPU_COOL_INTERVAL); temp=gpu_temp()
        print(f"  [epoch-cool] GPU {temp}C",flush=True)
        if temp<=GPU_COOL_TARGET: print("  [epoch-cool] ready",flush=True); break


# ---- データセット ----

class HandInferenceDataset(Dataset):
    def __init__(self, features, disc_tokens_list,
                 labels, labels_red, labels_block, labels_shanten,
                 noise_mask, retreat_label, push_label,
                 labels_yaku, labels_won):
        self.features       = torch.as_tensor(features,       dtype=torch.float32)
        self.disc_tokens    = disc_tokens_list   # List[List[np.ndarray(n,44)]]
        self.labels         = torch.as_tensor(labels,         dtype=torch.long).clamp(0, 4)
        self.labels_red     = torch.as_tensor(labels_red,     dtype=torch.long)
        self.labels_block   = torch.as_tensor(labels_block,   dtype=torch.float32)
        self.labels_shanten = torch.as_tensor(labels_shanten, dtype=torch.long)
        self.noise_mask     = torch.as_tensor(noise_mask,     dtype=torch.bool)
        self.retreat_label  = torch.as_tensor(retreat_label,  dtype=torch.float32)
        self.push_label     = torch.as_tensor(push_label,     dtype=torch.float32)
        self.labels_yaku    = torch.as_tensor(labels_yaku,    dtype=torch.float32)
        self.labels_won     = torch.as_tensor(labels_won,     dtype=torch.bool)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx],
                self.disc_tokens[idx],   # List[np.ndarray] per player
                self.labels[idx], self.labels_red[idx],
                self.labels_block[idx], self.labels_shanten[idx],
                self.noise_mask[idx], self.retreat_label[idx], self.push_label[idx],
                self.labels_yaku[idx], self.labels_won[idx])


def collate_fn(batch):
    """Variable-length discard tokens をバッチ内最大長でパディングする。"""
    (features, disc_tok_list,
     labels, labels_red, labels_block, labels_shanten,
     noise_mask, retreat_label, push_label,
     labels_yaku, labels_won) = zip(*batch)

    B = len(features)
    P = 3
    TOKEN_DIM = HI_TOKEN_DIM

    # バッチ内最大トークン長
    max_len = 1
    for tok in disc_tok_list:
        for p in range(P):
            if tok[p].shape[0] > max_len:
                max_len = tok[p].shape[0]

    padded = torch.zeros(B, P, max_len, TOKEN_DIM, dtype=torch.float32)
    mask   = torch.ones(B, P, max_len, dtype=torch.bool)  # True = padding

    for i, tok in enumerate(disc_tok_list):
        for p in range(P):
            n = tok[p].shape[0]
            if n > 0:
                n_clip = min(n, max_len)
                padded[i, p, :n_clip] = torch.from_numpy(tok[p][:n_clip].astype(np.float32))
                mask[i, p, :n_clip]   = False

    return (torch.stack(features), padded, mask,
            torch.stack(labels), torch.stack(labels_red),
            torch.stack(labels_block), torch.stack(labels_shanten),
            torch.stack(noise_mask), torch.stack(retreat_label), torch.stack(push_label),
            torch.stack(labels_yaku), torch.stack(labels_won))


# ---- モデル ----

class DiscardEncoder(nn.Module):
    """統一イベントトークン列 → コンテキストベクトル (disc_d次元)"""

    def __init__(self, token_dim=44, disc_d=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(token_dim, disc_d)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, disc_d))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=disc_d, nhead=nhead, dim_feedforward=disc_d*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens, padding_mask):
        """
        tokens:       (B, N, token_dim)
        padding_mask: (B, N)  True=padding
        returns:      (B, disc_d)  CLS token 出力
        """
        B = tokens.shape[0]
        x = self.input_proj(tokens)                              # (B, N, disc_d)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, disc_d)
        x   = torch.cat([cls, x], dim=1)                        # (B, N+1, disc_d)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        mask = torch.cat([cls_mask, padding_mask], dim=1)       # (B, N+1)
        out  = self.transformer(x, src_key_padding_mask=mask)   # (B, N+1, disc_d)
        return out[:, 0, :]                                      # CLS → (B, disc_d)


class HandInferenceV37(nn.Module):
    """
    v41: DiscardEncoder + タイルレベル Transformer (統一イベントストリーム) + yaku_head 統合

    入力:
      x:            (B, 3, fixed_dim=674)
      disc_tokens:  (B, 3, N, token_dim=44)  パディング済み
      disc_mask:    (B, 3, N)  True=パディング
    """

    def __init__(self, fixed_dim, token_dim, disc_d, disc_nhead, disc_layers,
                 d_model, nhead, num_layers,
                 n_pai=34, n_count_cls=5, n_players=3, n_blocks=N_BLOCKS,
                 n_tatsu=N_TATSU, n_shanten_cls=N_SHANTEN_CLASSES,
                 n_yaku=N_YAKU, dropout=0.1):
        super().__init__()
        self.n_pai=n_pai; self.n_count_cls=n_count_cls
        self.n_players=n_players; self.n_blocks=n_blocks; self.n_tatsu=n_tatsu
        self.n_yaku=n_yaku

        self.discard_encoder = DiscardEncoder(token_dim, disc_d, disc_nhead, disc_layers, dropout)

        self.global_encoder = nn.Sequential(
            nn.Linear(fixed_dim + disc_d, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, d_model),
        )
        self.tile_embed   = nn.Embedding(n_pai, d_model)
        self.player_embed = nn.Embedding(n_players, d_model)
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head       = nn.Linear(d_model, n_count_cls)
        self.red_head   = nn.Linear(d_model, 2)
        self.block_head = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,1))
        self.wait_pool  = nn.Linear(d_model, 1, bias=False)
        self.wait_head  = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,n_tatsu))

        self.shanten_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, n_shanten_cls)
        )

        # yaku_head: per-player attention-pooled 表現から役を推定
        self.yaku_head = nn.Linear(d_model, n_yaku)

        # retreat/push: 視点プレイヤー固定特徴量 (95次元) から推定
        self.self_encoder = nn.Sequential(
            nn.Linear(SELF_FEAT_DIM, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        self.retreat_head = nn.Linear(32, 1)
        self.push_head    = nn.Linear(32, 1)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("player_ids",   torch.arange(n_players))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4,13,22]))
        self.register_buffer("block_sel",
            torch.tensor(build_block_selection_matrix(include_tatsu=True), dtype=torch.float32))

    def forward(self, x, disc_tokens, disc_mask):
        """
        x:           (B, P, fixed_dim)
        disc_tokens: (B, P, N, token_dim)
        disc_mask:   (B, P, N)  True=padding
        """
        B, P, F = x.shape
        _, _, N, TD = disc_tokens.shape

        # DiscardEncoder を 3プレイヤー分まとめて処理
        tok_flat  = disc_tokens.reshape(B*P, N, TD)        # (B*P, N, TD)
        mask_flat = disc_mask.reshape(B*P, N)              # (B*P, N)
        ctx_flat  = self.discard_encoder(tok_flat, mask_flat)   # (B*P, disc_d)
        ctx       = ctx_flat.reshape(B, P, -1)             # (B, P, disc_d)

        # 固定特徴量とコンテキストを結合
        x_combined = torch.cat([x, ctx], dim=-1)           # (B, P, fixed_dim+disc_d)
        g = self.global_encoder(x_combined.reshape(B*P, -1)).reshape(B, P, -1)  # (B, P, d_model)

        vis    = x[:, :, VISIBLE_OFFSET:VISIBLE_OFFSET+self.n_pai]
        tokens = (g.unsqueeze(2)
                  + self.tile_embed(self.tile_ids)
                  + self.player_embed(self.player_ids).view(1, P, 1, -1)
                  + self.visible_proj(vis.unsqueeze(-1)))
        out = self.transformer(tokens.reshape(B, P*self.n_pai, -1)).reshape(B, P, self.n_pai, -1)

        vis_raw    = (vis*4).round().long().clamp(0,4)
        logits_raw = self.head(out)
        mask_c     = self.count_range > (4-vis_raw).clamp(min=0).unsqueeze(-1)
        logits     = logits_raw.masked_fill(mask_c, float('-inf'))

        red_logits   = self.red_head(out[:, :, self.red_tile_idx, :])
        out_flat     = out.reshape(B*P, self.n_pai, -1)
        block_logits = self.block_head(
            torch.bmm(self.block_sel.unsqueeze(0).expand(B*P,-1,-1), out_flat)
        ).squeeze(-1).reshape(B, P, self.n_blocks)

        attn_w         = torch.softmax(self.wait_pool(out).squeeze(-1), dim=-1)
        pooled         = (out * attn_w.unsqueeze(-1)).sum(dim=2)
        wait_logits    = self.wait_head(pooled)
        shanten_logits = self.shanten_head(pooled)
        yaku_logits    = self.yaku_head(pooled)    # (B, P, N_YAKU)

        # 視点プレイヤー自身の固定特徴量 (全 target で同一の viewer 文脈)
        self_feat = torch.cat([
            x[:, 0, SELF_SCORE_START : SELF_SCORE_START + HI_SCORE_DIM],
            x[:, 0, SELF_GAME_START  : SELF_GAME_START  + HI_GAME_DIM],
            x[:, 0, SELF_SELF_MELD   : SELF_SELF_MELD   + HI_MELD_DIM],
            x[:, 0, SELF_VISIBLE     : SELF_VISIBLE     + 34],
            x[:, 0, SELF_RED_VIS     : SELF_RED_VIS     + 3],
        ], dim=-1)                                         # (B, 95)
        self_h        = self.self_encoder(self_feat)
        retreat_logit = self.retreat_head(self_h).squeeze(-1)
        push_logit    = self.push_head(self_h).squeeze(-1)

        return logits, logits_raw, red_logits, block_logits, wait_logits, retreat_logit, push_logit, shanten_logits, yaku_logits


# ---- 学習 ----

def train_epoch(model, loader, optimizer, device,
                pos_weight_block, pos_weight_wait, pos_weight_retreat, pos_weight_push):
    model.train()
    total_ce=0.0
    count_vals=torch.arange(model.n_count_cls, device=device, dtype=torch.float32)
    five_idx=model.red_tile_idx

    for batch in loader:
        (features, disc_tok, disc_mask,
         labels, labels_red, labels_block, labels_shanten,
         noise_mask, retreat_label, push_label,
         labels_yaku, labels_won) = [x.to(device) for x in batch]
        optimizer.zero_grad()

        logits, logits_raw, red_logits, block_logits, wait_logits, \
            retreat_logit, push_logit, shanten_logits, yaku_logits = model(features, disc_tok, disc_mask)
        B, P = features.shape[:2]

        sample_w    = (~noise_mask).float()
        infer_w     = torch.stack([get_inference_weights(features[:,p]) for p in range(P)], dim=1)
        effective_w = infer_w * sample_w.unsqueeze(1)

        loss_nll_per = F.cross_entropy(
            logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1), reduction='none'
        ).reshape(B, P, -1).mean(-1)
        denom = effective_w.sum()
        loss_nll = (loss_nll_per*effective_w).sum()/denom if denom>0 else torch.tensor(0.,device=device)

        probs=F.softmax(logits_raw,dim=-1); pred_sum=(probs*count_vals).sum(-1)
        loss_sum   = F.mse_loss(pred_sum.sum(-1), labels.float().sum(-1))
        visible_counts = features[:,0,VISIBLE_OFFSET:VISIBLE_OFFSET+model.n_pai]*4
        loss_cross = F.relu(pred_sum.sum(dim=1)-(4-visible_counts).clamp(min=0)).mean()

        loss_red_ce   = F.cross_entropy(red_logits.reshape(-1,2), labels_red.reshape(-1))
        prob_has_red  = F.softmax(red_logits,dim=-1)[:,:,:,1]
        prob_cnt_ge1  = 1-probs[:,:,five_idx,0]
        loss_red_cons = F.relu(prob_has_red-prob_cnt_ge1).mean()

        block_w   = torch.where(labels_shanten<=1, 0.5**labels_shanten.float(),
                                  torch.zeros_like(labels_shanten.float())) * sample_w.unsqueeze(1) * infer_w
        denom_blk = block_w.reshape(-1).sum()
        loss_block = (F.binary_cross_entropy_with_logits(
            block_logits.reshape(-1,model.n_blocks), labels_block.reshape(-1,model.n_blocks),
            pos_weight=pos_weight_block, reduction='none'
        ).mean(dim=-1)*block_w.reshape(-1)).sum()/denom_blk if denom_blk>0 else torch.tensor(0.,device=device)

        n_melds = (features[:, :, MELD_CHI_OFFSET]
                   + features[:, :, MELD_PON_OFFSET]
                   + features[:, :, MELD_KAN_OFFSET])  # (B, P)
        wait_mask  = (labels_shanten==0).float() * sample_w.unsqueeze(1) * (n_melds + 1) * infer_w
        denom_wait = wait_mask.sum()
        if denom_wait > 0:
            tatsu_labels = torch.tensor(
                make_tatsu_label_batch(labels.cpu().numpy(), labels_shanten.cpu().numpy()), device=device
            )
            loss_wait = (F.binary_cross_entropy_with_logits(
                wait_logits.reshape(-1,model.n_tatsu), tatsu_labels.reshape(-1,model.n_tatsu),
                pos_weight=pos_weight_wait, reduction='none'
            ).mean(dim=-1)*wait_mask.reshape(-1)).sum()/(denom_wait*model.n_tatsu)
        else:
            loss_wait = torch.tensor(0.0, device=device)

        sh_label_3cls = labels_shanten.clamp(0, N_SHANTEN_CLASSES - 1)
        loss_shanten = (F.cross_entropy(
            shanten_logits.reshape(-1, N_SHANTEN_CLASSES), sh_label_3cls.reshape(-1), reduction='none'
        )*effective_w.reshape(-1)).sum()/denom if denom>0 else torch.tensor(0.,device=device)

        infer_w_p0 = infer_w[:, 0]   # player 0 の推論重み (retreat/push/yaku は p0 のみ)
        rp_w   = sample_w * infer_w_p0
        denom_ret = rp_w.sum()
        loss_retreat = (F.binary_cross_entropy_with_logits(
            retreat_logit, retreat_label, pos_weight=pos_weight_retreat, reduction='none'
        )*rp_w).sum()/denom_ret if denom_ret>0 else torch.tensor(0.,device=device)
        loss_push = (F.binary_cross_entropy_with_logits(
            push_logit, push_label, pos_weight=pos_weight_push, reduction='none'
        )*rp_w).sum()/denom_ret if denom_ret>0 else torch.tensor(0.,device=device)

        furiten_pen = compute_furiten_penalties(
            wait_logits, disc_tok, disc_mask, labels_shanten)

        # 役損失: 和了サンプル (target player p=0) のみ、副露数に応じた重み付きBCE
        # 副露0→×1.0、副露1→×2.0、副露2→×3.0、副露3→×4.0、副露4→×0.0
        won_mask = labels_won  # (B,) bool
        if won_mask.any():
            meld_cnt = (features[:, 0, MELD_CHI_OFFSET]
                        + features[:, 0, MELD_PON_OFFSET]
                        + features[:, 0, MELD_KAN_OFFSET]).round().long().clamp(0, 4)
            _YAKU_MELD_W = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0], device=device)
            yaku_w = _YAKU_MELD_W[meld_cnt] * infer_w_p0   # 副露数 × 非公開牌逆数
            won_w  = yaku_w[won_mask]
            loss_per = F.binary_cross_entropy_with_logits(
                yaku_logits[:, 0, :][won_mask], labels_yaku[won_mask],
                reduction="none"
            ).mean(dim=-1)
            denom_y = won_w.sum()
            loss_yaku = (loss_per * won_w).sum() / denom_y if denom_y > 0 else torch.tensor(0.0, device=device)
        else:
            loss_yaku = torch.tensor(0.0, device=device)

        loss = (loss_nll
                + LAMBDA_SUM      * loss_sum
                + LAMBDA_CROSS    * loss_cross
                + LAMBDA_RED_CE   * loss_red_ce
                + LAMBDA_RED_CONS * loss_red_cons
                + LAMBDA_BLOCK    * loss_block
                + LAMBDA_WAIT     * loss_wait
                + LAMBDA_SHANTEN  * loss_shanten
                + LAMBDA_RETREAT  * loss_retreat
                + LAMBDA_PUSH     * loss_push
                + LAMBDA_FURITEN  * furiten_pen
                + LAMBDA_YAKU     * loss_yaku)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_ce += loss_nll.item() * len(features)

    return total_ce / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    """val_eae(inference_w主軸) と val_eae_stage(stage_w参考値) を両方返す。"""
    model.eval()
    total_wsum=0.0; total_w=0.0
    total_wsum_s=0.0; total_w_s=0.0
    correct=0; total=0
    for batch in loader:
        features = batch[0].to(device)
        disc_tok  = batch[1].to(device)
        disc_mask = batch[2].to(device)
        labels = batch[3].to(device)
        logits = model(features, disc_tok, disc_mask)[0]
        for p in range(3):
            k_p = labels[:,p].float().sum(dim=-1).long()
            probs_p = constrained_softmax_probs(logits[:,p], k_p)
            iw = get_inference_weights(features[:,p])
            total_wsum += weighted_eae(probs_p,labels[:,p],iw).item()*iw.sum().item()
            total_w    += iw.sum().item()
            sw = get_stage_weights(features[:,p])
            total_wsum_s += weighted_eae(probs_p,labels[:,p],sw).item()*sw.sum().item()
            total_w_s    += sw.sum().item()
            correct    += (probs_p.argmax(dim=-1)==labels[:,p]).sum().item()
            total      += labels[:,p].numel()
    val_eae       = total_wsum/total_w   if total_w>0   else float('inf')
    val_eae_stage = total_wsum_s/total_w_s if total_w_s>0 else float('inf')
    return val_eae, val_eae_stage, correct/total


@torch.no_grad()
def eval_shanten_acc(model, loader, device):
    """シャンテン数分類の精度（inference_w重みづけ）。"""
    model.eval()
    total_wsum = 0.0; total_w = 0.0
    for batch in loader:
        features       = batch[0].to(device)
        disc_tok       = batch[1].to(device)
        disc_mask      = batch[2].to(device)
        labels_shanten = batch[6].to(device)
        shanten_logits = model(features, disc_tok, disc_mask)[7]   # (B, P, N_SHANTEN_CLASSES)
        preds = shanten_logits.argmax(dim=-1)                      # (B, P)
        gt    = labels_shanten.clamp(0, N_SHANTEN_CLASSES - 1)
        correct_mask = (preds == gt).float()                       # (B, P)
        for p in range(3):
            iw = get_inference_weights(features[:, p])
            total_wsum += (correct_mask[:, p] * iw).sum().item()
            total_w    += iw.sum().item()
    return total_wsum / total_w if total_w > 0 else 0.0


def main():
    resume = "--resume" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  resume: {resume}", flush=True)

    ndjson_path = DATA_DIR / "hand_inference_v41.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}")
        print("先に以下を実行してください:")
        print("  node phase2/scripts/extract_features.js --src phase2/data/states/states_v22.ndjson --dest phase2/data/features/")
        print("  python phase2/scripts/add_tenpai_features.py  (add_yaku_features.py は不要)")
        print("  python phase2/scripts/prepare_v41_data.py")
        print("  python phase2/scripts/add_block_labels.py --src hand_inference_v41_raw.ndjson --dst hand_inference_v41.ndjson --v2 --include-tatsu")
        print("  python phase2/scripts/add_intent_labels.py (必要に応じて)")
        sys.exit(1)

    print(f"読み込み中: {ndjson_path}", flush=True)

    INPUT_DIM  = CONFIG["fixed_dim"]   # 674
    TOKEN_DIM  = CONFIG["token_dim"]   # 44

    n_lines = sum(1 for ln in open(ndjson_path, encoding="utf-8") if ln.strip())
    print(f"総サンプル数: {n_lines}", flush=True)

    feat_np     = np.empty((n_lines, 3, INPUT_DIM), dtype=np.float32)
    lab_np      = np.empty((n_lines, 3, 34),        dtype=np.int64)
    lred_np     = np.empty((n_lines, 3, 3),         dtype=np.int64)
    lblk_np     = np.empty((n_lines, 3, N_BLOCKS),  dtype=np.float32)
    lsh_np      = np.empty((n_lines, 3),            dtype=np.int64)
    lnoise_np   = np.empty((n_lines,),              dtype=bool)
    lretreat_np = np.empty((n_lines,),              dtype=np.float32)
    lpush_np    = np.empty((n_lines,),              dtype=np.float32)
    lyaku_np    = np.zeros((n_lines, N_YAKU),       dtype=np.float32)
    lwon_np     = np.zeros((n_lines,),              dtype=bool)

    # discard_tokens は可変長 → float16 で格納してメモリ節約 (float32比 -50%)
    disc_tokens_list = []  # List[List[np.ndarray(n,42) float16]]

    noise_cnt=retreat_cnt=push_cnt=0
    with open(ndjson_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            s = json.loads(line)
            feat_np[i]     = s["features"]
            lab_np[i]      = s["label_hand"]
            lred_np[i]     = s["label_red"]
            lblk_np[i]     = s.get("label_block", [[0.0]*N_BLOCKS]*3)
            lsh_np[i]      = s.get("label_shanten", [4, 4, 4])
            lnoise_np[i]   = bool(s.get("label_noise", False))
            lretreat_np[i] = 1.0 if s.get("label_retreat", {}).get("is_retreat", False) else 0.0
            lpush_np[i]    = 1.0 if s.get("label_push", {}).get("is_push", False) else 0.0
            lyaku_np[i]    = s.get("label_yaku", [0.0] * N_YAKU)
            lwon_np[i]     = bool(s.get("label_won", False))
            if lnoise_np[i]:   noise_cnt += 1
            if lretreat_np[i]: retreat_cnt += 1
            if lpush_np[i]:    push_cnt += 1

            raw_tok = s.get("discard_tokens", [[], [], []])
            player_tokens = []
            for p in range(3):
                toks = raw_tok[p] if p < len(raw_tok) else []
                player_tokens.append(
                    np.array(toks, dtype=np.float16) if toks       # float16 で格納
                    else np.zeros((0, TOKEN_DIM), dtype=np.float16)
                )
            disc_tokens_list.append(player_tokens)
            del s  # JSON パース分のメモリを即解放

            if (i+1) % 50000 == 0:
                print(f"  {i+1}/{n_lines}", flush=True)

    gc.collect()
    print(f"noise={noise_cnt}  retreat={retreat_cnt}  push={push_cnt}")

    # 有効インデックスをコピーなしで計算 → Subset で分割
    lab_sum  = lab_np.sum(axis=-1)
    meld_sum = feat_np[:,:,MELD_CHI_OFFSET] + feat_np[:,:,MELD_PON_OFFSET] + feat_np[:,:,MELD_KAN_OFFSET]
    keep = (lab_sum.sum(axis=1) > 0) | (meld_sum.sum(axis=1) > 0)
    valid_idx = np.where(keep)[0]
    print(f"有効サンプル: {len(valid_idx)}")

    # pos_weight 計算 (有効サンプルのみで集計)
    pw_block   = min((1-float(lblk_np[valid_idx].mean()))/(float(lblk_np[valid_idx].mean())+1e-9), POS_WEIGHT_MAX)
    pw_wait    = min(0.98/0.02, POS_WEIGHT_MAX)
    n_valid    = len(valid_idx)
    pw_retreat = min((n_valid-retreat_cnt)/(retreat_cnt+1e-9), POS_WEIGHT_MAX)
    pw_push    = min((n_valid-push_cnt)/(push_cnt+1e-9), POS_WEIGHT_MAX)
    pos_weight_block   = torch.full((N_BLOCKS,),  pw_block,   device=device)
    pos_weight_wait    = torch.full((N_TATSU,),   pw_wait,    device=device)
    pos_weight_retreat = torch.tensor(pw_retreat, device=device)
    pos_weight_push    = torch.tensor(pw_push,    device=device)
    print(f"pw: block={pw_block:.2f} wait={pw_wait:.2f} retreat={pw_retreat:.2f} push={pw_push:.2f}")

    # シャッフルはインデックスのみ操作 (データコピーなし)
    rng = np.random.default_rng(42)
    shuffled = valid_idx[rng.permutation(n_valid)]
    n_train = int(n_valid * 0.8); n_val = int(n_valid * 0.1)
    train_idx_list = shuffled[:n_train].tolist()
    val_idx_list   = shuffled[n_train:n_train+n_val].tolist()
    test_idx_list  = shuffled[n_train+n_val:].tolist()

    # Dataset は全データを参照、Subset でスプリット (コピー不要)
    full_ds  = HandInferenceDataset(
        feat_np, disc_tokens_list,
        lab_np, lred_np, lblk_np, lsh_np,
        lnoise_np, lretreat_np, lpush_np,
        lyaku_np, lwon_np,
    )
    del feat_np, lab_np, lred_np, lblk_np, lsh_np, lnoise_np, lretreat_np, lpush_np, lyaku_np, lwon_np
    gc.collect()

    train_ds = torch.utils.data.Subset(full_ds, train_idx_list)
    val_ds   = torch.utils.data.Subset(full_ds, val_idx_list)
    test_ds  = torch.utils.data.Subset(full_ds, test_idx_list)
    print(f"train:{len(train_ds)}, val:{len(val_ds)}, test:{len(test_ds)}")

    use_gpu = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=use_gpu,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=use_gpu,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=use_gpu,
                              collate_fn=collate_fn)

    model = HandInferenceV37(
        fixed_dim   = CONFIG["fixed_dim"],
        token_dim   = CONFIG["token_dim"],
        disc_d      = CONFIG["disc_d"],
        disc_nhead  = CONFIG["disc_nhead"],
        disc_layers = CONFIG["disc_layers"],
        d_model     = CONFIG["d_model"],
        nhead       = CONFIG["nhead"],
        num_layers  = CONFIG["num_layers"],
        n_pai       = CONFIG["n_pai"],
        n_count_cls = CONFIG["n_count_cls"],
        n_players   = CONFIG["n_players"],
        n_blocks    = CONFIG["n_blocks"],
        n_tatsu     = N_TATSU,
        n_shanten_cls = N_SHANTEN_CLASSES,
        dropout     = CONFIG["dropout"],
    ).to(device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4)
    best_eae   = math.inf; patience_cnt = 0; start_epoch = 1

    ckpt_path = MODEL_DIR / "checkpoint.pt"
    if resume and ckpt_path.exists() and (MODEL_DIR/"train_log.json").exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        logs       = [json.loads(l) for l in (MODEL_DIR/"train_log.json").read_text().splitlines() if l.strip()]
        best_eae   = min((e.get("val_eae", math.inf) for e in logs), default=math.inf)
        start_epoch = max(e["epoch"] for e in logs) + 1
        patience_cnt = 0
        print(f"resume: epoch {start_epoch}  best_eae={best_eae:.4f}", flush=True)
    else:
        print("フルスクラッチ学習開始", flush=True)

    for epoch in range(start_epoch, CONFIG["epochs"]+1):
        train_nll                      = train_epoch(model, train_loader, optimizer, device,
                                                     pos_weight_block, pos_weight_wait,
                                                     pos_weight_retreat, pos_weight_push)
        val_eae, val_eae_stage, val_acc = eval_epoch(model, val_loader, device)
        shanten_acc                    = eval_shanten_acc(model, val_loader, device)
        wait_m                         = eval_wait_metrics(model, val_loader, device)

        composite = shanten_acc * 0.40 + wait_m["wait_f1"] * 0.40 + 1.0 / (1.0 + val_eae / 15.0) * 0.20
        scheduler.step(val_eae)

        log_entry = {"epoch": epoch,
                     "val_eae": round(val_eae, 4),
                     "val_eae_stage": round(val_eae_stage, 4),
                     "val_acc": round(val_acc, 4),
                     "shanten_acc": round(shanten_acc, 4),
                     "composite": round(composite, 4),
                     **wait_m}
        print(f"epoch {epoch:3d}  nll={train_nll:.4f}  eae={val_eae:.4f}  eae_stage={val_eae_stage:.4f}"
              f"  acc={val_acc:.4f}  shanten={shanten_acc:.4f}"
              f"  wait_f1={wait_m['wait_f1']:.4f}  composite={composite:.4f}", flush=True)
        with open(MODEL_DIR/"train_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        wait_for_cool()
        cool_after_epoch()

        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, ckpt_path)

        if val_eae < best_eae:
            best_eae = val_eae; patience_cnt = 0
            torch.save(model.state_dict(), MODEL_DIR/"model.pt")
            print(f"  [saved best_eae={best_eae:.4f}]", flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["early_stop_patience"]:
                print("early stopping"); break

    # テスト評価
    model.load_state_dict(torch.load(MODEL_DIR/"model.pt", map_location=device, weights_only=True))
    test_eae, test_eae_stage, test_acc = eval_epoch(model, test_loader, device)
    test_wait_m                        = eval_wait_metrics(model, test_loader, device)
    print(f"\n[model.pt] test_eae={test_eae:.4f}  test_eae_stage={test_eae_stage:.4f}  test_acc={test_acc:.4f}")
    for k, v in test_wait_m.items(): print(f"  {k}: {v}")

    (MODEL_DIR/"eval_result.json").write_text(json.dumps(
        {"test_eae":test_eae, "test_eae_stage":test_eae_stage,
         "test_acc":test_acc, **test_wait_m, "config":CONFIG}, indent=2))
    (MODEL_DIR/"config.json").write_text(json.dumps(
        {**CONFIG, "n_tatsu":N_TATSU, "n_shanten_cls":N_SHANTEN_CLASSES}, indent=2))

    # ONNX エクスポート
    model.eval().cpu()
    dummy_feat  = torch.zeros(1, 3, CONFIG["fixed_dim"])
    dummy_tok   = torch.zeros(1, 3, 10, CONFIG["token_dim"])   # N=10 で export (dynamic)
    dummy_mask  = torch.ones(1, 3, 10, dtype=torch.bool)

    class _Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, features, disc_tokens, disc_mask):
            logits,_,red_logits,block_logits,wait_logits,_,_,shanten_logits,_ = self.m(features, disc_tokens, disc_mask)
            return logits, red_logits, block_logits, wait_logits, shanten_logits

    torch.onnx.export(
        _Wrap(model), (dummy_feat, dummy_tok, dummy_mask),
        str(MODEL_DIR/"model.onnx"),
        input_names=["features", "disc_tokens", "disc_mask"],
        output_names=["logits","red_logits","block_logits","wait_logits","shanten_logits"],
        dynamic_axes={
            "features":  {0:"batch"},
            "disc_tokens": {0:"batch", 3:"n_tokens"},
            "disc_mask":   {0:"batch", 3:"n_tokens"},
            "logits":    {0:"batch"}, "red_logits":  {0:"batch"},
            "block_logits": {0:"batch"}, "wait_logits": {0:"batch"},
            "shanten_logits": {0:"batch"},
        },
        opset_version=17,
    )
    print(f"\nONNX保存: {MODEL_DIR/'model.onnx'}")
    print("入力: features(1,3,674), disc_tokens(1,3,N,44), disc_mask(1,3,N)")
    print("出力: logits(1,3,34,5) red_logits(1,3,3,2) block_logits(1,3,N_BLOCKS) wait_logits(1,3,113) shanten_logits(1,3,3)")


if __name__ == "__main__":
    main()
