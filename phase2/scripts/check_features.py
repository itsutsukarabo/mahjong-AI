"""
Phase 2: 特徴量統計チェック

手牌推定データ (hand_inference_v26.ndjson) の全 442次元について
min/max/mean/std/zero_rate を集計し、異常次元を報告する。
"""

import json
import sys
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data" / "features" / "hand_inference_v26.ndjson"
N_PAI = 34

# feature_offsets.py のオフセット定義を使う
sys.path.insert(0, str(Path(__file__).parent))
from feature_offsets import (
    HI_TOTAL, HI_SCORE_START, HI_SCORE_DIM, HI_GAME_START, HI_GAME_DIM,
    HI_PON_PASS, HI_CHI_PASS, HI_WIND_START, HI_WIND_DIM,
    HI_YAKU_START, HI_YAKU_DIM, HI_TENPAI,
    HI_DISCARD_START, HI_MELD_START, HI_RIICHI,
    HI_SELF_DISC, HI_SELF_MELD, HI_VISIBLE,
    HI_RED_DISC_SIG, HI_RED_VIS, HI_OTHER1_DISC, HI_OTHER2_DISC,
    HI_CHI_CALLED,
)

BLOCK_NAMES = [
    (HI_DISCARD_START, HI_MELD_START,              "target_discard    (44)"),
    (HI_MELD_START,    HI_RIICHI,                  "target_meld       (38)"),
    (HI_RIICHI,        HI_RIICHI + 1,              "riichi            ( 1)"),
    (HI_SCORE_START,   HI_SCORE_START+HI_SCORE_DIM,"score             (11)"),
    (HI_GAME_START,    HI_GAME_START+HI_GAME_DIM,  "game_state        ( 9)"),
    (HI_SELF_DISC,     HI_SELF_MELD,               "self_discard      (44)"),
    (HI_SELF_MELD,     HI_VISIBLE,                 "self_meld         (38)"),
    (HI_VISIBLE,       HI_RED_DISC_SIG,            "visible_counts    (34)"),
    (HI_RED_DISC_SIG,  HI_RED_VIS,                 "red_discard_sig   ( 3)"),
    (HI_RED_VIS,       HI_OTHER1_DISC,             "red_visible       ( 3)"),
    (HI_OTHER1_DISC,   HI_OTHER2_DISC,             "other1_discard    (44)"),
    (HI_OTHER2_DISC,   HI_PON_PASS,                "other2_discard    (44)"),
    (HI_PON_PASS,      HI_PON_PASS+N_PAI,          "pass_pon_signal   (34)"),
    (HI_WIND_START,    HI_WIND_START+HI_WIND_DIM,  "wind              ( 5)"),
    (HI_CHI_PASS,      HI_CHI_PASS+N_PAI,          "pass_chi_signal   (34)"),
    (HI_CHI_CALLED,    HI_YAKU_START,              "chi_called_tile   (34)"),
    (HI_YAKU_START,    HI_YAKU_START+HI_YAKU_DIM,  "yaku_prob         (21)"),
    (HI_TENPAI,        HI_TENPAI+1,               "tenpai_prob       ( 1)"),
]


def main():
    print(f"読み込み中: {DATA_PATH}")
    all_feats = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            for feat in s["features"]:  # 3 players
                all_feats.append(feat)

    arr = np.array(all_feats, dtype=np.float32)
    n_samples, n_dims = arr.shape
    print(f"サンプル数: {n_samples} (={len(all_feats)//3} 局面 × 3 プレイヤー)")
    print(f"特徴量次元: {n_dims}")

    # NaN / Inf チェック
    nan_cnt = int(np.isnan(arr).sum())
    inf_cnt = int(np.isinf(arr).sum())
    print(f"\nNaN: {nan_cnt}  Inf: {inf_cnt}")
    if nan_cnt > 0 or inf_cnt > 0:
        print("  !! NaN/Inf が検出されました !!")

    print("\n--- ブロック別統計 ---")
    print(f"{'ブロック':<30} {'min':>8} {'max':>8} {'mean':>8} {'std':>7} {'zero%':>7} {'gt1%':>7}")
    print("-" * 82)

    warnings = []
    for (s, e, name) in BLOCK_NAMES:
        block = arr[:, s:e]
        bmin  = float(block.min())
        bmax  = float(block.max())
        bmean = float(block.mean())
        bstd  = float(block.std())
        zero_pct = float((block == 0).mean()) * 100
        gt1_pct  = float((block > 1.0).mean()) * 100

        flag = ""
        if gt1_pct > 0.01:
            flag = " ⚠ >1"
            warnings.append((name, f"max={bmax:.2f}, {gt1_pct:.2f}% of values > 1.0"))
        if bmin < -1.0:
            flag += " ⚠ neg"
            if not any(name in w[0] for w in warnings):
                warnings.append((name, f"min={bmin:.2f}"))

        print(f"{name:<30} {bmin:>8.3f} {bmax:>8.3f} {bmean:>8.4f} {bstd:>7.4f} {zero_pct:>6.1f}% {gt1_pct:>6.2f}%{flag}")

    # 完全ゼロ次元チェック
    dim_max = arr.max(axis=0)
    dead_dims = np.where(dim_max == 0)[0]
    if len(dead_dims) > 0:
        print(f"\n!! 常にゼロの次元: {dead_dims.tolist()}")
    else:
        print(f"\n常にゼロの次元: なし ✓")

    # 警告サマリー
    if warnings:
        print("\n--- 警告 ---")
        for name, msg in warnings:
            print(f"  {name}: {msg}")
    else:
        print("\n警告なし ✓")

    # features[441] (tenpai_prob) の分布
    tp = arr[:, HI_TENPAI]
    print(f"\n--- tenpai_prob [441] 分布 ---")
    print(f"  min={tp.min():.4f}  max={tp.max():.4f}  mean={tp.mean():.4f}  std={tp.std():.4f}")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        print(f"  P(tenpai_prob > {threshold}) = {(tp > threshold).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
