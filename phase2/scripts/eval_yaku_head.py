"""
v41 yaku_head の役推定精度を評価し、yaku_inference v1 と比較する。

評価指標: per-yaku accuracy (threshold=0.5) の macro 平均
          和了サンプル (label_won=True) のみ対象
"""

import json
import sys
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "phase2" / "train"))

from train_hand_inference_v41 import (
    HandInferenceV37, CONFIG, N_YAKU, N_BLOCKS, N_TATSU, N_SHANTEN_CLASSES,
    HandInferenceDataset, collate_fn,
)
from torch.utils.data import DataLoader, Subset

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH  = ROOT / "phase2" / "data" / "features" / "hand_inference_v41.ndjson"
MODEL_PATH = ROOT / "phase2" / "models" / "hand_inference" / "v41" / "model_best_wait.pt"
TOKEN_DIM  = CONFIG["token_dim"]   # 44
INPUT_DIM  = CONFIG["fixed_dim"]   # 674

YAKU_NAMES = [
    "tanyao","honitsu","chinitsu",
    "yakuhai_haku","yakuhai_hatsu","yakuhai_chun",
    "yakuhai_ba","yakuhai_ji",
    "pinfu","chanta","riichi",
    "ippeiko","sanshoku","ittsu",
    "toitoi","sananko","honroto",
    "chitoitsu","shosangen","iipeiko","ryanpeiko",
]


def load_data():
    n_lines = sum(1 for ln in open(DATA_PATH, encoding="utf-8") if ln.strip())
    print(f"総サンプル数: {n_lines}")

    feat_np     = np.empty((n_lines, 3, INPUT_DIM), dtype=np.float32)
    lab_np      = np.empty((n_lines, 3, 34),        dtype=np.int64)
    lred_np     = np.empty((n_lines, 3, 3),         dtype=np.int64)
    lblk_np     = np.zeros((n_lines, 3, N_BLOCKS),  dtype=np.float32)
    lsh_np      = np.full((n_lines, 3), 4,          dtype=np.int64)
    lnoise_np   = np.zeros((n_lines,),              dtype=bool)
    lretreat_np = np.zeros((n_lines,),              dtype=np.float32)
    lpush_np    = np.zeros((n_lines,),              dtype=np.float32)
    lyaku_np    = np.zeros((n_lines, N_YAKU),       dtype=np.float32)
    lwon_np     = np.zeros((n_lines,),              dtype=bool)
    disc_tokens_list = []

    with open(DATA_PATH, encoding="utf-8") as f:
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
            raw_tok = s.get("discard_tokens", [[], [], []])
            disc_tokens_list.append([
                np.array(raw_tok[p], dtype=np.float16) if p < len(raw_tok) and raw_tok[p]
                else np.zeros((0, TOKEN_DIM), dtype=np.float16)
                for p in range(3)
            ])
            del s

    gc.collect()
    return feat_np, lab_np, lred_np, lblk_np, lsh_np, lnoise_np, lretreat_np, lpush_np, lyaku_np, lwon_np, disc_tokens_list


@torch.no_grad()
def eval_yaku(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    all_won    = []

    for batch in loader:
        (features, disc_tok, disc_mask,
         labels, labels_red, labels_block, labels_shanten,
         noise_mask, retreat_label, push_label,
         labels_yaku, labels_won) = [x.to(DEVICE) for x in batch]

        _, _, _, _, _, _, _, _, yaku_logits = model(features, disc_tok, disc_mask)
        # yaku_logits: (B, P, N_YAKU) — player 0 のみ使用
        preds = torch.sigmoid(yaku_logits[:, 0, :]).cpu()
        all_preds.append(preds)
        all_labels.append(labels_yaku.cpu())
        all_won.append(labels_won.cpu())

    preds  = torch.cat(all_preds)   # (N, N_YAKU)
    labels = torch.cat(all_labels)  # (N, N_YAKU)
    won    = torch.cat(all_won)     # (N,)

    won_preds  = preds[won]
    won_labels = labels[won]
    print(f"\n和了サンプル数 (テスト): {won.sum().item()}")

    per_class_acc = ((won_preds > 0.5) == won_labels.bool()).float().mean(0)
    macro_acc = per_class_acc.mean().item()

    # ラベル出現率も表示（出現しないクラスは精度が trivially 高い）
    label_rate = won_labels.mean(0)

    print("\n--- per-yaku accuracy (threshold=0.5, won-only) ---")
    print(f"{'yaku':<20} {'acc':>6}  {'label_rate':>10}")
    for i, name in enumerate(YAKU_NAMES):
        n = name if i < len(YAKU_NAMES) else f"yaku_{i}"
        print(f"  {n:<18} {per_class_acc[i].item():.4f}  {label_rate[i].item():.4f}")
    print(f"\n  macro_acc = {macro_acc:.4f}")
    return macro_acc


def main():
    print(f"device: {DEVICE}")

    print("データ読み込み中...")
    (feat_np, lab_np, lred_np, lblk_np, lsh_np,
     lnoise_np, lretreat_np, lpush_np, lyaku_np, lwon_np,
     disc_tokens_list) = load_data()

    # v41 と同一の train/val/test 分割 (seed=42)
    from train_hand_inference_v41 import MELD_CHI_OFFSET, MELD_PON_OFFSET, MELD_KAN_OFFSET
    lab_sum  = lab_np.sum(axis=-1)
    meld_sum = feat_np[:,:,MELD_CHI_OFFSET] + feat_np[:,:,MELD_PON_OFFSET] + feat_np[:,:,MELD_KAN_OFFSET]
    keep = (lab_sum.sum(axis=1) > 0) | (meld_sum.sum(axis=1) > 0)
    valid_idx = np.where(keep)[0]
    print(f"有効サンプル: {len(valid_idx)}")

    rng = np.random.default_rng(42)
    shuffled = valid_idx[rng.permutation(len(valid_idx))]
    n_valid  = len(valid_idx)
    n_train  = int(n_valid * 0.8)
    n_val    = int(n_valid * 0.1)
    test_idx = shuffled[n_train + n_val:].tolist()

    full_ds  = HandInferenceDataset(
        feat_np, disc_tokens_list,
        lab_np, lred_np, lblk_np, lsh_np,
        lnoise_np, lretreat_np, lpush_np,
        lyaku_np, lwon_np,
    )
    test_ds  = Subset(full_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print(f"\nモデル読み込み: {MODEL_PATH}")
    model = HandInferenceV37(**{k: CONFIG[k] for k in
        ["fixed_dim","token_dim","disc_d","disc_nhead","disc_layers",
         "d_model","nhead","num_layers","n_pai","n_count_cls","n_players",
         "n_blocks","dropout"]},
        n_tatsu=N_TATSU, n_shanten_cls=N_SHANTEN_CLASSES, n_yaku=N_YAKU,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    v41_macro = eval_yaku(model, test_loader)

    print("\n" + "="*50)
    print("比較サマリー")
    print("="*50)
    print(f"  yaku_inference v1  macro_acc = 0.9662  (standalone MLP, 108次元入力, won専用学習)")
    print(f"  hand_inference v41 macro_acc = {v41_macro:.4f}  (yaku_head統合, ep162 model_best_wait.pt)")
    print()
    print("注: 両モデルはデータセット・入力特徴量が異なるため直接比較には注意が必要")


if __name__ == "__main__":
    main()
