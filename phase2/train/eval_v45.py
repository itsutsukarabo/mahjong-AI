"""
eval_v45.py — v45 model.pt のテスト評価のみ実行

学習ループをスキップして eval_epoch + eval_wait_metrics + 副露数別EAE だけ走らせる。
データ分割は訓練時と同一 (rng=42, 80/10/10)。
"""
import gc
import json
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_v45", Path(__file__).parent / "train_hand_inference_v45.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HandInferenceDataset       = _mod.HandInferenceDataset
collate_fn                 = _mod.collate_fn
eval_epoch                 = _mod.eval_epoch
eval_wait_metrics          = _mod.eval_wait_metrics
HandInferenceModel         = _mod.HandInferenceV37
CONFIG                     = _mod.CONFIG
MODEL_DIR                  = _mod.MODEL_DIR
DATA_DIR                   = _mod.DATA_DIR
N_BLOCKS                   = CONFIG["n_blocks"]
N_TATSU                    = _mod.N_TATSU
N_SHANTEN_CLASSES          = _mod.N_SHANTEN_CLASSES
N_YAKU                     = _mod.N_YAKU
MELD_CHI_OFFSET            = _mod.MELD_CHI_OFFSET
MELD_PON_OFFSET            = _mod.MELD_PON_OFFSET
MELD_KAN_OFFSET            = _mod.MELD_KAN_OFFSET
get_inference_weights      = _mod.get_inference_weights
weighted_eae               = _mod.weighted_eae
constrained_softmax_probs  = _mod.constrained_softmax_probs

INPUT_DIM  = CONFIG["fixed_dim"]   # 674
TOKEN_DIM  = CONFIG["token_dim"]   # 45
FEATURE_PATH = DATA_DIR / "hand_inference_v45.ndjson"


@torch.no_grad()
def eval_by_meld_count(model, loader, device):
    """副露数(0/1/2/3+)別EAE。各(sample, player)を副露数でグループ化して集計。
    logits shape: (B, P, 34, 5) — per-tile count distribution
    EAE per sample = sum over tiles of expected absolute count error
    """
    model.eval()
    count_vals = torch.arange(5, device=device, dtype=torch.float32)
    buckets = {k: {"wsum": 0.0, "w": 0.0, "n": 0} for k in [0, 1, 2, 3]}
    for batch in loader:
        features = batch[0].to(device)
        disc_tok  = batch[1].to(device)
        disc_mask = batch[2].to(device)
        labels    = batch[3].to(device)           # (B, P, 34)
        logits    = model(features, disc_tok, disc_mask)[0]  # (B, P, 34, 5)
        n_melds   = (features[:, :, MELD_CHI_OFFSET]
                     + features[:, :, MELD_PON_OFFSET]
                     + features[:, :, MELD_KAN_OFFSET]).round().long().clamp(0, 3)
        for p in range(3):
            k_p   = labels[:, p].float().sum(dim=-1).long()  # (B,)
            probs = constrained_softmax_probs(logits[:, p], k_p)  # (B, 34, 5)
            iw    = get_inference_weights(features[:, p])         # (B,)
            # EAE per sample: sum over tiles of E[|count - true|]
            dev    = (count_vals - labels[:, p].unsqueeze(-1).float()).abs()  # (B, 34, 5)
            eae_pp = (dev * probs).sum(dim=-1).sum(dim=-1)                   # (B,)
            mc = n_melds[:, p]
            for m in range(4):
                mask = (mc == m) if m < 3 else (mc >= 3)
                sel_w   = iw[mask]
                sel_eae = eae_pp[mask]
                buckets[min(m, 3)]["wsum"] += (sel_eae * sel_w).sum().item()
                buckets[min(m, 3)]["w"]    += sel_w.sum().item()
                buckets[min(m, 3)]["n"]    += int(mask.sum().item())
    result = {}
    for m, d in buckets.items():
        result[f"meld{m}_eae"] = d["wsum"] / d["w"] if d["w"] > 0 else float("nan")
        result[f"meld{m}_n"]   = d["n"]
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"FEATURE_PATH: {FEATURE_PATH}")

    # --- 訓練スクリプトと同一のデータ読み込み ---
    print("\nデータ読み込み中...", flush=True)
    n_lines = sum(1 for ln in open(FEATURE_PATH, encoding="utf-8") if ln.strip())
    print(f"総行数: {n_lines:,}")

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
    disc_tokens_list = []

    with open(FEATURE_PATH, encoding="utf-8") as f:
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
            lpush_np[i]    = 1.0 if s.get("label_push",    {}).get("is_push",    False) else 0.0
            lyaku_np[i]    = s.get("label_yaku", [0.0] * N_YAKU)
            lwon_np[i]     = bool(s.get("label_won", False))
            raw_tok = s.get("discard_tokens", [[], [], []])
            player_tokens = []
            for p in range(3):
                toks = raw_tok[p] if p < len(raw_tok) else []
                player_tokens.append(
                    np.array(toks, dtype=np.float16) if toks
                    else np.zeros((0, TOKEN_DIM), dtype=np.float16)
                )
            disc_tokens_list.append(player_tokens)
            del s
            if (i+1) % 50000 == 0:
                print(f"  {i+1}/{n_lines}", flush=True)

    gc.collect()

    # --- 訓練時と同一フィルタ・分割 ---
    lab_sum  = lab_np.sum(axis=-1)
    meld_sum = feat_np[:,:,MELD_CHI_OFFSET] + feat_np[:,:,MELD_PON_OFFSET] + feat_np[:,:,MELD_KAN_OFFSET]
    keep      = (lab_sum.sum(axis=1) > 0) | (meld_sum.sum(axis=1) > 0)
    valid_idx = np.where(keep)[0]
    print(f"有効サンプル: {len(valid_idx):,}")

    rng      = np.random.default_rng(42)
    n_valid  = len(valid_idx)
    shuffled = valid_idx[rng.permutation(n_valid)]
    n_train  = int(n_valid * 0.8)
    n_val    = int(n_valid * 0.1)
    test_idx_list = shuffled[n_train + n_val:].tolist()
    print(f"test size: {len(test_idx_list):,}")

    full_ds = HandInferenceDataset(
        feat_np, disc_tokens_list,
        lab_np, lred_np, lblk_np, lsh_np,
        lnoise_np, lretreat_np, lpush_np,
        lyaku_np, lwon_np,
    )
    del feat_np, lab_np, lred_np, lblk_np, lsh_np
    del lnoise_np, lretreat_np, lpush_np, lyaku_np, lwon_np
    gc.collect()

    test_ds     = torch.utils.data.Subset(full_ds, test_idx_list)
    use_gpu     = device.type == "cuda"
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"],
                             shuffle=False, num_workers=0,
                             pin_memory=use_gpu, collate_fn=collate_fn)

    # --- モデル読み込み ---
    model = HandInferenceModel(
        fixed_dim    = CONFIG["fixed_dim"],
        token_dim    = CONFIG["token_dim"],
        disc_d       = CONFIG["disc_d"],
        disc_nhead   = CONFIG["disc_nhead"],
        disc_layers  = CONFIG["disc_layers"],
        d_model      = CONFIG["d_model"],
        nhead        = CONFIG["nhead"],
        num_layers   = CONFIG["num_layers"],
        n_pai        = CONFIG["n_pai"],
        n_count_cls  = CONFIG["n_count_cls"],
        n_players    = CONFIG["n_players"],
        n_blocks     = CONFIG["n_blocks"],
        n_tatsu      = N_TATSU,
        n_shanten_cls= N_SHANTEN_CLASSES,
        dropout      = CONFIG["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt",
                                     map_location=device, weights_only=True))
    model.eval()
    print(f"\nmodel.pt ロード完了 (best_eae=4.2802 @ EP165)\n")

    # --- テスト評価 ---
    print("テスト評価実行中...", flush=True)
    test_eae, test_eae_stage, test_acc = eval_epoch(model, test_loader, device)
    test_wait_m = eval_wait_metrics(model, test_loader, device)

    print("副露数別EAE計算中...", flush=True)
    meld_m = eval_by_meld_count(model, test_loader, device)

    print(f"\n===== v45 FINAL TEST RESULT =====")
    print(f"  test_eae       = {test_eae:.4f}")
    print(f"  test_eae_stage = {test_eae_stage:.4f}")
    print(f"  test_acc       = {test_acc:.4f}")
    for k, v in test_wait_m.items():
        print(f"  {k}: {v}")

    print(f"\n===== 副露数別 EAE (希薄化確認) =====")
    for m in [0, 1, 2, 3]:
        suffix = "+" if m == 3 else ""
        label  = f"門前(meld=0)" if m == 0 else f"meld={m}{suffix}"
        eae    = meld_m[f"meld{m}_eae"]
        n      = meld_m[f"meld{m}_n"]
        print(f"  {label:14s}: eae={eae:.4f}  n={n:,}")

    print(f"\n===== v44 比較 =====")
    print(f"  v44 val_eae  = 3.8946 (EP195 best)")
    print(f"  v44 test_eae = 3.9184")
    print(f"  v45 val_eae  = 4.2802 (EP165 best) Δ={4.2802-3.8946:+.4f}")
    print(f"  v45 test_eae = {test_eae:.4f}         Δ={test_eae-3.9184:+.4f}")

    result = {
        "test_eae": test_eae, "test_eae_stage": test_eae_stage,
        "test_acc": test_acc, **test_wait_m, **meld_m,
        "best_val_eae": 4.2802, "best_epoch": 165,
        "stop_reason": "2epoch_nll_rise_EP178",
    }
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
    print(f"\neval_result.json 保存完了: {MODEL_DIR/'eval_result.json'}")


if __name__ == "__main__":
    main()
