"""
eval_wait_tiles.py

v29モデルのブロックデコーダーによる待ち牌推定F1ベースライン計測。

評価対象: テンパイサンプル (label_shanten == 0) のみ
評価指標:
  - wait_f1     : 待ち牌集合の F1 スコア (マクロ平均)
  - wait_prec   : 精度
  - wait_recall : 再現率
  - no_result_rate: ビームサーチが解を返せなかった割合

実行:
  python phase2/scripts/eval_wait_tiles.py [--n-samples 5000]
"""

import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from decode_hand import beam_search_decoder_free, compute_wait_probs_free, N_PAI
from enumerate_decompositions import (
    compute_shanten, build_block_selection_matrix, N_BLOCKS_WITH_TATSU as N_BLOCKS
)

TRAIN_DIR = Path(__file__).parent.parent / "train"
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v29"
DATA_DIR  = Path(__file__).parent.parent / "data" / "features"

sys.path.insert(0, str(TRAIN_DIR))

# ---- モデル定義 (train_hand_inference_v29.py から転載) ----

VISIBLE_OFFSET = 185


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
        self.head        = nn.Linear(d_model, n_count_cls)
        self.red_head    = nn.Linear(d_model, 2)
        self.block_head  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        self.register_buffer("tile_ids",    torch.arange(n_pai))
        self.register_buffer("player_ids",  torch.arange(n_players))
        self.register_buffer("count_range", torch.arange(n_count_cls))
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
        out    = self.transformer(tokens).reshape(B, P, self.n_pai, -1)

        logits_raw  = self.head(out)
        count_range = self.count_range
        vis_raw     = (vis * 4).round().long().clamp(0, 4)
        mask        = count_range > (4 - vis_raw).clamp(min=0).unsqueeze(-1)
        logits      = logits_raw.masked_fill(mask, float('-inf'))

        out_flat    = out.reshape(B * P, self.n_pai, -1)
        block_feats = torch.bmm(
            self.block_sel.unsqueeze(0).expand(B * P, -1, -1),
            out_flat,
        ).reshape(B, P, self.n_blocks, -1)
        block_logits = self.block_head(block_feats).squeeze(-1)

        return logits, logits_raw, None, block_logits


# ---- 待ち牌計算 (真値) ----

def compute_true_waits(counts34: list, n_melds: int) -> list:
    """
    真の待ち牌リストを返す。
    各牌 t について counts34[t]+1 でシャンテン==-1 になる牌を収集する。
    """
    waits = []
    for t in range(N_PAI):
        c = list(counts34)
        if c[t] >= 4:
            continue
        c[t] += 1
        if compute_shanten(c, n_melds) == -1:
            waits.append(t)
    return waits


# ---- F1計算 ----

def wait_f1(pred_set: set, true_set: set):
    if not true_set:
        return None, None, None
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return prec, recall, f1


# ---- メイン ----

def main():
    n_max = 5000
    for arg in sys.argv[1:]:
        if arg.startswith("--n-samples"):
            n_max = int(arg.split("=")[-1]) if "=" in arg else int(sys.argv[sys.argv.index(arg) + 1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # モデルロード
    cfg_path = MODEL_DIR / "config.json"
    cfg = json.load(open(cfg_path))
    model = HandInferenceV26(
        input_dim   = cfg["input_dim"],
        d_model     = cfg["d_model"],
        nhead       = cfg["nhead"],
        num_layers  = cfg["num_layers"],
        n_pai       = cfg["n_pai"],
        n_count_cls = cfg["n_count_cls"],
        n_players   = cfg["n_players"],
        n_blocks    = cfg["n_blocks"],
        dropout     = 0.0,
    ).to(device)
    ckpt = torch.load(MODEL_DIR / "model.pt", map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"モデルロード完了: {MODEL_DIR / 'model.pt'}")

    # データロード (test split: seed=42, 80/10/10 → 最後の10%)
    ndjson_path = DATA_DIR / "hand_inference_v26.ndjson"
    print(f"データ読み込み: {ndjson_path}")

    raw_lines = [ln.strip() for ln in open(ndjson_path, encoding="utf-8") if ln.strip()]
    n_total   = len(raw_lines)
    rng       = np.random.default_rng(42)
    idx_perm  = rng.permutation(n_total)
    n_test    = int(n_total * 0.1)
    test_idx  = idx_perm[n_total - n_test:]
    print(f"総サンプル数: {n_total}, テストセット: {n_test}")

    # テンパイサンプルを n_max 件収集
    tenpai_samples = []
    for i in test_idx:
        s = json.loads(raw_lines[i])
        features      = np.array(s["features"],       dtype=np.float32)   # (3, 442)
        label_hand    = np.array(s["label_hand"],     dtype=np.int64)     # (3, 34)
        label_shanten = np.array(s["label_shanten"],  dtype=np.int64)     # (3,)

        for p in range(3):
            if label_shanten[p] != 0:
                continue
            counts34 = label_hand[p].tolist()
            total    = sum(counts34)
            if total == 0 or total % 3 != 1:
                continue
            n_melds  = (13 - total) // 3
            true_waits = compute_true_waits(counts34, n_melds)
            if not true_waits:
                continue
            tenpai_samples.append({
                "features": features,           # (3, 442)
                "player":   p,
                "k_tiles":  total,
                "true_waits": true_waits,
            })
            if len(tenpai_samples) >= n_max:
                break
        if len(tenpai_samples) >= n_max:
            break

    print(f"テンパイサンプル数: {len(tenpai_samples)}")
    del raw_lines; gc.collect()

    # バッチ推論
    BATCH = 64
    all_block_logits = []  # [(player, block_logits_134), ...]

    feat_buf  = []
    pidx_buf  = []

    def run_batch():
        if not feat_buf:
            return
        feat_t = torch.tensor(
            np.stack(feat_buf, axis=0), dtype=torch.float32
        ).to(device)  # (B, 3, 442)
        with torch.no_grad():
            _, _, _, block_logits = model(feat_t)
        block_logits = block_logits.cpu().numpy()  # (B, 3, 134)
        for j, p in enumerate(pidx_buf):
            all_block_logits.append(block_logits[j, p])
        feat_buf.clear(); pidx_buf.clear()

    print("推論中...", flush=True)
    for k, s in enumerate(tenpai_samples):
        feat_buf.append(s["features"])
        pidx_buf.append(s["player"])
        if len(feat_buf) >= BATCH:
            run_batch()
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(tenpai_samples)}", flush=True)
    run_batch()
    print("推論完了")

    # ビームサーチ + F1計算
    prec_list   = []
    recall_list = []
    f1_list     = []
    no_result   = 0

    print("ビームサーチ評価中...", flush=True)
    for k, (s, blk) in enumerate(zip(tenpai_samples, all_block_logits)):
        beam = beam_search_decoder_free(blk, s["k_tiles"], beam_width=50)
        if not beam:
            no_result += 1
            prec_list.append(0.0); recall_list.append(0.0); f1_list.append(0.0)
            continue

        wait_probs = compute_wait_probs_free(beam, min_prob=0.05)
        pred_set   = set(wait_probs.keys())
        true_set   = set(s["true_waits"])
        prec, recall, f1 = wait_f1(pred_set, true_set)
        prec_list.append(prec); recall_list.append(recall); f1_list.append(f1)

        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(tenpai_samples)}", flush=True)

    n = len(f1_list)
    result = {
        "n_samples":       n,
        "wait_prec":       round(float(np.mean(prec_list)),   4),
        "wait_recall":     round(float(np.mean(recall_list)), 4),
        "wait_f1":         round(float(np.mean(f1_list)),     4),
        "no_result_rate":  round(no_result / n,               4),
        "model":           "hand_inference/v29",
        "decoder":         "beam_search_free (k_tiles, beam_width=50)",
    }

    print("\n=== 待ち牌推定F1 評価結果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    out_path = MODEL_DIR / "wait_f1_baseline.json"
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
