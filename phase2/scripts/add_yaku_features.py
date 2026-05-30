"""
Stage 1 役推定モデルで hand_inference.ndjson の各サンプルに yaku_prob(11次元) を付与する。

入力: phase2/data/features/hand_inference.ndjson  (344次元)
出力: phase2/data/features/hand_inference.ndjson  (355次元、上書き)

実行順序:
  1. node extract_features.js       → hand_inference.ndjson (344次元)
  2. python train_yaku_inference.py → yaku_inference/v1/model.pt
  3. python add_yaku_features.py    → hand_inference.ndjson (355次元)
  4. python train_hand_inference.py → hand_inference/v6/model.pt
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---- パス設定 ----

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
YAKU_MODEL_DIR = Path(__file__).parent.parent / "models" / "yaku_inference" / "v1"

INPUT_PATH  = FEATURES_DIR / "hand_inference.ndjson"
OUTPUT_PATH = FEATURES_DIR / "hand_inference.ndjson"

YAKU_INPUT_DIM  = 92
YAKU_OUTPUT_DIM = 15
EXPECTED_INPUT_DIM  = 344
EXPECTED_OUTPUT_DIM = 359
BATCH_SIZE = 1024


# ---- モデル定義（train_yaku_inference.py と同一） ----

class YakuInference(nn.Module):
    def __init__(self, input_dim, n_yaku, hidden, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_yaku))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    config_path = YAKU_MODEL_DIR / "config.json"
    model_path  = YAKU_MODEL_DIR / "model.pt"
    if not model_path.exists():
        print(f"モデルなし: {model_path}")
        print("python -u phase2/train/train_yaku_inference.py を先に実行してください")
        sys.exit(1)

    config = json.loads(config_path.read_text())
    model = YakuInference(
        input_dim = config["input_dim"],
        n_yaku    = config["n_yaku"],
        hidden    = config["hidden"],
        dropout   = 0.0,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("役推定モデル ロード完了")

    if not INPUT_PATH.exists():
        print(f"入力ファイルなし: {INPUT_PATH}")
        sys.exit(1)

    print(f"読み込み中: {INPUT_PATH}")
    with open(INPUT_PATH, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    print(f"サンプル数: {len(all_data)}")

    sample_dim = len(all_data[0]["features"])
    if sample_dim == EXPECTED_OUTPUT_DIM:
        print(f"既に {EXPECTED_OUTPUT_DIM}次元です。処理をスキップします。")
        return
    if sample_dim != EXPECTED_INPUT_DIM:
        print(f"次元数不一致: expected {EXPECTED_INPUT_DIM}, got {sample_dim}")
        sys.exit(1)

    # バッチ推論: 先頭92次元を役推定の入力として使用
    tmp_path = OUTPUT_PATH.with_suffix(".tmp.ndjson")
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for start in range(0, len(all_data), BATCH_SIZE):
            batch = all_data[start:start + BATCH_SIZE]
            x = torch.tensor(
                [[s["features"][i] for i in range(YAKU_INPUT_DIM)] for s in batch],
                dtype=torch.float32,
            )
            with torch.no_grad():
                logits = model(x)  # [B, 11]
                probs  = torch.sigmoid(logits).tolist()

            for sample, prob in zip(batch, probs):
                sample["features"] = sample["features"] + prob
                out_f.write(json.dumps(sample) + "\n")

            if (start // BATCH_SIZE) % 50 == 0:
                print(f"  {start + len(batch)} / {len(all_data)} 処理済み", flush=True)

    tmp_path.replace(OUTPUT_PATH)
    print(f"完了: {len(all_data)} サンプル → {EXPECTED_OUTPUT_DIM}次元 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
