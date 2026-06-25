"""
Stage 1 役推定モデルで hand_inference.ndjson の各サンプルに yaku_prob(21次元) を付与する。

使い方:
  python add_yaku_features.py                              # デフォルトパス
  python add_yaku_features.py --src foo.ndjson --dst bar.ndjson  # 任意パス指定

入力次元は問わない。

NOTE: hand_inference features の score/game_state 順序は yaku モデルの学習フォーマットと
異なるため、先頭 108次元をそのまま渡してはいけない。build_stage1_input() で正しく変換する。
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from feature_offsets import build_stage1_input, STAGE1_INPUT_DIM  # noqa: E402

# ---- CLI引数 ----

_args = sys.argv[1:]
_opts = {}
for i, a in enumerate(_args):
    if a.startswith("--"):
        key = a[2:]
        val = _args[i + 1] if i + 1 < len(_args) and not _args[i + 1].startswith("--") else True
        _opts[key] = val

# ---- パス設定 ----

FEATURES_DIR   = Path(__file__).parent.parent / "data" / "features"
YAKU_MODEL_DIR = Path(__file__).parent.parent / "models" / "yaku_inference" / "v1"

INPUT_PATH  = Path(_opts["src"]) if "src" in _opts else FEATURES_DIR / "hand_inference.ndjson"
OUTPUT_PATH = Path(_opts["dst"]) if "dst" in _opts else INPUT_PATH

YAKU_INPUT_DIM  = STAGE1_INPUT_DIM  # 108
YAKU_OUTPUT_DIM = 21
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
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    print("役推定モデル ロード完了")

    if not INPUT_PATH.exists():
        print(f"入力ファイルなし: {INPUT_PATH}")
        sys.exit(1)

    n_total = sum(1 for line in open(INPUT_PATH, encoding="utf-8") if line.strip())
    print(f"サンプル数: {n_total}  読み込み中: {INPUT_PATH}")

    # 1行目で次元確認
    with open(INPUT_PATH, encoding="utf-8") as f:
        first = json.loads(next(l for l in f if l.strip()))
    sample_dim = len(first["features"])
    if sample_dim < YAKU_INPUT_DIM:
        print(f"次元数不足: need at least {YAKU_INPUT_DIM}, got {sample_dim}")
        sys.exit(1)
    expected_out_dim = sample_dim + YAKU_OUTPUT_DIM
    print(f"入力次元: {sample_dim} → 出力次元: {expected_out_dim}")

    tmp_path = OUTPUT_PATH.with_suffix(".tmp.ndjson")
    n_done = 0
    with open(INPUT_PATH, encoding="utf-8") as in_f, \
         open(tmp_path, "w", encoding="utf-8") as out_f:
        batch = []
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(batch) < BATCH_SIZE:
                continue
            x = torch.tensor(
                [build_stage1_input(s["features"], s["target_discard"]) for s in batch],
                dtype=torch.float32,
            )
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).tolist()
            for sample, prob in zip(batch, probs):
                sample["features"] = sample["features"] + prob
                out_f.write(json.dumps(sample) + "\n")
            n_done += len(batch)
            if (n_done // BATCH_SIZE) % 50 == 0:
                print(f"  {n_done} / {n_total} 処理済み", flush=True)
            batch = []
        # 残余バッチ
        if batch:
            x = torch.tensor(
                [build_stage1_input(s["features"], s["target_discard"]) for s in batch],
                dtype=torch.float32,
            )
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).tolist()
            for sample, prob in zip(batch, probs):
                sample["features"] = sample["features"] + prob
                out_f.write(json.dumps(sample) + "\n")
            n_done += len(batch)

    tmp_path.replace(OUTPUT_PATH)
    print(f"完了: {n_done} サンプル → {expected_out_dim}次元 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
