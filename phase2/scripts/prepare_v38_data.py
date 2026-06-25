"""
v38 用データ準備スクリプト

hand_inference.ndjson (flat 695次元、3サンプル/ゲーム状態) を
hand_inference_v38.ndjson (3プレイヤー結合、discard_tokens付き) に変換する。

使い方:
  python prepare_v37_data.py                              # デフォルトパス
  python prepare_v37_data.py --src foo.ndjson --dst bar.ndjson

入力 (3行 = 1ゲーム状態):
  { "features": [695],
    "discard_tokens": [[42], ...],   # 可変長
    "target_discard": [44],
    "label_hand": [34], "label_red": [3],
    "label_block": [N_BLOCKS] or null,
    "meta": {...},
    "label_noise": ..., "label_retreat": ..., "label_push": ...  (省略可)
  }

出力 (1行 = 1ゲーム状態):
  {
    "features":        [[695], [695], [695]],
    "discard_tokens":  [[[42],...], [[42],...], [[42],...]],
    "target_discard":  [[44], [44], [44]],
    "label_hand":      [[34], [34], [34]],
    "label_red":       [[3],  [3],  [3] ],
    "label_block":     [[N_BLOCKS], [N_BLOCKS], [N_BLOCKS]],
    "label_shanten":   [sh0, sh1, sh2],
    "label_noise":     bool,
    "label_retreat":   dict,
    "label_push":      dict,
  }
"""

import json
import sys
from pathlib import Path

_args = sys.argv[1:]
_opts = {}
for i, a in enumerate(_args):
    if a.startswith("--"):
        key = a[2:]
        val = _args[i + 1] if i + 1 < len(_args) and not _args[i + 1].startswith("--") else True
        _opts[key] = val

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
INPUT_PATH   = Path(_opts["src"]) if "src" in _opts else FEATURES_DIR / "hand_inference.ndjson"
OUTPUT_PATH  = Path(_opts["dst"]) if "dst" in _opts else FEATURES_DIR / "hand_inference_v38.ndjson"

N_BLOCKS_DEFAULT = 134  # enumerate_decompositions での標準値


def _block_or_zeros(s, n_blocks):
    b = s.get("label_block")
    if b is None:
        return [0.0] * n_blocks
    return b


def main():
    if not INPUT_PATH.exists():
        print(f"入力ファイルなし: {INPUT_PATH}")
        sys.exit(1)

    print(f"読み込み中: {INPUT_PATH}", flush=True)

    n_out = 0
    n_skip = 0

    with open(INPUT_PATH, encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        while True:
            raw_lines = [f_in.readline() for _ in range(3)]
            if not raw_lines[0]:
                break

            stripped = [l.strip() for l in raw_lines if l.strip()]
            if len(stripped) != 3:
                n_skip += 1
                break

            samples = [json.loads(l) for l in stripped]
            m0 = samples[0]["meta"]

            # 同一ゲーム状態か検証
            same_state = all(
                s["meta"]["paipu_id"]  == m0["paipu_id"]  and
                s["meta"]["round_idx"] == m0["round_idx"] and
                s["meta"]["event_idx"] == m0["event_idx"] and
                s["meta"]["viewer_l"]  == m0["viewer_l"]
                for s in samples[1:]
            )
            if not same_state:
                n_skip += 1
                continue

            # ゴースト席フィルタ: 手牌0枚かつ副露なしのプレイヤーを含む状態を除外
            def has_ghost(s):
                hand = s.get("label_hand", [])
                feat = s.get("features", [])
                # meld features = feat[0:38] (HI_MELD_START=0, HI_MELD_DIM=38)
                has_meld = any(v != 0 for v in feat[:38]) if feat else False
                return sum(hand) == 0 and not has_meld
            if any(has_ghost(s) for s in samples):
                n_skip += 1
                continue

            # viewer_l からの相対順: right/across/left
            viewer_l = m0["viewer_l"]
            order = [(viewer_l + 1) % 4, (viewer_l + 2) % 4, (viewer_l + 3) % 4]
            sample_map = {s["meta"]["target_l"]: s for s in samples}
            sorted_samples = [sample_map.get(tl) for tl in order]
            if None in sorted_samples:
                n_skip += 1
                continue

            # N_BLOCKS は label_block 長から判定
            for s in sorted_samples:
                if s.get("label_block") is not None:
                    N_BLOCKS_DEFAULT = len(s["label_block"])
                    break

            combined = {
                "features":       [s["features"]       for s in sorted_samples],
                "discard_tokens": [s.get("discard_tokens", []) for s in sorted_samples],
                "target_discard": [s.get("target_discard", [0.0]*44) for s in sorted_samples],
                "label_hand":     [s["label_hand"]     for s in sorted_samples],
                "label_red":      [s["label_red"]       for s in sorted_samples],
                # label_block / label_shanten は add_block_labels.py で後から追加される
                "label_noise":    sorted_samples[0].get("label_noise", False),
                "label_retreat":  sorted_samples[0].get("label_retreat", {"is_retreat": False}),
                "label_push":     sorted_samples[0].get("label_push",    {"is_push":    False}),
            }
            f_out.write(json.dumps(combined) + "\n")
            n_out += 1

            if n_out % 20000 == 0:
                print(f"  {n_out} サンプル処理済み", flush=True)

    print(f"完了: {n_out} サンプル → {OUTPUT_PATH}")
    if n_skip:
        print(f"スキップ: {n_skip} グループ（整合性エラー）")


if __name__ == "__main__":
    main()
