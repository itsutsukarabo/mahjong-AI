"""
hand_inference.ndjson (N-dim、3サンプル/ゲーム状態) を
hand_inference_v15.ndjson (N-dim × 3プレイヤー、1サンプル/ゲーム状態) に変換する。

使い方:
  python prepare_v15_data.py                              # デフォルトパス
  python prepare_v15_data.py --src foo.ndjson --dst bar.ndjson  # 任意パス指定

形式変換:
  入力: { "features": [N], "label_hand": [34], "label_red": [3], "meta": {...} } × 3行
  出力: { "features": [[N],[N],[N]], "label_hand": [[34],[34],[34]], "label_red": [[3],[3],[3]] } × 1行

連続する3行が同じゲーム状態に対応するため、ストリーミング処理でメモリ効率良く変換する。
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
INPUT_PATH  = Path(_opts["src"]) if "src" in _opts else FEATURES_DIR / "hand_inference.ndjson"
OUTPUT_PATH = Path(_opts["dst"]) if "dst" in _opts else FEATURES_DIR / "hand_inference_v15.ndjson"


def main():
    if not INPUT_PATH.exists():
        print(f"入力ファイルなし: {INPUT_PATH}")
        sys.exit(1)

    print(f"読み込み中: {INPUT_PATH}")
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

            # 同一ゲーム状態かチェック
            same_state = all(
                s["meta"]["paipu_id"]  == m0["paipu_id"] and
                s["meta"]["round_idx"] == m0["round_idx"] and
                s["meta"]["event_idx"] == m0["event_idx"] and
                s["meta"]["viewer_l"]  == m0["viewer_l"]
                for s in samples[1:]
            )
            if not same_state:
                n_skip += 1
                continue

            # right/across/left の順（viewer_l+1, +2, +3）でソート
            viewer_l = m0["viewer_l"]
            order = [(viewer_l + 1) % 4, (viewer_l + 2) % 4, (viewer_l + 3) % 4]
            sample_map = {s["meta"]["target_l"]: s for s in samples}
            sorted_samples = [sample_map.get(tl) for tl in order]
            if None in sorted_samples:
                n_skip += 1
                continue

            combined = {
                "features":   [s["features"]  for s in sorted_samples],
                "label_hand": [s["label_hand"] for s in sorted_samples],
                "label_red":  [s["label_red"]  for s in sorted_samples],
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
