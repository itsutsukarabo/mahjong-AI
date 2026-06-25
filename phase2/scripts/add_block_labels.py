"""
grouped hand_inference ndjson に label_block / label_shanten フィールドを追加するスクリプト。

使い方:
  # v22 (89ブロック, 旧実装・テンパイのみ)
  python add_block_labels.py --src hand_inference_v22_prepared.ndjson --dst hand_inference_v22.ndjson

  # v23 (134ブロック: ターツ含む, 旧実装・テンパイのみ)
  python add_block_labels.py --src hand_inference_v22_prepared.ndjson --dst hand_inference_v23.ndjson --include-tatsu

  # v26 (134ブロック: ターツ含む, v2実装・全シャンテン数)
  python add_block_labels.py --src hand_inference_v22_prepared.ndjson --dst hand_inference_v26.ndjson --include-tatsu --v2

  # v27 (89ブロック, v2実装・全シャンテン数)
  python add_block_labels.py --src hand_inference_v22_prepared.ndjson --dst hand_inference_v27.ndjson --v2

  # デフォルト (旧パス、89ブロック)
  python add_block_labels.py
"""

import json
import sys
import time
from pathlib import Path

_args = sys.argv[1:]
_opts = {}
for i, a in enumerate(_args):
    if a.startswith("--"):
        key = a[2:]
        val = _args[i + 1] if i + 1 < len(_args) and not _args[i + 1].startswith("--") else True
        _opts[key] = val

INCLUDE_TATSU = "include-tatsu" in _opts
USE_V2        = "v2" in _opts          # v2実装: 全シャンテン数 + 排他的バックトラッキング

sys.path.insert(0, str(Path(__file__).parent))
from enumerate_decompositions import (
    compute_soft_labels, compute_soft_labels_v2,
    N_BLOCKS, N_BLOCKS_WITH_TATSU,
)

N_BLOCKS_OUT = N_BLOCKS_WITH_TATSU if INCLUDE_TATSU else N_BLOCKS

DATA_DIR = Path(__file__).parent.parent / "data" / "features"
SRC_PATH = Path(_opts["src"]) if "src" in _opts else DATA_DIR / "hand_inference_v15.ndjson"
DST_PATH = Path(_opts["dst"]) if "dst" in _opts else DATA_DIR / "hand_inference_v19.ndjson"

MAX_DECOMPS   = 200
MAX_REM_PATS  = 50


def process():
    if not SRC_PATH.exists():
        print(f"ファイルなし: {SRC_PATH}")
        sys.exit(1)

    total = sum(1 for _ in open(SRC_PATH, encoding="utf-8"))
    impl  = "v2 (全シャンテン数・排他的列挙)" if USE_V2 else "旧 (テンパイのみ・独立列挙)"
    print(f"総サンプル数: {total}")
    print(f"ブロック数  : {N_BLOCKS_OUT} ({'ターツ含む 134' if INCLUDE_TATSU else '基本 89'})")
    print(f"実装        : {impl}")
    print(f"出力先      : {DST_PATH}")

    t0 = time.time()
    with open(SRC_PATH, encoding="utf-8") as fin, \
         open(DST_PATH, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)

            label_hand    = s["label_hand"]   # (3, 34)
            label_block   = []
            label_shanten = []                # v2のみ追加

            for p in range(3):
                counts34 = label_hand[p]
                # n_meld を手牌枚数から計算: (13 - 手牌枚数) / 3
                n_meld = (13 - int(sum(counts34))) // 3

                if USE_V2:
                    soft, sh = compute_soft_labels_v2(
                        counts34, n_meld=n_meld,
                        max_face_results=MAX_DECOMPS,
                        max_rem_patterns=MAX_REM_PATS,
                        include_tatsu=INCLUDE_TATSU,
                    )
                    label_shanten.append(int(sh))
                else:
                    soft = compute_soft_labels(
                        counts34, n_meld=n_meld,
                        max_results=MAX_DECOMPS,
                        include_tatsu=INCLUDE_TATSU,
                    )

                label_block.append(soft.tolist())

            s["label_block"] = label_block
            if USE_V2:
                s["label_shanten"] = label_shanten

            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"  {i+1}/{total}  {rate:.0f} samples/s  ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"完了: {elapsed:.1f}s  ({total/elapsed:.0f} samples/s)")
    print(f"出力: {DST_PATH}")


if __name__ == "__main__":
    process()
