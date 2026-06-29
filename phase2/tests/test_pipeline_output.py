"""
データパイプライン出力の整合性テスト

手牌推定学習データ (hand_inference_vXX.ndjson) が
学習に必要な全フィールドを含んでいるかを検証する。

使い方:
    python phase2/tests/test_pipeline_output.py phase2/data/features/hand_inference_v41.ndjson
    python -m pytest phase2/tests/test_pipeline_output.py  # pytestから実行する場合は要 --path 渡し
"""

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = {
    "features":       (list, "674次元固定特徴量"),
    "discard_tokens": (list, "捨て牌トークン列"),
    "target_discard": (list, "ターゲット捨て牌"),
    "label_hand":     (list, "手牌カウントラベル"),
    "label_red":      (list, "赤牌ラベル"),
    "label_block":    (list, "ブロック分解ラベル (add_block_labels必須)"),
    "label_shanten":  (list, "シャンテン数ラベル (add_block_labels必須)"),
    "label_yaku":     (list, "役ラベル21次元 (extract_features必須)"),
    "label_won":      ((bool, int), "和了フラグ (extract_features必須)"),
    "label_noise":    ((bool, int), "ノイズフラグ"),
    "label_retreat":  (dict, "撤退ラベル (add_intent_labels必須)"),
    "label_push":     (dict, "プッシュラベル (add_intent_labels必須)"),
}

EXPECTED_DIMS = {
    "features":       674,
    "label_hand":     34,
    "label_red":       3,
    "label_block":    134,
    "label_shanten":    3,
    "label_yaku":      21,
    "target_discard":  44,
}

N_PLAYERS = 3


def check_record(rec: dict, idx: int) -> list[str]:
    errors = []

    for field, (expected_type, desc) in REQUIRED_FIELDS.items():
        val = rec.get(field)

        # 存在チェック
        if val is None:
            errors.append(f"[{idx}] {field} が None または欠損 — {desc}")
            continue

        # 型チェック
        if not isinstance(val, expected_type if isinstance(expected_type, tuple) else (expected_type,)):
            errors.append(f"[{idx}] {field} の型が {type(val).__name__} (期待: {expected_type})")
            continue

        # 3プレイヤー構造のチェック (features, label_block など)
        if isinstance(val, list) and len(val) == N_PLAYERS and isinstance(val[0], list):
            for p, pval in enumerate(val):
                fname = field
                if fname in EXPECTED_DIMS and len(pval) != EXPECTED_DIMS[fname]:
                    errors.append(f"[{idx}] {field}[{p}] 次元が {len(pval)} (期待: {EXPECTED_DIMS[fname]})")
        elif isinstance(val, list) and field in EXPECTED_DIMS:
            # 単一プレイヤー分 (label_yaku, label_won など)
            if len(val) != EXPECTED_DIMS[field]:
                errors.append(f"[{idx}] {field} 次元が {len(val)} (期待: {EXPECTED_DIMS[field]})")

    # label_won と label_yaku の整合性チェック (警告のみ: 21種外の役で和了する場合は全ゼロが正常)
    # won = rec.get("label_won")
    # yaku = rec.get("label_yaku")
    # if won and yaku is not None and all(v == 0 for v in yaku): warn (valid)

    # label_shanten が全て None でないか (add_block_labels が適用されているか)
    shanten = rec.get("label_shanten")
    if isinstance(shanten, list) and all(s is None for s in shanten):
        errors.append(f"[{idx}] label_shanten が全 None — add_block_labels 未適用の可能性")

    return errors


def validate(path: Path, n_check: int = 500, n_won_min: int = 10) -> bool:
    print(f"検証対象: {path}")
    if not path.exists():
        print(f"ERROR: ファイルが存在しません: {path}")
        return False

    all_errors = []
    n_total = 0
    n_won = 0
    n_with_block = 0
    n_with_shanten0 = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if n_total >= n_check:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            errs = check_record(rec, n_total)
            all_errors.extend(errs)

            # 統計
            if rec.get("label_won"):
                n_won += 1
            if rec.get("label_block") is not None:
                n_with_block += 1
            shanten = rec.get("label_shanten")
            if isinstance(shanten, list) and 0 in shanten:
                n_with_shanten0 += 1
            n_total += 1

    print(f"  検証サンプル数   : {n_total}")
    print(f"  label_won=True   : {n_won} ({n_won/n_total*100:.1f}%)")
    print(f"  label_block あり : {n_with_block} ({n_with_block/n_total*100:.1f}%)")
    print(f"  shanten=0 あり   : {n_with_shanten0} ({n_with_shanten0/n_total*100:.1f}%)")

    # 必須チェック
    if n_won < n_won_min:
        all_errors.append(
            f"label_won=True が {n_won} 件 (期待: {n_won_min}件以上) — extract_features バグの可能性"
        )
    if n_with_block == 0:
        all_errors.append("label_block が全サンプルで欠損 — add_block_labels 未実行の可能性")
    if n_with_shanten0 == 0:
        all_errors.append("shanten=0 サンプルが 0 件 — n_tenpai=0 になり wait_f1 が計算不可")

    if all_errors:
        print(f"\n[FAIL] {len(all_errors)} 件のエラー:")
        for e in all_errors[:20]:
            print(f"  {e}")
        if len(all_errors) > 20:
            print(f"  ... 他 {len(all_errors)-20} 件")
        return False
    else:
        print("\n[OK] 全チェック通過")
        return True


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent.parent / "data" / "features" / "hand_inference_v41.ndjson"
    ok = validate(path)
    sys.exit(0 if ok else 1)
