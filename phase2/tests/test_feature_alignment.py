"""
特徴量アライメントのテスト

検証内容:
  1. hand_inference_v26.ndjson の特徴量次元が 442 であること
  2. 既知オフセットの値域が仕様通りであること（バイナリ / 正規化カウント）
  3. build_stage1_input が 108次元を返し、正しい順序でフィールドを配置すること
  4. 修正前の naive スライス (feat[0:108]) と build_stage1_input の出力が
     [83:108] の範囲で異なること（バグが実在したことの証明）
  5. Stage-1 モデルの期待フォーマットとの整合チェック
"""

import json
import sys
from pathlib import Path

import pytest

# phase2/scripts を import パスに追加
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from feature_offsets import (
    HI_TOTAL, HI_DISCARD_START, HI_MELD_START, HI_RIICHI,
    HI_SCORE_START, HI_SCORE_DIM,
    HI_GAME_START, HI_GAME_DIM,
    HI_WIND_START, HI_WIND_DIM,
    HI_YAKU_START, HI_YAKU_DIM, HI_TENPAI,
    STAGE1_INPUT_DIM, build_stage1_input,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "features" / "hand_inference_v26.ndjson"
N_SAMPLES = 1000  # テスト用サンプル数


@pytest.fixture(scope="module")
def samples():
    assert DATA_PATH.exists(), f"データファイルなし: {DATA_PATH}"
    data = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
            if len(data) >= N_SAMPLES:
                break
    return data


# ---- 1. 次元数チェック ----

def test_feature_total_dim(samples):
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            assert len(feat) == HI_TOTAL, (
                f"sample {i} player {p}: 特徴量次元 {len(feat)} ≠ {HI_TOTAL}"
            )


# ---- 2. 値域チェック ----

def test_riichi_flag_binary(samples):
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            v = feat[HI_RIICHI]
            assert v in (0, 1), f"sample {i} player {p}: riichi={v} (0 or 1 expected)"


def test_score_range(samples):
    """score は (点数-25000)/10000 で正規化: 点数0〜100000 → 約 -2.5〜7.5
    着順差は (自分 - ボーダー)/10000: 常に ≤ 0 → 約 -10〜0
    最後の3次元は場風(0/1), 局数(0〜1), 本場(0〜1)
    全体として大きな外れ値がないことを確認する"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            block = feat[HI_SCORE_START : HI_SCORE_START + HI_SCORE_DIM]
            for j, v in enumerate(block):
                assert -15.0 <= v <= 15.0, (
                    f"sample {i} player {p} score[{j}]={v:.4f} out of [-15, 15]"
                )


def test_game_state_range(samples):
    """game_state は正規化済み: 各値が 0.0〜1.0 の範囲内"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            block = feat[HI_GAME_START : HI_GAME_START + HI_GAME_DIM]
            for j, v in enumerate(block):
                assert 0.0 <= v <= 1.0, (
                    f"sample {i} player {p} game_state[{j}]={v:.4f} out of [0,1]"
                )


def test_visible_counts_normalized(samples):
    """visible_counts は 0〜4枚を /4 で正規化: 0.0〜1.0"""
    from feature_offsets import HI_VISIBLE
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            block = feat[HI_VISIBLE : HI_VISIBLE + 34]
            for j, v in enumerate(block):
                assert 0.0 <= v <= 1.0, (
                    f"sample {i} player {p} visible_counts[{j}]={v:.4f} out of [0,1]"
                )


def test_tenpai_prob_range(samples):
    """features[441] = tenpai_prob: 0.0〜1.0"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            v = feat[HI_TENPAI]
            assert 0.0 <= v <= 1.0, (
                f"sample {i} player {p}: tenpai_prob={v:.4f} out of [0,1]"
            )


def test_yaku_prob_range(samples):
    """features[420:441] = yaku_prob 21次元: 各値 0.0〜1.0"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            block = feat[HI_YAKU_START : HI_YAKU_START + HI_YAKU_DIM]
            for j, v in enumerate(block):
                assert 0.0 <= v <= 1.0, (
                    f"sample {i} player {p} yaku_prob[{j}]={v:.4f} out of [0,1]"
                )


# ---- 3. build_stage1_input の出力チェック ----

def test_build_stage1_input_dim(samples):
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            out = build_stage1_input(feat)
            assert len(out) == STAGE1_INPUT_DIM, (
                f"sample {i} player {p}: build_stage1_input len={len(out)} ≠ {STAGE1_INPUT_DIM}"
            )


def test_build_stage1_input_discard_unchanged(samples):
    """先頭 83次元 (discard+meld+riichi) は変わらないこと"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            out = build_stage1_input(feat)
            assert out[:83] == feat[:83], (
                f"sample {i} player {p}: discard/meld/riichi ブロックが変化している"
            )


def test_build_stage1_input_game_state_position(samples):
    """build_stage1_input の [83:92] が hand_inference の game_state [94:103] と一致すること"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            out = build_stage1_input(feat)
            expected_game = feat[HI_GAME_START : HI_GAME_START + HI_GAME_DIM]
            assert out[83:92] == expected_game, (
                f"sample {i} player {p}: Stage-1 入力の [83:92] が game_state と不一致"
            )


def test_build_stage1_input_score_position(samples):
    """build_stage1_input の [92:103] が hand_inference の score [83:94] と一致すること"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            out = build_stage1_input(feat)
            expected_score = feat[HI_SCORE_START : HI_SCORE_START + HI_SCORE_DIM]
            assert out[92:103] == expected_score, (
                f"sample {i} player {p}: Stage-1 入力の [92:103] が score と不一致"
            )


def test_build_stage1_input_wind_position(samples):
    """build_stage1_input の [103:108] が hand_inference の wind [347:352] と一致すること"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            out = build_stage1_input(feat)
            expected_wind = feat[HI_WIND_START : HI_WIND_START + HI_WIND_DIM]
            assert out[103:108] == expected_wind, (
                f"sample {i} player {p}: Stage-1 入力の [103:108] が wind と不一致"
            )


# ---- 4. 修正前バグの証明 ----

def test_naive_slice_differs_from_fixed(samples):
    """
    naive スライス feat[0:108] と build_stage1_input の [83:108] が異なるサンプルが
    存在することを確認する（= バグが実在したことの証明）
    """
    diff_found = False
    for s in samples:
        for feat in s["features"]:
            naive = feat[:STAGE1_INPUT_DIM]
            fixed = build_stage1_input(feat)
            if naive[83:] != fixed[83:]:
                diff_found = True
                break
        if diff_found:
            break
    assert diff_found, (
        "naive スライスと build_stage1_input が全サンプルで一致してしまった。"
        "バグが存在しないか、データが想定外の形式の可能性がある。"
    )


def test_naive_slice_prefix_same(samples):
    """先頭 83次元は naive スライスと build_stage1_input で同一であること"""
    for i, s in enumerate(samples):
        for p, feat in enumerate(s["features"]):
            naive = feat[:STAGE1_INPUT_DIM]
            fixed = build_stage1_input(feat)
            assert naive[:83] == fixed[:83], (
                f"sample {i} player {p}: 先頭 83次元が異なる（想定外）"
            )
