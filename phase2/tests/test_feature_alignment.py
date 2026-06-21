"""
特徴量アライメントのテスト (v37対応)

検証内容:
  1. hand_inference.ndjson の特徴量次元が HI_TOTAL (695) であること
  2. 既知オフセットの値域が仕様通りであること（バイナリ / 正規化カウント）
  3. build_stage1_input が 108次元を返し、正しい順序でフィールドを配置すること
  4. オフセット定数の連続性（コードレベルで検証）
  5. discard_tokens フィールドの形式（42次元/トークン、最大72トークン）
  6. target_discard フィールドの形式（44次元の集計ベクトル）
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from feature_offsets import (
    HI_TOTAL, HI_MELD_START, HI_RIICHI,
    HI_SCORE_START, HI_SCORE_DIM,
    HI_GAME_START, HI_GAME_DIM,
    HI_VISIBLE,
    HI_WIND_START, HI_WIND_DIM,
    HI_DORA_START, HI_DORA_DIM,
    HI_OTHER1_MELD, HI_OTHER2_MELD, HI_MELD_DIM,
    HI_SELF_PON_PASS, HI_O1_PON_PASS, HI_O2_PON_PASS, HI_PON_DIM,
    HI_SELF_CHI_PASS, HI_O1_CHI_PASS, HI_O2_CHI_PASS, HI_CHI_DIM,
    HI_LIZHIBANG,
    HI_SELF_JIKAZE, HI_O1_JIKAZE, HI_O2_JIKAZE,
    HI_SELF_CHI_CALLED, HI_O1_CHI_CALLED, HI_O2_CHI_CALLED,
    HI_YAKU_START, HI_YAKU_DIM, HI_TENPAI,
    HI_TOKEN_DIM, HI_TOKEN_MAX,
    HI_RED_DISC_SIG, HI_RED_VIS,
    HI_PON_PASS, HI_CHI_PASS, HI_CHI_CALLED,
    STAGE1_INPUT_DIM, build_stage1_input,
)

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
N_SAMPLES = 1000


def _find_data_file():
    """最新の hand_inference*.ndjson を返す（なければ None）"""
    candidates = sorted(DATA_DIR.glob("hand_inference*.ndjson"), reverse=True)
    return candidates[0] if candidates else None


@pytest.fixture(scope="module")
def samples():
    path = _find_data_file()
    if path is None:
        pytest.skip(f"データファイルなし: {DATA_DIR}/hand_inference*.ndjson")
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
            if len(data) >= N_SAMPLES:
                break
    if not data:
        pytest.skip("データが空")
    return data


# ---- 0. オフセット連続性チェック（コードレベル、データ不要） ----

def test_offset_continuity():
    """全ブロックが隙間なく連続して並んでいること"""
    blocks = [
        (HI_MELD_START,       38, "target_meld"),
        (HI_RIICHI,            1, "riichi"),
        (HI_SCORE_START,      11, "score"),
        (HI_GAME_START,        9, "game_state"),
        (59,                  38, "self_meld"),
        (HI_VISIBLE,          34, "visible_counts"),
        (HI_RED_DISC_SIG,      3, "red_discard_sig"),
        (HI_RED_VIS,           3, "red_visible"),
        (HI_PON_PASS,         34, "pass_pon(target)"),
        (HI_WIND_START,        5, "wind(target)"),
        (HI_CHI_PASS,         34, "pass_chi(target)"),
        (HI_CHI_CALLED,       34, "chi_called(target)"),
        (HI_DORA_START,       34, "dora"),
        (HI_OTHER1_MELD,      38, "other1_meld"),
        (HI_OTHER2_MELD,      38, "other2_meld"),
        (HI_SELF_PON_PASS,    34, "self_pon_pass"),
        (HI_O1_PON_PASS,      34, "other1_pon_pass"),
        (HI_O2_PON_PASS,      34, "other2_pon_pass"),
        (HI_SELF_CHI_PASS,    34, "self_chi_pass"),
        (HI_O1_CHI_PASS,      34, "other1_chi_pass"),
        (HI_O2_CHI_PASS,      34, "other2_chi_pass"),
        (HI_LIZHIBANG,         1, "lizhibang"),
        (HI_SELF_JIKAZE,       4, "self_jikaze"),
        (HI_O1_JIKAZE,         4, "other1_jikaze"),
        (HI_O2_JIKAZE,         4, "other2_jikaze"),
        (HI_SELF_CHI_CALLED,  34, "self_chi_called"),
        (HI_O1_CHI_CALLED,    34, "other1_chi_called"),
        (HI_O2_CHI_CALLED,    34, "other2_chi_called"),
        (HI_YAKU_START,       21, "yaku_prob"),
        (HI_TENPAI,            1, "tenpai_prob"),
    ]
    cursor = 0
    for start, dim, name in blocks:
        assert start == cursor, f"{name}: オフセット {start} ≠ 期待値 {cursor}"
        cursor += dim
    assert cursor == HI_TOTAL, f"合計次元 {cursor} ≠ HI_TOTAL={HI_TOTAL}"


# ---- 1. 次元数チェック ----

def test_feature_total_dim(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_TOTAL:
            pytest.skip(f"データ再生成前（{len(feat)}次元）のためスキップ")
        assert len(feat) == HI_TOTAL, (
            f"sample {i}: 特徴量次元 {len(feat)} ≠ {HI_TOTAL}"
        )


# ---- 2. 値域チェック ----

def test_riichi_flag_binary(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) <= HI_RIICHI:
            continue
        v = feat[HI_RIICHI]
        assert v in (0, 1), f"sample {i}: riichi={v} (0 or 1 expected)"


def test_score_range(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_SCORE_START + HI_SCORE_DIM:
            continue
        block = feat[HI_SCORE_START : HI_SCORE_START + HI_SCORE_DIM]
        for j, v in enumerate(block):
            assert -15.0 <= v <= 15.0, (
                f"sample {i} score[{j}]={v:.4f} out of [-15, 15]"
            )


def test_game_state_range(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_GAME_START + HI_GAME_DIM:
            continue
        block = feat[HI_GAME_START : HI_GAME_START + HI_GAME_DIM]
        for j, v in enumerate(block):
            assert 0.0 <= v <= 1.0, (
                f"sample {i} game_state[{j}]={v:.4f} out of [0,1]"
            )


def test_visible_counts_normalized(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_VISIBLE + 34:
            continue
        block = feat[HI_VISIBLE : HI_VISIBLE + 34]
        for j, v in enumerate(block):
            assert 0.0 <= v <= 1.0, (
                f"sample {i} visible_counts[{j}]={v:.4f} out of [0,1]"
            )


def test_tenpai_prob_range(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) <= HI_TENPAI:
            pytest.skip("データ再生成前のためスキップ")
        v = feat[HI_TENPAI]
        assert 0.0 <= v <= 1.0, f"sample {i}: tenpai_prob={v:.4f} out of [0,1]"


def test_yaku_prob_range(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_YAKU_START + HI_YAKU_DIM:
            pytest.skip("データ再生成前のためスキップ")
        block = feat[HI_YAKU_START : HI_YAKU_START + HI_YAKU_DIM]
        for j, v in enumerate(block):
            assert 0.0 <= v <= 1.0, (
                f"sample {i} yaku_prob[{j}]={v:.4f} out of [0,1]"
            )


# ---- 3. 新規特徴量の値域チェック ----

def test_dora_features_range(samples):
    """dora_features は 0.0〜1.0（最大5ドラ/5=1.0）"""
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_DORA_START + HI_DORA_DIM:
            pytest.skip("データ再生成前のためスキップ")
        block = feat[HI_DORA_START : HI_DORA_START + HI_DORA_DIM]
        for j, v in enumerate(block):
            assert 0.0 <= v <= 1.0, (
                f"sample {i} dora[{j}]={v:.4f} out of [0,1]"
            )


def test_other_meld_features_range(samples):
    """other1/other2 の meld_features: tile_presence は 0/1、カウントは 0〜4"""
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_OTHER2_MELD + HI_MELD_DIM:
            pytest.skip("データ再生成前のためスキップ")
        for start, name in [(HI_OTHER1_MELD, "other1_meld"), (HI_OTHER2_MELD, "other2_meld")]:
            tile_block  = feat[start : start + 34]
            count_block = feat[start + 34 : start + 38]
            for j, v in enumerate(tile_block):
                assert v in (0, 1), f"sample {i} {name} tile[{j}]={v} not binary"
            for j, v in enumerate(count_block):
                assert 0 <= v <= 4, f"sample {i} {name} count[{j}]={v} out of [0,4]"


def test_pon_pass_signals_range(samples):
    """self/other1/other2 の pon_pass_signal は 0.0 以上"""
    pass_starts = [
        (HI_SELF_PON_PASS, "self_pon"),
        (HI_O1_PON_PASS,   "other1_pon"),
        (HI_O2_PON_PASS,   "other2_pon"),
    ]
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_O2_PON_PASS + HI_PON_DIM:
            pytest.skip("データ再生成前のためスキップ")
        for start, name in pass_starts:
            for j, v in enumerate(feat[start : start + HI_PON_DIM]):
                assert v >= 0.0, f"sample {i} {name}[{j}]={v} < 0"


def test_chi_pass_signals_range(samples):
    """self/other1/other2 の chi_pass_signal は 0.0 以上"""
    pass_starts = [
        (HI_SELF_CHI_PASS, "self_chi"),
        (HI_O1_CHI_PASS,   "other1_chi"),
        (HI_O2_CHI_PASS,   "other2_chi"),
    ]
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_O2_CHI_PASS + HI_CHI_DIM:
            pytest.skip("データ再生成前のためスキップ")
        for start, name in pass_starts:
            for j, v in enumerate(feat[start : start + HI_CHI_DIM]):
                assert v >= 0.0, f"sample {i} {name}[{j}]={v} < 0"


def test_lizhibang_range(samples):
    """lizhibang は 0.0〜1.0（min(count,8)/8 で正規化）"""
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) <= HI_LIZHIBANG:
            pytest.skip("データ再生成前のためスキップ")
        v = feat[HI_LIZHIBANG]
        assert 0.0 <= v <= 1.0, f"sample {i} lizhibang={v:.4f} out of [0,1]"


def test_jikaze_onehot(samples):
    """self/other1/other2 の jikaze は one-hot（合計1、各値0/1）"""
    jikaze_blocks = [
        (HI_SELF_JIKAZE, "self_jikaze"),
        (HI_O1_JIKAZE,   "other1_jikaze"),
        (HI_O2_JIKAZE,   "other2_jikaze"),
    ]
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_O2_JIKAZE + 4:
            pytest.skip("データ再生成前のためスキップ")
        for start, name in jikaze_blocks:
            block = feat[start : start + 4]
            for j, v in enumerate(block):
                assert v in (0, 1), f"sample {i} {name}[{j}]={v} not binary"
            assert sum(block) == 1, f"sample {i} {name}={block} not one-hot"


def test_chi_called_binary(samples):
    """self/other1/other2 の chi_called_tile は 0/1 バイナリ"""
    chi_called_blocks = [
        (HI_SELF_CHI_CALLED, "self_chi_called"),
        (HI_O1_CHI_CALLED,   "other1_chi_called"),
        (HI_O2_CHI_CALLED,   "other2_chi_called"),
    ]
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_O2_CHI_CALLED + 34:
            pytest.skip("データ再生成前のためスキップ")
        for start, name in chi_called_blocks:
            for j, v in enumerate(feat[start : start + 34]):
                assert v in (0, 1), f"sample {i} {name}[{j}]={v} not binary"


# ---- 4. build_stage1_input のチェック ----

def test_build_stage1_input_dim(samples):
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_TOTAL:
            pytest.skip("データ再生成前のためスキップ")
        td = s["target_discard"]
        assert len(td) == 44, f"sample {i}: target_discard len={len(td)} ≠ 44"
        out = build_stage1_input(feat, td)
        assert len(out) == STAGE1_INPUT_DIM, (
            f"sample {i}: build_stage1_input len={len(out)} ≠ {STAGE1_INPUT_DIM}"
        )


def test_build_stage1_input_game_state_position(samples):
    """build_stage1_input の [83:92] が hand_inference の game_state と一致すること"""
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_TOTAL:
            pytest.skip("データ再生成前のためスキップ")
        out = build_stage1_input(feat, s["target_discard"])
        expected_game = feat[HI_GAME_START : HI_GAME_START + HI_GAME_DIM]
        assert out[83:92] == expected_game, (
            f"sample {i}: Stage-1 入力の [83:92] が game_state と不一致"
        )


def test_build_stage1_input_score_position(samples):
    """build_stage1_input の [92:103] が hand_inference の score と一致すること"""
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_TOTAL:
            pytest.skip("データ再生成前のためスキップ")
        out = build_stage1_input(feat, s["target_discard"])
        expected_score = feat[HI_SCORE_START : HI_SCORE_START + HI_SCORE_DIM]
        assert out[92:103] == expected_score, (
            f"sample {i}: Stage-1 入力の [92:103] が score と不一致"
        )


def test_build_stage1_input_wind_position(samples):
    """build_stage1_input の [103:108] が hand_inference の wind と一致すること"""
    for i, s in enumerate(samples):
        feat = s["features"]
        if len(feat) < HI_TOTAL:
            pytest.skip("データ再生成前のためスキップ")
        out = build_stage1_input(feat, s["target_discard"])
        expected_wind = feat[HI_WIND_START : HI_WIND_START + HI_WIND_DIM]
        assert out[103:108] == expected_wind, (
            f"sample {i}: Stage-1 入力の [103:108] が wind と不一致"
        )


# ---- 5. discard_tokens フィールドのチェック ----

def test_discard_tokens_format(samples):
    """discard_tokens: 各トークンが HI_TOKEN_DIM (42) 次元、最大 HI_TOKEN_MAX (72) トークン"""
    for i, s in enumerate(samples):
        if "discard_tokens" not in s:
            pytest.skip("データ再生成前（discard_tokensフィールドなし）のためスキップ")
        tokens = s["discard_tokens"]
        assert len(tokens) <= HI_TOKEN_MAX, (
            f"sample {i}: トークン数 {len(tokens)} > {HI_TOKEN_MAX}"
        )
        for t, tok in enumerate(tokens):
            assert len(tok) == HI_TOKEN_DIM, (
                f"sample {i} token {t}: 次元 {len(tok)} ≠ {HI_TOKEN_DIM}"
            )


def test_discard_tokens_tile_onehot(samples):
    """discard_tokens の tile_onehot (先頭34次元) は one-hot であること"""
    for i, s in enumerate(samples):
        if "discard_tokens" not in s:
            pytest.skip("データ再生成前のためスキップ")
        for t, tok in enumerate(s["discard_tokens"]):
            tile = tok[:34]
            for j, v in enumerate(tile):
                assert v in (0, 1), f"sample {i} token {t} tile[{j}]={v} not binary"
            assert sum(tile) == 1, f"sample {i} token {t} tile not one-hot: {tile}"


def test_discard_tokens_flags(samples):
    """discard_tokens のフラグ次元 (34-37) はすべて 0/1 であること"""
    for i, s in enumerate(samples):
        if "discard_tokens" not in s:
            pytest.skip("データ再生成前のためスキップ")
        for t, tok in enumerate(s["discard_tokens"]):
            flags = tok[34:38]   # turn_norm(1) + is_tsumogiri(1) + is_riichi_decl(1) + is_red_five(1)
            for j, v in enumerate(flags[1:], 1):   # turn_norm は 0〜1 の連続値なのでスキップ
                assert v in (0, 1), f"sample {i} token {t} flag[{j}]={v} not binary"


def test_discard_tokens_player_role(samples):
    """discard_tokens の player_role (末尾4次元) は one-hot であること"""
    for i, s in enumerate(samples):
        if "discard_tokens" not in s:
            pytest.skip("データ再生成前のためスキップ")
        for t, tok in enumerate(s["discard_tokens"]):
            role = tok[38:42]
            for j, v in enumerate(role):
                assert v in (0, 1), f"sample {i} token {t} role[{j}]={v} not binary"
            assert sum(role) == 1, f"sample {i} token {t} role not one-hot: {role}"


# ---- 6. target_discard フィールドのチェック ----

def test_target_discard_dim(samples):
    """target_discard は44次元の集計ベクトルであること"""
    for i, s in enumerate(samples):
        if "target_discard" not in s:
            pytest.skip("データ再生成前のためスキップ")
        td = s["target_discard"]
        assert len(td) == 44, f"sample {i}: target_discard len={len(td)} ≠ 44"
