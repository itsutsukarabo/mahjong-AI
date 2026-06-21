"""
Phase 2: 特徴量統計テスト

NaN/Inf ゼロ・値域・死亡次元を pytest で検証する。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from feature_offsets import (
    HI_TOTAL, HI_RIICHI, HI_TENPAI, HI_YAKU_START, HI_YAKU_DIM,
    HI_SCORE_START, HI_SCORE_DIM, HI_GAME_START, HI_GAME_DIM,
    HI_VISIBLE, HI_RED_DISC_SIG, HI_WIND_START, HI_WIND_DIM,
    HI_PON_PASS, HI_CHI_PASS,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "features" / "hand_inference_v26.ndjson"
N_SAMPLES = 5000


@pytest.fixture(scope="module")
def feat_array():
    assert DATA_PATH.exists(), f"データファイルなし: {DATA_PATH}"
    feats = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            feats.extend(s["features"])
            if len(feats) >= N_SAMPLES * 3:
                break
    return np.array(feats, dtype=np.float32)


# ---- NaN / Inf ----

def test_no_nan(feat_array):
    assert not np.isnan(feat_array).any(), "NaN が検出されました"


def test_no_inf(feat_array):
    assert not np.isinf(feat_array).any(), "Inf が検出されました"


# ---- バイナリ次元の値域 ----

def test_riichi_binary(feat_array):
    col = feat_array[:, HI_RIICHI]
    assert np.all((col == 0) | (col == 1)), "riichi フラグに 0/1 以外の値"


def test_game_state_riichi_flags(feat_array):
    """game_state の [1:5] (全プレイヤーリーチフラグ) はバイナリ"""
    block = feat_array[:, HI_GAME_START + 1 : HI_GAME_START + 5]
    assert np.all((block == 0) | (block == 1)), "game_state リーチフラグに 0/1 以外の値"


def test_yaku_prob_range(feat_array):
    block = feat_array[:, HI_YAKU_START : HI_YAKU_START + HI_YAKU_DIM]
    assert np.all(block >= 0.0), "yaku_prob に負値"
    assert np.all(block <= 1.0), "yaku_prob に 1.0 超過"


def test_tenpai_prob_range(feat_array):
    col = feat_array[:, HI_TENPAI]
    assert np.all(col >= 0.0), "tenpai_prob に負値"
    assert np.all(col <= 1.0), "tenpai_prob に 1.0 超過"


def test_visible_counts_range(feat_array):
    """visible_counts は /4 正規化: 0〜1.0"""
    block = feat_array[:, HI_VISIBLE : HI_RED_DISC_SIG]
    assert np.all(block >= 0.0), "visible_counts に負値"
    assert np.all(block <= 1.0), "visible_counts に 1.0 超過"


def test_wind_features_range(feat_array):
    """wind は one-hot(4) + 場風(1)
    自風 [0:4] はバイナリ。場風 [4] は 0=東/1=南/2=西 (3麻なし前提で 0〜2)"""
    block = feat_array[:, HI_WIND_START : HI_WIND_START + HI_WIND_DIM]
    # 自風 one-hot (4次元) はバイナリ
    jikaze = block[:, :4]
    assert np.all((jikaze == 0) | (jikaze == 1)), "自風 one-hot に 0/1 以外の値"
    # 場風は 0(東)/1(南)/2(西) のいずれか
    bakaze = block[:, 4]
    assert np.all(np.isin(bakaze, [0, 1, 2])), f"場風に想定外の値: {np.unique(bakaze)}"


def test_game_state_remaining_normalized(feat_array):
    """remaining/70: 0〜1.0 の範囲"""
    col = feat_array[:, HI_GAME_START]
    assert np.all(col >= 0.0), "remaining に負値"
    assert np.all(col <= 1.0), "remaining に 1.0 超過"


# ---- 非正規化次元の合理的な上限 ----

def test_pass_pon_signal_not_extreme(feat_array):
    """pass_pon_signal は累積加算されるが異常値でないこと (上限 10.0 で検査)"""
    block = feat_array[:, HI_PON_PASS : HI_PON_PASS + 34]
    assert np.all(block >= 0.0), "pass_pon_signal に負値"
    max_val = float(block.max())
    assert max_val < 10.0, f"pass_pon_signal に異常な大値: max={max_val:.3f}"


def test_pass_chi_signal_not_extreme(feat_array):
    """pass_chi_signal も同様"""
    block = feat_array[:, HI_CHI_PASS : HI_CHI_PASS + 34]
    assert np.all(block >= 0.0), "pass_chi_signal に負値"
    max_val = float(block.max())
    assert max_val < 10.0, f"pass_chi_signal に異常な大値: max={max_val:.3f}"


# ---- 死亡次元チェック ----

def test_no_unexpected_dead_dim(feat_array):
    """真の死亡次元（max==0 かつ std==0）が構造的ゼロのみであること。

    構造的ゼロ（仕様通り）:
      81  : meld n_kita       (北抜きは本データで未使用)
      184 : self_meld n_kita  (同上)
      379-385: pass_chi_signal z1-z7    (字牌はチー不可)
      413-419: chi_called_tile z1-z7    (字牌はチー不可)

    NOTE: dim 87 は score[4]=(my_score-max_score)/10000 で常に≤0 (max=0) だが
    分散あり → 有効な特徴量のため除外。
    """
    STRUCTURAL_ZEROS = {81, 184, 379, 380, 381, 382, 383, 384, 385,
                        413, 414, 415, 416, 417, 418, 419}
    # max==0 AND std==0 の次元を真の死亡次元とする
    truly_dead = set(
        np.where((feat_array.max(axis=0) == 0) & (feat_array.std(axis=0) == 0))[0].tolist()
    )
    unexpected = truly_dead - STRUCTURAL_ZEROS
    assert len(unexpected) == 0, (
        f"想定外の常ゼロ次元: {sorted(unexpected)}"
    )
    assert truly_dead == STRUCTURAL_ZEROS, (
        f"構造的ゼロが変化:\n  期待: {sorted(STRUCTURAL_ZEROS)}\n  実際: {sorted(truly_dead)}"
    )


def test_no_unexpected_constant_dim(feat_array):
    """分散ゼロ（定数）の次元が構造的ゼロのみであること。
    score[4] = (my_score - max_score)/10000 は常に≤0で最大値=0だが分散あり → 除外しない。"""
    STRUCTURAL_ZEROS = {81, 184, 379, 380, 381, 382, 383, 384, 385,
                        413, 414, 415, 416, 417, 418, 419}
    zero_var = set(np.where(feat_array.std(axis=0) == 0)[0].tolist())
    unexpected = zero_var - STRUCTURAL_ZEROS
    assert len(unexpected) == 0, (
        f"想定外の定数次元: {sorted(unexpected)}"
    )
