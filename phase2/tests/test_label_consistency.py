"""
Phase 3: ラベル整合性テスト

hand_inference_v26.ndjson のラベルが特徴量と矛盾していないかを検証する。
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from feature_offsets import HI_RIICHI, HI_TOTAL

DATA_PATH = Path(__file__).parent.parent / "data" / "features" / "hand_inference_v26.ndjson"
N_SAMPLES = 5000


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


# ---- ラベルキーの存在確認 ----

def test_required_label_keys(samples):
    required = {"features", "label_hand", "label_red", "label_block",
                "label_shanten", "label_noise", "label_retreat", "label_push"}
    for i, s in enumerate(samples):
        missing = required - set(s.keys())
        assert not missing, f"sample {i}: ラベルキー不足 {missing}"


# ---- label_shanten の値域 ----

def test_label_shanten_values(samples):
    """label_shanten は -1〜8 の整数（または None）
    -1 はアガリ状態（和了後）の有効値"""
    for i, s in enumerate(samples):
        for p, v in enumerate(s["label_shanten"]):
            if v is None:
                continue
            assert isinstance(v, int), f"sample {i} player {p}: label_shanten が int でない: {v}"
            assert -1 <= v <= 8, f"sample {i} player {p}: label_shanten={v} out of [-1,8]"


# ---- 立直者は必ずテンパイ ----

def test_riichi_implies_tenpai(samples):
    """features[82] (riichi) = 1 なら label_shanten = 0 であること"""
    violations = 0
    for i, s in enumerate(samples):
        for p, (feat, shanten) in enumerate(zip(s["features"], s["label_shanten"])):
            if feat[HI_RIICHI] == 1 and shanten is not None and shanten != 0:
                violations += 1
    assert violations == 0, f"立直なのにテンパイでないサンプルが {violations} 件"


# ---- label_noise の型 ----

def test_label_noise_type(samples):
    """label_noise はサンプル単位の bool（プレイヤーごとの配列ではない）"""
    for i, s in enumerate(samples):
        v = s["label_noise"]
        assert isinstance(v, bool), f"sample {i}: label_noise が bool でない: {type(v)}"


# ---- label_retreat の型と値 ----

def test_label_retreat_structure(samples):
    """label_retreat は {is_retreat, tile, certain_multi, reason} の dict"""
    for i, s in enumerate(samples):
        lr = s["label_retreat"]
        assert isinstance(lr, dict), f"sample {i}: label_retreat が dict でない: {type(lr)}"
        assert "is_retreat" in lr, f"sample {i}: label_retreat に is_retreat なし"
        assert isinstance(lr["is_retreat"], bool), (
            f"sample {i}: is_retreat が bool でない: {type(lr['is_retreat'])}"
        )


# ---- label_push の型と構造 ----

def test_label_push_structure(samples):
    for i, s in enumerate(samples):
        lp = s["label_push"]
        assert isinstance(lp, dict), f"sample {i}: label_push が dict でない"
        assert "is_push" in lp, f"sample {i}: label_push に is_push なし"
        assert isinstance(lp["is_push"], bool), f"sample {i}: is_push が bool でない"
        if lp["is_push"]:
            assert "tile" in lp, f"sample {i}: is_push=True なのに tile なし"


# ---- 分布チェック（異常な偏りがないか） ----

def test_shanten_distribution(samples):
    """label_shanten の分布: テンパイ(0) が極端に少なくないこと (>5%)"""
    tenpai = sum(
        1 for s in samples for v in s["label_shanten"] if v == 0
    )
    total = sum(
        1 for s in samples for v in s["label_shanten"] if v is not None
    )
    tenpai_rate = tenpai / total if total > 0 else 0
    assert tenpai_rate > 0.05, f"テンパイサンプルが少なすぎる: {tenpai_rate*100:.1f}%"


def test_noise_rate(samples):
    """label_noise=True の割合が 0〜5% に収まること"""
    noise = sum(1 for s in samples if s["label_noise"])
    rate = noise / len(samples)
    assert rate < 0.05, f"ノイズ率が高すぎる: {rate*100:.1f}%"
    assert rate > 0.0, "ノイズサンプルが1件もない（ラベル生成が疑われる）"


def test_retreat_rate(samples):
    """label_retreat['is_retreat']=True の割合が 0〜20% に収まること"""
    retreat = sum(1 for s in samples if s["label_retreat"]["is_retreat"])
    rate = retreat / len(samples)
    assert 0.0 < rate < 0.20, f"retreat 率が想定外: {rate*100:.1f}%"


def test_push_rate(samples):
    """label_push.is_push=True の割合が 0〜50% に収まること"""
    push = sum(1 for s in samples if s["label_push"]["is_push"])
    rate = push / len(samples)
    assert 0.0 < rate < 0.50, f"push 率が想定外: {rate*100:.1f}%"


# ---- label_hand の次元 ----

def test_label_hand_dim(samples):
    """label_hand は 3プレイヤー × 34次元"""
    for i, s in enumerate(samples):
        assert len(s["label_hand"]) == 3, f"sample {i}: label_hand のプレイヤー数不正"
        for p, hand in enumerate(s["label_hand"]):
            assert len(hand) == 34, f"sample {i} player {p}: label_hand 次元={len(hand)} ≠ 34"


def test_label_hand_count_range(samples):
    """label_hand の各牌枚数は 0〜4"""
    for i, s in enumerate(samples):
        for p, hand in enumerate(s["label_hand"]):
            for j, cnt in enumerate(hand):
                assert 0 <= cnt <= 4, (
                    f"sample {i} player {p} tile {j}: label_hand={cnt} out of [0,4]"
                )


def test_label_red_dim(samples):
    """label_red は 3プレイヤー × 3次元 (5m/5p/5s)"""
    for i, s in enumerate(samples):
        assert len(s["label_red"]) == 3, f"sample {i}: label_red のプレイヤー数不正"
        for p, red in enumerate(s["label_red"]):
            assert len(red) == 3, f"sample {i} player {p}: label_red 次元={len(red)} ≠ 3"
            for j, v in enumerate(red):
                assert v in (0, 1), f"sample {i} player {p} red[{j}]={v} not 0/1"
