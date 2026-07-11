"""
hand_inference_v*.ndjson の per-player 特徴量オフセット定義 (695次元)

extract_features.js の make_hand_inference_sample と
add_yaku_features.py / add_tenpai_features.py で共有する。

v37変更点:
  - discard_features(×4) を discard_tokens 別フィールドに移行 (-176次元)
  - self/other1/other2 jikaze を追加 (+12次元)
  - self/other1/other2 chi_called を追加 (+102次元)
  - is_red_five フラグをトークンに追加 (42次元/トークン)
  757次元 → 673次元固定 + discard_tokens別フィールド + 22 (yaku/tenpai) = 695次元
"""

# ---- hand_inference per-player 固定特徴量オフセット (673次元) ----
# ※ discard_features は discard_tokens (別フィールド) に移行済み

HI_MELD_START      = 0    # target_meld            [0:38]    38次元
HI_RIICHI          = 38   # target_riichi          [38]       1次元
HI_SCORE_START     = 39   # score                  [39:50]   11次元
HI_GAME_START      = 50   # game_state             [50:59]    9次元
HI_SELF_MELD       = 59   # self_meld              [59:97]   38次元
HI_VISIBLE         = 97   # visible_counts         [97:131]  34次元
HI_RED_DISC_SIG    = 131  # red_discard_signal     [131:134]  3次元
HI_RED_VIS         = 134  # red_visible            [134:137]  3次元
HI_PON_PASS        = 137  # pass_pon_signal(target)[137:171] 34次元
HI_WIND_START      = 171  # wind_features(target)  [171:176]  5次元
HI_CHI_PASS        = 176  # pass_chi_signal(target)[176:210] 34次元
HI_CHI_CALLED      = 210  # chi_called_tile(target)[210:244] 34次元
HI_DORA_START      = 244  # dora_features          [244:278] 34次元
HI_OTHER1_MELD     = 278  # other1_meld_features   [278:316] 38次元
HI_OTHER2_MELD     = 316  # other2_meld_features   [316:354] 38次元
HI_SELF_PON_PASS   = 354  # self_pon_pass_signal   [354:388] 34次元
HI_O1_PON_PASS     = 388  # other1_pon_pass_signal [388:422] 34次元
HI_O2_PON_PASS     = 422  # other2_pon_pass_signal [422:456] 34次元
HI_SELF_CHI_PASS   = 456  # self_chi_pass_signal   [456:490] 34次元
HI_O1_CHI_PASS     = 490  # other1_chi_pass_signal [490:524] 34次元
HI_O2_CHI_PASS     = 524  # other2_chi_pass_signal [524:558] 34次元
HI_LIZHIBANG       = 558  # lizhibang              [558]      1次元
HI_SELF_JIKAZE     = 559  # self_jikaze            [559:563]  4次元
HI_O1_JIKAZE       = 563  # other1_jikaze          [563:567]  4次元
HI_O2_JIKAZE       = 567  # other2_jikaze          [567:571]  4次元
HI_SELF_CHI_CALLED = 571  # self_chi_called        [571:605] 34次元
HI_O1_CHI_CALLED   = 605  # other1_chi_called      [605:639] 34次元
HI_O2_CHI_CALLED   = 639  # other2_chi_called      [639:673] 34次元
# HI_YAKU_START は廃止: yaku_head として手牌推定モデルに統合 (v41~)
HI_TENPAI          = 673  # tenpai_prob            [673]      1次元
HI_TOTAL           = 674  # 合計

# ---- 各ブロックの長さ ----

HI_SCORE_DIM  = 11
HI_GAME_DIM   = 9
HI_WIND_DIM   = 5
HI_DORA_DIM   = 34
HI_MELD_DIM   = 38
HI_PON_DIM    = 34
HI_CHI_DIM    = 34
HI_YAKU_DIM   = 21  # モデルのyaku_head出力次元（特徴量としては使わない）

# ---- 統一イベントトークン形式 (45次元/トークン) ----
# tile_onehot(34) + turn_norm(1) + is_tsumogiri(1) + is_riichi_decl(1) + is_red_five(1)
# + player_role(4) + event_is_pass_pon(1) + event_is_pass_chi(1) + danger_level_norm(1)
# player_role: [is_target, is_self, is_other1, is_other2]
# event: DISCARD=[0,0,dl/3], PASS_PON=[1,0,0], PASS_CHI=[0,1,0]
# danger_level_norm: 0=リーチなし, 1/3=現物, 2/3=スジ, 1=無スジ (v45追加)
HI_TOKEN_DIM = 45
HI_TOKEN_MAX = 144  # 最大トークン数 (捨て牌~72 + PASSイベント~72)

# ---- Stage-1 モデル入力フォーマット (108次元) ----
#
# tenpai_inference / yaku_inference は make_tenpai_sample / make_yaku_sample で
# 学習されており、入力フォーマットは:
#   discard(44) + meld(38) + riichi(1) + game_state(9) + score(11) + wind(5) = 108
#
# hand_inference の固定特徴量には discard_features が含まれないため、
# target_discard (別フィールド) を引数として別途渡す必要がある。

STAGE1_INPUT_DIM = 108


def build_stage1_input(feat: list, target_discard: list) -> list:
    """hand_inference 固定特徴量 (695次元) と target_discard (44次元) から Stage-1 モデル用 108次元入力を構築する。

    score と game_state の順序を修正し、wind を正しい位置から取得する。
    """
    _min_needed = HI_WIND_START + HI_WIND_DIM  # 176
    assert len(feat) >= _min_needed, f"特徴量次元不足: {len(feat)} < {_min_needed}"
    assert len(target_discard) == 44, f"target_discard次元不足: {len(target_discard)} != 44"
    return (
        target_discard                                                  # discard  (44)
        + feat[HI_MELD_START : HI_RIICHI]                             # meld     (38)
        + [feat[HI_RIICHI]]                                            # riichi   ( 1)
        + feat[HI_GAME_START : HI_GAME_START + HI_GAME_DIM]           # game     ( 9)
        + feat[HI_SCORE_START : HI_SCORE_START + HI_SCORE_DIM]        # score    (11)
        + feat[HI_WIND_START  : HI_WIND_START  + HI_WIND_DIM]         # wind     ( 5)
    )
