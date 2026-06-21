"""
hand_inference_v*.ndjson の per-player 特徴量オフセット定義 (442次元)

extract_features.js の make_hand_inference_sample と
add_yaku_features.py / add_tenpai_features.py で共有する。
"""

# ---- hand_inference per-player 特徴量オフセット (442次元) ----

HI_DISCARD_START  = 0    # target_discard         [0:44]    44次元
HI_MELD_START     = 44   # target_meld            [44:82]   38次元
HI_RIICHI         = 82   # target_riichi          [82]       1次元
HI_SCORE_START    = 83   # score                  [83:94]   11次元
HI_GAME_START     = 94   # game_state             [94:103]   9次元
HI_SELF_DISC      = 103  # self_discard           [103:147] 44次元
HI_SELF_MELD      = 147  # self_meld              [147:185] 38次元
HI_VISIBLE        = 185  # visible_counts         [185:219] 34次元
HI_RED_DISC_SIG   = 219  # red_discard_signal     [219:222]  3次元
HI_RED_VIS        = 222  # red_visible            [222:225]  3次元
HI_OTHER1_DISC    = 225  # other1_discard         [225:269] 44次元
HI_OTHER2_DISC    = 269  # other2_discard         [269:313] 44次元
HI_PON_PASS       = 313  # pass_pon_signal        [313:347] 34次元
HI_WIND_START     = 347  # wind_features(target)  [347:352]  5次元
HI_CHI_PASS       = 352  # pass_chi_signal        [352:386] 34次元
HI_CHI_CALLED     = 386  # chi_called_tile        [386:420] 34次元
HI_YAKU_START     = 420  # yaku_prob (Stage-1)    [420:441] 21次元
HI_TENPAI         = 441  # tenpai_prob (Stage-1)  [441]      1次元
HI_TOTAL          = 442  # 合計

# ---- 各ブロックの長さ ----

HI_SCORE_DIM  = 11
HI_GAME_DIM   = 9
HI_WIND_DIM   = 5
HI_YAKU_DIM   = 21

# ---- Stage-1 モデル入力フォーマット (108次元) ----
#
# tenpai_inference / yaku_inference は make_tenpai_sample / make_yaku_sample で
# 学習されており、入力フォーマットは:
#   discard(44) + meld(38) + riichi(1) + game_state(9) + score(11) + wind(5) = 108
#
# hand_inference features では score と game_state の順序が逆で、
# wind は [347:352] にある。そのまま先頭 108次元を切り出すと次元ズレが起きる。

STAGE1_INPUT_DIM = 108


def build_stage1_input(feat: list) -> list:
    """hand_inference 特徴量 (442次元リスト) から Stage-1 モデル用 108次元入力を構築する。

    score と game_state の順序を修正し、wind を正しい位置から取得する。
    """
    assert len(feat) >= HI_TOTAL, f"特徴量次元不足: {len(feat)} < {HI_TOTAL}"
    return (
        feat[HI_DISCARD_START : HI_MELD_START]              # discard  (44)
        + feat[HI_MELD_START  : HI_RIICHI]                  # meld     (38)
        + [feat[HI_RIICHI]]                                  # riichi   ( 1)
        + feat[HI_GAME_START  : HI_GAME_START + HI_GAME_DIM]  # game_state (9) ← 94:103
        + feat[HI_SCORE_START : HI_SCORE_START + HI_SCORE_DIM] # score    (11) ← 83:94
        + feat[HI_WIND_START  : HI_WIND_START + HI_WIND_DIM]  # wind      (5) ← 347:352
    )
