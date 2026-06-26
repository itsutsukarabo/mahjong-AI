"""
fix_late_game_weight: 終盤 (巡目15+) サンプルの loss_weight を 1.5 に設定

- 終盤は paishu が少なく枯れ牌問題が顕在化しやすい。
- データ再生成不要。
"""

LATE_GAME_SCALE = 1.5
LATE_GAME_TURN  = 15  # この巡目以降を「終盤」とみなす

_WEIGHT_FUNC = f'''
def get_late_game_weight(features, remaining_idx=None, base_weight=1.0):
    """
    HI_GAME_START + 残り牌数の正規化値から終盤を判定して weight を上げる。
    remaining が 0.3 以下 (巡目15+ 相当) なら {LATE_GAME_SCALE} 倍。
    features: (B, 3, D)
    """
    from feature_offsets import HI_GAME_START
    # HI_GAME_START+0: remaining/136 (残り牌数の正規化)
    remaining_norm = features[..., HI_GAME_START]  # (B, 3)
    # 136 * 0.3 ≈ 40.8 残 → 巡目≥15 相当
    late_game = (remaining_norm < 0.30).float()
    weight = base_weight + late_game * ({LATE_GAME_SCALE} - 1.0)
    return weight
'''

_IMPORT_PATCH = 'import sys\n'
_AFTER_IMPORT = 'import sys\n' + _WEIGHT_FUNC + '\n'

_LOSS_SEARCH  = 'loss = nll_loss'
_LOSS_REPLACE = (
    'late_w = get_late_game_weight(features)  # (B, 3)\n'
    '    nll_loss = (nll_loss_unreduced * late_w).mean()\n'
    '    loss = nll_loss'
)


def patch_train_script(code: str, base_ver: int, target_ver: int) -> str:
    if 'get_late_game_weight' not in code:
        code = code.replace(_IMPORT_PATCH, _AFTER_IMPORT, 1)

    if 'late_w' not in code and _LOSS_SEARCH in code:
        code = code.replace(
            "reduction='mean'",
            "reduction='none'",
            1
        )
        code = code.replace(_LOSS_SEARCH, _LOSS_REPLACE, 1)

    if 'LATE_GAME_SCALE' not in code:
        code = code.replace(
            'CONFIG = {',
            f'LATE_GAME_SCALE = {LATE_GAME_SCALE}\nCONFIG = {{',
            1
        )

    return code
