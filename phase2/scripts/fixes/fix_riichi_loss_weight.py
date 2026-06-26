"""
fix_riichi_loss_weight: リーチ局面サンプルの loss_weight を 2.0 に設定

- features の HI_RIICHI (offset 38) がターゲット player のリーチフラグ。
- データ再生成不要。
"""

RIICHI_WEIGHT = 2.0

_WEIGHT_FUNC = f'''
def get_riichi_sample_weight(features, base_weight=1.0):
    """
    HI_RIICHI (offset 38) のフラグでリーチ局面を 2.0 倍重み付け。
    features: (B, 3, D)
    returns:  (B, 3) weight tensor
    """
    from feature_offsets import HI_RIICHI
    riichi_flag = features[..., HI_RIICHI]  # (B, 3)
    weight = base_weight + riichi_flag * ({RIICHI_WEIGHT} - 1.0)
    return weight
'''

_IMPORT_PATCH  = 'import sys\n'
_AFTER_IMPORT  = 'import sys\n' + _WEIGHT_FUNC + '\n'

_LOSS_SEARCH   = 'loss = nll_loss'
_LOSS_REPLACE  = (
    'riichi_w = get_riichi_sample_weight(features)  # (B, 3)\n'
    '    nll_loss = (nll_loss_unreduced * riichi_w).mean()\n'
    '    loss = nll_loss'
)


def patch_train_script(code: str, base_ver: int, target_ver: int) -> str:
    if 'get_riichi_sample_weight' not in code:
        code = code.replace(_IMPORT_PATCH, _AFTER_IMPORT, 1)

    # nll_loss を unreduced で計算する形式を前提とする
    # もし既に reduction='mean' なら reduce=False に変更が必要
    if 'nll_loss_unreduced' not in code and _LOSS_SEARCH in code:
        # nll_loss = F.nll_loss(...) → reduction='none' に変更して後で mean
        code = code.replace(
            "reduction='mean'",
            "reduction='none'",
            1
        )
        code = code.replace(_LOSS_SEARCH, _LOSS_REPLACE, 1)

    if 'RIICHI_WEIGHT_SCALE' not in code:
        code = code.replace(
            'CONFIG = {',
            f'RIICHI_WEIGHT_SCALE = {RIICHI_WEIGHT}\nCONFIG = {{',
            1
        )

    return code
