"""
fix_impossible_tile_penalty: 枯れ牌への確率割当をペナルティ化

- feature_offsets.py の HI_VISIBLE (offset 97〜130) から paishu を推定し、
  枯れ牌(残0枚)への P(持っている) を loss に加算する。
- データ再生成不要。学習スクリプトのみ変更。
"""

LAMBDA = 0.5  # ペナルティ強度の初期値

# train_hand_inference_v*.py への差分コード
_PENALTY_FUNC = '''
def compute_impossible_tile_penalty(logits, features, fixed_dim=None):
    """
    枯れ牌(残0枚)への確率割当ペナルティ。
    logits:   (B, 3, 34, 5)
    features: (B, 3, D) — HI_VISIBLE (offset 97-130) から paishu を推定
    """
    import torch.nn.functional as F
    from feature_offsets import HI_VISIBLE
    # visible_counts_vec は /4 正規化済み → 4倍して四捨五入 → seen_count
    vis = features[..., HI_VISIBLE:HI_VISIBLE+34]  # (B, 3, 34)
    seen = (vis * 4).round().long().clamp(0, 4)
    impossible = (seen >= 4).float()               # (B, 3, 34): 1=枯れ牌
    # P(枚数>=1) = 1 - softmax(logits)[...,0]
    probs = F.softmax(logits, dim=-1)              # (B, 3, 34, 5)
    p_has = 1.0 - probs[..., 0]                   # (B, 3, 34)
    penalty = (p_has * impossible).mean()
    return penalty
'''

_IMPORT_PATCH = 'import sys\n'
_AFTER_IMPORT = 'import sys\n' + _PENALTY_FUNC + '\n'

# loss 計算箇所に挿入するコード
_LOSS_PATCH_SEARCH  = 'loss = nll_loss'
_LOSS_PATCH_REPLACE = (
    'impossible_penalty = compute_impossible_tile_penalty(logits, features)\n'
    f'    loss = nll_loss + {LAMBDA} * impossible_penalty'
)


def patch_train_script(code: str, base_ver: int, target_ver: int) -> str:
    # 1. ペナルティ関数を先頭付近に挿入
    if 'compute_impossible_tile_penalty' not in code:
        code = code.replace(_IMPORT_PATCH, _AFTER_IMPORT, 1)

    # 2. loss 行にペナルティ加算を挿入
    if _LOSS_PATCH_SEARCH in code and 'impossible_penalty' not in code:
        code = code.replace(_LOSS_PATCH_SEARCH, _LOSS_PATCH_REPLACE, 1)

    # 3. CONFIG に LAMBDA_IMPOSSIBLE を追記
    if 'LAMBDA_IMPOSSIBLE' not in code:
        code = code.replace(
            'CONFIG = {',
            f'LAMBDA_IMPOSSIBLE = {LAMBDA}\nCONFIG = {{',
            1
        )

    return code
