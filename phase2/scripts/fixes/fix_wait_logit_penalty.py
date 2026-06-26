"""
fix_wait_logit_penalty: リーチ後ツモ切り牌への wait_logits 矛盾ペナルティ

- リーチ後にツモ切りされた牌は待ち牌ではないはず。
- label_wait の生成時に「ツモ切り牌=0」マスクを適用し、
  対応する wait_logits が高い場合にペナルティを加算する。
- データ再生成不要。
"""

LAMBDA_WAIT_PENALTY = 0.3

_PENALTY_FUNC = f'''
def compute_wait_contradiction_penalty(wait_logits, disc_tokens, disc_mask):
    """
    リーチ後ツモ切り牌のインデックスを disc_tokens から取得し、
    wait_logits の単騎・双碰 head でそれらに高い確率が割り当てられていたら罰する。
    wait_logits: (B, 3, N_TATSU)
    disc_tokens: (B, 3, T, TD)  — token[..., :34] が tile_onehot
    disc_mask:   (B, 3, T)      — True=padding
    """
    import torch
    import torch.nn.functional as F
    B, P, T, TD = disc_tokens.shape
    # ツモ切りトークン: is_tsumogiri=1 (dim35) かつ is_riichi_decl=1 の次以降
    tsumogiri = disc_tokens[..., 35]           # (B, 3, T)
    valid      = (~disc_mask).float()           # (B, 3, T)
    tsumo_tiles = disc_tokens[..., :34] * tsumogiri.unsqueeze(-1) * valid.unsqueeze(-1)
    # 各牌の最大ツモ切りフラグ: (B, 3, 34)
    tsumo_tile_mask = tsumo_tiles.sum(dim=2).clamp(0, 1)

    # wait_logits の単騎(idx 45-78) と双碰(idx 79-112) を sigmoid
    wait_probs = torch.sigmoid(wait_logits)    # (B, 3, N_TATSU)
    tanki   = wait_probs[..., 45:79]           # (B, 3, 34)
    shanpon = wait_probs[..., 79:113]          # (B, 3, 34)

    # ツモ切り牌への確率が高い = 矛盾
    penalty = ((tanki + shanpon) * tsumo_tile_mask).mean()
    return penalty
'''

_IMPORT_PATCH = 'import sys\n'
_AFTER_IMPORT = 'import sys\n' + _PENALTY_FUNC + '\n'

_LOSS_SEARCH  = 'loss = nll_loss'
_LOSS_REPLACE = (
    f'wait_contra = compute_wait_contradiction_penalty(wait_logits, disc_tokens, disc_mask)\n'
    f'    loss = nll_loss + {LAMBDA_WAIT_PENALTY} * wait_contra'
)


def patch_train_script(code: str, base_ver: int, target_ver: int) -> str:
    if 'compute_wait_contradiction_penalty' not in code:
        code = code.replace(_IMPORT_PATCH, _AFTER_IMPORT, 1)

    if 'wait_contra' not in code and _LOSS_SEARCH in code:
        code = code.replace(_LOSS_SEARCH, _LOSS_REPLACE, 1)

    if 'LAMBDA_WAIT_PENALTY' not in code:
        code = code.replace(
            'CONFIG = {',
            f'LAMBDA_WAIT_PENALTY = {LAMBDA_WAIT_PENALTY}\nCONFIG = {{',
            1
        )

    return code
