"""
fix_furiten_penalty: フリテン系待ち確率の抑制ペナルティ (v38 → v39)

2種のフリテン原理を同時に学習させる:
  1. 自捨て牌フリテン: 自分が切った牌に絡む tatsu への高確率をペナルティ化
  2. スルーフリテン:   リーチ後に他家が切ってスルーした牌に絡む tatsu をペナルティ化

根拠: 当たり牌であれば「自己フリテンを避けて切らない」「ロンする」はずなので、
      それらの牌を高確率で待ちとして出力するのは誤り。フリテン立直など意図的な
      ケースはごく稀 (約1-3%) なので、確率を低く誘導する。
テンパイ (shanten=0) サンプルのみに適用。データ再生成不要。
"""

LAMBDA_FURITEN = 0.4

# ---- 学習スクリプトに挿入するコード ----

_MASK_AND_FUNCS = '''
# ---- fix_furiten_penalty: フリテン制約テーブルと損失関数 ----

def _build_furiten_mask():
    """tile t を保有する必要がある tatsu の (34, 113) 対応行列を構築する。"""
    import numpy as _np
    N_PAI = 34
    RYANMEN_BASE = 0; PENCHAN_BASE = 18; KANCHAN_BASE = 24
    TANKI_BASE = 45; SHANPON_BASE = 79
    mat = _np.zeros((N_PAI, 113), dtype=_np.float32)
    # 両面: index = suit*6+(n-1), n in 1-6, holds {suit*9+n, suit*9+n+1}
    for suit in range(3):
        for n in range(1, 7):
            t   = suit * 9 + n
            idx = RYANMEN_BASE + suit * 6 + (n - 1)
            mat[t,     idx] = 1.0
            mat[t + 1, idx] = 1.0
    # 辺張: [1,2]→PENCHAN+suit*2, [8,9]→PENCHAN+suit*2+1  (0-indexed: [0,1] [7,8])
    for suit in range(3):
        lo = PENCHAN_BASE + suit * 2
        mat[suit*9 + 0, lo] = 1.0;  mat[suit*9 + 1, lo] = 1.0
        hi = PENCHAN_BASE + suit * 2 + 1
        mat[suit*9 + 7, hi] = 1.0;  mat[suit*9 + 8, hi] = 1.0
    # 嵌張: index = suit*7+n, n in 0-6, holds {suit*9+n, suit*9+n+2}
    for suit in range(3):
        for n in range(7):
            idx = KANCHAN_BASE + suit * 7 + n
            mat[suit*9 + n,     idx] = 1.0
            mat[suit*9 + n + 2, idx] = 1.0
    # 単騎/双碰: tile t → TANKI_BASE+t / SHANPON_BASE+t
    for t in range(N_PAI):
        mat[t, TANKI_BASE  + t] = 1.0
        mat[t, SHANPON_BASE + t] = 1.0
    return mat

_FURITEN_MASK_NP = _build_furiten_mask()


def compute_furiten_penalties(wait_logits, disc_tok, disc_mask, labels_shanten):
    """
    フリテン系ペナルティ。テンパイサンプルのみ対象。

    wait_logits:    (B, P, 113)
    disc_tok:       (B, P, T, 44)
      dim34=turn_norm  dim35=tsumogiri  dim36=riichi_decl  dim38=self_role
    disc_mask:      (B, P, T)  True=padding
    labels_shanten: (B, P)     0=tenpai
    """
    import torch
    B, P, T, TD = disc_tok.shape
    device = wait_logits.device

    fm     = torch.tensor(_FURITEN_MASK_NP, dtype=torch.float32, device=device)  # (34, 113)
    valid  = (~disc_mask).float()                        # (B, P, T)
    tenpai = (labels_shanten == 0).float().to(device)   # (B, P)

    self_role  = disc_tok[..., 38]  # 1 = target player の自捨て
    turn_norm  = disc_tok[..., 34]  # 捨て牌通し番号 / 70
    riichi_bit = disc_tok[..., 36]  # 1 = リーチ宣言トークン

    # ---- (1) 自捨て牌フリテン ----
    self_tiles = (
        disc_tok[..., :34]
        * self_role.unsqueeze(-1)
        * valid.unsqueeze(-1)
    ).sum(dim=2).clamp(0, 1)                             # (B, P, 34)

    mask_self = (
        torch.matmul(self_tiles, fm).clamp(0, 1)
        * tenpai.unsqueeze(-1)
    )                                                    # (B, P, 113)

    # ---- (2) スルーフリテン: リーチ後他家捨て牌スルー ----
    riichi_self = self_role * riichi_bit * valid         # (B, P, T)
    in_riichi   = (riichi_self.sum(dim=2) > 0).float()  # (B, P)

    # リーチ宣言の turn_norm (未リーチは 999 で "全て対象外")
    riichi_turn = torch.where(
        in_riichi > 0,
        (turn_norm * riichi_self).max(dim=2).values,
        torch.full((B, P), 999.0, device=device),
    )                                                    # (B, P)

    after_riichi = (turn_norm > riichi_turn.unsqueeze(2)).float()  # (B, P, T)
    other_role   = 1.0 - self_role                                 # (B, P, T)

    passed_tiles = (
        disc_tok[..., :34]
        * other_role.unsqueeze(-1)
        * after_riichi.unsqueeze(-1)
        * valid.unsqueeze(-1)
    ).sum(dim=2).clamp(0, 1)                             # (B, P, 34)

    mask_passed = (
        torch.matmul(passed_tiles, fm).clamp(0, 1)
        * tenpai.unsqueeze(-1)
        * in_riichi.unsqueeze(-1)
    )                                                    # (B, P, 113)

    # ---- 合算してペナルティ計算 ----
    furiten_mask = (mask_self + mask_passed).clamp(0, 1)
    denom = furiten_mask.sum()
    if denom < 1.0:
        return torch.tensor(0.0, device=device)

    return (torch.sigmoid(wait_logits) * furiten_mask).sum() / denom
'''

_IMPORT_PATCH = 'import sys\n'
_AFTER_IMPORT = 'import sys\n' + _MASK_AND_FUNCS + '\n'

_LAMBDA_SEARCH  = 'LAMBDA_PUSH        = 0.1\n'
_LAMBDA_REPLACE = f'LAMBDA_PUSH        = 0.1\nLAMBDA_FURITEN     = {LAMBDA_FURITEN}\n'

_LOSS_SEARCH  = '        loss = (loss_nll'
_LOSS_REPLACE = (
    '        furiten_pen = compute_furiten_penalties(\n'
    '            wait_logits, disc_tok, disc_mask, labels_shanten)\n'
    '        loss = (loss_nll'
)

_LOSS_END_SEARCH  = '+ LAMBDA_PUSH     * loss_push)'
_LOSS_END_REPLACE = f'+ LAMBDA_PUSH     * loss_push\n                + LAMBDA_FURITEN  * furiten_pen)'


def patch_train_script(code: str, base_ver: int, target_ver: int) -> str:
    # 1. マスクテーブルとペナルティ関数を挿入
    if '_FURITEN_MASK_NP' not in code:
        code = code.replace(_IMPORT_PATCH, _AFTER_IMPORT, 1)

    # 2. LAMBDA 定数を追加
    if 'LAMBDA_FURITEN' not in code:
        code = code.replace(_LAMBDA_SEARCH, _LAMBDA_REPLACE, 1)

    # 3. loss 計算前にペナルティを計算
    if 'furiten_pen = compute' not in code and _LOSS_SEARCH in code:
        code = code.replace(_LOSS_SEARCH, _LOSS_REPLACE, 1)

    # 4. loss の合算に加算
    if 'LAMBDA_FURITEN  * furiten_pen' not in code and _LOSS_END_SEARCH in code:
        code = code.replace(_LOSS_END_SEARCH, _LOSS_END_REPLACE, 1)

    return code
