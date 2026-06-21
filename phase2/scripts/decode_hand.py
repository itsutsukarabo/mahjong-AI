"""
decode_hand.py

block_logits (134次元) から制約付きビームサーチで有効な聴牌分解を求め、
待ち牌確率を算出する。REINFORCE 学習用のサンプリング関数も提供する。

ブロックインデックス (enumerate_decompositions.py と共通):
  [0..20]    順子 21種  s*7+n → tiles=[s*9+n, s*9+n+1, s*9+n+2]
  [21..54]   刻子 34種  21+i  → tiles=[i, i, i]
  [55..88]   対子 34種  55+i  → tiles=[i, i]
  [89..112]  両面/辺張 24種  89+s*8+n → tiles=[s*9+n, s*9+n+1]
  [113..133] 嵌張 21種  113+s*7+n → tiles=[s*9+n, s*9+n+2]

牌インデックス (0-based):
  m1..m9: 0..8  /  p1..p9: 9..17  /  s1..s9: 18..26  /  z1..z7: 27..33
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ============================================================
# 定数
# ============================================================

N_PAI          = 34
N_BLOCKS       = 134

SHUNTSU_START  = 0    # [0..20]   順子
KOUTSU_START   = 21   # [21..54]  刻子
TOITSU_START   = 55   # [55..88]  対子
RYANMEN_START  = 89   # [89..112] 両面/辺張
KANCHAN_START  = 113  # [113..133] 嵌張

_SUIT_CHAR = 'mmmmmmmmmpppppppppssssssssszzzzzzz'

# ============================================================
# ブロック ↔ 牌 マッピング
# ============================================================

def block_to_tiles(b: int) -> List[int]:
    """
    ブロックインデックス → 構成牌の 0-based インデックスリスト（重複あり）。

    >>> block_to_tiles(0)    # m123
    [0, 1, 2]
    >>> block_to_tiles(21)   # m1刻子
    [0, 0, 0]
    >>> block_to_tiles(55)   # m1対子
    [0, 0]
    >>> block_to_tiles(89)   # m12 両面/辺張
    [0, 1]
    >>> block_to_tiles(113)  # m1_3 嵌張
    [0, 2]
    """
    if b < KOUTSU_START:
        s, n = divmod(b, 7)
        base = s * 9
        return [base+n, base+n+1, base+n+2]
    elif b < TOITSU_START:
        i = b - KOUTSU_START
        return [i, i, i]
    elif b < RYANMEN_START:
        i = b - TOITSU_START
        return [i, i]
    elif b < KANCHAN_START:
        idx = b - RYANMEN_START
        s, n = divmod(idx, 8)
        base = s * 9
        return [base+n, base+n+1]
    else:
        idx = b - KANCHAN_START
        s, n = divmod(idx, 7)
        base = s * 9
        return [base+n, base+n+2]


def block_to_waits(b: int) -> List[int]:
    """
    ターツまたは対子ブロック → 待ち牌の 0-based インデックスリスト。
    面子ブロック (b < 55) は空リストを返す。

    >>> block_to_waits(89)   # m12 辺張 → m3
    [2]
    >>> block_to_waits(90)   # m23 両面 → m1, m4
    [0, 4]
    >>> block_to_waits(96)   # m89 辺張 → m7
    [6]
    >>> block_to_waits(113)  # m1_3 嵌張 → m2
    [1]
    >>> block_to_waits(55)   # m1 対子 → m1 (単騎)
    [0]
    """
    if b < TOITSU_START:
        return []
    elif b < RYANMEN_START:
        return [b - TOITSU_START]
    elif b < KANCHAN_START:
        idx = b - RYANMEN_START
        s, n = divmod(idx, 8)
        base = s * 9
        if n == 0:
            return [base + 2]
        elif n == 7:
            return [base + 6]
        else:
            return [base+n-1, base+n+2]
    else:
        idx = b - KANCHAN_START
        s, n = divmod(idx, 7)
        return [s*9 + n + 1]


def tile_to_name(t: int) -> str:
    """牌インデックス → 文字列 (例: 0→'m1', 27→'z1')"""
    if t < 27:
        return _SUIT_CHAR[t] + str(t % 9 + 1)
    return 'z' + str(t - 26)


def block_to_name(b: int) -> str:
    """ブロックインデックス → 文字列 (例: 0→'m123', 89→'m12')"""
    seen = []
    for t in block_to_tiles(b):
        s = tile_to_name(t)
        if s not in seen:
            seen.append(s)
    return ''.join(seen)


# ============================================================
# デコーダー状態
# ============================================================

@dataclass
class DecoderState:
    """
    ビームサーチ中の1状態。

    聴牌形の種類:
      標準形  : mentsu×3 + head対子 + tatsu(両面/辺張/嵌張)
      単騎形  : mentsu×4 + tanki 1牌 (head=-1, tatsu=-1)
      双碰形  : mentsu×3 + head対子 + tatsu対子 (shanpon=True)

    オープンメルドがある場合は total_tiles が 10/7/4/1 になる。
    """
    remaining:   np.ndarray  # (34,) int8  残り牌枚数
    mentsu:      List[int]   # 選択済み面子ブロックインデックスリスト
    head:        int         # 雀頭ブロックインデックス (-1=未確定)
    tatsu:       int         # ターツブロックインデックス (-1=未確定)
    shanpon:     bool        # True → tatsu が2つ目の対子ブロック (双碰)
    tanki_tile:  int         # 単騎牌インデックス (-1=なし)
    log_score:   float       # Σ log σ(block_logit) for all chosen blocks

    @classmethod
    def initial(cls, counts: np.ndarray) -> 'DecoderState':
        return cls(
            remaining=counts.copy().astype(np.int8),
            mentsu=[],
            head=-1, tatsu=-1,
            shanpon=False, tanki_tile=-1,
            log_score=0.0,
        )

    def copy(self) -> 'DecoderState':
        return DecoderState(
            remaining=self.remaining.copy(),
            mentsu=list(self.mentsu),
            head=self.head, tatsu=self.tatsu,
            shanpon=self.shanpon, tanki_tile=self.tanki_tile,
            log_score=self.log_score,
        )

    @property
    def n_tiles(self) -> int:
        return int(self.remaining.sum())

    def is_complete(self) -> bool:
        """残り牌ゼロかつ有効な聴牌構造ならば True"""
        if self.n_tiles != 0:
            return False
        if self.tanki_tile >= 0:
            return True
        return self.head >= 0 and self.tatsu >= 0

    def get_wait_tiles(self) -> List[int]:
        """待ち牌の 0-based インデックスリスト"""
        if self.tanki_tile >= 0:
            return [self.tanki_tile]
        if self.tatsu < 0:
            return []
        if self.shanpon:
            return [self.head - TOITSU_START, self.tatsu - TOITSU_START]
        return block_to_waits(self.tatsu)

    def __repr__(self) -> str:
        mentsu_s = ' '.join(block_to_name(m) for m in self.mentsu)
        head_s   = block_to_name(self.head) if self.head >= 0 else '-'
        if self.tanki_tile >= 0:
            tatsu_s = f'tanki:{tile_to_name(self.tanki_tile)}'
        elif self.tatsu >= 0:
            tatsu_s = block_to_name(self.tatsu)
            if self.shanpon:
                tatsu_s += '(双碰)'
        else:
            tatsu_s = '-'
        waits = [tile_to_name(t) for t in self.get_wait_tiles()]
        return (f'[{mentsu_s}] [{head_s}] [{tatsu_s}]'
                f' waits={waits} score={self.log_score:.3f}')


# ============================================================
# ビームサーチ
# ============================================================

def _can_subtract(remaining: np.ndarray, tiles: List[int]) -> bool:
    tmp = remaining.copy()
    for t in tiles:
        tmp[t] -= 1
        if tmp[t] < 0:
            return False
    return True


def _subtract(remaining: np.ndarray, tiles: List[int]) -> np.ndarray:
    new = remaining.copy()
    for t in tiles:
        new[t] -= 1
    return new


def _expand_state(
    state:      DecoderState,
    log_probs:  np.ndarray,      # (134,) log σ(block_logits)
    candidates: np.ndarray,      # 閾値を超えたブロックインデックス
) -> List[DecoderState]:
    """
    1ステップ展開: state から到達可能な次状態を列挙する。

    head 未確定時の遷移:
      R == 1 → 単騎終端
      R == 4 → 面子追加(→単騎パス) と 雀頭選択(→標準/双碰パス) の両方
      R >  4 → 面子追加のみ

    head 確定・tatsu 未確定時:
      R == 2 → ターツ選択 or 双碰検出
    """
    nexts: List[DecoderState] = []
    R = state.n_tiles

    if state.head < 0:
        # 単騎終端 (R == 1)
        if R == 1:
            for t in np.where(state.remaining == 1)[0]:
                ns = state.copy()
                ns.tanki_tile = int(t)
                ns.remaining[t] = 0
                ns.log_score += log_probs[TOITSU_START + t]
                nexts.append(ns)
            return nexts

        # 面子追加 (R >= 4: 単騎パスと標準パスの共通ステップ)
        if R >= 4:
            for b in candidates:
                if b >= TOITSU_START:
                    continue
                tiles = block_to_tiles(b)
                if not _can_subtract(state.remaining, tiles):
                    continue
                ns = state.copy()
                ns.remaining = _subtract(ns.remaining, tiles)
                ns.mentsu.append(b)
                ns.log_score += log_probs[b]
                nexts.append(ns)

        # 雀頭選択 (R == 4 のみ: 標準/双碰パスへ分岐)
        if R == 4:
            for b in candidates:
                if b < TOITSU_START or b >= RYANMEN_START:
                    continue
                tiles = block_to_tiles(b)
                if not _can_subtract(state.remaining, tiles):
                    continue
                ns = state.copy()
                ns.remaining = _subtract(ns.remaining, tiles)
                ns.head = b
                ns.log_score += log_probs[b]
                nexts.append(ns)

        return nexts

    # ---- ターツフェーズ (head 確定、tatsu 未確定、R == 2) ----
    if state.tatsu < 0 and R == 2:
        # 双碰: 残り2牌が同種
        for t in range(N_PAI):
            if state.remaining[t] == 2:
                ns = state.copy()
                ns.tatsu    = TOITSU_START + t
                ns.shanpon  = True
                ns.remaining[t] = 0
                ns.log_score += log_probs[TOITSU_START + t]
                nexts.append(ns)
                return nexts  # 双碰は1パターン

        # 通常ターツ (両面/辺張/嵌張)
        for b in candidates:
            if b < RYANMEN_START:
                continue
            tiles = block_to_tiles(b)
            if not _can_subtract(state.remaining, tiles):
                continue
            ns = state.copy()
            ns.remaining = _subtract(ns.remaining, tiles)
            ns.tatsu = b
            ns.log_score += log_probs[b]
            nexts.append(ns)

    return nexts


def beam_search_decoder(
    block_logits:   np.ndarray,
    initial_counts: np.ndarray,
    beam_width:     int   = 20,
    prob_threshold: float = 0.03,
) -> List[DecoderState]:
    """
    block_logits に対して制約付きビームサーチを実行し、
    有効な聴牌分解を log_score 降順で返す。

    Args:
        block_logits   : (134,) float — block_head の raw logit (sigmoid 前)
        initial_counts : (34,) int   — 各牌の推定枚数 (sum = 13 - 3*melds)
        beam_width     : ビーム幅
        prob_threshold : この確率以上のブロックのみ候補に含める

    Returns:
        完成した DecoderState のリスト (最大 beam_width 件, log_score 降順)
        聴牌形が見つからない場合は空リスト
    """
    total = int(initial_counts.sum())
    # 有効な手牌枚数チェック (13 - 3*melds: total ≡ 1 mod 3)
    if total % 3 != 1:
        return []

    probs     = 1.0 / (1.0 + np.exp(-block_logits.astype(np.float64)))
    log_probs = np.log(probs + 1e-9)
    candidates = np.where(probs > prob_threshold)[0]

    initial   = DecoderState.initial(initial_counts)
    beam:      List[DecoderState] = [initial]
    completed: List[DecoderState] = []

    # 最大ステップ: 面子(最大4) + 雀頭1 + ターツ1 = 6、余裕込みで8
    for _ in range(8):
        if not beam:
            break
        next_beam: List[DecoderState] = []
        for state in beam:
            for ns in _expand_state(state, log_probs, candidates):
                if ns.is_complete():
                    completed.append(ns)
                else:
                    next_beam.append(ns)
        next_beam.sort(key=lambda s: -s.log_score)
        beam = next_beam[:beam_width]

    completed.sort(key=lambda s: -s.log_score)
    return completed[:beam_width]


# ============================================================
# 待ち牌確率の集計
# ============================================================

def compute_wait_probs(
    beam_results: List[DecoderState],
    min_prob:     float = 0.01,
) -> Dict[int, float]:
    """
    ビームサーチ結果からタイル別待ち牌確率を集計する。

    ビーム内の各状態を softmax で重みづけし、
    wait_probs[t] = Σ_i weight_i × 1[t ∈ waits_i]

    Args:
        beam_results : beam_search_decoder の戻り値
        min_prob     : この確率以下の牌は結果から除外

    Returns:
        {tile_idx: probability} の dict
    """
    if not beam_results:
        return {}

    scores = np.array([s.log_score for s in beam_results], dtype=np.float64)
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()

    wait_probs = np.zeros(N_PAI)
    for state, w in zip(beam_results, weights):
        for t in state.get_wait_tiles():
            wait_probs[t] += w

    return {int(i): float(wait_probs[i])
            for i in np.where(wait_probs >= min_prob)[0]}


# ============================================================
# フリーモードビームサーチ (initial_counts 不要)
# ============================================================

@dataclass
class FreeDecoderState:
    """
    フリーモードビームサーチの状態。
    initial_counts に依存せず、tile_used で使用枚数を管理する。
    """
    tile_used:  np.ndarray  # (34,) 各牌の使用枚数
    n_used:     int         # 選択済み合計枚数
    mentsu:     List[int]
    head:       int
    tatsu:      int
    shanpon:    bool
    tanki_tile: int
    log_score:  float

    @property
    def is_complete(self) -> bool:
        if self.tanki_tile >= 0:
            return True
        return self.head >= 0 and self.tatsu >= 0

    def get_wait_tiles(self) -> List[int]:
        if self.tanki_tile >= 0:
            return [self.tanki_tile]
        if self.tatsu < 0:
            return []
        if self.shanpon:
            return [self.head - TOITSU_START, self.tatsu - TOITSU_START]
        return block_to_waits(self.tatsu)


def _can_add(tile_used: np.ndarray, tiles: List[int]) -> bool:
    for t in tiles:
        if tile_used[t] >= 4:
            return False
    return True


def _add(tile_used: np.ndarray, tiles: List[int]) -> np.ndarray:
    r = tile_used.copy()
    for t in tiles:
        r[t] += 1
    return r


def _expand_free(state: FreeDecoderState, log_probs: np.ndarray, k_tiles: int) -> List[FreeDecoderState]:
    nexts: List[FreeDecoderState] = []
    R = k_tiles - state.n_used

    if state.head < 0:
        if R == 1:
            for t in range(N_PAI):
                if state.tile_used[t] >= 4:
                    continue
                nu = state.tile_used.copy(); nu[t] += 1
                nexts.append(FreeDecoderState(
                    tile_used=nu, n_used=state.n_used + 1,
                    mentsu=list(state.mentsu),
                    head=-1, tatsu=-1, shanpon=False, tanki_tile=t,
                    log_score=state.log_score + log_probs[TOITSU_START + t],
                ))
            return nexts

        if R >= 4:
            for b in range(TOITSU_START):
                tiles = block_to_tiles(b)
                if not _can_add(state.tile_used, tiles):
                    continue
                nexts.append(FreeDecoderState(
                    tile_used=_add(state.tile_used, tiles), n_used=state.n_used + 3,
                    mentsu=state.mentsu + [b],
                    head=-1, tatsu=-1, shanpon=False, tanki_tile=-1,
                    log_score=state.log_score + log_probs[b],
                ))

        if R == 4:
            for b in range(TOITSU_START, RYANMEN_START):
                tiles = block_to_tiles(b)
                if not _can_add(state.tile_used, tiles):
                    continue
                nexts.append(FreeDecoderState(
                    tile_used=_add(state.tile_used, tiles), n_used=state.n_used + 2,
                    mentsu=list(state.mentsu),
                    head=b, tatsu=-1, shanpon=False, tanki_tile=-1,
                    log_score=state.log_score + log_probs[b],
                ))
        return nexts

    if state.tatsu < 0 and R == 2:
        head_tile = state.head - TOITSU_START
        for t in range(N_PAI):
            if t == head_tile:
                continue
            if not _can_add(state.tile_used, [t, t]):
                continue
            nexts.append(FreeDecoderState(
                tile_used=_add(state.tile_used, [t, t]), n_used=state.n_used + 2,
                mentsu=list(state.mentsu),
                head=state.head, tatsu=TOITSU_START + t, shanpon=True, tanki_tile=-1,
                log_score=state.log_score + log_probs[TOITSU_START + t],
            ))
        for b in range(RYANMEN_START, N_BLOCKS):
            tiles = block_to_tiles(b)
            if not _can_add(state.tile_used, tiles):
                continue
            nexts.append(FreeDecoderState(
                tile_used=_add(state.tile_used, tiles), n_used=state.n_used + 2,
                mentsu=list(state.mentsu),
                head=state.head, tatsu=b, shanpon=False, tanki_tile=-1,
                log_score=state.log_score + log_probs[b],
            ))
    return nexts


def beam_search_decoder_free(
    block_logits: np.ndarray,
    k_tiles:      int,
    beam_width:   int = 50,
) -> List[FreeDecoderState]:
    """
    initial_counts 不要のフリーモードビームサーチ。
    tile_used[t] <= 4 制約のみで全ブロック組み合わせを探索する。

    Args:
        block_logits : (134,) float — block_head の raw logit
        k_tiles      : 手牌枚数 (13 - 3*n_melds)
        beam_width   : ビーム幅

    Returns:
        完成した FreeDecoderState のリスト (最大 beam_width 件, log_score 降順)
    """
    probs     = 1.0 / (1.0 + np.exp(-block_logits.astype(np.float64)))
    log_probs = np.log(probs + 1e-9)

    initial = FreeDecoderState(
        tile_used=np.zeros(N_PAI, dtype=np.int8), n_used=0,
        mentsu=[], head=-1, tatsu=-1, shanpon=False, tanki_tile=-1,
        log_score=0.0,
    )
    beam: List[FreeDecoderState] = [initial]
    completed: List[FreeDecoderState] = []

    for _ in range(8):
        if not beam:
            break
        next_beam: List[FreeDecoderState] = []
        for state in beam:
            for ns in _expand_free(state, log_probs, k_tiles):
                if ns.is_complete:
                    completed.append(ns)
                else:
                    next_beam.append(ns)
        next_beam.sort(key=lambda s: -s.log_score)
        beam = next_beam[:beam_width]

    completed.sort(key=lambda s: -s.log_score)
    return completed[:beam_width]


def compute_wait_probs_free(
    beam_results: List[FreeDecoderState],
    min_prob:     float = 0.01,
) -> Dict[int, float]:
    """beam_search_decoder_free の結果から待ち牌確率を集計する。"""
    if not beam_results:
        return {}
    scores = np.array([s.log_score for s in beam_results], dtype=np.float64)
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    wait_probs = np.zeros(N_PAI)
    for state, w in zip(beam_results, weights):
        for t in state.get_wait_tiles():
            wait_probs[t] += w
    return {int(i): float(wait_probs[i])
            for i in np.where(wait_probs >= min_prob)[0]}


# ============================================================
# REINFORCE 用サンプリングデコーダー
# ============================================================

Trajectory = List[Tuple[np.ndarray, int]]
"""trajectory の各要素: (valid_block_indices, chosen_block_index)"""


def sample_trajectory(
    block_logits:   np.ndarray,
    initial_counts: np.ndarray,
    temperature:    float = 1.0,
    prob_threshold: float = 0.03,
    rng:            Optional[np.random.Generator] = None,
) -> Tuple[Optional[DecoderState], Trajectory]:
    """
    確率的サンプリングで聴牌分解を1軌跡生成する (REINFORCE 学習用)。

    ビームサーチと同じ展開ロジックを使うが、
    各ステップで次状態を確率的にサンプリングする。

    Args:
        block_logits   : (134,) float — block_head の raw logit
        initial_counts : (34,) int
        temperature    : サンプリング温度 (高いほどランダム)
        prob_threshold : 候補ブロックの最低確率
        rng            : 乱数生成器 (None なら np.random.default_rng() を使用)

    Returns:
        (terminal_state, trajectory)
          terminal_state : 聴牌形に達した DecoderState (失敗時は None)
          trajectory     : [(valid_indices, chosen_idx), ...] — 各ステップの選択
    """
    if rng is None:
        rng = np.random.default_rng()

    total = int(initial_counts.sum())
    if total % 3 != 1:
        return None, []

    probs     = 1.0 / (1.0 + np.exp(-block_logits.astype(np.float64)))
    log_probs = np.log(probs + 1e-9)
    candidates = np.where(probs > prob_threshold)[0]

    state = DecoderState.initial(initial_counts)
    trajectory: Trajectory = []

    for _ in range(8):
        nexts = _expand_state(state, log_probs, candidates)
        if not nexts:
            return None, trajectory

        # 各次状態のスコア差分 = 追加したブロックの log_score 分
        delta_scores = np.array([ns.log_score - state.log_score for ns in nexts],
                                dtype=np.float64)
        # 温度スケーリング後に softmax でサンプリング確率を計算
        delta_scores /= temperature
        delta_scores -= delta_scores.max()
        weights = np.exp(delta_scores)
        weights /= weights.sum()

        chosen_idx = int(rng.choice(len(nexts), p=weights))
        chosen_ns  = nexts[chosen_idx]

        # 軌跡記録: このステップで選ばれたブロック
        # (単騎の場合は tanki_tile に対応する対子ブロックを記録)
        if chosen_ns.tanki_tile >= 0 and state.tanki_tile < 0:
            chosen_block = TOITSU_START + chosen_ns.tanki_tile
        elif chosen_ns.tatsu >= 0 and state.tatsu < 0:
            chosen_block = chosen_ns.tatsu
        elif chosen_ns.head >= 0 and state.head < 0:
            chosen_block = chosen_ns.head
        elif len(chosen_ns.mentsu) > len(state.mentsu):
            chosen_block = chosen_ns.mentsu[-1]
        else:
            chosen_block = -1

        if chosen_block >= 0:
            # valid_blocks: このステップで候補だったブロックのインデックス
            valid_blocks = np.array([
                (TOITSU_START + ns.tanki_tile if ns.tanki_tile >= 0 and state.tanki_tile < 0
                 else ns.tatsu if ns.tatsu >= 0 and state.tatsu < 0
                 else ns.head if ns.head >= 0 and state.head < 0
                 else ns.mentsu[-1] if len(ns.mentsu) > len(state.mentsu)
                 else -1)
                for ns in nexts
            ], dtype=np.int64)
            valid_blocks = valid_blocks[valid_blocks >= 0]
            valid_blocks = np.unique(valid_blocks)
            trajectory.append((valid_blocks, chosen_block))

        state = chosen_ns
        if state.is_complete():
            return state, trajectory

    return None, trajectory


# ============================================================
# テスト
# ============================================================

def _make_counts(tiles: List[int]) -> np.ndarray:
    """牌インデックスリスト → counts ベクトル"""
    c = np.zeros(N_PAI, dtype=np.int8)
    for t in tiles:
        c[t] += 1
    return c


def _logits_from_tiles(tiles: List[int], base: float = -3.0, boost: float = 5.0) -> np.ndarray:
    """
    指定した牌を構成するブロックのみ高いスコアを持つ logits を生成する。
    テスト用の簡易ヘルパー。
    """
    logits = np.full(N_BLOCKS, base)
    tile_set = list(tiles)
    for b in range(N_BLOCKS):
        btiles = block_to_tiles(b)
        counts_needed: Dict[int, int] = {}
        for t in btiles:
            counts_needed[t] = counts_needed.get(t, 0) + 1
        tile_counts: Dict[int, int] = {}
        for t in tile_set:
            tile_counts[t] = tile_counts.get(t, 0) + 1
        if all(tile_counts.get(t, 0) >= v for t, v in counts_needed.items()):
            logits[b] = boost
    return logits


def _run_tests():
    print("=== decode_hand.py テスト ===\n")

    # --- block_to_tiles / block_to_waits 単体テスト ---
    assert block_to_tiles(0)   == [0, 1, 2],   f"m123: {block_to_tiles(0)}"
    assert block_to_tiles(7)   == [9, 10, 11], f"p123: {block_to_tiles(7)}"
    assert block_to_tiles(21)  == [0, 0, 0],   f"m1刻: {block_to_tiles(21)}"
    assert block_to_tiles(55)  == [0, 0],       f"m1対: {block_to_tiles(55)}"
    assert block_to_tiles(89)  == [0, 1],       f"m12両: {block_to_tiles(89)}"
    assert block_to_tiles(97)  == [9, 10],      f"p12両: {block_to_tiles(97)}"
    assert block_to_tiles(113) == [0, 2],       f"m1_3嵌: {block_to_tiles(113)}"

    assert block_to_waits(89)  == [2],       f"m12辺張→m3: {block_to_waits(89)}"
    assert block_to_waits(90)  == [0, 3],    f"m23両面→m1/m4: {block_to_waits(90)}"
    assert block_to_waits(96)  == [6],       f"m89辺張→m7: {block_to_waits(96)}"
    assert block_to_waits(113) == [1],       f"m1_3嵌張→m2: {block_to_waits(113)}"
    assert block_to_waits(55)  == [0],       f"m1対子→m1単騎: {block_to_waits(55)}"
    assert block_to_waits(0)   == [],        f"m123順子→待ちなし: {block_to_waits(0)}"
    print("OK: block_to_tiles / block_to_waits")

    # --- ビームサーチ テスト1: 標準両面待ち ---
    # 手牌: m123 p456 s789 z1z1 + m23 (ryanmen → m1/m4待ち)
    # 0,1,2=m123  12,13,14=p456  20,21,22=s789  27=z1  1,2=m23(待ち牌 m1=0, m4=3)
    # ※ 0-based: m1=0,m2=1,m3=2,m4=3; p4=12,p5=13,p6=14; s7=24...
    # s7=18+6=24, s8=25, s9=26; z1=27
    tiles_hand = [0,1,2, 12,13,14, 24,25,26, 27,27, 1,2]
    counts1 = _make_counts(tiles_hand)
    logits1 = _logits_from_tiles(tiles_hand)
    beam1 = beam_search_decoder(logits1, counts1, beam_width=10)
    assert len(beam1) > 0, "テスト1: 聴牌形が見つからない"
    waits1 = compute_wait_probs(beam1)
    wait_tiles1 = set(waits1.keys())
    # m23両面 → m1(0) or m4(3)
    assert 0 in wait_tiles1 or 3 in wait_tiles1, f"テスト1: 待ち牌が不正 {wait_tiles1}"
    print(f"OK: テスト1 (両面待ち) waits={{{', '.join(tile_to_name(t) for t in sorted(wait_tiles1))}}}")
    print(f"  top3: {beam1[:3]}")

    # --- テスト2: 単騎待ち ---
    # 手牌: m123 m456 p789 s123 + z1 (tanki → z1待ち)
    tiles_hand2 = [0,1,2, 3,4,5, 15,16,17, 18,19,20, 27]
    counts2 = _make_counts(tiles_hand2)
    logits2 = _logits_from_tiles(tiles_hand2)
    beam2 = beam_search_decoder(logits2, counts2, beam_width=10)
    assert len(beam2) > 0, "テスト2: 聴牌形が見つからない"
    waits2 = compute_wait_probs(beam2)
    assert 27 in waits2, f"テスト2: z1待ちが検出されない {waits2}"
    print(f"OK: テスト2 (単騎待ち) waits={{{', '.join(tile_to_name(t) for t in sorted(waits2))}}}")

    # --- テスト3: 双碰待ち ---
    # 手牌: m123 p456 s789 + m1m1 + p2p2 (shanpon → m1/p2待ち)
    tiles_hand3 = [0,1,2, 12,13,14, 24,25,26, 0,0, 9,9]
    counts3 = _make_counts(tiles_hand3)
    logits3 = _logits_from_tiles(tiles_hand3)
    beam3 = beam_search_decoder(logits3, counts3, beam_width=10)
    assert len(beam3) > 0, "テスト3: 聴牌形が見つからない"
    waits3 = compute_wait_probs(beam3)
    # m1=0, p2=10
    assert 0 in waits3 or 10 in waits3, f"テスト3: 双碰待ちが検出されない {waits3}"
    print(f"OK: テスト3 (双碰待ち) waits={{{', '.join(tile_to_name(t) for t in sorted(waits3))}}}")

    # --- テスト4: 嵌張待ち ---
    # 手牌: m123 p456 s789 z1z1 + m13 (kanchan → m2待ち)
    tiles_hand4 = [0,1,2, 12,13,14, 24,25,26, 27,27, 0,2]
    counts4 = _make_counts(tiles_hand4)
    logits4 = _logits_from_tiles(tiles_hand4)
    beam4 = beam_search_decoder(logits4, counts4, beam_width=10)
    assert len(beam4) > 0, "テスト4: 聴牌形が見つからない"
    waits4 = compute_wait_probs(beam4)
    assert 1 in waits4, f"テスト4: m2待ちが検出されない {waits4}"
    print(f"OK: テスト4 (嵌張待ち) waits={{{', '.join(tile_to_name(t) for t in sorted(waits4))}}}")

    # --- テスト5: sample_trajectory ---
    state5, traj5 = sample_trajectory(logits1, counts1)
    assert state5 is not None, "テスト5: サンプリング失敗"
    assert state5.is_complete(), "テスト5: 終端状態でない"
    assert len(traj5) > 0, "テスト5: 軌跡が空"
    print(f"OK: テスト5 (sample_trajectory) len(traj)={len(traj5)} waits={[tile_to_name(t) for t in state5.get_wait_tiles()]}")

    # --- テスト6: 1副露ありの聴牌 (10牌) ---
    # 副露1つ(m123) → 手牌10枚: p456 s789 z1z1 + m23
    tiles_hand6 = [12,13,14, 24,25,26, 27,27, 1,2]
    counts6 = _make_counts(tiles_hand6)
    logits6 = _logits_from_tiles(tiles_hand6)
    beam6 = beam_search_decoder(logits6, counts6, beam_width=10)
    assert len(beam6) > 0, "テスト6: 1副露聴牌が見つからない"
    waits6 = compute_wait_probs(beam6)
    assert 0 in waits6 or 3 in waits6, f"テスト6: 待ち牌が不正 {waits6}"
    print(f"OK: テスト6 (1副露) waits={{{', '.join(tile_to_name(t) for t in sorted(waits6))}}}")

    print("\n全テスト通過")


if __name__ == '__main__':
    _run_tests()
