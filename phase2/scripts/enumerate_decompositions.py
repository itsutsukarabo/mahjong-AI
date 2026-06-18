"""
手牌分解の列挙器 (v20 ブロックラベル生成用)

89 ブロック種:
  [0..20]   順子 21種: m123..m789, p123..p789, s123..s789
  [21..54]  刻子 34種: m1..z7 (インデックス順)
  [55..88]  対子 34種: m1..z7 (インデックス順) — 雀頭役 + 浮き対子 を統一してラベル化

牌インデックス (0-indexed):
  m1..m9: 0..8
  p1..p9: 9..17
  s1..s9: 18..26
  z1..z7: 27..33
"""

from typing import Generator, List, Tuple
import numpy as np

# ---- ブロック定義 ----

SUIT_OFFSETS = [0, 9, 18]  # m, p, s の牌インデックス開始

def _block_defs():
    """89 ブロック定義を返す: [(block_name, [tile_indices])]"""
    blocks = []
    # 順子 21種 (m/p/s × 7)
    for offset in SUIT_OFFSETS:
        for start in range(7):
            tiles = [offset + start, offset + start + 1, offset + start + 2]
            suit = "mps"[SUIT_OFFSETS.index(offset)]
            blocks.append((f"{suit}{start+1}{start+2}{start+3}", tiles))
    # 刻子 34種
    for i in range(34):
        suit = "mpsz"
        if i < 9:   name = f"m{i+1}{i+1}{i+1}"
        elif i < 18: name = f"p{i-8}{i-8}{i-8}"
        elif i < 27: name = f"s{i-17}{i-17}{i-17}"
        else:        name = f"z{i-26}{i-26}{i-26}"
        blocks.append((name, [i, i, i]))
    # 雀頭 34種
    for i in range(34):
        if i < 9:   name = f"m{i+1}{i+1}"
        elif i < 18: name = f"p{i-8}{i-8}"
        elif i < 27: name = f"s{i-17}{i-17}"
        else:        name = f"z{i-26}{i-26}"
        blocks.append((name, [i, i]))
    return blocks

def _block_defs_with_tatsu():
    """134 ブロック定義: 89ブロック + リャンメン/ペンチャン(24) + カンチャン(21)"""
    blocks = list(_block_defs())
    # リャンメン/ペンチャン 24種 (m/p/s × 8)
    for s_idx, offset in enumerate(SUIT_OFFSETS):
        suit = "mps"[s_idx]
        for start in range(8):
            tiles = [offset + start, offset + start + 1]
            blocks.append((f"{suit}{start+1}{start+2}", tiles))
    # カンチャン 21種 (m/p/s × 7)
    for s_idx, offset in enumerate(SUIT_OFFSETS):
        suit = "mps"[s_idx]
        for start in range(7):
            tiles = [offset + start, offset + start + 2]
            blocks.append((f"{suit}{start+1}{start+3}", tiles))
    return blocks

BLOCK_DEFS = _block_defs()
N_BLOCKS = 89
N_BLOCKS_WITH_TATSU = 134

# ブロックインデックス定数
SHUNTSU_START  = 0    # [0..20]
KOUTSU_START   = 21   # [21..54]
TOITSU_START   = 55   # [55..88]
JANTOU_START   = TOITSU_START  # 後方互換エイリアス
RYANMEN_START  = 89   # [89..112]  リャンメン/ペンチャン (tatsu のみ)
KANCHAN_START  = 113  # [113..133] カンチャン (tatsu のみ)


def build_block_selection_matrix(include_tatsu: bool = False) -> np.ndarray:
    """(N_BLOCKS, 34) バイナリ行列: mat[b, i] = 1 ならブロック b に牌 i が含まれる
    include_tatsu=True で (134, 34)、False で (89, 34)"""
    block_defs = _block_defs_with_tatsu() if include_tatsu else BLOCK_DEFS
    n_b = N_BLOCKS_WITH_TATSU if include_tatsu else N_BLOCKS
    mat = np.zeros((n_b, 34), dtype=np.float32)
    for b_idx, (_, tiles) in enumerate(block_defs):
        for t in set(tiles):
            mat[b_idx, t] = 1
    return mat


# ---- シャンテン数計算 ----

def _xiangting_score(m, d, g, has_jantou):
    n = 4 if has_jantou else 5
    if m > 4:         d += m - 4;     m = 4
    if m + d > 4:     g += m + d - 4; d = 4 - m
    if m + d + g > n: g = n - m - d
    if has_jantou: d += 1
    return 13 - m * 3 - d * 2 - g


def _shanten_suit(counts: List[int]) -> Tuple[List[int], List[int]]:
    """
    数牌1スーツの面子/搭子をDPで最適化。
    returns: (best_a, best_b)
      best_a: [面子数, 搭子数, 孤立牌数] for min-isolated strategy
      best_b: [面子数, 搭子数, 孤立牌数] for max-mentsu strategy
    """
    counts = list(counts)

    def recurse(n, c):
        if n > 9:
            # 残り牌から搭子/孤立牌をカウント
            nd, ng = 0, 0
            i = 1
            while i <= 9:
                if c[i] == 0:
                    i += 1
                    continue
                if i <= 7 and c[i + 1] == 0 and (i > 7 or c[i + 2] == 0):
                    nd += c[i] // 2
                    ng += c[i] % 2
                    i += 1
                    continue
                # 連続グループ処理
                grp = c[i]
                grp_sum = grp
                j = i + 1
                while j <= 9 and (j - i) <= 2 and c[j] > 0:
                    grp_sum += c[j]
                    j += 1
                nd += grp_sum // 2
                ng += grp_sum % 2
                i = j
            # simplified dazi
            nd2, ng2 = 0, 0
            pair_sum = sum(c[1:])
            nd2 = pair_sum // 2
            ng2 = pair_sum % 2
            # use the simpler approach from xiangting.js dazi()
            nd3, ng3 = 0, 0
            p = 0
            for ni in range(1, 10):
                p += c[ni]
                if ni <= 7 and c[ni+1] == 0 and c[ni+2] == 0:
                    nd3 += p // 2
                    ng3 += p % 2
                    p = 0
            nd3 += p // 2
            ng3 += p % 2
            return ([0, nd3, ng3], [0, nd3, ng3])

        best_a, best_b = recurse(n + 1, c)

        # 順子
        if n <= 7 and c[n] > 0 and c[n+1] > 0 and c[n+2] > 0:
            c[n] -= 1; c[n+1] -= 1; c[n+2] -= 1
            ra, rb = recurse(n, c)
            c[n] += 1; c[n+1] += 1; c[n+2] += 1
            ra2 = [ra[0]+1, ra[1], ra[2]]
            rb2 = [rb[0]+1, rb[1], rb[2]]
            if ra2[2] < best_a[2] or (ra2[2] == best_a[2] and ra2[1] < best_a[1]):
                best_a = ra2
            if rb2[0] > best_b[0] or (rb2[0] == best_b[0] and rb2[1] > best_b[1]):
                best_b = rb2

        # 刻子
        if c[n] >= 3:
            c[n] -= 3
            ra, rb = recurse(n + 1, c)
            c[n] += 3
            ra2 = [ra[0]+1, ra[1], ra[2]]
            rb2 = [rb[0]+1, rb[1], rb[2]]
            if ra2[2] < best_a[2] or (ra2[2] == best_a[2] and ra2[1] < best_a[1]):
                best_a = ra2
            if rb2[0] > best_b[0] or (rb2[0] == best_b[0] and rb2[1] > best_b[1]):
                best_b = rb2

        return best_a, best_b

    c_arr = [0] + list(counts[:9]) + [0, 0]  # 1-indexed, padding
    return recurse(1, c_arr)


def compute_shanten_yiban(counts34: List[int], n_meld: int = 0) -> int:
    """通常手のシャンテン数を計算 (副露数 n_meld を加味)"""
    # 各スーツの面子/搭子
    suits_ab = []
    for s in range(3):
        a, b = _shanten_suit(counts34[s*9:(s+1)*9])
        suits_ab.append((a, b))

    # 字牌
    z_counts = counts34[27:34]
    z = [0, 0, 0]
    for v in z_counts:
        if v >= 3: z[0] += 1
        elif v == 2: z[1] += 1
        elif v == 1: z[2] += 1

    min_sh = 13
    for ma, mb in [suits_ab[0]]:
        for pa, pb in [suits_ab[1]]:
            for sa, sb in [suits_ab[2]]:
                for m_vec in [[ma[0],ma[1],ma[2]], [mb[0],mb[1],mb[2]]]:
                    for p_vec in [[pa[0],pa[1],pa[2]], [pb[0],pb[1],pb[2]]]:
                        for s_vec in [[sa[0],sa[1],sa[2]], [sb[0],sb[1],sb[2]]]:
                            x = [n_meld, 0, 0]
                            for i in range(3):
                                x[i] += m_vec[i] + p_vec[i] + s_vec[i] + z[i]
                            sh = _xiangting_score(x[0], x[1], x[2], False)
                            if sh < min_sh:
                                min_sh = sh

    # 雀頭ありパターン
    for s_idx in range(4):
        r = range(9) if s_idx < 3 else range(7)
        for n in r:
            tile = s_idx * 9 + n if s_idx < 3 else 27 + n
            if counts34[tile] >= 2:
                c2 = list(counts34)
                c2[tile] -= 2
                suits_ab2 = []
                for s in range(3):
                    a, b = _shanten_suit(c2[s*9:(s+1)*9])
                    suits_ab2.append((a, b))
                z2 = [0, 0, 0]
                for v in c2[27:34]:
                    if v >= 3: z2[0] += 1
                    elif v == 2: z2[1] += 1
                    elif v == 1: z2[2] += 1
                for m_vec2 in [suits_ab2[0][0], suits_ab2[0][1]]:
                    for p_vec2 in [suits_ab2[1][0], suits_ab2[1][1]]:
                        for s_vec2 in [suits_ab2[2][0], suits_ab2[2][1]]:
                            x = [n_meld, 0, 0]
                            for i in range(3):
                                x[i] += m_vec2[i] + p_vec2[i] + s_vec2[i] + z2[i]
                            sh = _xiangting_score(x[0], x[1], x[2], True)
                            if sh < min_sh:
                                min_sh = sh
    return min_sh


def compute_shanten_qidui(counts34: List[int]) -> int:
    pairs = sum(1 for c in counts34 if c >= 2)
    singles = sum(1 for c in counts34 if c == 1)
    pairs = min(pairs, 7)
    if pairs + singles > 7:
        singles = 7 - pairs
    return 13 - pairs * 2 - singles


def compute_shanten(counts34: List[int], n_meld: int = 0) -> int:
    sh = compute_shanten_yiban(counts34, n_meld)
    if n_meld == 0:
        sh = min(sh, compute_shanten_qidui(counts34))
    return sh


# ---- バックトラッキング分解列挙 ----

def _enumerate_suit_decompositions(
    counts: List[int],
    n: int,
    path: List[int],
    results: List[List[int]],
    max_results: int,
    suit_offset: int,
):
    """
    数牌1スーツをバックトラックで列挙。
    path: ブロックインデックスのリスト（このスーツで使ったブロック）
    suit_offset: 牌インデックスのオフセット (0/9/18)
    """
    if len(results) >= max_results:
        return
    if n > 9:
        results.append(list(path))
        return

    # 何もしない（n を飛ばす）
    _enumerate_suit_decompositions(counts, n + 1, path, results, max_results, suit_offset)

    if len(results) >= max_results:
        return

    # 刻子 (n, n, n)
    if counts[n] >= 3:
        tile_idx = suit_offset + n - 1
        block_idx = KOUTSU_START + tile_idx
        counts[n] -= 3
        path.append(block_idx)
        _enumerate_suit_decompositions(counts, n + 1, path, results, max_results, suit_offset)
        path.pop()
        counts[n] += 3

    if len(results) >= max_results:
        return

    # 順子 (n, n+1, n+2)
    if n <= 7 and counts[n] > 0 and counts[n+1] > 0 and counts[n+2] > 0:
        tile_idx = suit_offset + n - 1
        # 順子のブロックインデックス: suit_offset/9 * 7 + (n-1)
        suit_num = suit_offset // 9
        block_idx = SHUNTSU_START + suit_num * 7 + (n - 1)
        counts[n] -= 1; counts[n+1] -= 1; counts[n+2] -= 1
        path.append(block_idx)
        _enumerate_suit_decompositions(counts, n, path, results, max_results, suit_offset)
        path.pop()
        counts[n] += 1; counts[n+1] += 1; counts[n+2] += 1


def _dazi_count(remaining34: List[int]) -> Tuple[int, int]:
    """残り牌（面子・雀頭除去後）から搭子数と孤立牌数を計算 (xiangting.js dazi 相当)"""
    nd, ng = 0, 0
    for s in range(3):
        offset = s * 9
        c = [0] + list(remaining34[offset:offset+9]) + [0, 0]
        p = 0
        for ni in range(1, 10):
            p += c[ni]
            if ni <= 7 and c[ni+1] == 0 and c[ni+2] == 0:
                nd += p // 2; ng += p % 2; p = 0
        nd += p // 2; ng += p % 2
    for zi in range(7):
        v = remaining34[27 + zi]
        nd += v // 2; ng += v % 2
    return nd, ng


def _score_decomp(counts34: List[int], decomp: List[int], n_meld: int) -> int:
    """分解のシャンテン数を計算する"""
    remaining = list(counts34)
    has_jantou = False
    n_mentsu = n_meld
    for b in decomp:
        _, tiles = BLOCK_DEFS[b]
        for t in tiles:
            remaining[t] -= 1
        if b >= TOITSU_START:
            has_jantou = True
        else:
            n_mentsu += 1
    nd, ng = _dazi_count(remaining)
    return _xiangting_score(n_mentsu, nd, ng, has_jantou)


def _enumerate_with_jantou(
    counts34: List[int],
    jantou_tile: int,
    max_results: int,
) -> List[List[int]]:
    """雀頭 jantou_tile を固定して面子分解を列挙（生の全分解; フィルタなし）"""
    c = list(counts34)
    c[jantou_tile] -= 2
    jantou_block = TOITSU_START + jantou_tile

    suit_results = []
    for s in range(3):
        offset = s * 9
        c_suit = [0] + list(c[offset:offset+9]) + [0, 0]
        res = []
        _enumerate_suit_decompositions(c_suit, 1, [], res, max_results, offset)
        suit_results.append(res)

    z_blocks = [KOUTSU_START + 27 + zi for zi in range(7) if c[27+zi] >= 3]

    all_decomps = []
    for m_path in suit_results[0]:
        for p_path in suit_results[1]:
            for s_path in suit_results[2]:
                full = [jantou_block] + m_path + p_path + s_path + z_blocks
                all_decomps.append(full)
                if len(all_decomps) >= max_results:
                    return all_decomps
    return all_decomps


def enumerate_optimal_decompositions(
    counts34: List[int],
    n_meld: int = 0,
    max_results: int = 200,
) -> List[List[int]]:
    """
    シャンテン数を最小化する手牌分解を列挙する。

    Returns: 各要素はブロックインデックスのリスト（面子+雀頭のみ; tatsu/孤立牌は含まない）
    """
    counts34 = list(counts34)
    min_sh = compute_shanten(counts34, n_meld)

    # 七対子: shanten_qidui が min_sh を達成している場合
    # 完成形(len>=7)でも未完成形(len<7)でも、今ある対子全てをラベル化する
    if n_meld == 0 and min_sh == compute_shanten_qidui(counts34):
        pairs = [TOITSU_START + i for i in range(34) if counts34[i] >= 2]
        if len(pairs) >= 7:
            return [pairs[:7]]
        # テンパイ以下: 現在の対子をラベルとして返す（待ち牌は確定しないので含めない）
        if pairs:
            return [pairs]

    candidates = []

    # 雀頭候補を全パターン試す
    for tile_idx in range(34):
        if counts34[tile_idx] < 2:
            continue
        for d in _enumerate_with_jantou(counts34, tile_idx, max_results):
            candidates.append(d)

    # 雀頭なしパターン
    suit_results = []
    for s in range(3):
        offset = s * 9
        c_suit = [0] + list(counts34[offset:offset+9]) + [0, 0]
        res = []
        _enumerate_suit_decompositions(c_suit, 1, [], res, max_results, offset)
        suit_results.append(res)
    z_blocks = [KOUTSU_START + 27 + zi for zi in range(7) if counts34[27+zi] >= 3]
    for m_path in suit_results[0]:
        for p_path in suit_results[1]:
            for s_path in suit_results[2]:
                candidates.append(m_path + p_path + s_path + z_blocks)

    # min_shanten を達成する分解のみ保持
    optimal = []
    seen = set()
    for d in candidates:
        sh = _score_decomp(counts34, d, n_meld)
        if sh == min_sh:
            key = tuple(sorted(d))
            if key not in seen:
                seen.add(key)
                optimal.append(d)
                if len(optimal) >= max_results:
                    break

    return optimal


def _extract_floating_pairs(counts34: List[int], decomp: List[int]) -> List[int]:
    """面子（順子・刻子）を除去した残り牌から浮き対子ブロックインデックスを返す。
    すでに decomp に含まれている対子ブロックは除外する。"""
    remaining = list(counts34)
    for b in decomp:
        _, tiles = BLOCK_DEFS[b]
        for t in tiles:
            remaining[t] -= 1
    existing_toitsu = {b - TOITSU_START for b in decomp if b >= TOITSU_START}
    return [TOITSU_START + i for i in range(34)
            if remaining[i] >= 2 and i not in existing_toitsu]


def _extract_tatsu(counts34: List[int], decomp: List[int]) -> List[int]:
    """面子・対子を除去した残り牌から搭子ブロックインデックスを返す。

    浮き対子とは独立して抽出（残り牌から見える全搭子をラベル化）。
    同じ牌を使う搭子が複数あっても全て返す（ソフトラベルは平均で調整される）。
    """
    remaining = list(counts34)
    for b in decomp:
        _, tiles = BLOCK_DEFS[b]
        for t in tiles:
            remaining[t] -= 1

    tatsu = []
    for s_idx, offset in enumerate(SUIT_OFFSETS):
        # リャンメン/ペンチャン (連続2枚)
        for n in range(8):
            t1 = offset + n
            t2 = offset + n + 1
            if remaining[t1] >= 1 and remaining[t2] >= 1:
                tatsu.append(RYANMEN_START + s_idx * 8 + n)
        # カンチャン (間2枚)
        for n in range(7):
            t1 = offset + n
            t2 = offset + n + 2
            if remaining[t1] >= 1 and remaining[t2] >= 1:
                tatsu.append(KANCHAN_START + s_idx * 7 + n)
    return tatsu


def compute_soft_labels(
    counts34: List[int],
    n_meld: int = 0,
    max_results: int = 200,
    include_tatsu: bool = False,
) -> np.ndarray:
    """
    ソフトラベルを計算する（旧実装・後方互換用）。

    残り牌の対子・ターツを独立列挙するため、同じ牌を競合するブロック
    （例: 445s → s44対子 と s45ターツ）が両方ラベルに入る問題あり。
    新実装は compute_soft_labels_v2 を参照。
    """
    n_b = N_BLOCKS_WITH_TATSU if include_tatsu else N_BLOCKS
    block_defs_to_use = _block_defs_with_tatsu() if include_tatsu else BLOCK_DEFS

    decomps = enumerate_optimal_decompositions(counts34, n_meld, max_results)
    if not decomps:
        return np.zeros(n_b, dtype=np.float32)

    freq = np.zeros(n_b, dtype=np.float32)
    for d in decomps:
        floating = _extract_floating_pairs(counts34, d)
        if include_tatsu:
            tatsu = _extract_tatsu(counts34, d)
            full_d = d + floating + tatsu
        else:
            full_d = d + floating
        for b in set(full_d):
            if b < n_b:
                freq[b] += 1
    freq /= len(decomps)
    return freq


def _enumerate_remaining_patterns(
    remaining34: List[int],
    include_tatsu: bool = True,
    max_patterns: int = 50,
) -> List[List[int]]:
    """
    残り牌（面子・雀頭除去後）から対子・ターツの選択パターンを
    バックトラッキングで排他的に列挙する。

    compute_soft_labels の _extract_floating_pairs + _extract_tatsu による
    独立列挙（排他性なし）の問題を解消する。

    例: 残り牌 s4×2, s5×1
      旧: s44 と s45 を両方追加（s4を合計3枚必要だが実際は2枚）
      新: [[s44], [s45], []] の3パターンを個別分岐として列挙
          → ソフトラベルは s44≒0.5, s45≒0.5 になる

    Parameters
    ----------
    remaining34 : 面子・雀頭を除去した後の牌枚数配列 (len=34)
    include_tatsu : True でリャンメン・カンチャンも候補に含める
    max_patterns  : 列挙上限（組み合わせ爆発防止）

    Returns
    -------
    List[List[int]]  各要素はブロックインデックスのリスト（空リストも含む）
    """
    results: List[List[int]] = []

    def backtrack(c: List[int], start_tile: int, path: List[int]) -> None:
        results.append(list(path))
        if len(results) >= max_patterns:
            return

        # 対子 (start_tile 以降のタイルから選ぶ)
        for tile in range(start_tile, 34):
            if len(results) >= max_patterns:
                return
            if c[tile] >= 2:
                b = TOITSU_START + tile
                c[tile] -= 2
                path.append(b)
                backtrack(c, tile + 1, path)
                path.pop()
                c[tile] += 2

        if not include_tatsu:
            return

        # リャンメン/ペンチャン (連続2牌)
        for s_idx, offset in enumerate(SUIT_OFFSETS):
            for n in range(8):
                if len(results) >= max_patterns:
                    return
                t1 = offset + n
                t2 = offset + n + 1
                # start_tile 以降のみ（重複分岐防止）
                if t1 < start_tile:
                    continue
                if c[t1] >= 1 and c[t2] >= 1:
                    b = RYANMEN_START + s_idx * 8 + n
                    c[t1] -= 1
                    c[t2] -= 1
                    path.append(b)
                    backtrack(c, t1 + 1, path)
                    path.pop()
                    c[t1] += 1
                    c[t2] += 1

        # カンチャン (間隔2牌)
        for s_idx, offset in enumerate(SUIT_OFFSETS):
            for n in range(7):
                if len(results) >= max_patterns:
                    return
                t1 = offset + n
                t2 = offset + n + 2
                if t1 < start_tile:
                    continue
                if c[t1] >= 1 and c[t2] >= 1:
                    b = KANCHAN_START + s_idx * 7 + n
                    c[t1] -= 1
                    c[t2] -= 1
                    path.append(b)
                    backtrack(c, t1 + 1, path)
                    path.pop()
                    c[t1] += 1
                    c[t2] += 1

    backtrack(list(remaining34), 0, [])
    return results


def compute_soft_labels_v2(
    counts34: List[int],
    n_meld: int = 0,
    max_face_results: int = 200,
    max_rem_patterns: int = 50,
    include_tatsu: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    排他的バックトラッキングによる改良版ソフトラベル計算。

    旧 compute_soft_labels との違い:
    - 全シャンテン数のサンプルに対してラベルを生成（テンパイ限定でない）
    - 残り牌の対子・ターツ選択を排他的に列挙することで
      445s のような競合パターンを正確に等分できる

    Parameters
    ----------
    counts34        : 牌枚数配列 (len=34)
    n_meld          : 鳴き数
    max_face_results: 面子+雀頭分解の列挙上限
    max_rem_patterns: 残り牌パターンの列挙上限
    include_tatsu   : True で 134次元, False で 89次元

    Returns
    -------
    (label, shanten)
      label   : np.ndarray shape (134,) or (89,), dtype float32, values in [0,1]
      shanten : int  手牌のシャンテン数（-1=和了, 0=テンパイ, ...）
    """
    n_b = N_BLOCKS_WITH_TATSU if include_tatsu else N_BLOCKS
    shanten = compute_shanten(counts34, n_meld)

    face_decomps = enumerate_optimal_decompositions(counts34, n_meld, max_face_results)
    if not face_decomps:
        return np.zeros(n_b, dtype=np.float32), shanten

    all_patterns: List[List[int]] = []
    seen_keys: set = set()

    for fd in face_decomps:
        # 面子・雀頭を除去した残り牌を計算
        remaining = list(counts34)
        for b in fd:
            _, tiles = BLOCK_DEFS[b]
            for t in tiles:
                remaining[t] -= 1

        # 残り牌から対子・ターツを排他的に列挙
        rem_patterns = _enumerate_remaining_patterns(remaining, include_tatsu, max_rem_patterns)

        for rp in rem_patterns:
            full = fd + rp
            key = tuple(sorted(full))
            if key not in seen_keys:
                seen_keys.add(key)
                all_patterns.append(full)

    if not all_patterns:
        return np.zeros(n_b, dtype=np.float32), shanten

    freq = np.zeros(n_b, dtype=np.float32)
    for pat in all_patterns:
        for b in set(pat):
            if b < n_b:
                freq[b] += 1
    freq /= len(all_patterns)
    return freq, shanten


# ---- テスト ----

if __name__ == "__main__":
    import sys

    # テスト1: 123m 456m 789m 123p + 雀頭11s (テンパイ)
    c = [0] * 34
    for i in [0,1,2, 3,4,5, 6,7,8, 9,10,11]:  # 123m 456m 789m 123p
        c[i] += 1
    c[18] += 2  # 11s 雀頭
    print("テスト1: 123m456m789m123p11s")
    print(f"  シャンテン: {compute_shanten(c)}")
    decomps = enumerate_optimal_decompositions(c)
    print(f"  分解数: {len(decomps)}")
    labels = compute_soft_labels(c)
    nonzero = [(i, labels[i]) for i in range(89) if labels[i] > 0]
    print(f"  非ゼロブロック数: {len(nonzero)}")
    for i, v in nonzero:
        print(f"    block[{i:2d}] {BLOCK_DEFS[i][0]}: {v:.3f}")

    # テスト2: 1112345m (複数分解)
    c2 = [0] * 34
    for i in [0,0,0,1,2,3,4]:  # 111m 2345m
        c2[i] += 1
    c2[0] += 2  # 実際は 1112345m = [3,1,1,1,1,0,...]
    c2 = [0] * 34
    c2[0] = 3; c2[1] = 1; c2[2] = 1; c2[3] = 1; c2[4] = 1
    # 残り8枚: 111m2345m + 何か
    print("\nテスト2: 1112345m (8枚)")
    print(f"  シャンテン: {compute_shanten(c2)}")
    decomps2 = enumerate_optimal_decompositions(c2)
    print(f"  分解数: {len(decomps2)}")
    labels2 = compute_soft_labels(c2)
    nonzero2 = [(i, labels2[i]) for i in range(89) if labels2[i] > 0]
    for i, v in nonzero2:
        print(f"    block[{i:2d}] {BLOCK_DEFS[i][0]}: {v:.3f}")
