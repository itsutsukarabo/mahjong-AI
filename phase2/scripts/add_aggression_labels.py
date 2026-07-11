"""
add_aggression_labels.py
攻撃性スカラー (-1〜+1) の合成・ラベル試作スクリプト

## 部品一覧とリーク分類

  [可視情報 = 入力特徴として使用可]
    danger_level        : 0=リーチなし, 1=現物, 2=スジ, 3=無スジ
    meld_threat         : float (副露脅威度, 上限なし加算)
    tile_value          : float (ドラ=1.0, 赤五=0.5, 両方=1.5)

  [手牌依存 = ラベル生成専用、モデル入力には使わない]
    is_ryanmen_nochance : bool   (切ったプレイヤーの手牌 + 可視情報)
    delta_shanten       : int    (sh_after - sh_best, ≥0)
    ukeire_change       : int    (受け入れ枚数種: 打牌後 - 打牌前)
    has_safe_tile       : bool   (手牌中に安全牌選択肢があったか)

  [出力]
    aggression          : float  -1〜+1 (-:防御, 0:中立/手なり, +:攻撃)

参照禁止: 他家手牌・山牌

使い方:
  python add_aggression_labels.py             # 全件処理 + 分布表示
  python add_aggression_labels.py --check N   # 先頭N件のみ + 分布表示
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from enumerate_decompositions import compute_shanten

# ---- ファイルパス ----
_HERE = Path(__file__).parent
STATES_PATH = _HERE.parent / "data" / "states" / "states_v22.ndjson"

# =====================================
# 攻撃性スカラー合成定数 (全調整値をここに集約)
# =====================================

# --- 防御ゾーン (圧力下・手後退 → 負のスカラー) ---
W_DELTA_SHANTEN = 0.40
# Δsh=+1後退あたりの防御強度。(旧0.50→0.40: -0.25〜-0.50の谷を-0.30〜-0.40に圧縮)

W_UKEIRE_NEG = 0.05
# ΔSH=0時: 受け入れ枚数種が1減少するあたりの防御強度
# (旧0.04→0.05: 上限0.30までの到達感度を上げる。例uk=-6→-0.30)

UKEIRE_ONLY_MAX = 0.30
# ΔSH=0・受け入れ減少のみのケースでの防御スカラー上限(絶対値)
# (旧0.25→0.30: W_DELTA_SHANTEN=0.40との差を0.10に縮めてギャップ解消)

DEFENSE_MAX = 1.0
# 防御スカラーの絶対値上限 (-DEFENSE_MAX でクリップ)

# --- 攻撃ゾーン: 危険牌種別の基本強度 ---
DL1_STRENGTH = 0.00
# DL=1(現物切り): 無難処理扱い。riichi=True but 現物 = 最安全

DL2_STRENGTH = 0.30
# DL=2(スジ): スジは半安全。一定の攻撃性を認める

DL3_NC_STRENGTH = 0.40
# DL=3(無スジ) かつ 両面NC=True: 無スジだが両面に当たらない推定安全牌
# NC=False より弱い (実態は微妙に危ない)

DL3_NOT_NC_STRENGTH = 0.70
# DL=3(無スジ) かつ NC=False: 最危険牌を押す = 最強攻撃性
# (旧0.80→0.70: 単体×INTENT=0.84で+1.0未満。meld/tile加算がある場合のみ高値に)

# --- 攻撃ゾーン: 副露・牌価値・意図 ---
MELD_THREAT_SCALE = 0.20
# meld_threat(上限なし加算値) に掛けてbase攻撃強度に加算するスケール
# (旧0.25→0.20: DL3_NOT_NC下げと併せて+1.0飽和を緩和)

TILE_VALUE_BONUS = 0.10
# tile_value(ドラ=1.0, 赤五=0.5) に掛けてbase攻撃強度に加算
# ドラ切りならさらに攻撃的と判断

INTENT_MULTIPLIER = 1.20
# has_safe_tile=True 時に攻撃強度に掛ける意図係数
# "安全牌を持ちながら敢えて危険牌を切った" = 意図的な攻撃と判断

ATTACK_MAX = 1.0
# 攻撃スカラーの上限 (ATTACK_MAX でクリップ)

# ---- タイル変換 (add_intent_labels.py と同一ロジック) ----
_SUIT_MAP = {"m": 0, "p": 9, "s": 18, "z": 27}

def norm_tile(t: str) -> str:
    t = str(t).rstrip("_*")
    if len(t) == 2 and t[1] == "0":
        t = t[0] + "5"   # 赤五 p0 → p5
    return t

def tile_str_to_idx(tile_str: str) -> int:
    """'m9'→8, 's7_'→24, 'z5*'→31, 'p0'→13(赤五)"""
    t = norm_tile(tile_str)
    if not t or t[0] not in _SUIT_MAP:
        return -1
    try:
        return _SUIT_MAP[t[0]] + int(t[1]) - 1
    except (IndexError, ValueError):
        return -1

def parse_hand(hand_str: str) -> list:
    """'m334499p99s3448z3s7' → counts34"""
    counts = [0] * 34
    cur = None
    for ch in hand_str:
        if ch in "mpsz":
            cur = _SUIT_MAP[ch]
        elif ch.isdigit() and cur is not None:
            n = int(ch)
            if n == 0: n = 5
            counts[cur + n - 1] += 1
    return counts

def parse_meld_tiles(meld_str: str) -> list:
    """'z222-' → [28,28,28], 'p34-5' → [11,12,13]"""
    tiles = []
    suit = None
    for ch in meld_str:
        if ch in "mpsz":
            suit = ch
        elif ch.isdigit() and suit is not None:
            n = int(ch)
            if n == 0: n = 5
            tiles.append(_SUIT_MAP[suit] + (n - 1))
    return tiles

# ---- ドラ計算 (baopai → dora index) ----
def baopai_to_dora_idx(baopai_idx: int) -> int:
    suit = baopai_idx // 9
    n    = baopai_idx % 9   # 0-indexed
    if suit < 3:             # 数牌: 次の牌 (9→1 で周回)
        return suit * 9 + (n + 1) % 9
    elif n <= 3:             # 風牌 z1-z4: z4→z1 で周回
        return 27 + (n + 1) % 4
    else:                    # 三元牌 z5-z7: z7→z5 で周回
        return 27 + 4 + (n - 4 + 1) % 3

def build_dora_set(baopai: list) -> set:
    s = set()
    for bp in (baopai or []):
        idx = tile_str_to_idx(bp)
        if idx >= 0:
            s.add(baopai_to_dora_idx(idx))
    return s

# ---- danger_level (extract_features.js の _danger_level をPython移植) ----
# 参照: discards_l のみ。hands_l / shanten_l 不使用。
def _get_riichi_suji(discards_j: list) -> set:
    """リーチ宣言牌のスジインデックスセット。宣言牌は '*' サフィックス付き。"""
    for ts in reversed(discards_j or []):
        if "*" in str(ts):
            idx = tile_str_to_idx(ts)
            if idx < 0 or idx >= 27:
                return set()
            suit = idx // 9
            n    = idx % 9
            suji = set()
            if n >= 3: suji.add(suit * 9 + (n - 3))
            if n <= 5: suji.add(suit * 9 + (n + 3))
            return suji
    return set()

def compute_danger_level(act_idx: int, l: int, riichi_l: list, discards_l: list) -> int:
    """0=リーチなし, 1=現物, 2=スジ, 3=無スジ"""
    riichi_js = [j for j in range(4) if j != l and riichi_l[j]]
    if not riichi_js:
        return 0
    # 現物チェック: ANY リーチ者の捨て牌に act_idx があれば DL=1
    for j in riichi_js:
        for ts in (discards_l[j] or []):
            if tile_str_to_idx(ts) == act_idx:
                return 1
    # スジチェック: 全リーチ者にスジなら DL=2
    all_suji = all(_get_riichi_suji(discards_l[j]).issuperset({act_idx}) for j in riichi_js)
    return 2 if all_suji else 3

# ---- is_ryanmen_nochance (test_ryanmen_nochance.py の実装をそのまま採用) ----
def _get_post_riichi_discards(discards_l: list, riichi_l: list) -> dict:
    result = {}
    for j, (dis, r) in enumerate(zip(discards_l, riichi_l)):
        if not r:
            continue
        post = set()
        found_decl = False
        for ts in dis:
            if found_decl:
                idx = tile_str_to_idx(ts)
                if 0 <= idx < 34:
                    post.add(idx)
            if "*" in str(ts):
                found_decl = True
        result[j] = post
    return result

def _build_visible_ext(discards_l: list, melds_l: list, hand_str) -> list:
    vis = [0] * 34
    for dis in discards_l:
        for ts in (dis or []):
            idx = tile_str_to_idx(ts)
            if 0 <= idx < 34:
                vis[idx] += 1
    for meld_list in melds_l:
        for meld in (meld_list or []):
            for idx in parse_meld_tiles(str(meld)):
                if 0 <= idx < 34:
                    vis[idx] += 1
    if isinstance(hand_str, list):
        for ts in hand_str:
            idx = tile_str_to_idx(ts)
            if 0 <= idx < 34:
                vis[idx] += 1
    elif isinstance(hand_str, str):
        cur = None
        for ch in hand_str:
            if ch in "mpsz":
                cur = _SUIT_MAP[ch]
            elif ch.isdigit() and cur is not None:
                n = int(ch)
                if n == 0: n = 5
                vis[cur + (n - 1)] += 1
    return vis

def is_ryanmen_nochance(act_idx: int, vis_ext: list,
                         post_rd: dict, riichi_players: list) -> bool:
    if act_idx >= 27:
        return True
    if not riichi_players:
        return True
    suit = act_idx // 9
    n    = act_idx % 9
    lines = []
    if n <= 5:
        lines.append((suit*9+(n+1), suit*9+(n+2), suit*9+(n+3)))
    if n >= 3:
        lines.append((suit*9+(n-2), suit*9+(n-1), suit*9+(n-3)))
    if not lines:
        return True
    for (a, b, co) in lines:
        if vis_ext[a] >= 4 or vis_ext[b] >= 4:
            continue
        if all(co in post_rd.get(j, set()) for j in riichi_players):
            continue
        return False
    return True

# ---- delta_shanten & ukeire_change (手牌依存) ----
def count_ukeire_types(counts34: list, n_meld: int) -> int:
    """シャンテンを改善する牌種の数（枚数なし、計算速度優先）"""
    sh = compute_shanten(counts34, n_meld)
    if sh == -1:
        return 0
    count = 0
    for t in range(34):
        if counts34[t] < 4:
            counts34[t] += 1
            if compute_shanten(counts34, n_meld) < sh:
                count += 1
            counts34[t] -= 1
    return count

def compute_delta_shanten_and_ukeire(act_idx: int, drawn_idx: int,
                                      counts14: list, n_meld: int):
    """
    Returns: (delta_shanten, ukeire_change)
      delta_shanten : sh_after_discard - sh_best_possible (≥0: 退引き, 0: 効率的)
      ukeire_change : ukeire_types(後) - ukeire_types(前) (正: 改善, 負: 悪化)
    """
    if counts14[act_idx] <= 0:
        return None, None

    # sh_best: 最善打牌後のシャンテン
    sh_best = 99
    tmp = counts14[:]
    for t in range(34):
        if tmp[t] > 0:
            tmp[t] -= 1
            sh = compute_shanten(tmp, n_meld)
            if sh < sh_best:
                sh_best = sh
            tmp[t] += 1

    # sh_after: act_idx 打牌後のシャンテン
    counts_after = counts14[:]
    counts_after[act_idx] -= 1
    sh_after = compute_shanten(counts_after, n_meld)
    delta_sh = sh_after - sh_best   # 0=効率的, >0=退引き

    # ukeire_change: 打牌前後の受け入れ枚数種変化
    counts_before = counts14[:]
    if 0 <= drawn_idx < 34:
        counts_before[drawn_idx] -= 1   # ツモ牌を除いた13枚
    if sum(counts_before) == 13:
        uk_before = count_ukeire_types(counts_before, n_meld)
        uk_after  = count_ukeire_types(counts_after,  n_meld)
        uk_change = uk_after - uk_before
    else:
        uk_change = None    # drawn_tile 取得失敗

    return delta_sh, uk_change

# ---- meld_threat (可視情報のみ・ルールベース) ----
_SANGENHAI = {31, 32, 33}   # z5=白, z6=發, z7=中

def compute_meld_threat(l: int, melds_l: list, riichi_l: list,
                         baopai: list, zhuangfeng: int) -> float:
    """
    副露脅威度 (可視情報のみ・上限なし加算式) ルール:

      [参照変数] melds_l, riichi_l, baopai, zhuangfeng のみ。
      shanten_l / hands_l[j≠l] / 山牌 は一切参照しない。

      成分 (仮値・合成前に相場調整):
        鳴き数ティア: 1副露=0.15, 2副露=0.35, 3副露=0.55
          ※ 3副露(手牌4枚)はテンパイほぼ確定。可視情報によるテンパイ近似。
        役牌ポン/カン: 三元牌=+0.50, 場風=+0.40, 自風(簡易)=+0.40, 他字牌=+0.30
        染め傾向:  数牌副露が単一スーツ → +0.20
        タンヤオ:  副露牌が全て 2-8   → +0.10
        ドラ含み:  副露中ドラ1枚ごとに +0.15 (上限なし)

      全相手のmax値を返す (1.0上限撤廃)。
    """
    dora_set       = build_dora_set(baopai)
    round_wind_idx = 27 + zhuangfeng   # z1=東=27, z2=南=28, z3=西=29, z4=北=30
    _MELD_TIER     = {1: 0.15, 2: 0.35, 3: 0.55}
    max_threat     = 0.0

    def is_tanyao_tile(t):
        return t < 27 and (t % 9) not in (0, 8)

    for j in range(4):
        if j == l or riichi_l[j]:
            continue
        meld_list = melds_l[j] or []
        if not meld_list:
            continue

        all_tiles = []
        for meld_str in meld_list:
            all_tiles.extend(parse_meld_tiles(str(meld_str)))
        if not all_tiles:
            continue

        threat  = 0.0
        n_melds = len(meld_list)

        # (1) 鳴き数ティア (テンパイ接近度の可視代理)
        threat += _MELD_TIER.get(n_melds, 0.55)

        # (2) 役牌ポン/カン判定 (字牌 ≥27 が3枚以上)
        honor_tiles = [t for t in all_tiles if t >= 27]
        checked = set()
        for t in honor_tiles:
            if t in checked:
                continue
            checked.add(t)
            if honor_tiles.count(t) >= 3:
                if t in _SANGENHAI:
                    threat += 0.50   # 三元牌ポン: 確定役牌
                elif t == round_wind_idx:
                    threat += 0.40   # 場風ポン
                elif t == 27 + (j % 4):
                    threat += 0.40   # 自風ポン (player index → seat wind 簡易近似)
                else:
                    threat += 0.30   # 他字牌ポン

        # (3) 染め傾向: 複数副露が単一スーツ (1副露では定義上常に同スーツ=無意味)
        num_tiles = [t for t in all_tiles if t < 27]
        if num_tiles and n_melds >= 2:
            if len({t // 9 for t in num_tiles}) == 1:
                threat += 0.20

        # (4) タンヤオ: 副露牌が全て 2-8
        if all_tiles and all(is_tanyao_tile(t) for t in all_tiles):
            threat += 0.10

        # (5) ドラ含み (上限なし)
        dora_count = sum(1 for t in all_tiles if t in dora_set)
        threat += dora_count * 0.15

        max_threat = max(max_threat, threat)

    return max_threat   # 上限撤廃 (B案)

# ---- tile_value (可視情報のみ) ----
def compute_tile_value(act_idx: int, action_str: str, dora_set: set) -> float:
    """ドラ=+1.0, 赤五=+0.5"""
    val = 1.0 if act_idx in dora_set else 0.0
    raw = str(action_str).rstrip("_*")
    if len(raw) == 2 and raw[0] in "mps" and raw[1] == "0":
        val += 0.5  # 赤五
    return val

# ---- has_safe_tile (手牌依存) ----
def _is_tile_safe_simple(act_idx: int, l: int, riichi_l: list,
                          discards_l: list, melds_l: list, shanten_l: list) -> bool:
    """全リーチ者にスジ安全 AND 副露テンパイ相手なし"""
    has_meld_tenpai = any(
        j != l and not riichi_l[j] and shanten_l[j] == 0
        and any(melds_l[j] or [])
        for j in range(4)
    )
    if has_meld_tenpai:
        return False
    for j in range(4):
        if j == l or not riichi_l[j]:
            continue
        suji = _get_riichi_suji(discards_l[j])
        if act_idx not in suji:
            return False
    return True

def compute_has_safe_tile(l: int, counts14: list, riichi_l: list,
                           discards_l: list, melds_l: list, shanten_l: list) -> bool:
    """14牌手牌中に少なくとも1枚安全牌があるか"""
    for t in range(34):
        if counts14[t] > 0:
            if _is_tile_safe_simple(t, l, riichi_l, discards_l, melds_l, shanten_l):
                return True
    return False

# ---- 攻撃性スカラー合成 ----

def danger_to_strength(dl: int, is_nc: bool) -> float:
    """danger_level と 両面NC から基本攻撃強度を返す (定数参照)"""
    if dl <= 1:
        return DL1_STRENGTH
    if dl == 2:
        return DL2_STRENGTH
    return DL3_NC_STRENGTH if is_nc else DL3_NOT_NC_STRENGTH

def compute_aggression(
    danger_level: int,
    is_ryanmen_nochance: bool,
    meld_threat: float,
    delta_shanten,          # int or None
    ukeire_change,          # int or None
    tile_value: float,
    has_safe_tile: bool,
) -> float:
    """
    攻撃性スカラーを合成して返す (-1〜+1)

    フロー:
      圧力なし → 0 (中立・手なり)
      圧力あり + 手後退 → 防御ゾーン (負)
        ΔSH>0: -W_DELTA_SHANTEN × ΔSH, クリップ -DEFENSE_MAX
        ΔSH=0 + uk<0: -min(W_UKEIRE_NEG×|uk|, UKEIRE_ONLY_MAX)
      圧力あり + 手維持/前進 → 攻撃ゾーン (正)
        DL<=1 かつ meld_threat==0: 0 (無難処理・不明)
        else:
          base  = danger_to_strength(dl, nc)
                + meld_threat × MELD_THREAT_SCALE
                + tile_value  × TILE_VALUE_BONUS
          has_safe → base × INTENT_MULTIPLIER
          クリップ ATTACK_MAX
    """
    pressure = (danger_level >= 1) or (meld_threat > 0.0)
    if not pressure:
        return 0.0

    # 手後退判定
    ds = delta_shanten if delta_shanten is not None else 0
    uk = ukeire_change if ukeire_change is not None else 0
    is_retreat = (ds > 0) or (ds == 0 and uk < 0)

    if is_retreat:
        if ds > 0:
            raw = -W_DELTA_SHANTEN * ds
        else:
            raw = -min(W_UKEIRE_NEG * (-uk), UKEIRE_ONLY_MAX)
        return max(raw, -DEFENSE_MAX)

    # 攻撃ゾーン
    if danger_level <= 1 and meld_threat == 0.0:
        return 0.0   # 現物切り・圧力なしの手維持 → 無難処理

    base  = danger_to_strength(danger_level, is_ryanmen_nochance)
    base += meld_threat * MELD_THREAT_SCALE
    base += tile_value  * TILE_VALUE_BONUS
    if has_safe_tile:
        base *= INTENT_MULTIPLIER
    return min(base, ATTACK_MAX)

# ---- メインループ ----
def process(limit: int = None):
    stats       = defaultdict(int)
    agg_vals    = []   # 全攻撃性スカラー
    ex_attack   = []   # 強い攻撃候補
    ex_defense  = []   # 明確な防御候補
    ex_neutral  = []   # 無難処理候補

    with open(STATES_PATH, encoding="utf-8") as f:
        for lno, line in enumerate(f, 1):
            if limit and lno > limit:
                break

            rec    = json.loads(line)
            evtype = rec.get("event_type", "")
            if evtype not in ("zimo", "gangzimo"):
                continue

            action = rec.get("action", "")
            if not action:
                continue
            act_idx = tile_str_to_idx(str(action))
            if act_idx < 0:
                continue

            l          = rec["l"]
            riichi_l   = rec["riichi_l"]
            discards_l = rec["discards_l"]
            melds_l    = rec["melds_l"]
            shanten_l  = rec["shanten_l"]
            baopai     = rec.get("baopai", []) or []
            zhuangfeng = rec.get("zhuangfeng", 0)
            drawn_tile = rec.get("drawn_tile", "")

            stats["total"] += 1

            # --- 可視情報部品 ---
            dl       = compute_danger_level(act_idx, l, riichi_l, discards_l)
            dora_set = build_dora_set(baopai)
            mt       = compute_meld_threat(l, melds_l, riichi_l, baopai, zhuangfeng)
            tv       = compute_tile_value(act_idx, action, dora_set)

            # --- 手牌依存部品 ---
            hand_str = rec["hands_l"][l]
            try:
                counts14 = parse_hand(hand_str)
                n_meld   = sum(1 for m in (melds_l[l] or []) if m)
                if counts14[act_idx] <= 0:
                    stats["skip_act_not_in_hand"] += 1
                    continue

                drawn_idx = tile_str_to_idx(str(drawn_tile)) if drawn_tile else -1
                delta_sh, uk_change = compute_delta_shanten_and_ukeire(
                    act_idx, drawn_idx, counts14, n_meld)

                riichi_others = [j for j in range(4) if j != l and riichi_l[j]]
                vis_ext = _build_visible_ext(discards_l, melds_l, hand_str)
                post_rd = _get_post_riichi_discards(discards_l, riichi_l)
                nc      = is_ryanmen_nochance(act_idx, vis_ext, post_rd, riichi_others)

                has_safe = compute_has_safe_tile(l, counts14, riichi_l, discards_l,
                                                  melds_l, shanten_l)
            except Exception:
                stats["skip_error"] += 1
                continue

            # --- 攻撃性スカラー合成 ---
            agg = compute_aggression(dl, nc, mt, delta_sh, uk_change, tv, has_safe)

            stats["ok"] += 1
            agg_vals.append(agg)

            # 手検算用サンプル収集
            sample = {
                "lno": lno, "l": l, "action": action, "act_idx": act_idx,
                "danger_level": dl, "is_ryanmen_nochance": nc,
                "delta_shanten": delta_sh, "ukeire_change": uk_change,
                "meld_threat": round(mt, 3), "tile_value": tv,
                "has_safe_tile": has_safe, "aggression": round(agg, 4),
                "riichi_l": riichi_l,
                "melds_j": {j: [str(m) for m in (melds_l[j] or [])]
                             for j in range(4) if melds_l[j]},
                "hand": hand_str,
            }
            # 強い攻撃: DL=3 & NOT NC & agg>0.7
            if len(ex_attack) < 3 and dl == 3 and not nc and agg > 0.7:
                ex_attack.append(sample)
            # 明確な防御: 圧力下 & agg < -0.3
            if len(ex_defense) < 3 and agg < -0.3:
                ex_defense.append(sample)
            # 無難処理: 圧力あり & -0.05 <= agg <= 0.05
            if len(ex_neutral) < 3 and (dl >= 1 or mt > 0) and abs(agg) <= 0.05:
                ex_neutral.append(sample)

    # ---- 集計・表示 ----
    n = stats["ok"]
    from statistics import mean, stdev
    print(f"\n== 処理完了: 有効 {n:,} / 総行 {stats['total']:,} ==\n")

    # --- 攻撃性スカラー分布 ---
    pos   = [v for v in agg_vals if v >  0.0]
    neg   = [v for v in agg_vals if v <  0.0]
    zero  = [v for v in agg_vals if v == 0.0]
    print("【攻撃性スカラー 全体分布】")
    print(f"  攻撃(正) : {len(pos):6,} ({len(pos)/n*100:5.1f}%)  "
          f"平均={mean(pos):.3f} min={min(pos):.3f} max={max(pos):.3f}" if pos else
          f"  攻撃(正) : {len(pos):6,}")
    print(f"  中立(0)  : {len(zero):6,} ({len(zero)/n*100:5.1f}%)")
    print(f"  防御(負) : {len(neg):6,} ({len(neg)/n*100:5.1f}%)  "
          f"平均={mean(neg):.3f} min={min(neg):.3f} max={max(neg):.3f}" if neg else
          f"  防御(負) : {len(neg):6,}")

    # 正値ヒストグラム (0.1刻み)
    if pos:
        bk = defaultdict(int)
        for v in pos: bk[round(v * 10) / 10] += 1
        print(f"\n  正値ヒストグラム(0.1刻み, +1張り付き確認):")
        for k in sorted(bk): print(f"    {k:.1f}: {bk[k]:5,}")

    # 負値ヒストグラム (0.1刻み)
    if neg:
        bk = defaultdict(int)
        for v in neg: bk[round(v * 10) / 10] += 1
        print(f"\n  負値ヒストグラム(0.1刻み, -1張り付き確認):")
        for k in sorted(bk): print(f"    {k:.1f}: {bk[k]:5,}")

    # 圧力あり局面限定
    pressure_vals = [v for v in agg_vals if v != 0.0]  # 圧力なし=0除く (近似)
    # ※ 圧力なしも agg=0 でない場合があるが、0のほとんどが無圧力 or 無難処理
    print(f"\n【攻撃性スカラー 圧力あり局面(0以外)限定: {len(pressure_vals):,}件】")
    if pressure_vals:
        p_pos = [v for v in pressure_vals if v > 0]
        p_neg = [v for v in pressure_vals if v < 0]
        print(f"  攻撃(正): {len(p_pos):6,} ({len(p_pos)/len(pressure_vals)*100:5.1f}%)"
              + (f"  平均={mean(p_pos):.3f}" if p_pos else ""))
        print(f"  防御(負): {len(p_neg):6,} ({len(p_neg)/len(pressure_vals)*100:5.1f}%)"
              + (f"  平均={mean(p_neg):.3f}" if p_neg else ""))

    # --- 手検算サンプル ---
    def show_sample(s, title):
        agg = s["aggression"]
        print(f"\n  [{title}]")
        print(f"  line={s['lno']} 打牌={s['action']} aggression={agg:+.4f}")
        print(f"  danger_level={s['danger_level']}  NC={s['is_ryanmen_nochance']}"
              f"  meld_threat={s['meld_threat']}")
        print(f"  delta_sh={s['delta_shanten']}  uk_change={s['ukeire_change']}"
              f"  tile_value={s['tile_value']}  has_safe={s['has_safe_tile']}")
        print(f"  手牌={s['hand']}  riichi={s['riichi_l']}  melds={s['melds_j']}")
        # 手計算再現
        dl, nc, mt, ds, uk, tv, safe = (
            s['danger_level'], s['is_ryanmen_nochance'], s['meld_threat'],
            s['delta_shanten'], s['ukeire_change'], s['tile_value'], s['has_safe_tile'])
        pressure = dl >= 1 or mt > 0
        if not pressure:
            calc = "圧力なし → 0.0"
        else:
            _ds = ds if ds is not None else 0
            _uk = uk if uk is not None else 0
            is_ret = _ds > 0 or (_ds == 0 and _uk < 0)
            if is_ret:
                if _ds > 0:
                    raw = -W_DELTA_SHANTEN * _ds
                    calc = f"-W_DELTA_SH({W_DELTA_SHANTEN})*{_ds}={raw:.3f} clip({-DEFENSE_MAX})"
                else:
                    raw = -min(W_UKEIRE_NEG * (-_uk), UKEIRE_ONLY_MAX)
                    calc = f"-min(W_UK({W_UKEIRE_NEG})*{-_uk},{UKEIRE_ONLY_MAX})={raw:.3f}"
            elif dl <= 1 and mt == 0:
                calc = "DL<=1 & mt=0 → 0.0 (無難処理)"
            else:
                base = danger_to_strength(dl, nc)
                base2 = base + mt * MELD_THREAT_SCALE + tv * TILE_VALUE_BONUS
                calc = (f"danger({base:.2f}) + mt*{MELD_THREAT_SCALE}({mt*MELD_THREAT_SCALE:.3f})"
                        f" + tv*{TILE_VALUE_BONUS}({tv*TILE_VALUE_BONUS:.3f}) = {base2:.3f}")
                if safe:
                    calc += f" × INTENT({INTENT_MULTIPLIER}) = {min(base2*INTENT_MULTIPLIER,ATTACK_MAX):.3f}"
        print(f"  手計算: {calc}")

    print("\n" + "="*60)
    print("【手検算: 強い攻撃候補 (DL=3&NOT_NC&agg>0.7)】")
    if ex_attack:
        for s in ex_attack: show_sample(s, "強攻撃")
    else:
        print("  (該当なし、limit を増やすか条件を緩和)")

    print("\n【手検算: 明確な防御候補 (agg<-0.3)】")
    if ex_defense:
        for s in ex_defense: show_sample(s, "防御")
    else:
        print("  (該当なし)")

    print("\n【手検算: 無難処理候補 (圧力あり&|agg|<=0.05)】")
    if ex_neutral:
        for s in ex_neutral: show_sample(s, "無難")
    else:
        print("  (該当なし)")

if __name__ == "__main__":
    args  = sys.argv[1:]
    limit = None
    for i, a in enumerate(args):
        if a == "--check" and i + 1 < len(args):
            limit = int(args[i + 1])
    process(limit=limit)
