"""
hand_inference ndjson に打牌意図ラベルを付与するスクリプト。

出力フィールド:
  label_noise   : bool  — ① 最終局 + 点数孤立(30000点差) + 非親
  label_retreat : {
      "is_retreat"    : bool   — ④ 引き打牌（手出し + 圧力 + 向聴増加）
      "tile"          : int    — 捨て牌 global index (0-33)。False のとき null
      "certain_multi" : bool   — 字牌 or 隣接枯渇 → 複数持ち確度ほぼ100%
      "reason"        : str    — "honor" / "neighbor_Xz_dead" / null
  }
  label_push    : {
      "is_push"   : bool   — ③ 押し打牌（圧力下 + NOT① + 手出し向聴非増加 OR 危険ツモ切り）
      "tile"      : int    — 捨て牌 global index。False のとき null
      "via_tsumo" : bool   — True=危険牌ツモ切り, False=手出し押し
  }

判定フロー（圧力あり前提）:
  ① ノイズ         → retreat=False, push=False（スキップ）
  手出し + sh増加   → ④ retreat=True,  push=False
  手出し + sh非増加 → ③ retreat=False, push=True
  ツモ切り + 安全牌 → ② retreat=False, push=False
  ツモ切り + 危険牌 → ③ retreat=False, push=True

「安全牌」= 全リーチ相手にスジ安全 AND 副露テンパイ相手なし

使い方:
  python add_intent_labels.py
  python add_intent_labels.py --src hand_inference_v26.ndjson --states states.ndjson --dst hand_inference_v26.ndjson
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# ---- CLI ----
_args = sys.argv[1:]
_opts = {}
for i, a in enumerate(_args):
    if a.startswith("--"):
        key = a[2:]
        val = _args[i + 1] if i + 1 < len(_args) and not _args[i + 1].startswith("--") else True
        _opts[key] = val

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
STATES_DIR   = Path(__file__).parent.parent / "data" / "states"

SRC_PATH    = Path(_opts["src"])    if "src"    in _opts else FEATURES_DIR / "hand_inference_v26.ndjson"
STATES_PATH = Path(_opts["states"]) if "states" in _opts else STATES_DIR   / "states.ndjson"
DST_PATH    = Path(_opts["dst"])    if "dst"    in _opts else SRC_PATH

SCORE_ISOLATION_THRESH = 30_000

# ---- 依存ライブラリ ----
sys.path.insert(0, str(Path(__file__).parent))
from enumerate_decompositions import compute_shanten


# ---- ユーティリティ ----

def norm_tile(t: str) -> str:
    t = t.rstrip("_*")
    if len(t) == 2 and t[1] == "0":
        t = t[0] + "5"
    return t


def parse_hand(hand_str: str) -> list:
    counts = [0] * 34
    suit_map = {"m": 0, "p": 9, "s": 18, "z": 27}
    cur = None
    for c in hand_str:
        if c in "mpsz":
            cur = suit_map[c]
        elif c.isdigit() and cur is not None:
            n = int(c)
            if n == 0:
                n = 5
            counts[cur + n - 1] += 1
    return counts


def tile_str_to_idx(tile_str: str) -> int:
    suit_map = {"m": 0, "p": 9, "s": 18, "z": 27}
    t = norm_tile(tile_str)
    return suit_map[t[0]] + int(t[1]) - 1


def count_visible(discards_l: list, melds_l: list) -> list:
    vis = [0] * 34
    for dis in discards_l:
        if not dis:
            continue
        for tile in dis:
            try:
                vis[tile_str_to_idx(tile)] += 1
            except Exception:
                pass
    for meld_list in melds_l:
        if not meld_list:
            continue
        for meld in meld_list:
            if not meld:
                continue
            suit_map = {"m": 0, "p": 9, "s": 18, "z": 27}
            cur = None
            for c in str(meld):
                if c in "mpsz":
                    cur = suit_map[c]
                elif c.isdigit() and cur is not None:
                    n = int(c)
                    if n == 0:
                        n = 5
                    vis[cur + n - 1] += 1
    return vis


def is_sequence_isolated(act_idx: int, visible: list):
    """字牌、または全順子パターンが不可能な牌 → certain_multi=True"""
    if act_idx >= 27:
        return True, "honor"

    suit = act_idx // 9
    n    = act_idx % 9

    sequences = []
    if n >= 2:    sequences.append((n - 2, n - 1, n))
    if 0 < n < 8: sequences.append((n - 1, n,     n + 1))
    if n <= 6:    sequences.append((n,     n + 1,  n + 2))

    for seq in sequences:
        partners = [suit * 9 + i for i in seq if i != n]
        if all(visible[p] < 4 for p in partners):
            return False, None

    for seq in sequences:
        for i in seq:
            if i != n:
                p = suit * 9 + i
                if visible[p] >= 4:
                    return True, f"neighbor_{i+1}{'mps'[suit]}_dead"

    return True, "isolated"


def get_riichi_suji(discards_j: list) -> set:
    """
    リーチ宣言牌（'*'サフィックス）からスジ牌のglobal indexセットを返す。
    スジペア: n と n±3（同スーツ内、0-indexed で n-3, n+3）

    v45修正: '_'(ツモ切りマーカー)→'*'(リーチ宣言マーカー)。
    旧版は手出しリーチ('p6*'等)を見落とし94.9%のケースでスジ誤計算していた。
    """
    for t in reversed(discards_j or []):
        ts = str(t)
        if "*" in ts:
            try:
                decl = tile_str_to_idx(ts)
            except Exception:
                continue
            if decl >= 27:
                return set()        # 字牌はスジなし
            suit = decl // 9
            n    = decl % 9         # 0-indexed
            suji = set()
            if n >= 3: suji.add(suit * 9 + (n - 3))
            if n <= 5: suji.add(suit * 9 + (n + 3))
            return suji
    return set()


def is_tile_safe(act_idx: int, l: int, riichi_l: list, discards_l: list,
                 melds_l: list, sh_l: list) -> bool:
    """
    ツモ切り牌が②安全かどうか判定。
    True  = 全リーチ相手にスジ安全 AND 副露テンパイ相手なし → ②
    False = 危険 → ③押し扱い
    """
    # 副露テンパイ相手がいれば安全か不明
    has_meld_tenpai = any(
        j != l and not riichi_l[j] and sh_l[j] == 0 and any(melds_l[j] or [])
        for j in range(4)
    )
    if has_meld_tenpai:
        return False

    # リーチ相手全員にスジか確認
    for j in range(4):
        if j == l or not riichi_l[j]:
            continue
        suji = get_riichi_suji(discards_l[j])
        if act_idx not in suji:
            return False

    return True


def compute_pressure_labels(s: dict, is_noise: bool) -> tuple:
    """
    圧力状況下の③押し / ④退引きラベルを計算（排他的）。
    Returns: (label_retreat, label_push)
    """
    null_retreat = {"is_retreat": False, "tile": None, "certain_multi": False, "reason": None}
    null_push    = {"is_push": False, "tile": None, "via_tsumo": False}

    l   = s["l"]
    act = s.get("action")
    drt = s.get("drawn_tile")
    if not act or not drt:
        return null_retreat, null_push

    riichi = s["riichi_l"]
    sh_l   = s["shanten_l"]
    melds  = s["melds_l"]

    # 圧力判定
    has_pressure = any(
        riichi[j] or (not riichi[j] and sh_l[j] == 0 and any(melds[j] or []))
        for j in range(4) if j != l
    )
    if not has_pressure:
        return null_retreat, null_push

    # ① ノイズは③④ともスキップ
    if is_noise:
        return null_retreat, null_push

    is_tsumo = (norm_tile(act) == norm_tile(drt))

    try:
        act_idx = tile_str_to_idx(act)
    except Exception:
        return null_retreat, null_push

    if is_tsumo:
        # ツモ切り: 安全牌→② / 危険牌→③
        if is_tile_safe(act_idx, l, riichi, s["discards_l"], melds, sh_l):
            return null_retreat, null_push
        else:
            push = {"is_push": True, "tile": act_idx, "via_tsumo": True}
            return null_retreat, push

    # 手出し: 向聴数で③④判定
    try:
        counts14 = parse_hand(s["hands_l"][l])
        n_melds  = sum(1 for m in (melds[l] or []) if m)
        if counts14[act_idx] == 0:
            return null_retreat, null_push

        counts14[act_idx] -= 1
        sh_after = compute_shanten(counts14, n_melds)
        counts14[act_idx] += 1

        sh_best = 99
        for t in range(34):
            if counts14[t] > 0:
                counts14[t] -= 1
                sh = compute_shanten(counts14, n_melds)
                if sh < sh_best:
                    sh_best = sh
                counts14[t] += 1

        if sh_after > sh_best:
            # ④ 退引き打牌
            vis = count_visible(s["discards_l"], melds)
            isolated, reason = is_sequence_isolated(act_idx, vis)
            retreat = {
                "is_retreat":    True,
                "tile":          act_idx,
                "certain_multi": isolated,
                "reason":        reason,
            }
            return retreat, null_push
        else:
            # ③ 押し打牌（手出し）
            push = {"is_push": True, "tile": act_idx, "via_tsumo": False}
            return null_retreat, push

    except Exception:
        return null_retreat, null_push


# ---- メイン ----

def main():
    print(f"最終局マップ構築中: {STATES_PATH}")
    game_max_round: dict = {}
    with open(STATES_PATH, encoding="utf-8") as f:
        for line in f:
            s   = json.loads(line)
            pid = s["paipu_id"]
            key = (s["zhuangfeng"], s["jushu"])
            if pid not in game_max_round or key > game_max_round[pid]:
                game_max_round[pid] = key

    print(f"処理中: {SRC_PATH} + {STATES_PATH} → {DST_PATH}")

    stats    = defaultdict(int)
    tmp_path = DST_PATH.with_suffix(".tmp.ndjson")

    with open(SRC_PATH, encoding="utf-8") as src_f, \
         open(STATES_PATH, encoding="utf-8") as st_f, \
         open(tmp_path, "w", encoding="utf-8") as out_f:

        for i, (src_line, st_line) in enumerate(zip(src_f, st_f)):
            if i % 50000 == 0:
                print(f"  {i:,} / 205696 ...", flush=True)

            hi = json.loads(src_line)
            s  = json.loads(st_line)

            # ① ノイズ判定
            pid      = s["paipu_id"]
            l        = s["l"]
            scores   = s["scores"]
            pid_ids  = s["player_ids"]
            is_final = (s["zhuangfeng"], s["jushu"]) == game_max_round[pid]
            sc_sorted = sorted(scores)
            is_isol  = (sc_sorted[3] - sc_sorted[2] >= SCORE_ISOLATION_THRESH) or \
                       (sc_sorted[1] - sc_sorted[0] >= SCORE_ISOLATION_THRESH)
            is_oya   = (l == pid_ids[0])
            label_noise = bool(is_final and is_isol and not is_oya)

            # ③④ 圧力ラベル判定
            label_retreat, label_push = compute_pressure_labels(s, label_noise)

            # 統計
            stats["total"] += 1
            if label_noise:
                stats["noise"] += 1
            if label_retreat["is_retreat"]:
                stats["retreat"] += 1
                if label_retreat["certain_multi"]:
                    stats["retreat_certain"] += 1
            if label_push["is_push"]:
                stats["push"] += 1
                if label_push["via_tsumo"]:
                    stats["push_tsumo"] += 1

            hi["label_noise"]   = label_noise
            hi["label_retreat"] = label_retreat
            hi["label_push"]    = label_push
            out_f.write(json.dumps(hi, ensure_ascii=False) + "\n")

    tmp_path.replace(DST_PATH)

    total = stats["total"]
    print(f"\n完了: {DST_PATH}")
    print(f"  総サンプル              : {total:,}")
    print(f"  label_noise=True  (①)  : {stats['noise']:,} ({stats['noise']/total*100:.1f}%)")
    print(f"  label_retreat=True (④) : {stats['retreat']:,} ({stats['retreat']/total*100:.2f}%)")
    print(f"    うち certain_multi    : {stats['retreat_certain']:,} ({stats['retreat_certain']/max(stats['retreat'],1)*100:.1f}% of retreat)")
    print(f"  label_push=True   (③)  : {stats['push']:,} ({stats['push']/total*100:.2f}%)")
    print(f"    うち via_tsumo (危険ツモ切り): {stats['push_tsumo']:,} ({stats['push_tsumo']/max(stats['push'],1)*100:.1f}% of push)")


if __name__ == "__main__":
    main()
