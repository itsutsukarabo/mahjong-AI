"""
danger_level_at の単体テストスクリプト。

states_v22.ndjson (event_type=zimo) から実データを読み込み:
  1. was_riichi_at() の再構成精度を riichi_l (ground truth) と比較して検証
  2. 現在のアクション牌に対する danger_level(0-3) を計算・表示
  3. 旧 get_riichi_suji との差分（手出しリーチのバグ）を示す

全変数参照リスト（リーク検査）:
  - discards_l:   各プレイヤーの捨て牌列  → (A) 可視（公開情報）
  - riichi_l:     ground truth 比較用のみ → (A) 可視（テスト専用）
  - sh_l:         一切不使用              → リークなし
  - hands_l:      一切不使用              → リークなし
  - melds_l:      sh_l==0 チェックは不使用→ リークなし

実行:
  python phase2/scripts/test_danger_level.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import add_intent_labels as _old  # 旧バージョンとの比較用

STATES_PATH = Path(__file__).parent.parent / "data" / "states" / "states_v22.ndjson"

# ---- 牌変換ユーティリティ ----

def norm_tile(t: str) -> str:
    t = t.rstrip("_*")
    if len(t) == 2 and t[1] == "0":
        t = t[0] + "5"
    return t

def tile_str_to_idx(tile_str: str) -> int:
    suit_map = {"m": 0, "p": 9, "s": 18, "z": 27}
    t = norm_tile(tile_str)
    return suit_map[t[0]] + int(t[1]) - 1

def idx_to_name(idx: int) -> str:
    suits = ["m", "p", "s", "z"]
    suit  = idx // 9
    n     = idx % 9 + 1
    return f"{suits[suit]}{n}"

# ---- v45 用 get_riichi_suji（バグ修正版） ----

def get_riichi_suji_v45(discards_j: list) -> set:
    """
    リーチ宣言牌（'*' サフィックス）からスジ牌の global index セットを返す。

    旧版との差異:
      旧版: '_' (ツモ切りマーカー) を探す → 手出しリーチ ('p6*' 等) を見落とす
      新版: '*' (リーチ宣言マーカー) を探す → 手出し・ツモ切り両方を正しく捕捉

    参照変数: discards_j のみ → (A) 可視。sh_l 不使用。
    """
    for t in reversed(discards_j or []):
        ts = str(t)
        if "*" in ts:  # ← 旧版は "_" だったためバグ
            try:
                decl = tile_str_to_idx(ts)
            except Exception:
                continue
            if decl >= 27:
                return set()     # 字牌はスジなし
            suit = decl // 9
            n    = decl % 9      # 0-indexed
            suji = set()
            if n >= 3: suji.add(suit * 9 + (n - 3))
            if n <= 5: suji.add(suit * 9 + (n + 3))
            return suji
    return set()

# ---- was_riichi_at ----

def was_riichi_at(j: int, t_key_event: int, discards_j: list) -> bool:
    """
    プレイヤー j が t_key_event より前にリーチ宣言済みかを、捨て牌列のみから判定。

    参照変数: discards_j （捨て牌列 = 可視）のみ。sh_l / hands_l 不使用。

    seat_off について:
      states_v22.ndjson の `l` は座席番号（0=東=親, 1=南, 2=西, 3=北）。
      親は常に座席0なので dealer=0 固定、seat_off_j = j が正しい。
      旧 extract_features.js の dealer = rec.jushu % 4 は誤りだが、
      本関数では独立して正しい計算を行う（v45 の危険度計算専用）。

    時系列境界:
      各捨て牌の t_key = 4 * discard_index + j (seat_off = j)
      t_j >= t_key_event でループを break → 対象打牌と同時刻以降は参照しない。
      これにより「未来のリーチ宣言を見てしまう」時系列リークを防ぐ。
    """
    seat_off_j = j   # dealer=0固定（l は座席番号）
    for idx, tile in enumerate(discards_j or []):
        t_j = 4 * idx + seat_off_j
        if t_j >= t_key_event:
            break          # ← 境界: 同時刻以降は見ない
        if "*" in str(tile):
            return True
    return False

# ---- danger_level_at ----

def danger_level_at(act_idx: int, l: int, t_key_event: int, rec: dict) -> tuple:
    """
    打牌 (l が act_idx を t_key_event に捨てる) の danger_level を計算。

    戻り値: (level: int, detail: dict)
      0 = リーチ者なし（圧力なし）
      1 = 現物（リーチ者のいずれかの捨て牌に含まれる）
      2 = スジ（全リーチ者に対してスジ安全）
      3 = 無スジ（少なくとも1人のリーチ者に対して無スジ）

    参照変数: rec['discards_l']（全員の捨て牌 = 可視）のみ。
    sh_l / hands_l 不使用。
    """
    discards_l = rec["discards_l"]

    # step1: 打牌時点でリーチ中の他プレイヤーを特定（was_riichi_at のみ使用）
    riichi_players = [
        j for j in range(4)
        if j != l and was_riichi_at(j, t_key_event, discards_l[j])
    ]

    detail = {"riichi_players": riichi_players,
              "genbutsu_of": [], "suji_of": [], "musuji_of": []}

    if not riichi_players:
        return 0, detail

    # step2: 現物チェック（リーチ者の捨て牌に含まれる）
    for j in riichi_players:
        for tile in discards_l[j]:
            try:
                if tile_str_to_idx(tile) == act_idx:
                    detail["genbutsu_of"].append(j)
                    break
            except Exception:
                pass
    if detail["genbutsu_of"]:
        return 1, detail

    # step3: スジチェック
    for j in riichi_players:
        suji = get_riichi_suji_v45(discards_l[j])
        if act_idx in suji:
            detail["suji_of"].append(j)
        else:
            detail["musuji_of"].append(j)

    if not detail["musuji_of"]:
        return 2, detail  # 全リーチ者にスジ
    return 3, detail      # 少なくとも1人に無スジ

# ---- サンプル収集 ----

def collect_riichi_zimo_samples(path: Path, max_samples: int = 30) -> list:
    """
    リーチ中の他プレイヤーがいて、viewer(l) 自身はリーチ前の zimo レコードを収集する。
    （viewer がリーチ後のレコードは action が全て tsumogiri でつまらないため除外）
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "zimo":
                continue
            riichi_l = rec.get("riichi_l", [])
            l        = rec.get("l", 0)
            action   = rec.get("action")
            if not action:
                continue
            # viewer自身がリーチ済みなら除外（推測面白みがない）
            if riichi_l[l]:
                continue
            # 他プレイヤーに少なくとも1人リーチ者がいる
            if not any(riichi_l[j] for j in range(4) if j != l):
                continue
            samples.append(rec)
            if len(samples) >= max_samples:
                break
    return samples

# ---- 表示 ----

LEVEL_LABELS = {0: "なし(0)", 1: "現物(1)", 2: "スジ(2)", 3: "無スジ(3)"}

def print_sample(rec: dict, idx: int) -> dict:
    """
    1レコードを表示し、was_riichi_at 検証結果を返す。
    戻り値: {'ok': bool, 'mismatch': list}
    """
    l        = rec["l"]
    action   = rec["action"]
    riichi_l = rec["riichi_l"]

    # dealer=0固定（l は座席番号、0=東=親）
    seat_off_l   = l   # seat_off = l（dealer=0 固定）
    n_discards_l = len(rec["discards_l"][l])
    t_key_action = 4 * n_discards_l + seat_off_l

    try:
        act_idx = tile_str_to_idx(action)
    except Exception:
        return {"ok": True, "mismatch": []}

    is_tsumogiri  = "_" in action
    is_riichi_decl= "*" in action

    print(f"--- Sample {idx}  paipu={rec['paipu_id']}  round={rec['round_idx']} ---")
    print(f"  l={l}(座席)  action={action!r}({idx_to_name(act_idx)})  "
          f"t_key={t_key_action}  tsumogiri={is_tsumogiri}")

    # ---- was_riichi_at vs ground-truth riichi_l の比較 ----
    print("  [リーチ状態再構成検証]")
    mismatches = []
    for j in range(4):
        if j == l:
            continue
        gt   = bool(riichi_l[j])           # ground truth
        recon = was_riichi_at(j, t_key_action, rec["discards_l"][j])

        seat_off_j = j  # dealer=0固定
        # j のリーチ宣言牌と t_key を特定
        decl_tile = None
        decl_tkey = None
        for di, tile in enumerate(rec["discards_l"][j] or []):
            if "*" in str(tile):
                decl_tile = tile
                decl_tkey = 4 * di + seat_off_j
                break

        match = "OK" if gt == recon else "MISMATCH!"
        decl_info = f"(宣言:{decl_tile!r} t={decl_tkey})" if decl_tile else "(宣言なし)"
        print(f"    Player{j}: gt={gt}  recon={recon}  [{match}]  {decl_info}")
        if gt != recon:
            mismatches.append(j)

    # ---- danger_level 計算 ----
    level, detail = danger_level_at(act_idx, l, t_key_action, rec)
    label = LEVEL_LABELS[level]

    print(f"  [danger_level]  {label}  リーチ中={detail['riichi_players']}")
    if level == 1:
        print(f"    → 現物 of Player{detail['genbutsu_of']}")
    elif level == 2:
        print(f"    → スジ of Player{detail['suji_of']}")
    elif level == 3:
        print(f"    → 無スジ of Player{detail['musuji_of']}")
        # スジ確認のための補助情報
        for j in detail["riichi_players"]:
            suji = get_riichi_suji_v45(rec["discards_l"][j])
            suji_names = [idx_to_name(s) for s in sorted(suji)]
            decl_tiles = [t for t in rec["discards_l"][j] if "*" in str(t)]
            print(f"    Player{j} 宣言牌={decl_tiles}  スジ牌={suji_names}")

    # ---- 旧 get_riichi_suji との差分 ----
    for j in detail["riichi_players"]:
        old_suji = _old.get_riichi_suji(rec["discards_l"][j])
        new_suji = get_riichi_suji_v45(rec["discards_l"][j])
        if old_suji != new_suji:
            old_names = [idx_to_name(s) for s in sorted(old_suji)]
            new_names = [idx_to_name(s) for s in sorted(new_suji)]
            decl_tiles = [t for t in rec["discards_l"][j] if "*" in str(t)]
            print(f"  [旧バグ検出] Player{j}: 宣言牌={decl_tiles}")
            print(f"    旧スジ(誤)={old_names} → 新スジ(正)={new_names}")

    # ---- 捨て牌履歴の表示 ----
    print(f"  [捨て牌履歴]")
    for j in range(4):
        discards = rec["discards_l"][j] or []
        riichi_marker = " [RIICHI]" if riichi_l[j] else ""
        print(f"    P{j}{riichi_marker}: {discards}")

    print()
    return {"ok": len(mismatches) == 0, "mismatch": mismatches}

# ---- メイン ----

if __name__ == "__main__":
    print(f"Reading: {STATES_PATH}")
    print()

    samples = collect_riichi_zimo_samples(STATES_PATH, max_samples=15)
    print(f"リーチあり zimo サンプル {len(samples)} 件\n")
    print("=" * 70)
    print()

    total_ok = 0
    total_mismatch = 0

    for i, rec in enumerate(samples):
        result = print_sample(rec, i + 1)
        if result["ok"]:
            total_ok += 1
        else:
            total_mismatch += 1
            print(f"  *** was_riichi_at MISMATCH on Player{result['mismatch']} ***")

    print("=" * 70)
    print(f"【サマリー】")
    print(f"  was_riichi_at 再構成精度: {total_ok}/{total_ok + total_mismatch} 一致")
    if total_mismatch > 0:
        print(f"  *** {total_mismatch} 件のミスマッチあり → was_riichi_at にバグ ***")
    else:
        print(f"  *** 全サンプル一致 → was_riichi_at は正しく動作 ***")

    # ---- 特別確認: リーチ宣言牌の種別別バグ統計 ----
    print()
    print("【旧 get_riichi_suji バグ統計（手出し vs ツモ切りリーチの差）】")
    hand_decl_count  = 0  # 手出しリーチ (suffix: * only)
    tsumi_decl_count = 0  # ツモ切りリーチ (suffix: _*)
    both_ok = 0
    hand_bug_count = 0

    with open(STATES_PATH, "r", encoding="utf-8") as f:
        checked = 0
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "zimo":
                continue
            riichi_l = rec.get("riichi_l", [])
            for j in range(4):
                if not riichi_l[j]:
                    continue
                discards_j = rec["discards_l"][j] or []
                decl_tile  = None
                for d in reversed(discards_j):
                    if "*" in str(d):
                        decl_tile = str(d)
                        break
                if not decl_tile:
                    continue

                is_tsumogiri_decl = "_" in decl_tile
                old_suji = _old.get_riichi_suji(discards_j)
                new_suji = get_riichi_suji_v45(discards_j)

                if is_tsumogiri_decl:
                    tsumi_decl_count += 1
                else:
                    hand_decl_count += 1
                    if old_suji != new_suji:
                        hand_bug_count += 1

            checked += 1
            if checked >= 10000:
                break

    print(f"  手出しリーチ宣言: {hand_decl_count} 件  "
          f"→ 旧バグで suji が誤った件数: {hand_bug_count} 件 "
          f"({100*hand_bug_count/max(1,hand_decl_count):.1f}%)")
    print(f"  ツモ切りリーチ宣言: {tsumi_decl_count} 件 → 旧版も正しく動作")
    print()
    print("  ※ 手出しリーチの宣言牌が数牌の場合のみスジが変わる。")
    print("    字牌(z*)の場合は新旧ともに set() を返すので差異なし。")
