# v22 実装計画: チースルー信号 + チー受け取り牌信号の追加

## 概要

v21 (test_eae=6.78) と v15 (test_eae=6.38) の差 0.40 の主因は **precision の低下（0.688 vs 0.861）**。

モデルが「ターツがある牌を広く予測しすぎる」問題を解決するため、2つの信号を追加する：

1. **`pass_chi_signal(34)`**: チーをスルーした = その牌に関わるターツを持っていない（否定証拠）
2. **`chi_called_tile_signal(34)`**: チーで実際に受け取った牌 = そのターツが完成している（肯定証拠）

---

## 2つの新信号の設計

### 信号1: `pass_chi_signal(34)` — チースルー信号

上家の捨て牌にチーできたのにしなかった牌を集計。
ターツが手牌にあったにもかかわらずチーをしなかった → そのターツは今も手牌にある可能性が高い。

**計算**: `t / max_discards` で直近のスルーほど大きい値（`pass_pon_signal` と同方式）

### 信号2: `chi_called_tile_signal(34)` — チー受け取り牌信号

ターゲットのチー面子から「どの牌を鳴いた（上家から受け取った）か」を抽出。

**面子文字列の読み方（majiang-core/lib/shoupai.js:284-326 より確認済み）：**

```
direction marker (+/=/-) の直前の数字 = 鳴いた牌

m3-45  → m3 鳴き  (Pattern C: 手牌m4,m5 で上家m3を鳴き)
m23-4  → m3 鳴き  (Pattern B: 手牌m2,m4 で上家m3を鳴き ― カンチャン)
m234-  → m4 鳴き  (Pattern A: 手牌m2,m3 で上家m4を鳴き)
s06-7  → s6 鳴き  (赤牌 s0=s5 が手牌)
```

**なぜ方向（鳴き元プレイヤー）は不要か：**

捨て牌特徴量（`discards_l`）には鳴かれた牌も含まれており、誰が何を切ったかが記録されている。
チー受け取り牌 X とプレイヤー別捨て牌特徴量を組み合わせることで、モデルは「誰から鳴いたか」を学習できる。
→ 方向を明示する特徴量は冗長なため追加しない。

**2信号の相補性：**
```
chi_called_tile_signal: 「X を鳴いた → X に関わるターツが手牌にあった」（完成の証拠）
pass_chi_signal:         「X を鳴けたのに鳴かなかった → そのターツが残っている可能性」（継続の証拠）
```

---

## v22 の変更範囲

| 変更 | 詳細 |
|------|------|
| `parse_paipu.js` | `chi_passes_l` 追跡を追加 |
| `extract_features.js` | `pass_chi_signal(34)` + `chi_called_tile_signal(34)` を feature 末尾に追加 |
| feature 次元数 | 374 → **442** (+68) |
| データ再生成 | states_v22.ndjson → hand_inference_v22.ndjson の全パイプライン再実行 |
| 学習スクリプト | `train_hand_inference_v22.py`（input_dim=442, それ以外は v21 と同じ） |

**既存オフセットは変更なし:**
- `VISIBLE_OFFSET = 185`（visible_counts の先頭）
- `REMAINING_OFFSET = 94`（残り牌枚数）
- offsets 78, 79, 80（n_chi, n_pon, n_kan）

---

## Step 1: `parse_paipu.js` 修正

### 1a. ヘルパー関数を追加

```javascript
/**
 * 手牌 shoupai が tile_norm に対するチーターツを持つか判定。
 * チーは数牌のみ（mps）、上家の捨て牌のみ（d='-'）。
 * tile_norm 例: 'm4', 'p7', 's2'
 */
function can_chi(shoupai, tile_norm) {
    const suit = tile_norm[0];
    if (!'mps'.includes(suit)) return false;  // 字牌はチー不可
    const n = parseInt(tile_norm[1]);
    if (n < 1 || n > 9) return false;

    // 3パターンのターツ確認
    // パターンA: (n-2, n-1) があれば (n-2)(n-1)(n) のチー可
    const hasA = n >= 3
        && count_tile_in_shoupai(shoupai, `${suit}${n-2}`) >= 1
        && count_tile_in_shoupai(shoupai, `${suit}${n-1}`) >= 1;
    // パターンB: (n-1, n+1) があれば (n-1)(n)(n+1) のチー可
    const hasB = n >= 2 && n <= 8
        && count_tile_in_shoupai(shoupai, `${suit}${n-1}`) >= 1
        && count_tile_in_shoupai(shoupai, `${suit}${n+1}`) >= 1;
    // パターンC: (n+1, n+2) があれば (n)(n+1)(n+2) のチー可
    const hasC = n <= 7
        && count_tile_in_shoupai(shoupai, `${suit}${n+1}`) >= 1
        && count_tile_in_shoupai(shoupai, `${suit}${n+2}`) >= 1;

    return hasA || hasB || hasC;
}

/**
 * meld_str が tile_norm を含むチー面子か判定。
 * チー面子フォーマット: suit + digits + direction_marker（方向マーカーが1つだけ）
 */
function is_chi_of(meld_str, tile_norm) {
    if (!meld_str) return false;
    const match = meld_str.match(/^([mps])(\d*)([\+\=\-])(\d*)$/);
    if (!match) return false;
    const suit = match[1];
    if (tile_norm[0] !== suit) return false;
    const before = match[2];
    const after = match[4];
    const digits = (before + after).split('').map(c => parseInt(c) === 0 ? 5 : parseInt(c));
    const tile_n = parseInt(tile_norm[1]) === 0 ? 5 : parseInt(tile_norm[1]);
    return digits.includes(tile_n);
}
```

### 1b. `parse_round` 関数内のローカル変数に追加

```javascript
// 既存
const pon_passes_l = [[], [], [], []];
// 追加
const chi_passes_l = [[], [], [], []];
```

### 1c. `dapai` イベント処理にチースルー検出を追加

pon_passes 検出の直後（`total_discards++` の前）に追加：

```javascript
// 【追加】チースルー検出
// チーを呼べるのは捨てた人の次のプレイヤー ((val.l + 1) % 4) のみ
const chi_caller_l = (val.l + 1) % 4;
if (
    !riichi_l[chi_caller_l] &&          // リーチ中はチー不可
    board.shoupai[chi_caller_l] &&       // 手牌が存在する
    can_chi(board.shoupai[chi_caller_l], tile_norm)  // ターツを持っている
) {
    const is_chi = next_ev?.fulou
        && is_chi_of(next_ev.fulou.m, tile_norm)
        && next_ev.fulou.l === chi_caller_l;
    if (!is_chi) {
        chi_passes_l[chi_caller_l].push({ p: tile_norm, t: total_discards });
    }
}
```

### 1d. レコード出力に `chi_passes_l` を追加

```javascript
records.push({
    // ...既存フィールド...
    pon_passes_l: pon_passes_l.map(a => [...a]),
    chi_passes_l: chi_passes_l.map(a => [...a]),   // ← 追加
    // ...
});
```

---

## Step 2: `extract_features.js` 修正

### 2a. `pass_chi_signal` 関数を追加（`pass_pon_signal` の直後）

```javascript
/**
 * チースルー信号: 上家の捨て牌にチーできたのにしなかった牌を集計（34次元）
 * t / max_discards で直近のスルーほど大きい値
 */
function pass_chi_signal(chi_passes, max_discards = 70) {
    const signal = new Array(N_PAI).fill(0);
    for (const { p, t } of (chi_passes || [])) {
        const pi = pai_to_idx(p);
        if (pi >= 0) signal[pi] += t / max_discards;
    }
    return signal;
}
```

### 2b. `chi_called_tile_signal` 関数を追加

```javascript
/**
 * チー受け取り牌信号: ターゲットのチー面子で上家から受け取った牌を返す（34次元バイナリ）
 *
 * 面子文字列の規則（majiang-core/lib/shoupai.js:284-326 より確認済み）：
 *   direction marker (+/=/-) の直前の数字が鳴いた牌
 *   例: m3-45 → m3, m23-4 → m3, m234- → m4
 *
 * 0 は赤牌（実質 5 相当）として扱う。
 */
function chi_called_tile_signal(melds) {
    const signal = new Array(N_PAI).fill(0);
    for (const m of (melds || [])) {
        const match = m.match(/^([mps])(\d*)([\+\=\-])(\d*)$/);
        if (!match) continue;  // ポン・カン・キタ はスキップ
        const suit = match[1];
        const before = match[2];
        if (before.length === 0) continue;
        const raw_digit = parseInt(before[before.length - 1]);
        const called_n = raw_digit === 0 ? 5 : raw_digit;  // 赤牌0→5
        const pi = pai_to_idx(`${suit}${called_n}`);
        if (pi >= 0) signal[pi] = 1;
    }
    return signal;
}
```

### 2c. `hand_inference_feature` の feature ベクトル末尾に追加

```javascript
// 既存（374次元）の末尾に:
...pass_chi_signal(rec.chi_passes_l?.[target_l]),      // +34 → 408次元
...chi_called_tile_signal(rec.melds_l?.[target_l]),    // +34 → 442次元
```

**feature レイアウト（v22）:**
```
[0..43]    target_discard              44
[44..81]   target_meld                38  (offset 78,79,80 = n_chi/n_pon/n_kan)
[82]       riichi                      1
[83..93]   score                      11
[94..102]  game                        9  ← REMAINING_OFFSET=94
[103..146] self_discard               44
[147..184] self_meld                  38
[185..218] visible_counts             34  ← VISIBLE_OFFSET=185
[219..221] red_discard_signal          3
[222..224] red_visible                 3
[225..268] other1_discard             44
[269..312] other2_discard             44
[313..346] pass_pon_signal            34
[347..351] wind                        5
[352..373] tenpai/yaku features       22
[374..407] pass_chi_signal            34  ← NEW (parse_paipu.js の chi_passes_l から)
[408..441] chi_called_tile_signal     34  ← NEW (メルド文字列から直接抽出)
合計: 442次元
```

---

## Step 3: データ再生成パイプライン

```bash
# 3-1. states を再生成（chi_passes_l を含む新フォーマット）
node phase2/scripts/parse_paipu.js phase2/data/raw/xml/ \
    > phase2/data/states/states_v22.ndjson

# 3-2. feature 抽出（442次元）
node phase2/scripts/extract_features.js \
    phase2/data/states/states_v22.ndjson \
    > phase2/data/features/hand_inference_raw_v22.ndjson

# 3-3. 3プレイヤー1サンプルにグループ化
python phase2/scripts/prepare_v15_data.py \
    --src phase2/data/features/hand_inference_raw_v22.ndjson \
    --dst phase2/data/features/hand_inference_v22.ndjson

# 3-4. ブロックラベル追加（enumerate_decompositions.py は v20 修正済みを利用）
python phase2/scripts/add_block_labels.py \
    --src phase2/data/features/hand_inference_v22.ndjson \
    --dst phase2/data/features/hand_inference_v22.ndjson
```

**注意:** prepare_v15_data.py と add_block_labels.py は現在 src/dst のコマンドライン引数がハードコードされているため、スクリプト内のパス定数を v22 用に変更するか、引数対応が必要。

---

## Step 4: `train_hand_inference_v22.py` 作成

v21 からの変更点のみ:

```python
# 定数変更
MODEL_DIR  = .../hand_inference/v22
INPUT_DIM  = 442   # v21: 374 → v22: 442

# VISIBLE_OFFSET, REMAINING_OFFSET は変更なし
VISIBLE_OFFSET   = 185
REMAINING_OFFSET = 94

# モデルクラス名変更
class HandInferenceV22(nn.Module): ...

# 設定変更
CONFIG = {
    "input_dim": 442,   # ← ここだけ変更
    # 他は v21 と同じ
}
```

LAMBDA_BLOCK=0.3・pos_weight 上限 10・block_head 2層・終盤重み付き NLL は v21 から引き継ぐ。

---

## 期待効果

### なぜ precision が改善するか

```
現状（v21）:
  モデルが「この牌はターツかも？」と広く予測 → precision 低下

v22 追加後:
  pass_chi_signal:
    「target が上家の m4 捨てにチーしなかった → m35/m24/m56 ターツがない可能性」
    → 「この牌は確実にターツでない」という判断が容易になる

  chi_called_tile_signal:
    「target が m4 を鳴いた → m4 を含むターツが手牌にあった（そして完成した）」
    → 鳴き済みの牌は手牌の推測から除外できる
```

### 定量目標

| 指標 | v15 | v21 | v22 (目標) |
|------|-----|-----|-----------|
| test_eae | **6.38** | 6.78 | **< 6.38** |
| precision_nonzero | **0.861** | 0.688 | **> 0.75** |
| recall_nonzero | 0.614 | 0.725 | > 0.70 |
| hand_exact_acc | 0.207 | 0.247 | > 0.25 |

---

## 工数・注意事項

- parse_paipu.js 修正後のデータ再生成は 500 XML × 全局処理で **数分〜十数分**
- `chi_called_tile_signal` は parse_paipu.js の変更不要（既存の melds_l から直接抽出）
- prepare_v15_data.py と add_block_labels.py はパス変更が必要（要確認）
- 学習は v21 と同エポック数（約 90〜110 epoch）を想定
- v23（ターツブロック追加）は v22 完了後に着手

## 着手タイミング

計画確認後、Step 1（parse_paipu.js 修正）から着手。
