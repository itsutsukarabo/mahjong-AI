# hand_inference v11 実装計画

作成日: 2026-06-09  
関連チケット: TICKET-029

---

## 概要

v10 は per-tile の枚数分布（0〜4枚）を独立に予測する。v11 ではテンパイ時のブロック構成（刻子/順子/対子）を直接学習し、テンパイ確率でブレンドすることで **per-tile 枚数分布の精度を改善** する。

---

## 設計方針

### 3種類の出力とその役割

| 出力 | 用途 | ブレンド |
|---|---|---|
| **per-tile 枚数分布** | 既存テーブル（0枚/1枚/2枚...） | **あり**: `(1-tp)×model + tp×block逆算` |
| **ブロック期待値** | ブロックテーブル | **なし**: 常に「聴牌条件付き」出力をそのまま表示 |
| **最頻手牌** | 最尤聴牌形 + テンパイ確率 | **なし**: 常に「聴牌条件付き」出力をそのまま表示 |

### ブレンドの目的

テンパイ時はブロック構造が手牌を強く拘束する。  
例: m123+p456+s789+z1z1 なら m1〜m3 は exactly 1枚、z1 は exactly 2枚。  
独立近似のモデルより遥かに鋭い分布 → per-tile 精度の大幅改善。

```
per_tile_final[i] = (1 - tenpai_p) × per_tile_model[i]
                  + tenpai_p       × block_to_per_tile(block_tenpai)[i]
```

---

## 変更ファイル一覧

| ファイル | 種別 | 変更内容 |
|---|---|---|
| `phase2/scripts/extract_features.js` | 改修 | `label_block` 追加 |
| `phase2/train/train_hand_inference_v11.py` | 新規 | v11 学習スクリプト |
| `phase2/browser/ai_phase2.js` | 改修 | block head 推論 + ブレンド |
| `tmp_clone/dist/js/ai_phase2.js` | 改修 | 同上（browser 側と同内容） |
| `majiang-ui-local/lib/paipu.js` | 改修 | 表示変更（テンパイ確率追加等） |
| `tmp_clone/src/css/ai-modal.styl` | 改修 | 新UIラベルの CSS |

---

## Step 1: `extract_features.js` の改修

### 追加する関数

```javascript
const Majiang = require('../../majiang-core/lib');

/**
 * 面子文字列 → triplet/seq/pair カウント配列に加算
 * 例: 'm123' → seq[0]++, 'm111' → triplet[0]++, 'm11' → pair[0]++
 */
function meld_str_to_block(meld_str, triplet, seq, pair) {
    const suit = meld_str[0];
    const nums = meld_str.slice(1).replace(/[^0-9]/g, '').split('').map(Number);
    const suit_offset = { m: 0, p: 9, s: 18, z: 27 }[suit];
    if (suit_offset === undefined || nums.length === 0) return;

    if (nums.length === 3 && nums[0] === nums[1] && nums[1] === nums[2]) {
        // 刻子
        triplet[suit_offset + nums[0] - 1]++;
    } else if (nums.length === 3) {
        // 順子 (m/p/s のみ)
        const seq_base = { m: 0, p: 7, s: 14 }[suit];
        if (seq_base !== undefined) seq[seq_base + nums[0] - 1]++;
    } else if (nums.length === 2 && nums[0] === nums[1]) {
        // 対子
        pair[suit_offset + nums[0] - 1]++;
    }
    // 単張は無視（テンパイ待ち牌として別途 tingpai で管理）
}

/**
 * テンパイ手牌 → { triplet:[34], seq:[21], pair:[34] } のソフトラベル
 * 全待ち × 全有効分解 を列挙し、各ブロックの出現頻度を正規化する
 */
function compute_block_labels(shoupai) {
    const triplet = new Array(34).fill(0);
    const seq     = new Array(21).fill(0);
    const pair    = new Array(34).fill(0);
    let total = 0;

    try {
        const waiting = Majiang.Util.tingpai(shoupai);
        for (const wp of waiting) {
            const decomps = Majiang.Util.hule_mianzi(shoupai, wp);
            for (const decomp of decomps) {
                total++;
                for (const meld of decomp) {
                    meld_str_to_block(meld, triplet, seq, pair);
                }
            }
        }
    } catch (e) {
        return null;
    }

    if (total === 0) return null;

    return {
        triplet: triplet.map(v => v / total),
        seq:     seq.map(v => v / total),
        pair:    pair.map(v => v / total),
    };
}
```

### `make_hand_inference_sample` への追記

```javascript
// 既存の shanten チェック後に追加
const label_block = (rec.shanten_l && rec.shanten_l[target_l] === 0)
    ? compute_block_labels(Majiang.Shoupai.fromString(rec.hands_l[target_l]))
    : null;

return { features, label_hand, label_red, label_block, meta };
// label_block: { triplet:[34], seq:[21], pair:[34] } or null
```

---

## Step 2: `train_hand_inference_v11.py`（新規）

### CONFIG

```python
CONFIG = {
    "input_dim":    374,   # v10 と同じ
    "d_model":      256,
    "nhead":        4,
    "num_layers":   3,
    "dropout":      0.1,
    "batch_size":   512,
    "epochs":       150,
    "early_stop_patience": 7,
    "lr":           0.001,
    "weight_decay": 0.0001,
    "lambda_red":   0.3,
    "lambda_block": 0.3,   # block head の損失重み（要調整）
}
MODEL_DIR = ".../v11"
```

### モデルアーキテクチャ

バックボーンは v10 と同一。block heads を追加。

```python
class HandInferenceV11(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, ...):
        # ... v10 と同じバックボーン ...

        # 新規: block heads
        self.triplet_head = nn.Linear(d_model, 1)   # (B, 34) → BCE
        self.pair_head    = nn.Linear(d_model, 1)   # (B, 34) → BCE
        # 順子: tokens の m1-m7, p1-p7, s1-s7 インデックスを使用
        self.seq_head     = nn.Linear(d_model, 1)   # (B, 21) → BCE

    # 順子トークンのインデックス (m1-m7, p1-p7, s1-s7)
    SEQ_TILE_INDICES = list(range(0, 7)) + list(range(9, 16)) + list(range(18, 25))

    def forward(self, x, vis):
        # ... v10 と同じ forward ...
        tokens = ...  # (B, 34, d_model)

        logits     = self.head(tokens)                                         # (B, 34, 5)
        red_logits = self.red_head(tokens[:, [4, 13, 22], :])                 # (B, 3, 2)

        # block heads
        triplet_logits = self.triplet_head(tokens).squeeze(-1)                # (B, 34)
        pair_logits    = self.pair_head(tokens).squeeze(-1)                   # (B, 34)
        seq_tokens     = tokens[:, self.SEQ_TILE_INDICES, :]                  # (B, 21, d_model)
        seq_logits     = self.seq_head(seq_tokens).squeeze(-1)                # (B, 21)

        return logits, red_logits, triplet_logits, seq_logits, pair_logits
```

### 損失関数

```python
def compute_loss(logits, red_logits, triplet_logits, seq_logits, pair_logits,
                 label_hand, label_red, label_block_tensor, tenpai_mask):
    # 既存損失
    ce_loss  = cross_entropy_per_tile(logits, label_hand)
    red_loss = bce_red(red_logits, label_red)

    # block 損失（tenpai サンプルのみ）
    block_loss = 0.0
    n_tenpai = tenpai_mask.sum().item()
    if n_tenpai > 0:
        tri = triplet_logits[tenpai_mask]           # (N, 34)
        seq = seq_logits[tenpai_mask]               # (N, 21)
        pai = pair_logits[tenpai_mask]              # (N, 34)
        t_tri = label_block_tensor[tenpai_mask, :34]
        t_seq = label_block_tensor[tenpai_mask, 34:55]
        t_pai = label_block_tensor[tenpai_mask, 55:89]
        block_loss = (F.binary_cross_entropy_with_logits(tri, t_tri) +
                      F.binary_cross_entropy_with_logits(seq, t_seq) +
                      F.binary_cross_entropy_with_logits(pai, t_pai)) / 3

    return ce_loss + CONFIG["lambda_red"] * red_loss + CONFIG["lambda_block"] * block_loss
```

### データローダー

```python
# label_block が null のサンプル → tenpai_mask=False, label_block_tensor=zeros
# label_block が存在するサンプル → tenpai_mask=True, label_block_tensor=[triplet(34)+seq(21)+pair(34)] = 89次元
```

### ONNX エクスポート

```python
torch.onnx.export(
    model, (dummy_x, dummy_vis),
    output_names=['logits', 'red_logits', 'triplet_logits', 'seq_logits', 'pair_logits'],
    ...
)
```

---

## Step 3: `ai_phase2.js` の改修（browser + dist 両方）

### 追加関数

```javascript
// ---- ブロック → per-tile 逆算 ----
function seq_contribution(tile_idx, seq_probs) {
    // tile_idx に関わる順子の確率合計を返す
    // 例: m3(idx=2) → m123(seq[0]), m234(seq[1]), m345(seq[2]) の合計
    const suit = Math.floor(tile_idx / 9);
    if (suit >= 3) return 0;  // 字牌は順子なし
    const pos = tile_idx % 9;  // 0-8
    const seq_base = suit * 7;
    let total = 0;
    for (let s = Math.max(0, pos - 2); s <= Math.min(6, pos); s++) {
        total += seq_probs[seq_base + s];
    }
    return Math.min(1, total);
}

function block_to_per_tile_dist(block_tenpai) {
    // block_tenpai: { triplet:[34], seq:[21], pair:[34] }
    // 戻り値: [[p0,p1,p2,p3,p4], ...] × 34
    return Array.from({length: 34}, (_, i) => {
        const p_tri  = block_tenpai.triplet[i];
        const p_seq  = seq_contribution(i, block_tenpai.seq) * (1 - p_tri);
        const p_pair = block_tenpai.pair[i] * (1 - p_tri) * (1 - p_seq);
        const p_none = Math.max(0, 1 - p_tri - p_seq - p_pair);
        return [p_none, p_seq, p_pair, p_tri, 0];
        // [0枚, 1枚(順子), 2枚(対子), 3枚(刻子), 4枚≈0]
    });
}

// ---- per-tile ブレンド ----
function blend_per_tile(model_probs, block_tenpai, tenpai_p) {
    // model_probs: [[p0,p1,p2,p3,p4], ...] × 34
    // 戻り値: ブレンド済み同形式
    const block_dists = block_to_per_tile_dist(block_tenpai);
    return model_probs.map((probs, i) =>
        probs.map((p, k) => (1 - tenpai_p) * p + tenpai_p * block_dists[i][k])
    );
}

// ---- 最尤聴牌形の復元 ----
function get_best_tenpai_hand(block_tenpai, melds) {
    // block_tenpai からスコア上位ブロックを greedy 選択 → Shoupai 文字列を組み立て
    // Majiang.Util.hule_mianzi で分解を列挙して返す
    // 戻り値: { hand_str, decomps, tingpai } or null
    // （詳細実装は Step 3 の実装フェーズで確定）
}
```

### 推論フローへの組み込み

```javascript
// 既存の per_tile logits → softmax の後に追加
const per_tile_model = probs_per_tile;  // softmax 済み (34×5)

// block head outputs（v11 モデルから）
const block_tenpai = {
    triplet: Array.from(out['triplet_logits'].data).map(sigmoid),  // [34]
    seq:     Array.from(out['seq_logits'].data).map(sigmoid),       // [21]
    pair:    Array.from(out['pair_logits'].data).map(sigmoid),      // [34]
};

// per-tile ブレンド
const probs_per_tile_blended = blend_per_tile(per_tile_model, block_tenpai, tenpai_prob ?? 0);

// make_block_display_data は block_tenpai を直接使う（compute_block_ev の代替）
const block_ev = make_block_display_data_v11(block_tenpai, tenpai_prob, melds);

players.push({
    ...,
    probs_per_tile: probs_per_tile_blended,  // ブレンド済みをテーブルに渡す
    block_ev,
});
```

### `make_block_display_data_v11`

```javascript
function make_block_display_data_v11(block_tenpai, tenpai_prob, melds) {
    // block_tenpai をそのまま triplet_dist/seq_dist/pair_dist 形式に変換
    // (1個確率 = block_tenpai[i], 0個確率 = 1 - block_tenpai[i])
    const to_dist = p => [1 - p, p];  // [0個, 1個]

    const triplet_dist = block_tenpai.triplet.map(to_dist);
    const seq_dist     = block_tenpai.seq.map(to_dist);
    const pair_dist    = block_tenpai.pair.map(to_dist);

    const best_tenpai  = get_best_tenpai_hand(block_tenpai, melds);

    return {
        triplet_dist,
        seq_dist,
        pair_dist,
        tenpai_prob,       // 表示用
        hand_str:    best_tenpai?.hand_str  ?? null,
        decomps:     best_tenpai?.decomps   ?? null,
        tingpai:     best_tenpai?.tingpai   ?? null,
        shanten:     null,  // v11 では「聴牌形」固定なので不要
    };
}
```

---

## Step 4: `paipu.js` の表示変更

### 変更点

1. ブロックテーブルのタイトル: 「ブロック期待値」→「ブロック期待値（聴牌形）」
2. 最頻手牌の表示:
   - 旧: `{n}向聴: m123 p456...`
   - 新: `聴牌推定形  テンパイ確率: XX%` の後に分解を表示
3. shanten 数ラベルを削除（常に聴牌形として表示）

```javascript
// make_best_hand_section の変更
const make_best_hand_section = (block_ev) => {
    const div = $('<div class="ai-hi-best-hand">');
    const tp_pct = block_ev.tenpai_prob !== null
        ? Math.round(block_ev.tenpai_prob * 100) + '%'
        : '-';
    div.append($('<span class="ai-hi-shanten-label">').text('聴牌推定形 '));
    div.append($('<span class="ai-hi-tenpai-prob">').text('テンパイ確率: ' + tp_pct));

    if (block_ev.decomps && block_ev.decomps.length > 0) {
        const tp_str = (block_ev.tingpai && block_ev.tingpai.length > 0)
            ? ' [待: ' + block_ev.tingpai.join(' ') + ']' : '';
        for (const d of block_ev.decomps.slice(0, 3)) {
            div.append($('<div class="ai-hi-decomp">').text(d.join(' ') + tp_str));
        }
    } else if (block_ev.hand_str) {
        div.append($('<span class="ai-hi-hand-str">').text(block_ev.hand_str));
    }
    return div;
};
```

---

## Step 5: CSS 追加 (`ai-modal.styl`)

```stylus
.ai-hi-tenpai-prob
  margin-left 8px
  font-family monospace
  color #8af
  font-size 10px
```

---

## データパイプライン

```
states.ndjson
    ↓ parse_paipu.js（変更なし）
    ↓ extract_features.js ← ★ Majiang import + label_block 追加
hand_inference.ndjson（label_block フィールド付き）
    ↓ add_yaku_features.py（変更なし）
    ↓ add_tenpai_features.py（変更なし）
    ↓ train_hand_inference_v11.py ← ★ block heads 追加
phase2/models/hand_inference/v11/model.onnx（出力 +3テンソル）
    ↓ ai_phase2.js ← ★ blend_per_tile + make_block_display_data_v11
    ↓ paipu.js ← ★ 表示変更
```

---

## 実装順序

1. `extract_features.js` 改修 → `hand_inference.ndjson` 再生成（データ再生成が必要）
2. `train_hand_inference_v11.py` 作成 → 学習実行
3. `ai_phase2.js` 改修（browser + dist）
4. `paipu.js` + CSS 改修
5. `npm run build:js` → 動作確認

## 注意事項

- `extract_features.js` にはすでに `module.exports` があるが、`Majiang` の require を追加するとブラウザ側から `require.main` ガードなしで読まれる恐れがないか確認すること
- `hule_mianzi` は七対子・国士形も返すため、`meld_str_to_block` でそれらの面子文字列形式（例: `z77`, `m1` 等）を正しくハンドルすること
- `label_block` が null（非テンパイ）のサンプルは `tenpai_mask=False` として block 損失をスキップ
- `lambda_block` の最適値は学習後に eval で検証（初期値 0.3）
- データ再生成が必要なため、v11 学習前に `node extract_features.js` を再実行すること
