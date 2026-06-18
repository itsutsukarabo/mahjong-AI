# v19 実装計画: Multi-task 手牌構成予測

## 動機

v15〜v18 の実験を通じて判明した根本的な問題:

- 現状モデルは 34 牌を**独立に**予測している（独立分解の仮定）
- 実際の手牌は面子・対子・搭子で構成される**構造的制約**を持つ
- v17/v18 の自己回帰はこの独立性を崩す試みだったが露出バイアスで失敗

v19 では **Multi-task learning** によって構造的制約を学習に組み込む。  
自己回帰を使わないため露出バイアスは発生しない。

---

## アプローチ: Multi-task Learning

```
共有エンコーダ（v15 と同一）
        ↓
  ┌─────────────────────┐
  ↓                     ↓
枚数ヘッド（既存）    構成ヘッド（新規）
(B, 3, 34, 5)         (B, 3, 89)
 各牌の枚数分布         各ブロックの存在確率
```

構成ヘッドが「123m順子がある確率 = 0.7」と出力すると、
エンコーダは「1m, 2m, 3m が同時に存在しやすい」という特徴を学習するよう促される。
これが枚数ヘッドにも波及し、3牌の整合性が取れた予測になる。

---

## ブロック定義 (計 89 種)

| 種類 | 数 | 例 |
|------|----|----|
| 順子 (shuntsu) | 21 | 123m, 234m, ..., 789s |
| 刻子 (koutsu)  | 34 | 111m, ..., 777z |
| 雀頭 (jantou)  | 34 | 11m, ..., 77z |

搭子・孤立牌は v19 では対象外（v20 以降で追加検討）。

### ブロック→構成牌マッピング (block_selection_matrix)

固定の (89, 34) バイナリ行列。各行が「そのブロックを構成する牌のインデックス」を示す。

```python
# 例: 123m = インデックス [0, 1, 2]
# 111z = インデックス [27, 27, 27] → ones_hot で [27] に 3
block_selection_matrix[idx_123m, [0,1,2]] = 1
```

この行列は学習パラメータではなく定数バッファとして登録する。

---

## ラベル生成パイプライン

### 問題: xiangting.js はシャンテン数のみ返す

現状の JS 実装は `[面子数, 搭子数, 孤立牌数]` のカウントのみを追跡しており、
「どの牌がどのブロックを構成したか」の情報は途中で捨てられている。

### 解決策: Python でバックトラッキング列挙器を新規実装

```
phase2/scripts/enumerate_decompositions.py
```

#### アルゴリズム

```python
def enumerate_optimal_decompositions(counts_34):
    """
    counts_34: shape (34,) の整数配列（各牌の枚数）
    
    1. 既存シャンテン計算で min_shanten を取得
    2. 再帰バックトラックで全分解を列挙
    3. min_shanten を達成する分解のみ残す
    4. 各ブロックの出現頻度をソフトラベル化
    """
```

#### バックトラック構造

```
enumerate_suit(suit_counts, n=1, path=[]):
    try shuntsu at n   → recurse with n (same position)
    try koutsu at n    → recurse with n+1
    try nothing at n   → recurse with n+1
    if n > 9: yield path (tatsu/isolated は残り牌から自動計算)
```

#### ソフトラベル化

```python
# 同一シャンテンの最良分解が N 個ある場合
label_block_b = count(分解中に block_b を含むもの) / N

# 例: 1112345m でシャンテン -1 の最良分解が 3 通り
#   分解A: 111m刻子 + 234m順子 + 5m単騎
#   分解B: 111m刻子 + 23m搭子 + 45m搭子  ← シャンテン0なので除外
#   分解C: 123m順子 + 145m... ← シャンテン数次第
# → 各ブロックに [0, 1] の値を割り当て
```

#### 注意点

- 列挙数の上限を設ける（例: 最大 200 分解）
- 七対子・国士無双は別処理（それぞれ 7 対子ブロック / 13 種ブロック）
- 副露（鳴き）がある場合は残り手牌枚数に応じて処理

### ラベル追加スクリプト

```
phase2/scripts/add_block_labels.py
```

既存の `hand_inference_v15.ndjson` を読み込み、各サンプルに `label_block` フィールドを追加して `hand_inference_v19.ndjson` として保存する。

```json
{
  "features": [...],
  "label_hand": [...],
  "label_red": [...],
  "label_block": [0.0, 1.0, 0.5, ...]   // 89次元ソフトラベル × 3プレイヤー
}
```

---

## モデルアーキテクチャ変更

### 追加コンポーネント

```python
class HandInferenceV19(nn.Module):
    def __init__(self, ...):
        # 既存コンポーネント (v15 と同一)
        self.global_encoder = ...
        self.transformer     = ...
        self.tile_attn       = ...
        self.tile_norm       = ...
        self.head            = ...   # 枚数ヘッド
        self.red_head        = ...

        # 新規: 構成ヘッド
        self.block_head = nn.Linear(d_model, 1)   # 各ブロックに共有

        # 固定バッファ: (89, 34) ブロック選択行列
        self.register_buffer('block_sel', build_block_selection_matrix())

    def forward(self, x):
        # ... 既存処理 ...
        out = ...  # (B, 3, 34, d_model)

        # 構成ヘッド: ブロックを構成する牌の表現を集約
        # block_sel: (89, 34) → (B, 3, 89, d_model)
        block_feats = torch.einsum('bd, nbd -> nb', ...)
        # 簡略化: block_sel @ out.mean(-1)
        block_feats = self.block_sel @ out          # (B, 3, 89, d_model)
        block_logits = self.block_head(block_feats).squeeze(-1)  # (B, 3, 89)

        return logits, logits_raw, red_logits, block_logits
```

### block_selection_matrix の構築

```python
def build_block_selection_matrix():
    """89 × 34 の固定バイナリ行列を構築"""
    mat = torch.zeros(89, 34)
    # 順子 21 種: m[0-6], p[9-15], s[18-24] (開始インデックス)
    for suit_offset, suit_start in [(0, 0), (9, 21*1), (18, 21*2)]:
        for start in range(7):  # 1〜7 始まりの順子
            idx = ...  # ブロックインデックス
            for k in range(3):
                mat[idx, suit_offset + start + k] = 1
    # 刻子・雀頭も同様
    return mat  # (89, 34)
```

---

## 損失関数

```python
# 既存損失 (v15 と同一)
loss_nll      = F.cross_entropy(logits_raw, labels)
loss_sum      = ...
loss_cross    = ...
loss_red_ce   = ...
loss_red_cons = ...

# 新規: 構成予測損失 (ソフトラベル BCE)
loss_block = F.binary_cross_entropy_with_logits(
    block_logits.reshape(-1, N_BLOCKS),
    block_labels.reshape(-1, N_BLOCKS)   # ソフトラベル [0,1]
)

total_loss = (loss_nll
              + LAMBDA_SUM      * loss_sum
              + LAMBDA_CROSS    * loss_cross
              + LAMBDA_RED_CE   * loss_red_ce
              + LAMBDA_RED_CONS * loss_red_cons
              + LAMBDA_BLOCK    * loss_block)   # LAMBDA_BLOCK = 0.5 から試す
```

---

## 評価指標

既存指標 (EAE, hand_exact_acc, pred_total_mae, ...) に加え:

| 指標 | 内容 |
|------|------|
| `block_f1` | 各ブロックタイプの予測 F1 (二値化閾値 0.5) |
| `block_precision` | ブロック予測精度 |
| `block_recall` | ブロック予測再現率 |

---

## 実装ステップ

| ステップ | 内容 | ファイル |
|----------|------|---------|
| 1 | Python バックトラッキング列挙器 | `phase2/scripts/enumerate_decompositions.py` |
| 2 | ラベル追加スクリプト | `phase2/scripts/add_block_labels.py` |
| 3 | データ生成 | `hand_inference_v19.ndjson` |
| 4 | モデル実装 | `phase2/train/train_hand_inference_v19.py` |
| 5 | 学習・評価 | — |

ステップ 1〜2 が最も工数のかかる部分。  
ステップ 3 のデータ生成は一度だけ実行すれば良い（既存データの拡張）。

---

## v15〜v19 比較予測

| 指標 | v15 | v16 | v17/v18 | v19 (目標) |
|------|-----|-----|---------|-----------|
| test_eae | 6.38 | 9.42 | 14〜16 | **< 6.38** |
| hand_exact_acc | 20.7% | 12.4% | ~7% | **> 20.7%** |
| pred_total_mae | 3.61 | 0.0 | 0.0 | **0.0** (DP維持) |

---

## 着手タイミング

v18 の結果確定後（完了済み）、ステップ 1 から着手する。
