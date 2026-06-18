# v23 実装計画: ターツブロック追加（v21 + v22 特徴量 + 搭子ラベル）

## 概要

v23 = v21 の学習設定 + v22 の特徴量（442次元）+ **搭子ブロック**（89→134ブロック）

v22 では特徴量側から「チーした/しなかった」を与えてモデルの precision を改善する。
v23 ではそれに加えてブロックラベル側にも搭子（リャンメン・カンチャン・ペンチャン）を追加し、
「手牌のどの搭子が残っているか」を明示的に予測させる。

```
v22: 442次元特徴量 × 89ブロック予測
v23: 442次元特徴量 × 134ブロック予測 （+45ブロック）
```

---

## 搭子ブロックの定義

### 追加するブロック（45種）

```
リャンメン/ペンチャン（連続2枚）— 24種:
  m12, m23, m34, m45, m56, m67, m78, m89  ← 8種 × 3スーツ
  ※ m12, m89 はペンチャン、m23〜m78 はリャンメン（役論的には区別なし）

カンチャン（間2枚）— 21種:
  m13, m24, m35, m46, m57, m68, m79       ← 7種 × 3スーツ
```

### ブロックインデックス（v23）

```
[0..20]    順子                  21種  ← 変更なし
[21..54]   刻子                  34種  ← 変更なし
[55..88]   対子（浮き対子含む）   34種  ← 変更なし
[89..112]  リャンメン/ペンチャン  24種  ← NEW
[113..133] カンチャン             21種  ← NEW
──────────────────────────────────────
合計 134ブロック
```

### ブロックインデックスの詳細

リャンメン/ペンチャン `[89..112]`:
```
89: m12, 90: m23, 91: m34, 92: m45, 93: m56, 94: m67, 95: m78, 96: m89
97: p12, 98: p23, ..., 104: p89
105: s12, 106: s23, ..., 112: s89
```

カンチャン `[113..133]`:
```
113: m13, 114: m24, 115: m35, 116: m46, 117: m57, 118: m68, 119: m79
120: p13, 121: p24, ..., 126: p79
127: s13, 128: s24, ..., 133: s79
```

---

## v23 の変更範囲

| 変更 | 詳細 |
|------|------|
| `enumerate_decompositions.py` | ターツブロック定義 + `_extract_tatsu()` + `compute_soft_labels()` 更新、N_BLOCKS: 89→134 |
| `add_block_labels.py` | パスを v23 向けに変更（SRC=v22 準備済みデータ, DST=v23） |
| 特徴量データ | v22 と共有（再生成不要）— 同じ 442次元特徴量を利用 |
| ブロックラベルデータ | 134ブロックで再生成（`hand_inference_v23.ndjson`） |
| 学習スクリプト | `train_hand_inference_v23.py`（N_BLOCKS=134, input_dim=442） |

**注意: enumerate_decompositions.py の変更は v22 のブロックラベル生成完了後に実施。**
v22 と v23 のブロックラベル生成で同じスクリプトを共用するため、順序を守ること。

---

## Step 1: `enumerate_decompositions.py` 修正

### 1a. ブロック定義の追加

`_block_defs()` 関数に搭子ブロックを追加：

```python
def _block_defs():
    """134 ブロック定義を返す: [(block_name, [tile_indices])]"""
    blocks = []
    # 順子 21種 (m/p/s × 7)  ← 変更なし
    for offset in SUIT_OFFSETS:
        for start in range(7):
            tiles = [offset + start, offset + start + 1, offset + start + 2]
            suit = "mps"[SUIT_OFFSETS.index(offset)]
            blocks.append((f"{suit}{start+1}{start+2}{start+3}", tiles))
    # 刻子 34種  ← 変更なし
    for i in range(34):
        ...  # 既存コードのまま
    # 対子 34種  ← 変更なし
    for i in range(34):
        ...  # 既存コードのまま

    # ---- 以下、NEW ----
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
```

### 1b. 定数を更新

```python
N_BLOCKS = 134  # v22: 89 → v23: 134

SHUNTSU_START  = 0    # [0..20]
KOUTSU_START   = 21   # [21..54]
TOITSU_START   = 55   # [55..88]
JANTOU_START   = TOITSU_START  # 後方互換エイリアス
RYANMEN_START  = 89   # [89..112]  ← NEW
KANCHAN_START  = 113  # [113..133] ← NEW
```

### 1c. `_extract_tatsu()` 関数を追加

`_extract_floating_pairs()` の直後に追加：

```python
def _extract_tatsu(counts34: List[int], decomp: List[int]) -> List[int]:
    """面子・対子を除去した残り牌から搭子ブロックインデックスを返す。

    浮き対子とは独立して抽出（残り牌から見える全搭子をラベル化）。
    例: 残り牌 m1,m2,m3 → m12(89), m23(90) の両方を返す。
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
```

### 1d. `compute_soft_labels()` を更新

ターツを加算するように修正：

```python
def compute_soft_labels(
    counts34: List[int],
    n_meld: int = 0,
    max_results: int = 200,
) -> np.ndarray:
    """
    134次元のソフトラベルを計算する。  # 89 → 134 に変更

    各分解に浮き対子・搭子を加えた上で頻度を計算する。
    """
    decomps = enumerate_optimal_decompositions(counts34, n_meld, max_results)
    if not decomps:
        return np.zeros(N_BLOCKS, dtype=np.float32)

    freq = np.zeros(N_BLOCKS, dtype=np.float32)
    for d in decomps:
        floating = _extract_floating_pairs(counts34, d)
        tatsu    = _extract_tatsu(counts34, d)          # ← NEW
        full_d   = d + floating + tatsu                 # ← tatsu を追加
        for b in set(full_d):
            freq[b] += 1
    freq /= len(decomps)
    return freq
```

---

## Step 2: データ再生成

### 2-0. 前提: v22 の準備済みデータを共用

v22 のパイプライン（states_v22.ndjson → extract → prepare）で生成した
**特徴量のみのファイル**をそのまま利用する。

v22 パイプラインで以下の名前で保存しておく（v22 実装時に確認）:
```
phase2/data/features/hand_inference_v22_prepared.ndjson  ← Step 3-3 の出力
phase2/data/features/hand_inference_v22.ndjson           ← Step 3-4 の出力（89ブロック付き）
```

### 2-1. ブロックラベル追加（134ブロック版）

```bash
# enumerate_decompositions.py を 134ブロック版に更新済みであること
# add_block_labels.py のパス定数を v23 向けに変更:
#   SRC_PATH = hand_inference_v22_prepared.ndjson
#   DST_PATH = hand_inference_v23.ndjson

python phase2/scripts/add_block_labels.py
```

出力: `phase2/data/features/hand_inference_v23.ndjson`
- 特徴量: 442次元（v22 と同じ）
- ブロックラベル: (3, 134) — v23 の 134ブロック

**注意:** このステップを実行する前に、v22 の `hand_inference_v22.ndjson`（89ブロック）が
既に生成済みであることを確認すること（enumerate_decompositions.py を上書きするため）。

---

## Step 3: `train_hand_inference_v23.py` 作成

v21 からの変更点のみ（v22 学習スクリプトと比較した変更点）：

```python
# 定数変更
MODEL_DIR  = .../hand_inference/v23
INPUT_DIM  = 442    # v22 と同じ（pass_chi + chi_called_tile 込み）
N_BLOCKS   = 134    # v22: 89 → v23: 134

# VISIBLE_OFFSET, REMAINING_OFFSET は変更なし
VISIBLE_OFFSET   = 185
REMAINING_OFFSET = 94

# 学習設定は v21 から引き継ぐ
LAMBDA_BLOCK    = 0.3   # v21 と同じ
POS_WEIGHT_MAX  = 10.0  # v21 と同じ（pos_rate は上昇するが cap は維持）

# モデルクラス名変更
class HandInferenceV23(nn.Module): ...

CONFIG = {
    "input_dim": 442,    # ← v22 と同じ
    "n_blocks":  134,    # ← v23 で変更
    # その他は v21 と同じ
}
```

**v21 から引き継ぐ設定:**
- LAMBDA_BLOCK=0.3、pos_weight 上限 10
- block_head 2層 (`nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))`)
- 終盤重み付き NLL（remaining≤10: ×4, ≤30: ×3, ≤50: ×2）
- d_model=256, nhead=4, num_layers=3, dropout=0.1

---

## 実験の意義: v22 との比較設計

### v22 vs v23 の違い

| 項目 | v22 | v23 |
|------|-----|-----|
| 特徴量次元 | 442 | 442 |
| ブロック数 | 89 | **134** |
| 予測対象 | 順子・刻子・対子 | 順子・刻子・対子 + **搭子** |
| pos_rate | ~3% (cap 常に発動) | ~4-6% (cap 緩和) |
| pos_weight | 上限 10 | 上限 10 (自然低下) |

### v22 との比較で検証できること

1. **搭子ラベルの有効性**: 搭子を明示的に予測させることで eae が改善するか？
2. **特徴量との相乗効果**: pass_chi_signal が「チーしなかった → その搭子がまだある」という推論を強化できるか？
3. **precision の追加改善**: 搭子ブロックを別途予測することで手牌枚数推測の ambiguity が減るか？

---

## pos_rate の推定

v23 で追加される搭子ブロックの正例数推定：

```
典型的な手牌（例: シャンテン数 1〜2）
  面子/面子候補: 3〜4個  → 順子/刻子ブロック 約 2.6 個 (v21 と同じ)
  対子: 1個             → 対子ブロック 約 1 個
  搭子: 1〜4個           → 搭子ブロック 約 2〜3 個（分解によって変動）

v21: 正例 ≈ 2.6 / 89 ブロック → pos_rate ≈ 2.9%  pos_weight ≈ 33 → cap で 10
v23: 正例 ≈ 5〜6 / 134ブロック → pos_rate ≈ 4〜5% pos_weight ≈ 19〜24 → cap で 10

pos_weight の cap は引き続き 10 が適切。
LAMBDA_BLOCK=0.3 × pos_weight=10 = 実効重み 3（NLL に対して）は v21 と同じ。
```

---

## 定量目標

| 指標 | v15 | v21 | v22 (目標) | v23 (目標) |
|------|-----|-----|-----------|-----------|
| test_eae | **6.38** | 6.78 | < 6.38 | **< v22** |
| precision_nonzero | **0.861** | 0.688 | > 0.75 | > v22 |
| recall_nonzero | 0.614 | 0.725 | > 0.70 | > 0.70 |
| hand_exact_acc | 0.207 | 0.247 | > 0.25 | > v22 |
| block_f1 | — | 0.392 | > 0.42 | **> 0.50** |

### 搭子ブロック固有の期待効果

```
現状（v21/v22）: モデルは手牌枚数 + 順子/刻子/対子の有無を予測
  → 「残り牌がどのターツになっているか」は完全に implicit

v23 追加後:
  モデルが「m23 の搭子が残っている」と予測できる
    → 対応する pass_chi_signal（m1 や m4 のスルー）と組み合わせて確信度が上がる
    → eae の改善（特に中盤の手牌推測精度向上）が期待できる
```

---

## 実装順序と依存関係

```
【v22 完了後に着手】

v22:
  [済] parse_paipu.js 修正（chi_passes_l 追加）
  [済] extract_features.js 修正（pass_chi_signal + chi_called_tile_signal 追加）
  [済] データ再生成（states_v22 → features_v22_prepared）
  [済] add_block_labels.py で 89ブロックラベル付与 → hand_inference_v22.ndjson
  [済] train_hand_inference_v22.py で学習

v23:
  1. enumerate_decompositions.py に搭子ブロックを追加（N_BLOCKS: 89→134）
  2. add_block_labels.py パス変更（SRC=v22_prepared, DST=v23）
  3. add_block_labels.py 実行 → hand_inference_v23.ndjson（134ブロック付き）
  4. train_hand_inference_v23.py 作成・実行

※ v23 Step 1 を実施すると enumerate_decompositions.py の N_BLOCKS が 134 になり、
   v22 の学習スクリプト（N_BLOCKS=89 を期待）とは非互換になる。
   v22 学習が完了してから v23 の enumerate 修正を行うこと。
```

---

## 付録: ブロック競合を厳密に扱う場合の選択肢

現状の設計では block BCE は **独立 Bernoulli**（ブロック間の排他制約なし）。
「m123 と m12 が同時に高い」は学習データ上の相関で暗黙的に抑制されるが、
推論時の二重カウントを完全には排除できない。

厳密にしたい場合は以下のアプローチが考えられる:

| アプローチ | 概要 | コスト |
|-----------|------|--------|
| **一貫性損失** | 「同じタイルを共有するブロックが両方 high = ペナルティ」という追加損失項。 例: `loss_consistency += ReLU(pred[m123] + pred[m12] - 1.0)` | 損失設計・重みチューニングが追加で必要 |
| **ターツを独立ヘッドで予測** | 順子/刻子/対子ヘッドとは別に tatsu_head を持ち、mentsu 予測に conditioned して tatsu を予測する。例: `tatsu_logits = tatsu_head(hidden - stop_grad(mentsu_attention))` | アーキテクチャ変更が必要。実装難度高 |
| **DP post-processing** | 推論後に予測ブロックリストを「タイルを消費しながらグリーディに選択」するDP処理で排他性を強制 | 評価指標は改善するが学習目標との乖離が生じる |

**推奨**: v23 では現状設計のまま実験し、`block_f1` と `test_eae` への影響を数字で確認してから判断。
ソフトラベルの設計上、排他的なケース（例: 常に m123 が最適な手牌）では 0/1 明確なラベルで学習されるため、
多くのケースでは問題にならないと予想される。

---

## 工数・注意事項

- データ生成: v22 の特徴量データを再利用するため、Step 2-1 のみ実行（約 30分〜1時間）
- 学習: v21/v22 と同エポック数（約 90〜110 epoch）を想定
- `build_block_selection_matrix()` は N_BLOCKS を参照するため、自動的に (134, 34) になる（変更不要）
- 評価スクリプト側で `N_BLOCKS` を参照している箇所があれば合わせて確認すること
