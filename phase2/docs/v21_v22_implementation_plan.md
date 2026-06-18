# v21・v22 実装計画

## v20 の結果と反省

| 指標 | v15 | v19 | v20 | 備考 |
|------|-----|-----|-----|------|
| test_eae | 6.38 | 6.70 | **7.71** | v20 は悪化 |
| hand_exact_acc | 20.7% | 23.7% | **15.5%** | 大幅悪化 |
| block_recall | — | 0.184 | **0.919** | 狙い通り改善 |
| block_precision | — | 0.539 | **0.183** | 大幅悪化 |
| block_f1 | — | 0.275 | 0.305 | ほぼ変わらず |

### v20 が失敗した原因

```
pos_weight = 32.73 × LAMBDA_BLOCK = 1.0
→ 実効的な block 損失の重み ≈ 33

手牌枚数 NLL の重み: 1.0
block BCE の重み:    33  ← 33倍の力でモデルを引っ張る
```

- モデルは「block を全部 positive と予測すれば loss が最小化できる」ことを学習してしまった
  - recall=0.919（ほぼ全部拾う）
  - precision=0.183（大量の誤検出）
- 手牌推測 (NLL) の学習が block 損失に圧迫され、test_eae が悪化

---

## v21: block 損失の過剰を修正

### 変更点（v20 との差分のみ）

| 項目 | v20 | v21 |
|------|-----|-----|
| LAMBDA_BLOCK | 1.0 | **0.3** |
| pos_weight | 自動計算 ≈ 32.73 | **上限 10 でクリップ** |
| データ | hand_inference_v19.ndjson | 同じ（変更なし） |
| モデル構造 | block_head 2層 | 同じ（変更なし） |
| ラベル | 浮き対子含む 89ブロック | 同じ（変更なし） |

### 狙い

```
block 損失の実効重み = LAMBDA_BLOCK × pos_weight
v20: 1.0 × 32.73 ≈ 33  ← 大きすぎ
v21: 0.3 × 10    =  3  ← NLL(1.0) の 3 倍程度に抑える
```

pos_weight を 10 に抑えることで precision/recall のバランスを取りつつ、
LAMBDA_BLOCK を 0.3 に下げて NLL を主役に戻す。

### 実装差分

`train_hand_inference_v21.py` の変更箇所（v20 からの差分）:

```python
# main(): pos_weight 計算部分
POS_WEIGHT_MAX = 10.0   # 追加定数

pos_rate = float(lblk_np.mean())
pw_val = min((1.0 - pos_rate) / (pos_rate + 1e-9), POS_WEIGHT_MAX)  # クリップ追加
print(f"pos_rate={pos_rate:.4f}  pos_weight={pw_val:.2f}")
pos_weight = torch.full((N_BLOCKS,), pw_val, device=device)
```

```python
# 定数
LAMBDA_BLOCK = 0.3   # v20: 1.0 → v21: 0.3
```

モデルクラス名: `HandInferenceV21`
出力ディレクトリ: `phase2/models/hand_inference/v21/`

### 期待結果

| 指標 | v19 | v20 | v21 (目標) |
|------|-----|-----|-----------|
| test_eae | 6.70 | 7.71 | **< 6.70** |
| hand_exact_acc | 23.7% | 15.5% | **> 23%** |
| block_recall | 0.184 | 0.919 | **> 0.4** |
| block_precision | 0.539 | 0.183 | **> 0.4** |
| block_f1 | 0.275 | 0.305 | **> 0.4** |

---

## v22: 搭子ブロックを追加

### 背景

v20・v21 では block_f1 の改善が限定的。原因の一つに **正例が少なすぎる** 問題がある。

```
v20 の pos_rate = 2.96%（89ブロック中、1プレイヤー平均 ≈ 2.6 ブロック）
```

搭子（リャンメン・カンチャン・ペンチャン）を追加することで：
- 正例数が増える（1プレイヤー平均 ≈ 2.6 + 3〜5 搭子 ≈ 6〜8 ブロック）
- pos_rate が 7〜10% 程度に上昇 → pos_weight が自然に 9〜13 程度まで低下

### 追加するブロック定義

```
数牌のみ（字牌に搭子はない）

リャンメン/ペンチャン（連続2枚）:
  m12, m23, m34, m45, m56, m67, m78, m89  = 8種 × 3スーツ = 24種

カンチャン（間2枚）:
  m13, m24, m35, m46, m57, m68, m79       = 7種 × 3スーツ = 21種

合計: 45種
```

```
ブロックインデックス（v22）:
  [0..20]    順子  21種
  [21..54]   刻子  34種
  [55..88]   対子  34種
  [89..112]  リャンメン/ペンチャン  24種  ← NEW
  [113..133] カンチャン             21種  ← NEW
  ────────────────────────────────────────
  合計 134ブロック
```

### 搭子ラベル生成ロジック

各分解において、面子・対子を除去した「残り牌」から搭子を抽出する。

```python
def _extract_tatsu(counts34: List[int], decomp: List[int]) -> List[int]:
    """面子・対子を除去した残り牌から搭子ブロックインデックスを返す"""
    remaining = list(counts34)
    for b in decomp:
        _, tiles = BLOCK_DEFS[b]
        for t in tiles:
            remaining[t] -= 1

    tatsu = []
    for s in range(3):
        offset = s * 9
        # リャンメン/ペンチャン (連続2枚)
        for n in range(8):
            t1 = offset + n
            t2 = offset + n + 1
            if remaining[t1] >= 1 and remaining[t2] >= 1:
                block_idx = RYANMEN_START + s * 8 + n
                tatsu.append(block_idx)
        # カンチャン (間2枚)
        for n in range(7):
            t1 = offset + n
            t2 = offset + n + 2
            if remaining[t1] >= 1 and remaining[t2] >= 1:
                block_idx = KANCHAN_START + s * 7 + n
                tatsu.append(block_idx)

    return tatsu
```

```python
# compute_soft_labels の変更
for d in optimal_decomps:
    floating_toitsu = _extract_floating_pairs(counts34, d)
    tatsu = _extract_tatsu(counts34, d)
    full_d = d + floating_toitsu + tatsu
    for b in set(full_d):
        freq[b] += 1
freq /= len(optimal_decomps)
```

### 実装ステップ

1. `enumerate_decompositions.py` にブロック定義と `_extract_tatsu` を追加
2. `add_block_labels.py` を実行してデータ再生成
   - 出力: `hand_inference_v22.ndjson`（N_BLOCKS=134）
3. `train_hand_inference_v22.py` を作成
   - N_BLOCKS: 89 → 134
   - LAMBDA_BLOCK: v21 の最適値を引き継ぐ
   - pos_weight: 自動計算（pos_rate が上昇するため pos_weight は自然に低下）
   - モデルクラス: `HandInferenceV22`

### 期待効果

```
pos_rate の変化:
  v21: 2.96% (89ブロック中 ≈ 2.6 正例)
  v22: 推定 6〜9% (134ブロック中 ≈ 8〜12 正例)

pos_weight の変化:
  v21: 10 (クリップ)
  v22: (1 - 0.075) / 0.075 ≈ 12 程度 (クリップ不要に)
```

| 指標 | v19 | v21 (目標) | v22 (目標) |
|------|-----|-----------|-----------|
| test_eae | 6.70 | < 6.70 | **< 6.38 (v15超え)** |
| block_f1 | 0.275 | > 0.4 | **> 0.5** |
| block_recall | 0.184 | > 0.4 | **> 0.5** |
| block_precision | 0.539 | > 0.4 | **> 0.5** |

---

## 実装順序

```
v21 実装（小変更: v20 の train スクリプト修正のみ）
  → 学習・評価
  → v19 超えを確認

v22 実装（中規模: enumerate_decompositions.py 拡張 + データ再生成 + 新スクリプト）
  → 学習・評価
  → v15 (test_eae 6.38) 超えを目標
```

## 着手タイミング

v20 終了確認後、v21 から着手する。
