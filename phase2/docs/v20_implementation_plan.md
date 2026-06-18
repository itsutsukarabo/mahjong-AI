# v20 実装計画: 対子ブロック追加 + 学習改善

## v19からの変更点サマリー

| 変更 | 内容 |
|------|------|
| ラベル再生成 | 七対子テンパイバグ修正済み（6対子+孤立 → 空ラベルになる問題を修正） |
| 対子ブロック化 | 雀頭→対子に改名 + 浮き対子もラベル化（89ブロックのまま） |
| pos_weight BCE | block_recall=0.184 の改善（正例比率からpos_weightを自動計算） |
| 終盤重み付き損失 | train_epoch にも get_stage_weights を適用（現在はeval側のみ） |
| block_head 強化 | Linear(d,1) → Linear(d,64)→ReLU→Linear(64,1) |
| LAMBDA_BLOCK | 0.5 → 1.0 に増加 |

---

## 背景: 七対子テンパイバグ（修正済み）

`enumerate_decompositions.py` の七対子処理が `len(pairs) >= 7`（完成形のみ）だったため、
6対子+孤立（テンパイ形）で分解数=0・ラベル全0になっていた。

**修正内容** (enumerate_decompositions.py 済み):
```python
# 修正前
if len(pairs) >= 7:
    return [pairs[:7]]

# 修正後
if len(pairs) >= 7:
    return [pairs[:7]]
if pairs:          # テンパイ以下: 今ある対子全てをラベル化
    return [pairs]
```

---

## ブロック定義の変更: 雀頭 → 対子

### 現状（89ブロック）
```
[0..20]  順子 21種
[21..54] 刻子 34種
[55..88] 雀頭 34種  ← 「構造上の雀頭役を担う対子」のみ
```

### 問題
- 雀頭ブロックは「その分解で雀頭役を担う対子」としてのみラベル化
- 面子除去後に残る**浮き対子**（雀頭でも刻子でもない対子）がラベルなし
- 例: `11m 22p 123m 456m 789m` でsoft labels = m11=0.5, p22=0.5 のみ
  → p22が雀頭でない分解では p22は浮き対子になるが現在ラベルなし

### v20の変更（89ブロックのまま）
```
[0..20]  順子 21種
[21..54] 刻子 34種
[55..88] 対子 34種  ← 雀頭役 + 浮き対子 を統一してラベル化
```

名前を `JANTOU_START → TOITSU_START` に変更し、ソフトラベル生成時に
面子除去後の残り牌の対子も加算する。

### 刻子との競合（解決済み）

m1=3枚のケース:
```
分解A: 刻子m111 → 対子m11ラベルなし  → 刻子m111 += 1
分解B: 雀頭m11  → 残りm1=孤立       → 対子m11  += 1

soft labels: 刻子m111=0.5, 対子m11=0.5
```
バックトラッキング＋シャンテン最小フィルタが自然に解決する。

---

## 実装ステップ

### Step 1: enumerate_decompositions.py 修正

```
phase2/scripts/enumerate_decompositions.py
```

**1a. JANTOU_START → TOITSU_START に改名**

**1b. 浮き対子の抽出関数を追加**
```python
def _extract_floating_pairs(counts34: List[int], decomp: List[int]) -> List[int]:
    """面子（順子・刻子）を除去した残り牌から浮き対子を取得。
    雀頭（対子ブロック）は除く（すでに decomp に含まれているため）。"""
    remaining = list(counts34)
    for b in decomp:
        _, tiles = BLOCK_DEFS[b]
        for t in tiles:
            remaining[t] -= 1
    # TOITSU_START 以上のブロック（対子）はすでに decomp に含まれているので除外
    existing_toitsu = {b - TOITSU_START for b in decomp if b >= TOITSU_START}
    return [TOITSU_START + i for i in range(34)
            if remaining[i] >= 2 and i not in existing_toitsu]
```

**1c. compute_soft_labels でラベル加算を変更**
```python
for d in optimal_decomps:
    floating = _extract_floating_pairs(counts34, d)
    full_d = d + floating
    for b in set(full_d):
        freq[b] += 1
freq /= len(optimal_decomps)
```

**1d. 七対子テンパイバグ修正は保持（修正済み）**

### Step 2: データ再生成

```
python phase2/scripts/add_block_labels.py
```

`hand_inference_v19.ndjson` を上書き（ブロック定義が変わるため再生成必須）

- 処理時間目安: 約3〜4分（206034サンプル × 3プレイヤー）

### Step 3: train_hand_inference_v20.py 実装

v19からの変更点:

**3a. pos_weight BCE（block_recall 改善）**
```python
# main() のデータロード直後に自動計算
pos_rate = float(lblk_np.mean())
pw = (1.0 - pos_rate) / (pos_rate + 1e-9)
pos_weight = torch.full((N_BLOCKS,), pw, device=device)

# train_epoch の loss_block
loss_block = F.binary_cross_entropy_with_logits(
    block_logits.reshape(-1, N_BLOCKS),
    labels_block.reshape(-1, N_BLOCKS),
    pos_weight=pos_weight,
)
```

**3b. 訓練側の終盤重み付き NLL 損失**
```python
# train_epoch 内
loss_nll_per = F.cross_entropy(
    logits_raw.reshape(-1, model.n_count_cls),
    labels.reshape(-1),
    reduction='none'
).reshape(B, P, -1).mean(-1)                   # (B, P)

stage_w = torch.stack(
    [get_stage_weights(features[:, p]) for p in range(model.n_players)],
    dim=1
)                                               # (B, P)

loss_nll = (loss_nll_per * stage_w).sum() / stage_w.sum()
```

**3c. block_head 2層化**
```python
# 変更前
self.block_head = nn.Linear(d_model, 1)

# 変更後
self.block_head = nn.Sequential(
    nn.Linear(d_model, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

**3d. LAMBDA_BLOCK: 0.5 → 1.0**

### Step 4: 学習・評価

```
python phase2/train/train_hand_inference_v20.py >> .../v20/train.log 2>&1 &
```

---

## 期待される改善

| 指標 | v15 | v19 | v20 (目標) |
|------|-----|-----|-----------|
| test_eae | 6.38 | 6.70 | **< 6.38** |
| hand_exact_acc | 20.7% | 23.7% | **> 24%** |
| block_recall | — | 0.184 | **> 0.4** |
| block_f1 | — | 0.275 | **> 0.4** |
| pred_total_mae | 3.61 | 0.0 | 0.0 (維持) |

---

## 着手タイミング

compact後、Step 1から着手する。
