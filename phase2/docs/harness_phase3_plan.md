# ハーネスエンジニアリング Phase 3: 修正実行

## 概要

Phase 2 で選択された対応策を自動実行する CLI ツール。
コード修正・パイプライン再実行・学習スクリプト生成を担う。

---

## ファイル構成

```
phase2/scripts/
  harness_fix.py         ← 新規: 対応策の実行エントリポイント

phase2/scripts/fixes/
  fix_add_paishu_feature.py      ← extract_features.js に残り枚数特徴を追加
  fix_impossible_tile_penalty.py ← 学習スクリプトに枯れ牌ペナルティ追加
  fix_riichi_loss_weight.py      ← リーチ局面の loss_weight 変更
  fix_late_game_weight.py        ← 終盤サンプルの loss_weight 変更
  fix_wait_logit_penalty.py      ← wait_logitsへの矛盾ペナルティ追加
```

---

## 実行方法

```bash
# Phase 2 の出力（選択番号）を引き渡す
C:\ml\venv\Scripts\python.exe phase2/scripts/harness_fix.py \
    --fixes fix_add_paishu_feature fix_impossible_tile_penalty \
    --base-version 38 \
    --target-version 39
```

---

## 各 fix の詳細

### `fix_add_paishu_feature`

**対象**: `phase2/scripts/extract_features.js`
**内容**: 各牌の残り枚数（34次元）を固定特徴量に追加する。

```
変更内容:
  - make_hand_inference_sample() で visible_count から paishu を計算し追加
  - feature_offsets.py の HI_TOTAL を 695 → 729 に更新
  - 変更後: パイプラインの extract ステップを再実行
```

**副作用**: 特徴量次元が変わるためパイプライン全再実行・再学習が必要。

---

### `fix_impossible_tile_penalty`

**対象**: `phase2/train/train_hand_inference_v39.py` (新規生成)
**内容**: 学習 loss に「枯れ牌ペナルティ項」を追加する。

```python
# 既存の loss（NLL）に加算
def compute_impossible_penalty(logits, paishu_tensor):
    # paishu_tensor: (B, 3, 34) → 0 の牌は予測も 0 であるべき
    probs = constrained_softmax(logits)  # (B, 3, 34, 5)
    p_has = 1.0 - probs[..., 0]          # (B, 3, 34) P(枚数>=1)
    impossible_mask = (paishu_tensor == 0).float()
    penalty = (p_has * impossible_mask).mean()
    return penalty

# 合計 loss
loss = nll_loss + LAMBDA_IMPOSSIBLE * impossible_penalty
# LAMBDA_IMPOSSIBLE: 初期値 0.5、調整可能
```

**副作用**: 学習スクリプトのみ変更。データパイプライン再実行は不要。

---

### `fix_riichi_loss_weight`

**対象**: `phase2/train/train_hand_inference_v39.py`
**内容**: リーチ中のプレイヤーを持つサンプルの loss_weight を増加。

```python
# 現在: リーチフラグで weight 増減なし
# 変更: riichi=True サンプルに weight *= RIICHI_WEIGHT_SCALE (初期値 2.0)
```

---

### `fix_late_game_weight`

**対象**: `phase2/train/train_hand_inference_v39.py`
**内容**: 終盤（巡目15+）サンプルの loss_weight を増加。

```python
# 現在: 全サンプル等重み
# 変更: turn >= 15 サンプルに weight *= LATE_GAME_SCALE (初期値 1.5)
```

---

### `fix_wait_logit_penalty`

**対象**: `phase2/train/train_hand_inference_v39.py`
**内容**: リーチ後にツモ切りされた牌への wait_logits に矛盾ペナルティを追加。

```python
# リーチ後ツモ切り牌は待ち牌ではないはず
# label_wait のマスクに「ツモ切り牌=0」を強制し、
# 対応する wait_logits が高い場合にペナルティを加算
```

---

## 実行フロー（harness_fix.py の内部処理）

```
1. 各 fix を順番に適用
2. パイプライン再実行が必要な fix があれば確認してから実行
   $ python run_pipeline_v39.sh
3. 学習スクリプト (train_hand_inference_v39.py) を自動生成
   (v38 スクリプトをベースに diff を適用)
4. 確認を求めてから学習を開始
   $ python train_hand_inference_v39.py
5. 完了後に git commit を提案
```

---

## v39 への引き継ぎ事項

| 項目 | v38 | v39 予定 |
|------|-----|---------|
| 特徴量次元 | 695 | 695 or 729 (fix次第) |
| 枯れ牌ペナルティ | なし | あり (λ=0.5) |
| リーチ weight | 1.0 | 2.0 |
| 終盤 weight | 1.0 | 1.5 |
| データ | 54,804 samples | 同左（fix_add_paishu_feature 適用時は再生成） |

---

## チェックリスト

- [ ] `harness_fix.py` エントリポイント作成
- [ ] `fixes/` ディレクトリ作成
- [ ] `fix_add_paishu_feature.py` 実装
- [ ] `fix_impossible_tile_penalty.py` 実装
- [ ] `fix_riichi_loss_weight.py` 実装
- [ ] `fix_late_game_weight.py` 実装
- [ ] `fix_wait_logit_penalty.py` 実装
- [ ] パイプライン再実行の自動化
- [ ] `train_hand_inference_v39.py` 自動生成ロジック
