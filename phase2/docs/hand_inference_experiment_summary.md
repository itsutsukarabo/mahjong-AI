# hand_inference 実験結果サマリー

最終更新: 2026-06-18

## モデル一覧と最終評価指標

### 主要指標比較

| バージョン | test_eae | test_acc | recall_nz | precision_nz | hand_exact_acc | block_f1 | epochs |
|-----------|---------|---------|----------|-------------|---------------|---------|--------|
| v15       | 6.379   | 0.858   | 0.614    | 0.861       | 0.207         | —       | —      |
| v22       | 6.220   | 0.848   | 0.739    | 0.705       | 0.274         | 0.409   | —      |
| v23       | 6.431   | 0.846   | 0.741    | 0.701       | 0.268         | 0.317   | —      |
| v24       | 6.310   | 0.858   | 0.612    | 0.864       | 0.210         | —       | —      |
| v26       | 6.708   | 0.841   | 0.733    | 0.689       | 0.265         | 0.290   | 〜100  |
| v27       | 6.385   | 0.846   | 0.738    | 0.698       | 0.279         | 0.369   | 137    |
| v28       | —       | 0.811   | 0.687    | 0.632       | 0.236         | 0.222   | 失敗   |
| **v29**   | **5.895** | **0.853** | **0.753** | **0.712** | **0.309** | 0.265   | **116** |

※ v15/v22〜v27 の test_eae は旧 stage_w（残り牌重みのみ）基準。v29 は副露補正 stage_w 基準のため直接比較不可。バケット別 eae で比較すること。

### バケット別 eae（残り牌数ベース、直接比較可能）

| バケット | 残り牌数 | v15   | v24   | v26   | v27   | **v29** |
|---------|---------|-------|-------|-------|-------|---------|
| early   | 51枚〜  | 13.01 | 13.08 | 14.30 | 14.14 | **13.89** |
| mid     | 31〜50枚 | 5.79  | 5.77  | 6.79  | 6.27  | **5.63** |
| late    | 11〜30枚 | 2.74  | 2.58  | 1.86  | 1.61  | **1.37** |
| endgame | 0〜10枚 | 2.37  | 2.18  | 0.97  | 0.89  | **0.82** |

**v29 は全バケットで過去最良を更新。**

---

## 各バージョンの設計

### v15（ベースライン）
- 入力: 374次元
- block head なし
- eval_by_remaining あり

### v22（block head 追加）
- 入力: 442次元（pass_chi/chi_called_tile 追加）
- N_BLOCKS=89（順子21+刻子34+対子34）
- テンパイ時のみ block_loss 適用
- ラベル生成: 旧実装（445s バグあり）

### v23（ターツ追加、失敗）
- N_BLOCKS=134（v22 + 両面24 + 嵌張21）
- テンパイ時のみ block_loss
- ラベル生成: 旧実装（445s バグあり）→ 精度低下

### v24（block head なし、比較用）
- v22 と同アーキテクチャだが block head を削除
- block_loss の寄与を確認するための ablation

### v26（全シャンテン対応、失敗）
- N_BLOCKS=134
- ラベル生成: 全シャンテン数 + 排他的バックトラッキング（445s バグ修正）
- block_loss: `0.5^shanten` 重み（最低 0.0625）
- → 序盤の高シャンテンサンプルのノイジーなラベルが学習を阻害

### v27（シャンテン重み、全シャンテン）
- N_BLOCKS=89（ターツなし）
- v26 と同データ・同手法だが 89 ブロックに絞った
- late/endgame は改善したが early/mid は v22 に劣る

### v28（失敗）
- v26 ベース + NLL loss にシャンテン重み追加 + block_loss shanten≤1 限定
- NLL シャンテン重みが逆効果：endgame eae が 0.89 → 3.37 に悪化
- **教訓: NLL loss のシャンテン重みは不要かつ有害**

### v29（現行最良モデル）
- N_BLOCKS=134（ターツ含む、将来の当たり牌推測に対応）
- データ: hand_inference_v26.ndjson 流用（再生成不要）
- **変更①: block_loss を shanten≤1 のみに限定**（shanten≥2 はゼロ）
- **変更②: get_stage_weights に副露補正を追加**（副露1枚=壁牌2枚相当）
- NLL loss: stage_w のみ（シャンテン重みなし）

---

## v29 設計詳細

### block_loss 重み
```python
block_w = torch.where(
    labels_shanten <= 1,
    (0.5 ** labels_shanten.float()),        # shanten=0: 1.0 / shanten=1: 0.5
    torch.zeros_like(labels_shanten.float()) # shanten>=2: 0.0（勾配なし）
)
```

### 副露補正 stage_w
```python
meld_tiles = features_p[:, 78] * 3 + features_p[:, 79] * 3 + features_p[:, 80] * 4
effective_remaining = (remaining - meld_tiles * 2).clamp(min=0)
# 以降は従来の 4 段階バケット（effective_remaining で判定）
```

### ブロック構成（134 ブロック）
```
[0..20]   順子 21種 (m/p/s × 7)
[21..54]  刻子 34種
[55..88]  対子 34種（雀頭 + 浮き対子）
[89..112] 両面/辺張 24種（ターツ）← 当たり牌推測に使用予定
[113..133] 嵌張 21種（ターツ）← 当たり牌推測に使用予定
```

### モデル構成
```
input: (B, 3, 442)
global_encoder: Linear(442→256) → BN → ReLU → Linear(256→256)
tile_embed: Embedding(34, 256)
player_embed: Embedding(3, 256)
transformer: 3層, nhead=4, d_model=256
head: Linear(256→5)          → 牌枚数分布（0〜4枚）
red_head: Linear(256→2)      → 赤牌所持
block_head: Linear(256→64) → ReLU → Linear(64→1)  → block スコア
```

---

## 将来の当たり牌推測への接続イメージ

v29 の block_head 出力（134 次元の block_logits）を使って待ち牌を推定：

| block インデックス | 待ち種別 | 待ち牌の導出方法 |
|-----------------|---------|--------------|
| [55..88] 対子    | 単騎待ち | 対子の牌インデックス |
| [89..112] 両面/辺張 | 両面/辺張待ち | ターツ位置から ±1 または端から |
| [113..133] 嵌張   | 嵌張待ち  | ターツの中間牌 |

---

## ファイルパス

| 種別 | パス |
|------|------|
| 学習スクリプト | `phase2/train/train_hand_inference_v29.py` |
| モデル (PyTorch) | `phase2/models/hand_inference/v29/model.pt` |
| モデル (ONNX) | `phase2/models/hand_inference/v29/model.onnx` |
| 評価結果 | `phase2/models/hand_inference/v29/eval_result.json` |
| 学習ログ | `phase2/models/hand_inference/v29/train_log.json` |
| 訓練データ | `phase2/data/features/hand_inference_v26.ndjson` |
