# hand_inference v28 実装計画

## 背景と動機

### これまでの結果サマリー

| バージョン | ブロック数 | ラベル対象 | block_loss重み | test_eae | endgame eae |
|-----------|-----------|-----------|---------------|---------|------------|
| v22       | 89        | テンパイのみ | 固定          | **6.220** | — |
| v23       | 134       | テンパイのみ | 固定          | 6.431   | — |
| v24       | —（block headなし）| — | —         | 6.310   | 2.18 |
| v26       | 134       | 全シャンテン | 0.5^shanten  | 6.708   | 0.97 |
| v27       | 89        | 全シャンテン | 0.5^shanten  | 6.385   | 0.89 |

### v26/v27の問題分析

**全シャンテン対応がearly eaeを悪化させた理由：**

- シャンテン数が大きい（序盤）ほど最良分解の数が爆発的に増加
- 分解を平均した soft label が非常にフラットになる（全ブロックに微小な値が散らばる）
- 序盤サンプルは数的に最多 → フラットなラベルへの BCE 勾配がメインタスクに干渉
- `0.5^shanten` の重みでも shanten≥4 で重み 0.0625 が残り、ゼロにならない

**v26(134ブロック)がv27(89ブロック)より悪かった理由：**

- ターツ（両面・嵌張）ラベルはテンパイ以外で曖昧さが大きい
- block_f1: v26=0.290 vs v27=0.369 → ターツラベル自体の学習が困難
- ただし**問題はターツ追加ではなく全シャンテン適用**にある

### early eaeについての割り切り

残り牌50枚以上では対手13枚の分布が本質的に予測不能（情報論的限界）。
モデルの性能向上で改善できる余地は小さく、学習リソースを集中させる対象ではない。

---

## v28 設計

### 目標

1. **ターツ込み134ブロック維持** — 将来の当たり牌推測に必要（待ち牌種別→ターツ種別が1対1対応）
2. **block_lossをテンパイ・1シャンテン限定** — ノイジーな高シャンテンラベルを排除
3. **NLL lossにシャンテン重みを追加** — テンパイ・1シャンテン予測精度を優先

### 変更点（v26からの差分）

#### 1. block_loss の限定（核心）

```python
# v26/v27: 全シャンテン数で block_loss を適用（高シャンテンは重みで軽減）
block_w = (0.5 ** labels_shanten.float()).clamp(min=0.0625)

# v28: shanten <= 1 のみ適用、それ以外はゼロ
block_w = torch.where(
    labels_shanten <= 1,
    (0.5 ** labels_shanten.float()),        # shanten=0: 1.0 / shanten=1: 0.5
    torch.zeros_like(labels_shanten.float()) # shanten>=2: 0.0（勾配なし）
)
```

#### 2. NLL loss へのシャンテン重み追加

```python
# v26/v27: 残り牌バケット重みのみ
stage_w = torch.stack(
    [get_stage_weights(features[:, p]) for p in range(model.n_players)], dim=1
)  # (B, 3)
loss_nll = (loss_nll_per * stage_w).sum() / stage_w.sum()

# v28: 残り牌重み × シャンテン重み
stage_w = torch.stack(
    [get_stage_weights(features[:, p]) for p in range(model.n_players)], dim=1
)  # (B, 3)
shanten_w = (0.5 ** labels_shanten.float()).clamp(min=0.25)  # (B, 3)
combined_w = stage_w * shanten_w
loss_nll = (loss_nll_per * combined_w).sum() / combined_w.sum()
```

重みの対応：

| shanten | shanten_w | stage_w (endgame) | combined_w |
|---------|-----------|-------------------|------------|
| 0 (テンパイ) | 1.00 | 4.0 | **4.00** |
| 1       | 0.50 | 4.0 | **2.00** |
| 2       | 0.25 (下限) | 4.0 | **1.00** |
| 4+      | 0.25 (下限) | 1.0 | **0.25** |

#### 3. eval_epoch の val_eae もシャンテン重みで統一

```python
# v26/v27: 残り牌重みのみ
stage_w = get_stage_weights(features[:, p])
eae_val = weighted_eae(probs_p, labels[:, p], stage_w)

# v28: 残り牌重み × シャンテン重み（訓練目標と統一）
shanten_w  = (0.5 ** labels_shanten[:, p].float()).clamp(min=0.25)
combined_w = stage_w * shanten_w
eae_val    = weighted_eae(probs_p, labels[:, p], combined_w)
```

モデル選択基準（val_eae）を訓練目標と一致させることで、
テンパイ・1シャンテン予測が良いモデルが選ばれるようになる。

#### 4. その他変更なし（v26と同一）

- モデルアーキテクチャ: 同一（HandInferenceV26クラスを流用）
- N_BLOCKS = 134（ターツ含む）
- LAMBDA_BLOCK = 0.3（変更なし）
- ハイパーパラメータ全て同一

### データ

**`hand_inference_v26.ndjson` を流用（再生成不要）**

- 134ブロック + `label_shanten` フィールド付き
- 全シャンテンのラベルが含まれているが、学習時に shanten>1 の block_loss をゼロにする

---

## ファイル構成

| ファイル | 操作 |
|---------|------|
| `phase2/train/train_hand_inference_v28.py` | v26から複製 → 2箇所変更 |
| `phase2/models/hand_inference/v28/` | 学習時に自動生成 |
| `phase2/data/features/hand_inference_v26.ndjson` | 流用（変更なし） |

---

## 実行コマンド

```bash
python phase2/train/train_hand_inference_v28.py > /tmp/train_v28.log 2>&1 &
tail -f /tmp/train_v28.log
```

---

## 期待される効果

| 指標 | 予測 | 根拠 |
|------|------|------|
| test_eae | v22(6.22)以下を目標 | ターツラベル + 適切な学習集中 |
| early eae | v26/v27(14.1〜14.3)より改善 | 高シャンテンでblock_loss勾配なし |
| late/endgame eae | v26/v27並み(1.6〜1.9) | ターツブロック維持 |
| block_f1 | v26(0.290)より改善 | テンパイ時のクリーンなラベルで学習 |

---

## 将来への接続

当たり牌推測モデルへの接続想定：

```
hand_inference_v28 の block_head 出力
  → block_logits (134次元)
      [0..20]   順子21 → 面子確定 (not waiting)
      [21..54]  刻子34 → 面子確定
      [55..88]  対子34 → 雀頭 or 単騎待ち
      [89..112] 両面/辺張24 → 2方向待ち or 端待ち
      [113..133] 嵌張21 → 1方向待ち（真ん中）
  → 高スコアのターツ種別から待ち牌を推定
```

v28 の block_head がテンパイ時に正確にターツ種別を予測できれば、
当たり牌推測は「block_logits をフィルタして待ち牌インデックスを逆引き」で実装可能。
