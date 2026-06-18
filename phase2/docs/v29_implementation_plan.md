# hand_inference v29 実装計画

## 背景：v28の失敗から学んだこと

v28では2つの変更を同時に導入した結果、NLL lossへのシャンテン重み追加が逆効果となった：

| 変更 | 結果 |
|------|------|
| block_loss を shanten≤1 限定 | 検証できなかった（NLL重みと混在） |
| NLL loss にシャンテン重み追加 | ❌ 逆効果：endgame eae 0.89→3.37 |

NLLシャンテン重みの問題：終盤でも相手がshanten≥2の場合（重み0.25）に学習信号が弱まり、
残り牌stage_wの効果を打ち消した。基本的な牌枚数分布の学習が壊れた。

## v29 設計方針

**v26をベースに2点だけ変更する（v28の反省を活かした最小差分）：**

1. `block_loss` を shanten≤1 限定（v28から流用、NLL重みなしで単独検証）
2. `get_stage_weights` に副露情報を追加

---

## 変更点詳細

### 変更①：block_loss を shanten≤1 限定

```python
# v26: 全シャンテンに block_loss（高シャンテンは重みで軽減）
block_w = (0.5 ** labels_shanten.float()).clamp(min=0.0625)

# v29: shanten>=2 は完全にゼロ（勾配なし）
block_w = torch.where(
    labels_shanten <= 1,
    (0.5 ** labels_shanten.float()),        # shanten=0: 1.0 / shanten=1: 0.5
    torch.zeros_like(labels_shanten.float()) # shanten>=2: 0.0
)
```

### 変更②：get_stage_weights に副露情報を追加

**特徴量インデックス（確認済み）：**
- `features_p[:, 78]` = n_chi（チー数、生の整数: 0,1,2,3）
- `features_p[:, 79]` = n_pon（ポン数）
- `features_p[:, 80]` = n_kan（カン数）

**考え方：**
副露が発生すると相手の手牌から3枚（カンは4枚）が公開され、不確実性が減少する。
これは局の進行（壁牌消費）と同様の情報量増加をもたらす。
公開された副露牌1枚 ≈ 壁牌2枚消費相当 として傾斜を補正する（係数k=2）。

```python
def get_stage_weights(features_p):
    remaining = (features_p[:, REMAINING_OFFSET] * 70).round()

    # 副露による公開牌数（targetプレイヤーの生カウント）
    n_chi = features_p[:, 78]  # チー数
    n_pon = features_p[:, 79]  # ポン数
    n_kan = features_p[:, 80]  # カン数
    meld_tiles = n_chi * 3 + n_pon * 3 + n_kan * 4  # 公開された牌枚数

    # 有効残り牌数：副露1枚 = 壁牌2枚相当として補正
    effective_remaining = (remaining - meld_tiles * 2).clamp(min=0)

    w = torch.ones(len(features_p), device=features_p.device)
    w[effective_remaining <= 10] = 4.0
    w[(effective_remaining > 10) & (effective_remaining <= 30)] = 3.0
    w[(effective_remaining > 30) & (effective_remaining <= 50)] = 2.0
    return w
```

**補正の効果例：**

| 残り牌 | 副露なし | ポン1回(3枚) | ポン2回(6枚) | カン1回(4枚) |
|--------|---------|------------|------------|------------|
| 40枚   | 2.0     | 2.0 (34枚相当) | 3.0 (28枚相当) | 2.0 (32枚相当) |
| 35枚   | 2.0     | 3.0 (29枚相当) | 3.0 (23枚相当) | 3.0 (27枚相当) |
| 20枚   | 3.0     | 3.0 (14枚相当) | 4.0 ( 8枚相当) | 3.0 (12枚相当) |
| 12枚   | 3.0     | 4.0 ( 6枚相当) | 4.0 ( 0枚相当) | 4.0 ( 4枚相当) |

### 変更なし（v26と同一）

- NLL loss重み：`stage_w` のみ（シャンテン重みなし）
- eval_epoch：`stage_w` のみ（モデル選択基準）
- モデルアーキテクチャ：N_BLOCKS=134（ターツ含む）
- データ：`hand_inference_v26.ndjson` 流用（再生成不要）
- LAMBDA_BLOCK=0.3、ハイパーパラメータ全て同一

---

## ファイル構成

| ファイル | 操作 |
|---------|------|
| `phase2/train/train_hand_inference_v29.py` | v26から複製 → 2箇所変更 |
| `phase2/models/hand_inference/v29/` | 学習時に自動生成 |
| `phase2/data/features/hand_inference_v26.ndjson` | 流用（変更なし） |

---

## 期待される効果

| 指標 | 予測 | 根拠 |
|------|------|------|
| test_eae | v22(6.22)付近を目標 | block_lossノイズ除去 |
| early eae | v26/v27(14.1〜14.3)と同程度 | NLL損失変更なし |
| late/endgame eae | v26/v27並み(0.9〜1.9) | ターツブロック維持 |
| block_f1 | v26(0.290)より改善 | テンパイ時クリーンなラベルで学習 |

---

## 実行コマンド

```bash
python phase2/train/train_hand_inference_v29.py > /tmp/train_v29.log 2>&1 &
```

---

## v28の教訓

NLL lossのシャンテン重みは **不要かつ有害**。
- 終盤でもshanten≥2のサンプルが存在し、そこの学習が弱まる
- stage_w（残り牌ベース）で十分に終盤を重視できる
- シャンテン情報はblock_lossの「適用/不適用の二値判定」にのみ使うべき
