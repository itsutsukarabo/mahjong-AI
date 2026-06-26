# v38 ONNX エクスポート計画

## 前提・注意点

v38モデルは v32 と入力構造が異なる（**2入力**）。

| | v32（現在ブラウザ） | v38（新） |
|---|---|---|
| 固定特徴量 | 757次元 | 695次元 |
| トークン列 | なし | 44次元/トークン × 最大144トークン（別テンソル） |
| ブラウザ関数 | `make_hi_features_v29` | `make_hi_features_v38`（新規作成） |

---

## Step 1: 学習完了確認

v38学習が early stopping で終了したことを確認する。

```
phase2/models/hand_inference/v38/model.pt       ← best val_eae
phase2/models/hand_inference/v38/model_best_wait.pt ← best wait_f1
```

→ `model_best_wait.pt`（wait_f1最良）をエクスポート対象とする。

---

## Step 2: ONNX エクスポートスクリプト作成

**ファイル**: `phase2/scripts/export_onnx_v38.py`

```python
# 入力テンソル（バッチ = 1, プレイヤー数 = 3）
#   features:     (1, 3, 695)        固定特徴量
#   tokens:       (1, 3, T_max, 44)  discardトークン列（T_max=144でパディング）
#   token_mask:   (1, 3, T_max)      パディング位置 True
#
# 出力テンソル
#   logits:       (1, 3, 34, 5)      手牌枚数分布
#   red_logits:   (1, 3, 3, 2)       赤五分布
#   block_logits: (1, 3, 134)        ブロック分布
#   wait_logits:  (1, 3, 113)        待ち形分布
```

### エクスポート手順

```bash
C:\ml\venv\Scripts\python.exe phase2/scripts/export_onnx_v38.py \
    --model phase2/models/hand_inference/v38/model_best_wait.pt \
    --out   phase2/models/hand_inference/v38/model.onnx \
    --opset 17
```

### 検証（PyTorch出力 vs ONNX出力の誤差確認）

```bash
C:\ml\venv\Scripts\python.exe phase2/scripts/validate_onnx_v38.py \
    --model_pt  phase2/models/hand_inference/v38/model_best_wait.pt \
    --model_onnx phase2/models/hand_inference/v38/model.onnx
# 目標: max abs error < 1e-4
```

---

## Step 3: ブラウザ側特徴量エンジンの更新

**ファイル**: `phase2/browser/ai_phase2.js`

### 3-1. `make_hi_features_v38()` 新規作成

`make_hi_features_v29()` の代替。695次元固定特徴量を生成。

```
feature_offsets.py の HI_* 定数に完全準拠:
  meld(38) + riichi(1) + score(11) + game(9) + self_meld(38)
  + visible(34) + red_disc_sig(3) + red_vis(3)
  + pon_pass(34) + wind(5) + chi_pass(34) + chi_called(34)
  + dora(34) + other1_meld(38) + other2_meld(38)
  + self_pon_pass(34) + o1_pon_pass(34) + o2_pon_pass(34)
  + self_chi_pass(34) + o1_chi_pass(34) + o2_chi_pass(34)
  + lizhibang(1) + self_jikaze(4) + o1_jikaze(4) + o2_jikaze(4)
  + self_chi_called(34) + o1_chi_called(34) + o2_chi_called(34)
  + yaku_probs(21) + tenpai_prob(1)
  = 695次元
```

### 3-2. `make_discard_tokens_v38()` 新規作成

44次元/トークン × 最大144トークンのシーケンスを生成。

```
各トークン（44次元）:
  tile_onehot(34) + turn_norm(1) + is_tsumogiri(1)
  + is_riichi_decl(1) + is_red_five(1)
  + player_role(4)  [is_target, is_self, is_other1, is_other2]
  + event_is_pass_pon(1) + event_is_pass_chi(1)
```

### 3-3. `run_phase2()` 内の推論呼び出しを更新

```javascript
// v38: 2テンソル入力
const feat_tensor  = new ort.Tensor('float32', flat_features, [1, 3, 695]);
const token_tensor = new ort.Tensor('float32', flat_tokens,   [1, 3, 144, 44]);
const mask_tensor  = new ort.Tensor('bool',    flat_mask,     [1, 3, 144]);

const out = await sessions.hand_inference.run({
    features:   feat_tensor,
    tokens:     token_tensor,
    token_mask: mask_tensor,
});
```

### 3-4. モデルパス更新

```javascript
// before
['hand_inference', MODEL_BASE + 'hand_inference/v32/model.onnx'],
// after
['hand_inference', MODEL_BASE + 'hand_inference/v38/model.onnx'],
```

---

## Step 4: デプロイ・動作確認

1. ONNX ファイルを `tmp_clone/dist/models/hand_inference/v38/model.onnx` に配置
2. ブラウザで牌譜を開き、AI解析ボタン → コンソールに `AI Phase2: loaded hand_inference` が出ることを確認
3. 手牌推定パネルが表示されることを確認
4. 従来の v32 表示と比較して推定が改善されていることを目視確認

---

## チェックリスト

- [ ] v38 学習完了確認
- [ ] `export_onnx_v38.py` 作成・実行
- [ ] `validate_onnx_v38.py` 実行（誤差 < 1e-4）
- [ ] `make_hi_features_v38()` 実装（695次元一致確認）
- [ ] `make_discard_tokens_v38()` 実装（44次元/トークン確認）
- [ ] `run_phase2()` 推論呼び出し更新
- [ ] ブラウザ動作確認
