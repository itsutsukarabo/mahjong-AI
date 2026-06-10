# hand_inference v15 実装計画

作成日: 2026-06-10

---

## 実験履歴サマリー（v11〜v14）

### 各バージョンの結果

| バージョン | 主な変更点 | val_eae | recall | pred_total_mae | エポック数 | 結論 |
|---|---|---|---|---|---|---|
| v10（基準） | NLL + MSE合計(λ=0.05) | — | 68.5% | 3.0枚 | — | 最高性能 |
| v11 | NLL損失に変更 + block heads(λ=0.3) | 12.60 | 32.2% | 7.3枚 | 24 | block headsが競合 |
| v12 | 制約NLL + MSE合計 + block heads | 14.69 | 18.9% | 9.3枚 | 19 | epoch13で勾配爆発 |
| v13 | NLL + MSE合計 + 勾配clip + block heads | 14.17 | 23.7% | 8.6枚 | 33 | 安定したが低水準 |
| v14 | v10アーキ + val_eae early stopping | **6.64** | 61.9% | 3.9枚 | 116 | block headsが原因確定 |

### 主な知見

1. **block heads（刻子/順子/対子、λ=0.3）が主要因**: v11〜v13 は全て v10 を大幅に下回った。v14（block heads なし）で val_eae が 12〜15 → 6.64 に大幅改善。

2. **val_eae を early stopping 指標にすることで長く学習できた**: v14 は 116 エポックまで改善し続けた（v11〜v13 は 19〜33 エポックで停止）。

3. **EAE を学習損失に使うと「動く目標問題」が発生**: λ（ラグランジュ乗数）を定数扱いで勾配更新するが、次ステップで λ が変化するため val_eae が epoch 1 で固着。

4. **3人独立推測の限界**: 同一牌を複数のプレイヤーが「持っている」と同時に予測できてしまう。v15 で解消する。

### v14 バケット別精度（テスト）

| ステージ | acc | eae |
|---|---|---|
| endgame（残り 0〜10枚） | 96.7% | 2.4 |
| late（11〜30枚） | 96.4% | 2.7 |
| mid（31〜50枚） | 91.2% | 6.0 |
| early（51枚〜） | 74.3% | 14.2 |

---

## v15 設計方針

### 目的

3人の相手手牌を**同時に推測・学習**することで、以下を実現する：

- 同一牌を複数プレイヤーが持つという矛盾を損失で直接ペナルティ化
- プレイヤー間の牌の競合をAttentionで学習（「Aが持つならBは持ちにくい」）

### アーキテクチャ概要

| 項目 | v14（現行） | v15（新規） |
|---|---|---|
| 入力 | (B, 374) × 3回 | (B, 3, 374) × 1回 |
| Transformerトークン数 | 34 | **102**（34 × 3プレイヤー） |
| 出力 | (B, 34, 5) | (B, 3, 34, 5) |
| early stopping指標 | val_eae（1人分） | **val_eae（3人合計）** |

---

## Step 1: `extract_features.js` の変更

### 現在の出力形式（1ゲーム状態 → 3サンプル）

```json
{ "features": [...374...], "label_hand": [...34...], "label_red": [0,0,0] }  // target=right
{ "features": [...374...], "label_hand": [...34...], "label_red": [0,1,0] }  // target=across
{ "features": [...374...], "label_hand": [...34...], "label_red": [0,0,0] }  // target=left
```

### v15 の出力形式（1ゲーム状態 → 1サンプル）

```json
{
  "features":   [[...374...], [...374...], [...374...]],
  "label_hand": [[...34...],  [...34...],  [...34...]],
  "label_red":  [[0,0,0],     [0,1,0],     [0,0,0]]
}
```

374次元の特徴量設計は**変更なし**。1レコードが3人分をまとめた構造になる。

### 変更点

`make_hand_inference_sample(state, target_l)` を3回呼ぶ処理を、
`make_hand_inference_sample_all(state)` として1回で3人分まとめて出力するよう変更。

データ再生成が必要（`node extract_features.js` を再実行）。

---

## Step 2: モデルアーキテクチャ（`train_hand_inference_v15.py`）

### モデルクラス

```python
class HandInferenceV15(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers,
                 n_pai=34, n_count_cls=5, n_players=3, dropout=0.1):
        super().__init__()
        self.n_pai       = n_pai
        self.n_count_cls = n_count_cls
        self.n_players   = n_players

        # グローバルエンコーダ（3プレイヤー間で重み共有）
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

        # 埋め込み
        self.tile_embed   = nn.Embedding(n_pai,     d_model)  # タイルID
        self.player_embed = nn.Embedding(n_players, d_model)  # プレイヤーID
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        # Transformer（102トークン）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力ヘッド（3プレイヤー間で重み共有）
        self.head     = nn.Linear(d_model, n_count_cls)
        self.red_head = nn.Linear(d_model, 2)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("player_ids",   torch.arange(n_players))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))

    def forward(self, x):
        # x: (B, 3, 374)
        B, P, F = x.shape

        # 1. グローバルエンコード（重み共有）
        g = self.global_encoder(x.reshape(B * P, F)).reshape(B, P, -1)  # (B, 3, d_model)

        # 2. 102トークンを構築
        tile_emb   = self.tile_embed(self.tile_ids)              # (34, d_model)
        player_emb = self.player_embed(self.player_ids)          # (3,  d_model)
        vis        = x[:, :, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai]  # (B, 3, 34)
        vis_emb    = self.visible_proj(vis.unsqueeze(-1))        # (B, 3, 34, d_model)

        tokens = (g.unsqueeze(2)                    # (B, 3,  1, d_model)
                + tile_emb                          # (   34,    d_model)
                + player_emb.view(1, P, 1, -1)      # (1, 3,  1, d_model)
                + vis_emb)                          # (B, 3, 34, d_model)
        tokens = tokens.reshape(B, P * self.n_pai, -1)  # (B, 102, d_model)

        # 3. Transformer（102トークン間でAttention）
        out = self.transformer(tokens)                   # (B, 102, d_model)
        out = out.reshape(B, P, self.n_pai, -1)          # (B, 3, 34, d_model)

        # 4. 枚数分類ヘッド
        logits_raw = self.head(out)                      # (B, 3, 34, 5)

        # 5. マスク（visible枚数を超える予測を -inf に）
        vis_raw          = (vis * 4).round().long().clamp(0, 4)     # (B, 3, 34)
        hidden_remaining = (4 - vis_raw).clamp(min=0)               # (B, 3, 34)
        mask = self.count_range > hidden_remaining.unsqueeze(-1)    # (B, 3, 34, 5)
        logits = logits_raw.masked_fill(mask, float('-inf'))

        # 6. 赤牌ヘッド（5m/5p/5s）
        red_tokens = out[:, :, self.red_tile_idx, :]                # (B, 3, 3, d_model)
        red_logits = self.red_head(red_tokens)                      # (B, 3, 3, 2)

        return logits, logits_raw, red_logits
```

---

## Step 3: 損失関数

```python
count_vals = torch.arange(5, device=device, dtype=torch.float32)

# ① NLL（3プレイヤー合計平均）
loss_nll = F.cross_entropy(logits_raw.reshape(-1, 5), labels.reshape(-1))

# ② MSE合計制約（プレイヤーごと: E[Σ c_i] ≈ k）
probs    = F.softmax(logits_raw, dim=-1)              # (B, 3, 34, 5)
pred_sum = (probs * count_vals).sum(-1)               # (B, 3, 34)
loss_sum = F.mse_loss(pred_sum.sum(-1), labels.float().sum(-1))  # (B, 3) vs (B, 3)

# ③ タイル横断制約（3人合計 ≤ 残り枚数）← v15 の新規追加
pred_tile_total = pred_sum.sum(dim=1)                 # (B, 34): 3人の期待値合計
visible_counts  = x[:, 0, VISIBLE_OFFSET:VISIBLE_OFFSET+34] * 4  # (B, 34)
max_hidden      = (4 - visible_counts).clamp(min=0)
loss_cross      = F.relu(pred_tile_total - max_hidden).mean()

# ④ 赤牌損失
loss_red_ce = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))

# 合計
loss = loss_nll + LAMBDA_SUM * loss_sum + LAMBDA_CROSS * loss_cross + LAMBDA_RED_CE * loss_red_ce
```

### ハイパーパラメータ（初期値）

| パラメータ | 値 | 備考 |
|---|---|---|
| LAMBDA_SUM | 0.05 | v10/v14 と同値 |
| LAMBDA_CROSS | 0.1 | タイル横断制約（調整余地あり） |
| LAMBDA_RED_CE | 0.3 | v14 と同値 |
| GRAD_CLIP_NORM | 1.0 | v14 と同値 |

---

## Step 4: val_eae（3人合計、early stopping 指標）

```python
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_wsum = 0.0
    total_w    = 0.0
    correct    = 0
    total      = 0

    for features, labels, labels_red in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _, _ = model(features)                    # logits: (B, 3, 34, 5)

        # 3プレイヤー分の EAE を合計
        for p in range(3):
            k_p    = labels[:, p].float().sum(dim=-1).long()
            probs_p = constrained_softmax_probs(logits[:, p], k_p)
            stage_w = get_stage_weights(features[:, p])
            total_wsum += weighted_eae(probs_p, labels[:, p], stage_w).item() * stage_w.sum().item()
            total_w    += stage_w.sum().item()

        # accuracy（3プレイヤー合計）
        k_all   = labels.float().sum(dim=-1).long()       # (B, 3)
        for p in range(3):
            probs_p = constrained_softmax_probs(logits[:, p], k_all[:, p])
            preds_p = probs_p.argmax(dim=-1)
            correct += (preds_p == labels[:, p]).sum().item()
            total   += labels[:, p].numel()

    val_eae = total_wsum / total_w if total_w > 0 else float('inf')
    return val_eae, correct / total
```

---

## Step 5: データ形式とデータセット

### HandInferenceDataset

```python
class HandInferenceDataset(Dataset):
    def __init__(self, features, labels, labels_red):
        # features:   (N, 3, 374)
        # labels:     (N, 3, 34)
        # labels_red: (N, 3, 3)
        self.features   = torch.as_tensor(features,   dtype=torch.float32)
        self.labels     = torch.as_tensor(labels,     dtype=torch.long).clamp(0, 4)
        self.labels_red = torch.as_tensor(labels_red, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.labels_red[idx]
```

---

## 実装順序

1. **`extract_features.js` 修正** — `make_hand_inference_sample_all()` 実装
2. **データ再生成** — `node extract_features.js` 再実行（時間がかかる）
3. **`train_hand_inference_v15.py` 作成**
4. **学習実行**（102トークンでGPU使用量増加に注意）
5. **`ai_phase2.js` 更新** — 3人同時推論に対応

---

## 注意事項

- 102トークンのTransformerはv14（34トークン）の約3倍の計算量。GPU発熱に注意。
- `BatchNorm1d` は `x.reshape(B*P, F)` 形式で入力すること（B×P=バッチとして処理）。
- visible_counts の正規化スケール確認（`* 4` で0〜4枚に戻す）。
- LAMBDA_CROSS は学習初期は小さく（0.1）、効果を見ながら調整。
