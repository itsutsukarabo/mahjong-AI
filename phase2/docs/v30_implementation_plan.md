# v30 実装計画: 推論デコーダー + REINFORCE による待ち牌推定

最終更新: 2026-06-19

---

## 概要・設計方針

### 問題の所在

v29 の block_logits は block_recall=0.69 と手牌構造を十分捉えているが、
134 個の独立した sigmoid には「4面子1雀頭」という**大局的な整合性がない**。
argmax 手牌が聴牌形になる保証もなく、待ち牌を正しく計算できていない。

### v30 の方針

```
現在:
  block_logits (134, 独立) → 表示のみ（整合性なし）

v30:
  block_logits (134, スコア) → 制約付きビームサーチ
                                  → 有効な聴牌分解
                                  → 待ち牌確率
                            ↑
                    REINFORCE で待ち牌F1を最大化するよう学習
```

- **フェーズ1**: 推論デコーダーを推論時のみ追加（学習変更なし）
  → 待ち牌 F1 のベースラインを計測
- **フェーズ2**: REINFORCE で block_logits を end-to-end 学習 (v30)
  → 評価軸を block_f1 → **待ち牌 F1** に切り替える

適用範囲: shanten ≤ 1（v29 の block_loss と同一マスク）

---

## フェーズ1: 推論デコーダー（学習変更なし）

### 1.1 ブロック→牌インデックス マッピング

```python
def block_to_tiles(block_idx):
    """
    ブロックインデックス → 構成牌の 0-based インデックスリスト
    戻り値の牌は重複あり (刻子なら [i, i, i])
    """
    if block_idx < 21:                          # 順子 [0..20]
        s, n = divmod(block_idx, 7)
        base = s * 9
        return [base+n, base+n+1, base+n+2]

    if block_idx < 55:                          # 刻子 [21..54]
        i = block_idx - 21
        return [i, i, i]

    if block_idx < 89:                          # 対子 [55..88]
        i = block_idx - 55
        return [i, i]

    if block_idx < 113:                         # 両面/辺張 [89..112]
        idx = block_idx - 89
        s, n = divmod(idx, 8)
        base = s * 9
        return [base+n, base+n+1]

    # 嵌張 [113..133]
    idx = block_idx - 113
    s, n = divmod(idx, 7)
    base = s * 9
    return [base+n, base+n+2]


def block_to_waits(block_idx):
    """
    ターツ/単騎ブロック → 待ち牌の 0-based インデックスリスト
    面子・対子は [] を返す（wait を生まない）
    """
    if block_idx < 55:                          # 順子・刻子
        return []

    if block_idx < 89:                          # 対子 → 単騎待ち
        i = block_idx - 55
        return [i]                              # 同牌を引いて和了

    if block_idx < 113:                         # 両面/辺張
        idx = block_idx - 89
        s, n = divmod(idx, 8)
        base = s * 9
        if n == 0:
            return [base + 2]                   # 辺張 12 → 3待ち
        elif n == 7:
            return [base + 6]                   # 辺張 89 → 7待ち
        else:
            return [base+n-1, base+n+2]         # 両面

    # 嵌張
    idx = block_idx - 113
    s, n = divmod(idx, 7)
    base = s * 9
    return [base + n + 1]                       # 中間牌
```

### 1.2 デコーダーの状態と遷移

**探索する聴牌形の種類:**

| パターン | 構成 | 待ちの種類 |
|---------|------|-----------|
| 標準 | 面子×3 + 雀頭×1 + ターツ×1 | 両面/辺張/嵌張 |
| 単騎 | 面子×4 + 単牌×1 | 単騎 |
| 双碰 | 面子×3 + 対子×2 | 双碰（どちらか） |

**状態:**

```python
@dataclass
class DecoderState:
    remaining: np.ndarray   # shape (34,) 残り牌枚数
    mentsu: list[int]       # 選択済み面子ブロックインデックス
    head: int | None        # 雀頭ブロックインデックス
    tatsu: int | None       # ターツ/単騎ブロックインデックス (None=未確定)
    log_score: float        # Σ log σ(block_logits[chosen])
```

**遷移ルール:**

```
Step A: len(mentsu) < 4 の間 → 順子/刻子ブロックを追加
Step B: head is None         → 対子ブロックを雀頭として確定
Step C: tatsu is None        → 両面/辺張/嵌張/対子(単騎) を追加
Step D: 終端チェック         → remaining が all-zero ならば有効
```

双碰は Step B で1つ目の対子を head に、Step C で残り対子を tatsu に割り当てる。

### 1.3 ビームサーチ アルゴリズム

```python
def beam_search_decoder(block_logits, initial_counts, beam_width=20, prob_threshold=0.03):
    """
    block_logits : (134,) float  — block_head の raw logit
    initial_counts : (34,) int   — 推定手牌枚数 (probs_per_tile argmax でよい)
    """
    probs = sigmoid(block_logits)
    log_probs = np.log(probs + 1e-9)

    # 確率閾値で候補を絞り込む（計算量削減）
    candidate_blocks = np.where(probs > prob_threshold)[0]

    initial = DecoderState(
        remaining=initial_counts.copy(),
        mentsu=[], head=None, tatsu=None, log_score=0.0
    )
    beam = [initial]
    completed = []

    for _ in range(6):   # 面子4 + 雀頭1 + ターツ1 = 最大6ステップ
        next_beam = []
        for state in beam:
            for b in candidate_blocks:
                tiles = block_to_tiles(b)
                if not can_use(state.remaining, tiles):
                    continue
                new_state = apply_block(state, b, tiles, log_probs[b])
                if new_state is None:
                    continue   # 遷移ルール違反
                if is_terminal(new_state):
                    completed.append(new_state)
                else:
                    next_beam.append(new_state)

        # スコア上位 beam_width を残す
        next_beam.sort(key=lambda s: -s.log_score)
        beam = next_beam[:beam_width]
        if not beam:
            break

    completed.sort(key=lambda s: -s.log_score)
    return completed[:beam_width]
```

### 1.4 待ち牌確率の集計

```python
def compute_wait_probs(beam_results):
    """
    beam_results: list of DecoderState (log_score 降順)
    戻り値: dict {tile_idx: prob}
    """
    if not beam_results:
        return {}

    scores = np.array([s.log_score for s in beam_results])
    weights = softmax(scores)   # ビーム内での確率化

    wait_probs = np.zeros(34)
    for state, w in zip(beam_results, weights):
        for t in get_wait_tiles(state):
            wait_probs[t] += w

    return {i: float(wait_probs[i]) for i in np.where(wait_probs > 0.01)[0]}
```

### 1.5 新評価指標: 待ち牌 F1

**待ち牌ラベルの生成:**

訓練データの tile count ラベルから推論時に計算する。

```python
def derive_wait_labels(tile_count_labels, melds, shanten_labels):
    """
    shanten ≤ 1 のサンプルのみ待ち牌ラベルを生成
    """
    for i, (counts, sh) in enumerate(zip(tile_count_labels, shanten_labels)):
        if sh > 1:
            yield i, None
            continue
        hand_str = counts_to_hand_str(counts, melds[i])
        sp = Shoupai.fromString(hand_str)
        waits = Util.tingpai(sp) if sh == 0 else estimate_1shanten_waits(sp)
        yield i, waits
```

**F1 計算:**

```python
def wait_tile_f1(pred_waits: set, true_waits: set):
    if not true_waits:
        return 1.0 if not pred_waits else 0.0
    if not pred_waits:
        return 0.0
    tp = len(pred_waits & true_waits)
    precision = tp / len(pred_waits)
    recall    = tp / len(true_waits)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

**ファイル:** `phase2/scripts/eval_wait_tiles.py`

---

## フェーズ2: REINFORCE による end-to-end 学習 (v30)

### 2.1 なぜ REINFORCE か

デコーダーのブロック選択は離散操作であり、
「待ち牌が正解かどうか」という報酬信号を通常の逆伝播で
block_logits に届けることができない。

REINFORCE（policy gradient）は離散決定の系列に対して
報酬の対数確率勾配を推定する標準的な手法：

```
J(θ) = E_π_θ [R(trajectory)]

∇_θ J(θ) ≈ (R - b) × Σ_t ∇_θ log π_θ(a_t | valid_t)

θ        : block_head + Transformer の重み
π_θ(a|V) : valid ブロック V の中から a を選ぶ確率
R        : 待ち牌 F1（報酬）
b        : ベースライン（分散削減）
```

この勾配は `block_logits → block_head → Transformer → θ` と
連鎖律で流れるため、通常の `loss.backward()` で計算できる。

### 2.2 報酬関数

```python
def reward(pred_waits: set, true_waits: set) -> float:
    return wait_tile_f1(pred_waits, true_waits)  # [0.0, 1.0]
```

完全一致報酬（0/1）より F1 の方が部分点がつき学習が安定する。

| 予測 | 正解 | R |
|------|------|---|
| {m1, m4} | {m1, m4} | 1.00 |
| {m1}     | {m1, m4} | 0.67 |
| {m1}     | {m4}     | 0.00 |
| {}       | {m1, m4} | 0.00 |

### 2.3 サンプリング付きデコーダー（学習時）

学習時はビームサーチ（greedy）ではなく確率的サンプリングを使う。

各ステップ t で valid ブロック集合 V_t から：

```python
def sample_action(block_logits, valid_blocks, temperature=1.0):
    logits = block_logits[valid_blocks] / temperature
    probs  = softmax(logits)           # valid ブロック間で正規化
    action = np.random.choice(valid_blocks, p=probs)
    log_pi = np.log(probs[action == valid_blocks][0])
    return action, log_pi
```

軌跡全体の log 確率：

```python
log_pi_total = sum(log_pi_t for each step t)
```

### 2.4 REINFORCE 損失

```python
def reinforce_loss(block_logits_batch, trajectories, rewards, baseline):
    """
    block_logits_batch : (B, 134) Tensor — block_head 出力
    trajectories       : list of [(valid_t, action_t)] per sample
    rewards            : (B,) — wait_tile_F1
    baseline           : float — EMA of recent rewards
    """
    loss = 0.0
    for i, (traj, R) in enumerate(zip(trajectories, rewards)):
        log_pi = 0.0
        for valid_t, action_t in traj:
            # valid ブロックの logit を抽出して正規化
            logits_valid = block_logits_batch[i][valid_t]
            log_probs    = log_softmax(logits_valid)
            action_mask  = (valid_t == action_t)
            log_pi       += log_probs[action_mask].squeeze()

        loss += -(R - baseline) * log_pi   # REINFORCE

    return loss / len(trajectories)
```

### 2.5 ベースライン（分散削減）

指数移動平均（EMA）で実装。シンプルで十分安定する。

```python
class EMABaseline:
    def __init__(self, alpha=0.99, init=0.5):
        self.value = init
        self.alpha = alpha

    def update(self, R_batch):
        mean_R = np.mean(R_batch)
        self.value = self.alpha * self.value + (1 - self.alpha) * mean_R
        return self.value
```

不安定な場合は value head（学習可能ベースライン）に切り替える。

### 2.6 損失の合算と学習スケジュール

```python
# 総損失
L = L_nll                                           # tile count (常時)
  + λ_block     * L_block                           # block BCE (shanten ≤ 1)
  + λ_reinforce * L_reinforce                       # REINFORCE (shanten ≤ 1)

# λ スケジュール
epoch  1-10:  λ_reinforce = 0.0     (ウォームアップ: 既存損失で収束させる)
epoch 11-30:  λ_reinforce = 0.01    (REINFORCE を小さく導入)
epoch 31-  :  λ_reinforce = 0.05    (本格化)
```

### 2.7 N サンプリングによる分散削減

同一サンプルで N=4 軌跡をサンプリングし損失を平均する。

```python
for _ in range(N):
    traj, log_pi = sample_trajectory(block_logits, ...)
    R = reward(get_waits(traj), true_waits)
    losses.append(-(R - baseline) * log_pi)

L_reinforce = mean(losses)
```

### 2.8 実装上の注意点

**勾配クリッピング（必須）:**
REINFORCE は勾配分散が大きいため、クリッピングを適用する。

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**有効ブロックの絞り込み（計算量削減）:**
prob > 0.03 のブロックのみサンプリング候補に含める。
ゼロ確率ブロックを除外するだけで探索空間が大幅に縮小する。

**shanten ≤ 1 マスク:**
REINFORCE 損失は shanten ≤ 1 サンプルにのみ適用。
高シャンテンのサンプルでは L_reinforce = 0 とする。

---

## モデルアーキテクチャ (v29 → v30)

| コンポーネント | v29 | v30 |
|--------------|-----|-----|
| Transformer backbone | 変更なし | 変更なし |
| count_head | 変更なし | 変更なし |
| red_head | 変更なし | 変更なし |
| block_head | 変更なし | 変更なし（出力を decoder に渡す） |
| **DecodeBeam** | なし | **追加** |
| **SampleDecoder** | なし | **追加（学習時）** |
| **EMABaseline** | なし | **追加** |

学習パラメータ数は変化なし。デコーダーは微分不要のアルゴリズムモジュール。

---

## 評価指標の変更

| 指標 | 役割 | v29 結果 |
|------|------|---------|
| test_eae | 枚数精度（主指標、退行検出） | 5.895 |
| block_f1 | 参考値（廃止ではないが主軸ではない） | 0.265 |
| **wait_f1** | **新主指標（shanten≤1 のみ）** | フェーズ1で計測 |
| wait_precision | 補助 | — |
| wait_recall | 補助 | — |

**目標:** wait_f1 > 0.55（フェーズ1 ベースライン後に再設定）

---

## ファイル構成

```
phase2/
├── scripts/
│   ├── decode_hand.py             # デコーダー本体 (beam search + sampling)
│   └── eval_wait_tiles.py         # 待ち牌 F1 評価スクリプト
├── train/
│   └── train_hand_inference_v30.py
└── docs/
    └── v30_implementation_plan.md  # 本文書
```

---

## 実装順序

### Step 1: decode_hand.py（1〜2日）
- `block_to_tiles`, `block_to_waits` 実装
- `DecoderState` + `beam_search_decoder` 実装
- `compute_wait_probs` 実装
- 単体テスト: 既知の聴牌形でデコーダー動作確認

### Step 2: eval_wait_tiles.py（0.5日）
- 訓練データから待ち牌ラベルを生成
- v29 の block_logits に対してデコーダーを走らせ wait_f1 を計測
- **ここで「デコーダーだけでどこまで行けるか」を確認**

### Step 3: train_hand_inference_v30.py（2〜3日）
- EMABaseline 実装
- SampleDecoder 実装（学習時 stochastic）
- REINFORCE 損失関数実装
- λ スケジューラー実装
- 既存学習ループに組み込み（既存損失は変更なし）

### Step 4: 実験・調整（継続）
- ウォームアップ10エポック後に REINFORCE 導入
- wait_f1 推移を監視
- 必要に応じて N サンプリング数・温度τ・λ を調整
- test_eae の退行がないことを確認

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| REINFORCE の分散が大きく学習不安定 | N=4 サンプリング、勾配クリッピング、λを小さく始める |
| test_eae が退行する | λ_reinforce を小さく保つ、L_nll の重みを下げない |
| デコーダーが聴牌形を見つけられない | prob_threshold を下げる、beam_width を増やす |
| value_head を持たないので baseline が粗い | EMA で不十分なら小さな value head を追加 |
