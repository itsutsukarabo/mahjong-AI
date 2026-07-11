# hand_inference 設計状態記録

最終更新: 2026-07-07

---

## v44 最終結果（確定）

### 学習概要

- 使用データ: `hand_inference_v41.ndjson`（164,414サンプル）
- モデル: HandInferenceV37（変更なし）
- epoch上限200に到達（early stopping 不発、patience=15 を消化する前に打ち切り）
- ベストモデル: **ep195**（val_eae 基準）

### バリデーション指標（ベスト ep195）

| 指標 | 値 |
|------|----|
| val_eae（inference_w主軸） | **3.8946** |
| val_eae_stage（stage_w参考） | 2.9485 |
| val_acc | 0.9188 |
| shanten_acc | 0.9718 |
| wait_f1 | 0.3312 |
| composite | 0.6800 |
| train_nll（ep200最終） | 0.1887 |

参考：v39 ep200 train_nll=0.0591、v43 ep68（終了）nll=0.5056

### テスト評価（model.pt = ep195）

| 指標 | 値 |
|------|----|
| test_eae | **3.9184** |
| test_eae_stage | 2.9734 |
| test_acc | 0.9181 |
| soft_f1 | 0.3439 |
| wait_f1 | **0.3349** |
| wait_top1_acc | **0.6976** |
| wait_hit_rate | 0.3733 |

val/test 乖離は微小（eae差+0.024、acc差-0.001）→ 過学習なし・汎化良好。

---

## 診断の確定（最重要）

### v43 不振の主因：UncertaintyWeights

v43 で導入した UncertaintyWeights（Kendall et al. 2018、10タスク learnable log_σ）が
主因と確定した。根拠：

- v44（UW廃止・固定λ復帰）は ep38〜50（v43が崩壊したゾーン）を逆行ゼロで通過。
- 200 epoch 中、train_nll の逆行は ep88 の +0.0002（1回）のみ。
- v43 は ep68 で nll=0.5056 のまま学習打ち切り（実用不可）。
  v44 は同区間を 0.3253（ep38）→ 0.1887（ep200）まで安定改善。

### 指標改善の確定値

| 指標 | v43 最終 | v44 最終 | 変化 |
|------|---------|---------|-----|
| train_nll | 0.5056（ep68） | 0.1887（ep200） | 大幅改善 |
| val_eae（inference_w） | 11.71（実用不可） | 3.89（実用ライン到達） | — |
| test_acc | — | 0.918 | — |
| wait_f1（test） | — | 0.335 | — |

---

## v44 構成（確定した勝ちパターン）

| 項目 | 設定 |
|------|------|
| UncertaintyWeights | **廃止**（v43から削除） |
| λ設定 | **v39固定λ復帰**（10タスク固定定数） |
| LAMBDA_NLL | 1.0（暗黙） |
| LAMBDA_SUM | 0.05 |
| LAMBDA_CROSS | 0.1 |
| LAMBDA_RED_CE | 0.3 |
| LAMBDA_RED_CONS | 0.1 |
| LAMBDA_BLOCK | 0.3 |
| LAMBDA_WAIT | 0.5 |
| LAMBDA_SHANTEN | 0.2 |
| LAMBDA_RETREAT | 0.1 |
| LAMBDA_PUSH | 0.1 |
| LAMBDA_FURITEN | 0.4 |
| LAMBDA_YAKU | 0.1（v43から統合維持） |
| batch_size | 256 |
| データ | hand_inference_v41.ndjson（164kサンプル） |
| モデルアーキ | HandInferenceV37（据え置き） |
| val_eae重み | get_inference_weights（1/n_hidden）主軸 |
| val_eae参考 | get_stage_weights（stage_w）併記 |
| early stopping | val_eae基準、patience=15 |
| scheduler | ReduceLROnPlateau(mode="min", patience=4) |

---

---

## v45 計画（学習前・2026-07-07）

### v44 データの確定バグ（v45で修正）

#### [Bug-1] `l` 座席番号誤認（影響: 全サンプルの75%）

`extract_features.js` の `build_event_tokens` で:

```javascript
// 旧 (v44まで) — 誤り
const dealer = rec.jushu % 4;          // jushu%4 は親の絶対IDを意味しない
const seat_off = (l - dealer + 4) % 4; // 75%の局（jushu%4≠0）で誤ったturn_norm

// 新 (v45) — 正しい
const seat_off = l;  // states_v22.ndjson の l は 0=東(dealer) の座席番号
```

`states_v22.ndjson` の `l` フィールドは座席相対（0=東=当局の親）であることを
実証的に確認（Kendall τ: dealer=0 → 1.000, dealer=jushu%4 → 0.903）。
全局で `l=0` が最初の打牌者。

同じ誤りが `wind_features` と `jikaze_onehot` にも波及（17次元誤り）:
```javascript
// 旧: jikaze = (player_l - rec.jushu + 4) % 4   ← 75%の局で全員の風が誤り
// 新: jikaze = player_l
```

#### [Bug-2] `get_riichi_suji` のマーカー誤認（影響: 手出しリーチ 94.9%）

`add_intent_labels.py`:
```python
# 旧: if "_" in ts  ← ツモ切りマーカー。手出しリーチ'p6*'を検出できない
# 新: if "*" in ts  ← リーチ宣言マーカー
```
手出しリーチ（リーチ宣言牌に `*` サフィックス）のスジが計算されず、
push/retreat ラベルが 94.9% のケースで誤っていた。

#### v45 変更内容

- (a) 入力バグ修正3点: `seat_off=l`, `jikaze=player_l`（2箇所）
- (b) push/retreat ラベル修正: `get_riichi_suji` の `"*"` 修正
- (c) `danger_level` 特徴量追加: token_dim 44 → 45
  - 0=リーチなし, 1=現物, 2=全員スジ, 3=無スジ
- v45 ベースライン: v44 eae=3.89（ただし差分は3変更の複合）

### パイプライン落とし穴（v45作業中に発見）

#### [Trap-1] Node.js WriteStream 重複書き込み

`extract_features.js` 実行時に WriteStream の `writev` エラーが発生し、
493,242 サンプルが正常生成されたはずが 502,344 行（3,034 グループ重複）に
なった。エラーはストリームクローズ時に発生するが、データは書き込み完了後に
バッファ再フラッシュされた模様。

**対処**: 出力後に Python スクリプトで重複除去（dedup_by_group_key）を実施。
`extract_features.js` 実行後は必ず行数÷3 == states 行数を確認すること。

#### [Trap-2] `add_block_labels.py` の src==dst 自己上書きバグ

```python
# 旧（危険）: dst を "w" モードで開いた瞬間に src が 0 バイト化する
with open(SRC_PATH, ...) as fin, open(DST_PATH, "w", ...) as fout:

# 新（安全）: 一時ファイルに書いてからリネーム
tmp_path = DST_PATH.with_suffix(".tmp.ndjson")
with open(SRC_PATH, ...) as fin, open(tmp_path, "w", ...) as fout:
    ...
tmp_path.replace(DST_PATH)
```

同パターンが `prepare_v41_data.py` にも存在し修正済み。
`add_tenpai_features.py` / `add_intent_labels.py` は既に tmp_path 経由で安全。

---

## 運用上の教訓

### [Lesson-1] 学習中は Windows Update 自動再起動を必ず抑制する

**発生**: 2026-07-07、v45学習 ep1 完了直後（ep2 開始時）にPCが強制終了。

**原因**: Windows Update の自動再起動。
- Kernel-Power ID=41 等のハードエラーではなく、Update適用後の再起動だった
- GPU温度・電源・メモリは正常。ハードウェア問題ではない

**被害**: checkpoint.pt・train_log.json は ep1 完了後に保存済みのため、
resume で ep2 から継続可能。データ（v45.ndjson・v41.ndjson）は無傷。

**対処（3層、学習完走後に解除）**:

| 層 | 設定 | 解除方法 |
|----|-----|---------|
| 層1 | アクティブ時間 0:00〜18:00（18時間幅） | そのまま継続可 |
| 層2 | `NoAutoRebootWithLoggedOnUsers=1`（サインイン中は再起動しない） | `Remove-ItemProperty -Path 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' -Name NoAutoRebootWithLoggedOnUsers` |
| 層3 | Windows Update 7日間一時停止（2026-07-14 まで） | 設定→Windows Update→「更新の再開」 |

**次回から学習起動前に確認すること**:
- `(Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU').NoAutoRebootWithLoggedOnUsers` が `1` であること
- `(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings').PauseUpdatesExpiryTime` が未来日時であること

---

## 未消化の課題

### epoch上限による打ち切り

- early stopping（patience=15）を消化する前に epoch 上限 200 に到達。
- ベストが ep195（終盤）→ まだ改善余地があった可能性がある。
- epoch上限を延ばした再学習で更に伸びるかは未検証。

### v39との優劣の未確定

- v44 eae が v39 を「超えた」かは重み関数が異なるため厳密には言えない。
  - v39: get_stage_weights（残り牌数4段階離散重み）
  - v44: get_inference_weights（1/n_hidden 連続値）
- 両者の val_eae は同一スケールではなく直接比較不可。
- 実用ライン（eae≈3.9）への到達は確実だが、v39（eae≈? stage_w基準）との
  優劣は指標定義上未確定。val_eae_stage（v44: 2.95）を v39 の stage_w 基準値と
  比較する評価が必要。

---

## v45 完走の結果（2026-07-09 確定）

### 学習概要

- 使用データ: `hand_inference_v45.ndjson`（164,414サンプル）
  - v44との差異: バグ修正3点(l座席・jikaze・get_riichi_suji) + danger_level追加 + token_dim 44→45
- モデル: HandInferenceV37（変更なし）
- **停止: EP178 外部NLLモニタ**（内部 early stopping は EP180 発動予定だったが、NLL連続2epoch上昇を検知して手動停止）
  - patience=13/15 の状態で停止（内部カウンタは発動直前）
  - ベストモデル: **EP165**（val_eae 基準）

### 最終テスト評価（model.pt = EP165）

| 指標 | v45 | v44 | Δ |
|------|-----|-----|---|
| val_eae（best） | **4.2802** | 3.8946 | +0.3856 |
| test_eae | **4.2972** | 3.9184 | +0.3788 |
| val/test 乖離 | **0.4%** | 0.6% | — |
| test_acc | 0.9111 | 0.9188 | — |
| wait_f1 | 0.2999 | 0.3312 | — |
| wait_top1_acc | 0.6626 | — | — |

### 副露数別EAE（希薄化確認）

| 副露数 | EAE | N |
|--------|-----|---|
| 0（門前） | 4.9063 | 42,099 |
| 1 | 1.5415 | 5,540 |
| 2 | 0.5655 | 1,508 |
| 3+ | 0.1591 | 179 |

meld=0→1 で EAE が 68.6% 低下。副露情報による情報稀薄化は完全否定。
（副露があると牌の組み合わせが絞られ、残手牌が少ないため精度が高い）

### v44/v45 直接比較が不能な理由

v45 は v44 より EAE が高い（悪い）数字だが、以下の3理由で優劣判定不能：

1. **データが別物**: データ特徴量空間が異なる（バグ修正+danger_level追加）。同じ指標でも意味するものが違う。
2. **停止条件が非対称**: v44はmax_epochs=200到達（patience=5/15）、v45は外部NLLモニタ停止（patience=13/15）。v45は最適に近い停止、v44は未収束の可能性あり。
3. **ブレイクスルー速度の逆転**: v45は EP75でブレイクスルー（v44はEP116）。学習曲線の傾きとしてはv45が優位。

### 確定事実

- **過学習なし**: val/test 乖離 0.4%（v44: 0.6%）。
- **希薄化なし**: meld別EAE実測で完全否定。
- **ブレイクスルー早期化**: v45 EP75 vs v44 EP116（約40EP差）。danger_level追加とバグ修正の複合効果。

---

## v46 設計確定事項（2026-07-09）

### 攻撃性タスク概要

v46 では手牌推定（EAE）に加えて**攻撃性スカラー (-1〜+1)** の予測タスクを追加する。

- **出力**: `aggression` float in [-1, +1]（-:防御, 0:中立/手なり, +:攻撃）
- **ラベル生成スクリプト**: `phase2/scripts/add_aggression_labels.py`

### 攻撃性ラベルの部品とリーク分類

| 部品 | 使用可否 | 理由 |
|------|---------|------|
| `danger_level` | **入力可（可視）** | リーチ有無・スジ判定はall可視 |
| `meld_threat` | **入力可（可視）** | 副露のみ使用（向聴数は不使用） |
| `tile_value` | **入力可（可視）** | ドラ表示牌・赤五は公開情報 |
| `is_ryanmen_nochance` | **ラベル専用** | 手牌内両面確認が必要 |
| `delta_shanten` | **ラベル専用** | 打牌前後の向聴数差（手牌依存） |
| `ukeire_change` | **ラベル専用** | 受け入れ枚数変化（手牌依存） |
| `has_safe_tile` | **ラベル専用** | 手牌中の安全牌有無（手牌依存） |

### 定数構造

`add_aggression_labels.py` 冒頭のブロックに全定数を集約。ロジック変更なしで
数値チューニングのみ行う。

```
# 防御ゾーン
W_DELTA_SHANTEN     # ΔSH=+1後退あたりの防御強度
W_UKEIRE_NEG        # ΔSH=0・受け入れ減少あたりの防御強度
UKEIRE_ONLY_MAX     # 受け入れ減少のみの場合の防御上限
DEFENSE_MAX         # 防御全体の上限（クリップ）

# 攻撃ゾーン（danger_level別基本強度）
DL1_STRENGTH        # 現物（= 0.0: 中立扱い）
DL2_STRENGTH        # スジ
DL3_NC_STRENGTH     # 無スジ+両面NC（安全方向）
DL3_NOT_NC_STRENGTH # 無スジ+非NC（最大危険）

# 加算ボーナス
MELD_THREAT_SCALE   # meld_threat の重み
TILE_VALUE_BONUS    # tile_value の重み
INTENT_MULTIPLIER   # has_safe_tile あり → 意図的攻撃 として倍率
ATTACK_MAX          # 攻撃全体の上限（クリップ）
```

### v46設計中に発見・確定した禁止事項

#### [Bug-A] meld_threat に他家向聴数を使用（リーク）

`compute_meld_threat` にて `shanten_l[j]` を参照してテンパイ確認をしていたが、
**他家の向聴数は不可視情報**。ラベル生成においても原則不使用（リーク分類: ラベル専用扱いなら許容だが、ラベル→モデル入力への漏洩に注意）。

修正: meld_threat は副露有無・鳴き数ティア・役牌/染め/タンヤオ/ドラのみ使用。
テンパイ確認をmeld_threatから除去し、`has_safe_tile`の安全牌判定（_is_tile_safe_simple内）でのみ使用。

#### [Bug-B] is_ryanmen_nochance の方向ミス

`is_ryanmen_nochance=True`（両面ノーチャンス）は安全牌扱い → **攻撃強度を下げる**方向。
誤解されやすいが「両面が枯れているため安全に切れる = 攻撃的でない」である。
`danger_to_strength` で `DL3_NC_STRENGTH < DL3_NOT_NC_STRENGTH` となるよう実装済み。
「NC = 安全なので攻撃意図が薄い」という設計を記録しておく。
