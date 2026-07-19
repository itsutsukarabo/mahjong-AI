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

---

## v46 完走の結果（2026-07-18 確定）

### 学習概要

- 使用データ: `hand_inference_v46.ndjson`（164,414サンプル）
  - v45との差異: `aggression` スカラー回帰ヘッド追加（push/retreat → [-1,+1]）
- モデル: HandInferenceV37 + aggression_head（Linear(d_model, 1) + Tanh）
- **ep200 到達**（early stopping 不発、best は途中で確定）
- ベストモデル: **ep185**（val_eae 基準、best_eae=4.0983）

### バリデーション指標（best ep185）

| 指標 | v46 | v45 | Δ |
|------|-----|-----|---|
| val_eae（best） | **4.0983** | 4.2802 | **-0.1819**（改善） |
| wait_f1 | 0.820（途中）| 0.2999 | — |

### テスト評価（model.pt = ep185）

| 指標 | v46 | v45 | v44 |
|------|-----|-----|-----|
| test_eae | **4.1170** | 4.2972 | 3.9184 |
| test_eae_stage | 3.1501 | — | — |
| test_acc | 0.9159 | 0.9111 | 0.9188 |
| wait_f1 | 0.3149 | 0.2999 | 0.3312 |
| wait_top1_acc | 0.6616 | 0.6626 | 0.6976 |
| agg_all_mae | **0.1501** | — | — |
| agg_pressure_mae | **0.1800** | — | — |
| agg_sign_acc | **0.8353** | — | — |

### v46 新規タスク（aggression）評価

| 指標 | 値 | 意味 |
|------|-----|-----|
| agg_all_mae | 0.1501 | 全局面の平均絶対誤差（[-1,+1]スケール） |
| agg_pressure_mae | 0.1800 | 圧力局面（攻防どちらかが明確）限定MAE |
| agg_sign_acc | 0.8353 | 符号一致率（攻撃 vs 防御の方向正解率）|

- 符号一致率 83.5% は実用レベル（ランダム=50%）
- MAE 0.15（[-1,+1]スケール）→ 15%ポイント誤差

### v46 構成（確定）

| 項目 | 設定 |
|------|------|
| ベースライン | v45 完全継承（データ・λ・アーキ） |
| 追加ヘッド | `aggression_head = Linear(256, 1) + Tanh` |
| LAMBDA_AGG | 0.3 |
| データ | hand_inference_v46.ndjson（aggression ラベル付き） |
| ラベル生成 | `phase2/scripts/add_aggression_labels.py` |
| early stopping | val_eae 基準、patience=15 |

---

## 新PC移行手順（2026-07-18 作成）

### 背景

旧PC（THPC-L01, RTX 3080）から新PC（RTX 5080）へ学習環境を移行する。
git 管理外の大容量ファイル（学習データ・モデル）はzipで移送する。

### 移行ファイルセット

| ファイル | 内容 | 圧縮後サイズ | 必須度 |
|---------|------|------------|-------|
| `data_all_20260718.zip` | 全 ndjson データファイル 16本（~23 GB 非圧縮） | 1.06 GB | **必須** |
| `models_v44v45v46_20260718.zip` | v44/v45/v46 model.pt + checkpoint.pt + v46 ONNX | 151 MB | **必須** |
| `models_full_20260718.zip` | phase2/models/ 配下 全114重みファイル | 1.13 GB | 任意（他バージョンが必要な場合） |
| `paipu_raw_20260718.zip` | 生牌譜 XML 500ファイル | 3.1 MB | 任意（states 再生成が必要な場合） |

- Google Drive 経由でアップロード・ダウンロード（認証はユーザーが実施）。
- 全 zip のSHA256は `phase2/migration/checksums_20260718.txt` の `[ZIP_SHA256]` セクションを参照。
- 個別ファイルの SHA256（v44/v45/v46 モデル + 全16 ndjson）は同ファイルの `[FILE_SHA256]` セクション、
  models_full の全114ファイルは `phase2/migration/models_full_hashes_20260718.txt` を参照。
- **15 GB 制限**: 単一ファイルは全て 1.13 GB 以下。Google Drive の個別ファイルサイズ制限（通常 5TB）に抵触なし。

### 新PCでの再現手順

```
# 1. リポジトリ取得
git clone https://github.com/itsutsukarabo/mahjong-AI.git
cd mahjong-AI
git checkout feature/v46-aggression

# 2. Google Drive から zip ダウンロード（最低限: 必須2本）
#    data_all_20260718.zip         → 学習・評価データ
#    models_v44v45v46_20260718.zip → v44/v45/v46 モデル本体
#    （任意）models_full_20260718.zip  → 全バージョン重みファイル
#    （任意）paipu_raw_20260718.zip    → 生牌譜XML（states再生成用）

# 3. zip 全体の SHA256 照合（破損チェック）
#    Windows PowerShell:
(Get-FileHash data_all_20260718.zip -Algorithm SHA256).Hash
(Get-FileHash models_v44v45v46_20260718.zip -Algorithm SHA256).Hash
#    → phase2/migration/checksums_20260718.txt の [ZIP_SHA256] セクションと照合

# 4. リポジトリルートで展開（7-Zip CLI 推奨。Expand-Archive は 2GB 制限あり）
7z x data_all_20260718.zip -o. -y
7z x models_v44v45v46_20260718.zip -o. -y

# 4b. 任意: models_full 展開（展開先: phase2/models/）
#     models_v44v45v46 と重複するファイルは上書きされる（同一バイナリのため問題なし）
7z x models_full_20260718.zip -o. -y

# 4c. 任意: 生牌譜 XML 展開（展開先: phase2/data/raw/xml/）
#     states 再生成が必要な場合（v48以降で特徴量仕様変更時）
7z x paipu_raw_20260718.zip -o. -y
#     展開後ファイル数確認:
(Get-ChildItem phase2\data\raw\xml\*.xml | Measure-Object).Count  # → 500

# 5. 個別ファイルの SHA256 照合（全ファイル完全性確認）
python phase2/migration/verify_checksums.py
#    OK: 24件（モデル8 + データ16）が全て OK と表示されれば成功
#    任意: models_full の照合は models_full_hashes_20260718.txt を手動で参照

# 6. 展開後の確認事項
#    (a) 必須モデルの存在確認
Test-Path phase2\models\hand_inference\v46\model.pt       # → True
Test-Path phase2\models\hand_inference\v46\model.onnx    # → True
Test-Path phase2\models\hand_inference\v45\model.pt      # → True
Test-Path phase2\models\hand_inference\v44\model.pt      # → True
#    (b) 学習データの行数確認（states 再利用の場合）
(Get-Content phase2\data\states\states_v22.ndjson | Measure-Object -Line).Lines  # → 1,153,236
(Get-Content phase2\data\features\hand_inference_v46.ndjson | Measure-Object -Line).Lines  # → 164,414
#    (c) 他バージョン重みの存在確認（models_full 展開時）
Test-Path phase2\models\tenpai_inference\v1\model.pt     # → True（例）

# 7. CUDA/PyTorch 環境構築（RTX 5080 向け）
python -m venv C:\ml\venv
C:\ml\venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r phase2/requirements_v46.txt

# 8. v46 評価（移行成功の最終判定）
#    注: --eval-only フラグは未実装。評価専用スクリプトを用意して実行すること
#    （詳細は「環境移行」セクションの注記を参照）
#    test_eae ≈ 4.117, agg_sign_acc ≈ 0.835 が出れば移行成功
```

### 移行成功の判定基準

| 指標 | 旧PCの値 | 許容範囲 |
|------|---------|---------|
| test_eae | 4.1170 | ±0.01 |
| test_acc | 0.9159 | ±0.001 |
| agg_sign_acc | 0.8353 | ±0.005 |

微小な数値差はハードウェア・cuDNN バージョン差異によるもので正常。
大きな乖離（>±0.1）はデータ破損・モデル不整合を示す。

---

## 環境移行（2026-07-18）

- 移行元: THPC-L01（旧ワークステーション、RTX 3080）
- 移行先: 新PC（RTX 5080 Blackwell, Windows, `C:\Users\tetsu\mahjong-AI`, ディスク1.9TB）
- 移行方法:
  * コード・設計・テスト: gitリモート（github.com/itsutsukarabo/mahjong-AI）経由。
    main と feature/v46-aggression 両ブランチをpush→新PCでclone。
  * データ・モデル: ディレクトリ構成を保ったzip化 → Google Drive → 新PCで展開。
    `data_all_20260718.zip`（~23GB展開）/ `models_v44v45v46_20260718.zip`（~166MB展開）
  * 完全性検証: SHA256ハッシュ一覧（`phase2/migration/checksums_20260718.txt`）をgitにコミットし、
    新PCでzip単位・個別ファイル単位（全24件）を照合（全一致）。
- 環境: RTX5080(Blackwell, sm_120)対応のPyTorch/CUDAを新規構築
  （旧PCのpip freezeは参照に留め、5080が動く組み合わせを確認して導入）。
  * Python: 3.12.10
  * PyTorch: **torch==2.11.0+cu128**（`pip install torch --index-url https://download.pytorch.org/whl/cu128`）
  * CUDA: ビルドに同梱のCUDA 12.8（`torch.version.cuda == "12.8"`）。
    ドライバは610.62でCUDA 13.3まで対応、cu128ビルドは後方互換で動作。
  * `torch.cuda.get_device_capability()` → `(12, 0)`（sm_120）で認識、動作確認済み。
  * その他依存は `phase2/requirements_v46.txt`（torch行以外）をそのまま導入、`pip check`で競合なし。
- 移行検証: 移したv46モデル（`model.pt`）+ 移したv46データ（`hand_inference_v46.ndjson`,
  164,414行）で同一test split（rng seed=42, 80/10/10分割）を再現し評価。

  | 指標 | 旧PC(RTX3080) | 新PC(RTX5080) | 差分 |
  |------|--------------|--------------|------|
  | test_eae | 4.116973118 | 4.116973331 | 2.1e-7 |
  | test_eae_stage | 3.150114642 | 3.150114833 | 1.9e-7 |
  | test_acc | 0.9159046297 | 0.9159046297 | 0 |
  | soft_f1 | 0.3237 | 0.3237 | 0 |
  | wait_f1 | 0.3149 | 0.3149 | 0 |
  | wait_hit_rate | 0.354 | 0.354 | 0 |
  | wait_top1_acc | 0.6616 | 0.6616 | 0 |
  | agg_all_mae | 0.1501 | 0.1501 | 0 |
  | agg_pressure_mae | 0.18 | 0.18 | 0 |
  | agg_sign_acc | 0.8353 | 0.8353 | 0 |

  判定基準（test_eae ±0.01, test_acc ±0.001, agg_sign_acc ±0.005）を全てクリア。
  他指標もほぼ完全一致。**移行成功。**

  注: `train_hand_inference_v46.py` に `--eval-only` フラグは実装されていない
  （sys.argv手動パースで `--resume`/`--stop-after` のみ対応、未知フラグは無視されて
  フルスクラッチ学習が誤起動しうる）。新PCでの再評価は、学習ループを呼ばず
  `eval_epoch`/`eval_wait_metrics`/`eval_aggression` のみを使う評価専用スクリプトを
  別途用意して実施した。

- 教訓: gitリモートへのpush状態を移行前に必ず確認する。
  今回、push直後にcloneして確認したところ main は origin と同期済み、
  feature/v46-aggression は main から **4コミット**先行した状態で push 済みだった
  （`v46: ep200完走・攻撃性ヘッド統合`、`push/retreat→aggression回帰ヘッド実装`、
  `v39〜v44実験記録追加`、`migration: 移行用zip・チェックサム・手順書追加`）。
  pushを確認せずcloneしていた場合、これらのコミット（v46の学習成果本体を含む）を
  取りこぼす可能性があった。

---

## v46 ONNXエクスポート: aggression_logit 欠落バグと修正（2026-07-18）

### 真因

`train_hand_inference_v46.py` の `main()` 末尾に埋め込まれたONNXエクスポート処理
（学習完走時に自動実行される）の `_Wrap.forward()` が、モデルの `forward()` が返す
8つの戻り値のうち5番目 `aggression_logit` を `_` で握り潰していた:

```python
# train_hand_inference_v46.py:1210（バグ）
logits,_,red_logits,block_logits,wait_logits,_,shanten_logits,_ = self.m(features, disc_tokens, disc_mask)
return logits, red_logits, block_logits, wait_logits, shanten_logits  # aggressionが無い
```

`forward()` の実際の戻り値順序（`train_hand_inference_v46.py:712`）:
`logits, logits_raw, red_logits, block_logits, wait_logits, aggression_logit, shanten_logits, yaku_logits`
（5番目=aggression_logit、`_Wrap`はこの位置を`_`で捨てていた）。

既存の `phase2/models/hand_inference/v46/model.onnx` をバイナリ文字列検索で確認した結果、
出力ノード名は `logits, red_logits, block_logits, wait_logits, shanten_logits` の5つのみで
`aggression_logit` は存在しないことを確認済み（推論は実行せず静的確認）。

### 修正

`phase2/scripts/export_onnx_v46.py` を新規作成し、`aggression_logit` を出力に追加した
6出力（`logits, red_logits, block_logits, wait_logits, aggression_logit, shanten_logits`）
でモデル本体（`model.pt`、best val_eae=ep185相当のstate_dict）から再エクスポート。
既存の `model.onnx` は上書きせず、別名 `model_aggression_candidate.onnx` として出力。

検証（ダミー入力・実データ1サンプルの両方でPyTorch出力とONNX出力を比較）:
全6出力の最大誤差 < 1e-5、aggression値は範囲内(-1〜+1)、実データサンプルでは
PyTorchとONNXの差が7.45e-09（実質一致）。

---

## フロントエンド v46 対応（2026-07-18）

### 背景

`phase2/browser/ai_phase2.js`（ブラウザ側ONNX推論）は v38 固定（695次元固定特徴 +
44次元/トークン、`hand_inference/v38/model.onnx` 参照）のままで、v45/v46 のデータ
仕様変更（`danger_level` 特徴追加によるtoken_dim 44→45、`yaku_probs`廃止による
fixed_dim 695→674）に未対応だった。

### 実施内容

1. **674/45次元対応の特徴量エンジンをJS側に新規実装**
   `make_hi_features_v46()` / `make_discard_tokens_v46()` / `build_event_tokens_v46()` /
   `_danger_level_v46()` を `phase2/browser/ai_phase2.js` に追加（既存のv38関数は
   比較・切り戻し用に無変更のまま保持）。`danger_level`（現物/スジ/無スジ判定）は
   `phase2/scripts/extract_features.js` の `_danger_level()` から1対1で移植。

2. **一致検証**
   Node.js（新規インストール、v24.18.0）で `extract_features.js` のロジックと
   新実装を、実データ（`states_v22.ndjson`）4局面×計10通りのtarget_lで突き合わせ。
   固定特徴量401次元（比較不能な既知近似272次元・tenpai_prob 1次元を除く）は
   float32丸め誤差の範囲で完全一致、トークン列（danger_level全4パターン: 無リーチ/
   現物/スジ/無スジ）も完全一致を確認。

3. **【重要な既知の不具合】JS側 v38 関数に座席番号バグが未修正で残存**

   `ai_phase2.js` の `wind_features()`（124行目付近）と `jikaze_onehot()`
   （127行目付近）が、v45でPython側（`extract_features.js`）が修正したはずの
   座席番号バグの**旧式のまま**になっている:

   ```js
   // ai_phase2.js（v38用、未修正のまま）
   const jikaze = (player_l - state.jushu + 4) % 4;   // 旧式・誤り
   ```
   ```js
   // extract_features.js（v45で修正済み）
   const jikaze = player_l;   // 座席番号がそのまま自風インデックス
   ```

   v46対応では `wind_features_v46()` / `jikaze_onehot_v46()` として修正版を
   新規実装し、v46経路はこの修正版を使用する。**v38関数（`wind_features`/
   `jikaze_onehot`）自体は今回修正していない** — v38経路は現状ONNXモデルが
   存在せず動作していない（`hand_inference/v38/model.onnx` は未エクスポート）
   ため、実害は今のところ無いが、将来 v38 のONNXを用意して動かす場合は
   このバグの移植修正が必要。

4. **モデル配置・配線・UI追加**
   - `model_aggression_candidate.onnx` を `tmp_clone/dist/models/hand_inference/v46/model.onnx`
     に配置（`dist/` はgitignore対象のため、ビルドのたびに手動配置が必要）。
   - `ai_phase2.js` の `load_sessions()` のモデルパスを v38→v46 に更新。
   - `run_phase2()` を v46特徴量関数の呼び出しに変更し、ONNX出力から
     `aggression_logit` を取り出して `result.hand_inference.aggression` に格納。
     **`aggression_logit` は (1,) 形状で視点プレイヤー1名分のスカラーであり、
     3人の対象プレイヤーそれぞれの値ではない**（モデルの `aggression_encoder` が
     `x[:,0,...]` のみを参照するため）。`players[i].aggression` ではなく
     `hand_inference.aggression`（単一値）として実装。
   - `paipu.js` の `_render_phase2()` に「局面情報」（順位・場況・残り牌数・
     終盤フラグ）と「自分の攻撃性 (v46)」の表示セクションを追加、
     `ai-modal.pug` にUIスロットを追加。
   - `behavior_clone`/`value_function`/`yaku_inference`/`tenpai_inference` は
     ONNX未エクスポートのままで、既存のtry/catchによる握り潰し設計を維持
     （hand_inferenceのみ動けば今回の照合検証には十分なため）。

5. **ビルド・ブラウザ動作確認**
   Node.js未インストール環境だったため winget で新規インストール。
   4パッケージ（majiang-core / majiang-ui-local / tenhou-log-local / tmp_clone）で
   `npm install` 実行（`tmp_clone` の `patch-package` postinstall は
   `@kobalab/majiang-ui` 用の旧パッチが `majiang-ui-local`（既に同内容を
   直接含む独自フォーク）に対して二重適用となり失敗するが、対象ファイルは
   無傷でビルドに支障なし。既知の問題として記録）。
   `npm run build` でHTML/CSS/JS生成に成功。
   Playwright（新規インストール）でヘッドレスブラウザから実際に牌譜
   （`majiang-core/test/data/script.json` の実データ、東1局・親リーチあり）を開き、
   AI解析モーダルで aggression が表示されることを確認:

   | 視点 | 局面 | aggression | 表示ラベル |
   |------|------|-----------|-----------|
   | 下家(l=1)、親のリーチ後 | 残り40枚 | -0.125 | 中立 |
   | 対面(l=2)、親のリーチ後 | 残り39枚 | 0.144 | 中立 |
   | 親本人(l=0、リーチ者本人) | 残り41枚 | 0.047 | 中立 |

   いずれも -1〜+1 の範囲内、局面ごとに異なる値を示すことを確認
   （常に同一値に固定されていない）。コンソールログで
   `AI Phase2: loaded hand_inference` と `AI Phase2: aggression = ...` を確認。
