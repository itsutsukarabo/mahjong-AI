# タスク一覧

最終更新: 2026-06-07（手牌推測強化グループ TICKET-027〜035 追加）

> **優先順位のルール**: 「新規タスク」セクションのチケットを最優先で処理する。「特殊牌対応」グループは新規タスクがなくなった後に着手する。

---

## 完了済み

### TICKET-001: majiang-core をローカルパス参照に切り替える

**状態**: 完了  
**優先度**: 高  
**依存**: なし

**概要**  
`tmp_clone/package.json` の依存を `file:../majiang-core` に変更し、ローカルの `majiang-core/` を直接参照できるようにする。  
`node_modules/@kobalab/majiang-core` は Junction として `majiang-core/` を指しており、ファイル編集後に `npm run build:js` だけで即時反映される。

---

## 新規タスク

> **三麻ビューア対応グループ**: TICKET-010 が他すべての前提。TICKET-012/014 は並行着手可。TICKET-015 は TICKET-014 完了後、TICKET-013 と TICKET-015 が揃ったら TICKET-016 に着手。

---

### TICKET-010: Git リポジトリと GitHub Actions CI 基盤を構築する

**状態**: 完了  
**優先度**: 高  
**依存**: なし

**概要**  
`mahjong-AI/` ワークスペースを https://github.com/itsutsukarabo/mahjong-AI.git へ push し、以降の全チケットが「テストが CI で自動検証される」状態で開発できるようにする。

**進め方**  
1. `mahjong-AI/` で `git init` → リモートを追加 → `.gitignore` を作成（除外対象: `*/node_modules/`, `tmp_clone/dist/`）
2. `.github/workflows/ci.yml` を作成（ジョブ設計は下記テスト戦略を参照）
3. 既存ファイル（`majiang-core/`, `tmp_clone/src/`, ドキュメント群）を初回コミットして push
4. GitHub Actions の画面でパイプラインが green になることを確認する

**CI ジョブ構成**

```yaml
jobs:
  test-majiang-core:       # majiang-core/ で npm ci && npm test
  test-tenhou-log-local:   # tenhou-log-local/ で npm ci && npm test
  build:                   # needs: [test-majiang-core, test-tenhou-log-local]
                           # tmp_clone/ で npm ci && npm run build:js
```

---

### TICKET-011: tenhou-log-local パッケージの雛形を作成する

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-010

**概要**  
`@kobalab/tenhou-log` のソースを `mahjong-AI/tenhou-log-local/` にコピーし、独自改修とテストを加えられる独立パッケージとして整備する。

**進め方**  
1. `tenhou-log-local/` ディレクトリを作成し、`package.json` を用意する（`name: "tenhou-log-local"`, `scripts.test: "mocha -u tdd"`, `devDependencies: mocha`）
2. `@kobalab/tenhou-log` の `lib/convlog.js` と `lib/getlog.js` をコピーする
3. `test/` ディレクトリを作成し、`npm test` が空でも通ることを確認する
4. コミットして CI が green になることを確認する

---

### TICKET-012: convlog.js を三麻対応に改修し、ユニットテストを書く

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-011

**概要**  
三麻（sanma）の天鳳 XML を電脳麻将 JSON に変換できるよう `convlog.js` を改修する。仕様はテストによって担保する。

**改修内容**  
| 変更箇所 | 内容 |
|---|---|
| GO タグ | `if (type.sanma) throw` を削除 |
| N タグ（北抜き判定） | `type.sanma && s=='z' && n==4` のとき `{kita:{l, p:'z4'}}` + `gang=true` を emit |
| `qipai()` | `hai3=""` → `''` そのまま渡す（board.js 側が `'_'.repeat(13)` にフォールバック）|
| `hule()` / `pingju()` | 4つ目の `sc` ペアが `0,0` → `fenpei[3]=0` で現コードのまま動作する。変更不要 |
| `owari` | 4つ目ペアが `0,0.0` → `defen[3]=0`, `point[3]=0.0`, `rank[3]=4` （強制最下位）|

**進め方**  
1. `test/fixtures/` に実際の天鳳三麻 XML（`sanma_sample.xml`）と期待出力 JSON（`sanma_expected.json`）を保存する  
2. `test/convlog.js` に以下のテストスイートを実装する（`mocha -u tdd`）

   ```
   suite('parse_type(185)', ()=>{
     test('三麻フラグが true になること');
     test('戻り値が "三" で始まること');
   });
   suite('北抜き N タグ変換', ()=>{
     test('m=30752 → {kita:{l:0, p:"z4"}}');
     test('m=31008 → {kita:{l:0, p:"z4"}}');
     test('m=31264 → {kita:{l:0, p:"z4"}}');
     test('北抜き後の draw が gangzimo として変換されること');
   });
   suite('qipai 三麻', ()=>{
     test('hai3="" のとき shoupai[3] が "" になること');
     test('ten の4番目が 0 のとき defen[3]=0 になること');
   });
   suite('owari 三麻', ()=>{
     test('4番目プレイヤーの rank が 4（最下位）になること');
     test('4番目プレイヤーの defen が 0 になること');
   });
   suite('full conversion（スナップショット）', ()=>{
     test('sanma_sample.xml の変換結果が sanma_expected.json と一致すること');
   });
   ```

3. 全テストが通ることを確認し、CI が green になることを確認する

---

### TICKET-013: tenhou-log-local サーバーを実装する

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-012

**概要**  
`kobalab.net/majiang/tenhou-log/` の代替となるローカルサーバーを実装する。天鳳から XML を取得し、改修済み `convlog.js` で JSON に変換して返す。

**進め方**  
1. `tenhou-log-local/server.js` を実装する（express, GET `/tenhou-log/:id.json`）
2. `dependencies` に `express` を追加する
3. `test/server.js` を実装する（`supertest` を使用）

   ```
   suite('GET /tenhou-log/:id.json', ()=>{
     test('200 と JSON が返ること（getlog をフィクスチャ XML でモック）');
     test('天鳳が 404 を返すとき 404 が伝播すること');
     test('三麻 XML を変換したとき kita アクションが含まれること');
   });
   ```

   ※ CI 内では天鳳へのリアル通信をしない。`getlog.js` をモック（`sinon` 等）してフィクスチャ XML を返す。

4. `devDependencies` に `supertest`, `sinon` を追加する
5. 全テストが通ることを確認する

---

### TICKET-014: majiang-core/lib/board.js に kita() を追加し、ユニットテストを書く

**状態**: 完了  
**優先度**: 高  
**依存**: なし（TICKET-010 と並行可）

**概要**  
牌譜再生エンジン（`Majiang.Board`）に 北抜きアクション `kita()` を追加する。`shoupai.dapai(p, false)` で手牌から z4 を除去し、`lunban` を更新する。

**進め方**  
1. `majiang-core/lib/board.js` の `gang()` の直後に追加する

   ```javascript
   kita(kita) {
       this.lunban = kita.l;
       this.shoupai[kita.l].dapai(kita.p, false);
   }
   ```

2. `majiang-core/test/board.js` に `suite('kita(kita)', ...)` を追加する

   ```
   suite('kita(kita)', ()=>{
     test('手番が更新されること');
     test('手牌から z4 が 1 枚除去されること');
     test('複数回連続して呼べること（残り枚数が毎回 1 減ること）');
     test('kita 後に zimo / dapai が正常に呼べること');
   });
   ```

3. `npm test` が全て通ることを確認し、CI が green になることを確認する

---

### TICKET-015: majiang-ui の paipu.js kita パッチを patch-package で管理する

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-014

**概要**  
`node_modules/@kobalab/majiang-ui/lib/paipu.js` に `kita` ディスパッチを追加し、その差分を `patch-package` で `patches/` ファイルに保存する。`npm install` のたびに自動適用されるようにする。

**進め方**  
1. `tmp_clone/` で `npm install --save-dev patch-package` を実行する
2. `tmp_clone/package.json` の `scripts` に `"postinstall": "patch-package"` を追加する
3. `node_modules/@kobalab/majiang-ui/lib/paipu.js` を直接編集して `kita` を追加する

   ```javascript
   // seek() 内（kaigang の直後）
   else if (data.kita)     this._model.kita(data.kita);

   // next() 内（kaigang の直後）
   else if (data.kita)     this.kita(data);

   // メソッドを追加
   kita(data) {
       this._model.kita(data.kita);
       this._view.update(data);
   }
   ```

4. `npx patch-package @kobalab/majiang-ui` でパッチファイルを生成する（`patches/@kobalab+majiang-ui+*.patch`）
5. `patches/` を git に追加してコミットする
6. `npm ci` → `postinstall` が自動実行されてパッチが当たることをローカルで確認する

---

### TICKET-016: paipu.js proxy URL 変更・ビルド・動作確認

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-013, TICKET-015

**概要**  
アプリの proxy URL をローカルサーバーへ切り替え、ビルドして実際の三麻牌譜 URL で表示が通ることを確認する。

**進め方**  
1. `tmp_clone/src/js/paipu.js` L.14 を変更する

   ```javascript
   // 変更前
   const tenhou_log = 'https://kobalab.net/majiang/tenhou-log/';
   // 変更後
   const tenhou_log = 'http://localhost:8001/tenhou-log/';
   ```

2. `npm run build:js` でビルドが通ることを確認する
3. `tenhou-log-local/` でローカルサーバーを起動する（`node server.js --port 8001`）
4. `npx http-server dist -p 8080 --cors` でアプリを起動する
5. 牌譜ビューアに `https://tenhou.net/0/?log=2025080617gm-00b9-0000-104adf08&tw=2` を入力して再生できることを確認する
6. 北抜きが発生する場面で手牌から z4 が消えることを目視で確認する

---

### TICKET-017: 対局中に北抜き枚数を表示する

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-015

**概要**  
三麻対局の再生中、プレイヤーが北を何枚抜いたかを手牌エリアに表示する。現状は `board.kita()` が `shoupai.dapai()` で z4 を手牌から除去するだけで、抜いた z4 の視覚的な記録が残らない。

**根本原因**  
- `majiang-ui/lib/board.js` の `update()` に `data.kita` ハンドラが存在しない
- `majiang-ui/lib/shoupai.js` に北抜き枚数を保持・描画する仕組みがない

**実装方針**  
patch-package で `@kobalab/majiang-ui` の 2 ファイルを修正する。

1. **`shoupai.js`** に以下を追加する
   - コンストラクタに `this._n_kita = 0`
   - `kita()` メソッド（`_n_kita` をインクリメントし、fulou エリアに z4 タイルを 1 枚追加）
   - `redraw()` 内の fulou ループの直後に、`_n_kita` 枚分の z4 タイルを描画（再描画しても消えないようにする）
   ```javascript
   kita() {
       this._n_kita++;
       this._node.fulou.append($('<span class="mianzi kita">').append(this._pai('z4')));
       return this.adjust();
   }
   // redraw() 内 fulou ループの直後:
   for (let i = 0; i < this._n_kita; i++) {
       this._node.fulou.append($('<span class="mianzi kita">').append(this._pai('z4')));
   }
   ```

2. **`board.js`** の `update()` に `data.kita` ハンドラを追加する（kaigang の直後）
   ```javascript
   else if (data.kita) {
       this._view.shoupai[data.kita.l].kita();
   }
   ```

3. **動作確認**: `npm run build:js` → 三麻牌譜を再生し、北抜きのたびに手牌エリアに z4 タイルが 1 枚ずつ追加されることを目視確認。局が変わったら 0 枚にリセットされること。

---

### TICKET-018: アガリ画面での北抜きの表示を修正する（暗槓→北として表示）

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-017（shoupai.js の kita() メソッドが必要）

**概要**  
アガリ確認画面で、抜いた北の枚数分だけ「暗槓（z4 4 枚）」が表示されるバグを修正する。北抜き 3 回なら北の z4 タイルが 3 枚並ぶ表示に変える。

**根本原因**  
`convlog.js` の `hule()` が AGARI タグの北抜き副露値を `mianzi(mc)` で `z4444` に変換し、shoupai 文字列に含める。`Majiang.Shoupai.fromString()` はこれを `_fulou` に格納し、`mianzi.js` が `/^[mpsz](\d)\1\1\1$/` にマッチして暗槓として描画する。

**実装方針**  
patch-package で `@kobalab/majiang-ui` の `dialog.js` を修正する（`convlog.js` 側は変更不要）。

1. **`dialog.js`** の `hule()` メソッド内で、shoupai 文字列から `z4444` を取り除いてから `Shoupai` を生成し、その後 kita 枚数分 `shoupai_view.kita()` を呼ぶ
   ```javascript
   // 変更前
   new Shoupai($('.shoupai', this._node.hule), this._pai,
               Majiang.Shoupai.fromString(hule.shoupai)).redraw(true);

   // 変更後
   let sp_str   = hule.shoupai;
   let n_kita   = (sp_str.match(/,z4{4}/g) || []).length;
   sp_str       = sp_str.replace(/,z4{4}/g, '');
   let sp_view  = new Shoupai($('.shoupai', this._node.hule), this._pai,
                               Majiang.Shoupai.fromString(sp_str));
   sp_view.redraw(true);
   for (let i = 0; i < n_kita; i++) sp_view.kita();
   ```

2. **四麻への影響**: 四麻では z4 の暗槓が理論上存在するが、天鳳ログにほぼ現れない。またこのロジックは sanma 判定を行わず `z4444` を一律に北として扱う。今後問題になれば `hule.n_kita` を `convlog.js` 側で付与する方式に切り替える。

3. **動作確認**: `npm run build:js` → 三麻アガリ画面で北が暗槓ではなく z4 タイル（1 枚ずつ）として表示されること。

---

## AI作成グループ

> フェーズ順に進める。TICKET-019（フェーズ1）は他フェーズの可視化基盤であり最優先。  
> TICKET-020（フェーズ2）は対局ループ基盤なしで独立して開始できる。  
> TICKET-021（フェーズ3）はTICKET-020の完了後に着手する。

---

### TICKET-019: AI作成フェーズ1 — 牌譜分析モーダル実装

**状態**: 未着手  
**優先度**: 高  
**依存**: なし  
**詳細計画**: [`AI_CREATION_PHASE1_ANALYSIS_MODAL.md`](AI_CREATION_PHASE1_ANALYSIS_MODAL.md)

**概要**  
牌譜再生機能を拡張し、切り番プレイヤーの手番ごとに現行ルールベースAI（`@kobalab/majiang-ai`）の分析結果をモーダルで表示する。  
フェーズ2（教師あり学習）の出力を差し込めるUIスロットも同時に設計する。

**主な実装内容**
- `minipaipu()` を使った任意局面の再現モジュール
- `select_dapai(info)` / `select_fulou(info)` / `select_gang(info)` の `info` 配列を活用した候補ランキング表示
- 期待値 (`ev`)・向聴数・テンパイ牌枚数・危険度 (`weixian`) のテーブル表示
- `eval_shoupai()` の内訳ログ化（どのテンパイ牌が何枚残ってどう加算されたか）
- 残り牌数テーブル・牌価値スコア (`paijia`)・危険度マップの特徴量パネル
- フェーズ2モデル（手牌推定・聴牌確率・当たり牌確率・行動確率）用のUIスロット（プレースホルダー）
- フェーズ2モデルの差し込み口インターフェース (`HandInferenceModel`, `PolicyModel`) の定義
- Worker による `eval_shoupai()` の非同期化

---

### TICKET-020: AI作成フェーズ2 — 教師あり学習パイプライン

**状態**: 未着手  
**優先度**: 高  
**依存**: TICKET-019（UIスロットの定義が確定してから開始推奨）  
**詳細計画**: [`AI_CREATION_PHASE2_SUPERVISED_LEARNING.md`](AI_CREATION_PHASE2_SUPERVISED_LEARNING.md)

**概要**  
天鳳の人間牌譜を教師データとして3つのモデルを学習する。フェーズ3（強化学習）の出発点として機能させる。  
対局ループ基盤（フェーズ3）がなくても独立して開始できる。

**主な実装内容**
- 天鳳牌譜ダウンロードスクリプト（20分間隔・単一セッション・圧縮対応、ユーザー手動実行）
- 牌譜XMLパーサー・局面状態の時系列変換モジュール
- **手牌類推モデル**: 公開情報 → 他家の各牌保有枚数の周辺確率・聴牌確率・当たり牌確率
- **行動クローン**: ゲーム状態 → アクション確率分布（人間の打牌を模倣）
- **V(状態)初期モデル**: ゲーム状態 → 対局終了着順点の期待値（フェーズ3 Critic の初期値）
- 特徴量設計（レベル1: 捨て牌パターン・副露パターン・点数状況等の手作り特徴量）
- モデルバージョン管理（`models/[種別]/v[N]/model + config + eval_result`）
- TICKET-019のUIスロットへの学習済みモデル差し込み

**天鳳規約上の注意**  
個人AI研究目的での利用は黙認（積極許可ではない）。  
再配布・競合サービス利用・企業利用は禁止。  
ダウンロードは必ずユーザー自身が手動またはルール遵守スクリプトで実行すること。

---

### TICKET-021: AI作成フェーズ3 — 強化学習フレームワーク

**状態**: 未着手  
**優先度**: 中  
**依存**: TICKET-020  
**詳細計画**: [`AI_CREATION_PHASE3_REINFORCEMENT_LEARNING.md`](AI_CREATION_PHASE3_REINFORCEMENT_LEARNING.md)

**概要**  
`do_sync()` を使った高速対局ループでAI同士をセルフプレイさせ、強化学習（Actor-Critic）でフェーズ2モデルを育成する。  
2種類のAI（Type A: 局内最大得点、Type B: ゲーム終了逆算）を別々に学習し比較する。

**主な実装内容**
- `do_sync()` を使ったN局バッチ実行・牌譜収集スクリプト
- 状態エンコーダー・行動エンコーダー
- Actor-Critic 実装（Actorはフェーズ2行動クローンで初期化、CriticはV(状態)初期モデルで初期化）
- **Type A（局内最大得点AI）**: `eval_shoupai()` ベースの報酬整形 + RL
- **Type B（ゲーム終了逆算AI）**: ウマ後ポイントベースの報酬整形 + RL・着順を意識した戦略を学習
- 報酬設計: `V(状態_{t+1}) - V(状態_t)` による中間報酬（TD法）
- 対戦相手管理: Fictitious Self-Play（過去世代との対戦プール）
- Elo レーティング計算・ベンチマーク結果記録
- TICKET-019のモーダルでType A/B の判断を表示できることの確認

---

## 手牌推測強化グループ（v10〜）

> 赤牌修正: TICKET-030 → TICKET-031 → TICKET-032 の順。  
> ブロック表示: TICKET-027 → TICKET-028。TICKET-029 は TICKET-027 後に設計確定。  
> TICKET-033（解釈説明）・TICKET-034（順目別評価）は他と独立して着手可。

---

### TICKET-027: ブロック期待値計算エンジン実装（フロント後処理モジュール）

**状態**: 完了  
**優先度**: 高  
**依存**: なし（既存モデル出力を利用）

**概要**  
per-tile 確率分布（現行 `logits` 出力）から、スート別の刻子・順子・対子の所持期待値を算出するJSモジュールを実装する。新モデル不要でフロント後処理のみで実現する。

**入出力仕様**

入力: `probs_per_tile[34][5]` — 各牌の枚数確率分布（probs[k] = P(count=k)）

出力:
| 種別 | 算出式 | 次元数 |
|------|--------|--------|
| 刻子期待値 `triplet_ev[34]` | `probs[3] + probs[4]` （P(count≥3)） | 34 |
| 対子期待値 `pair_ev[34]` | `probs[2] + probs[3] + probs[4]` （P(count≥2)） | 34 |
| 順子期待値 `seq_ev[21]` | `(1-probs_i[0])×(1-probs_{i+1}[0])×(1-probs_{i+2}[0])` の独立近似 | 各スート7種×3=21 |

**最尤手牌の面子分解**
1. 各牌の argmax を手牌として構築（`Shoupai` 文字列に変換）
2. `Majiang.Util.xiangting()` でシャンテン数を計算
3. シャンテン数 ≦ 0 の場合: `hule.js` の面子分解ロジックを呼び出し、全有効分解を列挙
4. シャンテン数 > 0 の場合: 各ブロック期待値を降順に並べ、上位N件を「候補ブロック」として提示

**多重分解の扱い（近似の限界について）**  
順子・刻子の期待値は各牌の枚数を独立に扱うため、タイル共有（例: m234 と m345 が m34 を共有）を考慮しない。  
これは値の過大評価要因となるが、「どのブロックが形成されやすいか」のランキング指標としては有効。  
厳密計算（全手牌分布の列挙）は TICKET-029 で設計する。

**実装場所**  
`phase2/browser/ai_phase2.js` 内に `compute_block_ev(probs_per_tile)` 関数として追加する（ビルド不要）。  
同内容を `ai_phase2.js` にも反映する（`phase2/browser/` と `tmp_clone/dist/js/` の両ファイル）。

**進め方**  
1. `compute_block_ev(probs_per_tile)` を実装し、返り値 `{ triplet_ev, pair_ev, seq_ev }` の形式を確定する  
2. コンソールで既存の hand_inference 出力に適用して値を手動検証する  
3. 最尤手牌の面子分解ロジック（`xiangting` / `hule` 呼び出し）を組み込む  
4. `make_block_display_data(player)` ヘルパーを追加して描画側（TICKET-028）に渡せる形にする  

---

### TICKET-028: フロント描画 — ブロック所持期待値テーブル

**状態**: 完了  
**優先度**: 高  
**依存**: TICKET-027

**概要**  
TICKET-027 で算出したブロック期待値を「他家手牌推定」セクションに常時表示する。テンパイ確率による表示条件は設けない。

**表示レイアウト（スート別ブロックテーブル）**

各相手プレイヤーに対して、現行の per-tile 枚数テーブルの下に以下を追加:

```
[ 刻子 ] m111 m222 m333 m444 m555 m666 m777 m888 m999
         XX%  XX%  XX%  XX%  XX%  XX%  XX%  XX%  XX%

[ 順子 ] m123 m234 m345 m456 m567 m678 m789
         XX%  XX%  XX%  XX%  XX%  XX%  XX%

[ 対子 ] m11  m22  m33  m44  m55  m66  m77  m88  m99
         XX%  XX%  XX%  XX%  XX%  XX%  XX%  XX%  XX%
```

同様に p/s/z スートも表示（z は刻子・対子のみ、z1〜z7）。

**最尤手牌の面子構成表示**  
テーブルの下部に「最尤構成」として表示:
- シャンテン数 + 最も可能性の高い面子分解の文字列（例: `m123 m456 p789 s11 [待: p2]`）
- 有効な分解が複数ある場合は上位2〜3件を併記

**実装場所**  
`tmp_clone/dist/js/majiang-2.5.1.js`（および `tmp_clone/src/` の対応ソース）の AI モーダル描画部分。  
`ai-modal.pug` にブロックテーブル用の HTML 構造を追加し、`majiang-2.5.1.js` の `_render_ai_hand_inference()` 相当部分に描画ロジックを実装する。

**進め方**  
1. `ai-modal.pug` にブロックテーブルの HTML 構造（`.ai-hi-block-section` 等）を追加してビルドする  
2. `majiang-2.5.1.js` の hand_inference 描画部分（L.8386〜8443）に `compute_block_ev` 呼び出しとテーブル描画を追加する  
3. CSS でスコアに応じた色強調（0%=暗、>30%=中、>70%=強調）を設定する  
4. 最尤手牌・面子構成セクションを追加する  
5. `localhost:8080` で既存の手牌推測表示と並べて目視確認する

---

### TICKET-029: 学習ラベルのブロック構成対応設計（v11以降モデル向け）

**状態**: 未着手  
**優先度**: 中  
**依存**: TICKET-027（実装経験を踏まえて設計確定）

**概要**  
per-tile count ラベルのみでは面子分解が一意に定まらない問題に対し、将来モデル（v11以降）でブロック構成を直接学習するためのラベル設計を決定する。

**問題の整理**

現行 `label_hand`（34次元 per-tile count）から面子分解を後処理するとき:
- m1234 保持 → {m123+4} と {m1+234} の両方が有効な分解
- どちらが「正解の面子構成か」は per-tile count のみからは判断できない
- 例外: リーチ時は捨て牌から待ち牌が特定できるため分解が絞れる

**設計選択肢**

| 案 | 内容 | メリット | デメリット |
|----|------|---------|-----------|
| A（現行維持） | per-tile count のみ。後処理で近似ブロック確率 | 学習データ変更不要 | ブロック表示は近似 |
| B（OR ラベル） | 全有効分解を列挙し、いずれかに含まれるブロックに 1 を付与（multi-label BCE） | 「持てる可能性のある」ブロックを学習 | 過剰包含。全分解列挙が重い |
| C（ソフトラベル） | 各ブロックの出現頻度 / 全有効分解数 を期待値として付与 | 分解の多様性を反映 | 列挙コスト大、実装複雑 |
| D（テンパイ限定ラベル） | テンパイ局面のみ、待ち形を特定してラベル化 | 高精度な制約 | テンパイ以外に適用不可 |

**検討課題**  
- 全有効分解の列挙コスト（B/C案）: `hule.js` の `mianzi_all` を利用すれば `O(hand_size!)` を実用時間で解ける見込みだが、学習データ全サンプル（数十万件）への適用時間を検証する必要がある  
- 案B/Cで追加するラベル次元数: 刻子34 + 順子21 + 対子34 = 89次元  
- 損失関数: multi-label BCE を追加し、主損失（per-tile CE）と重み付き合算

**このチケットのゴール**  
- 上記選択肢のどれを採用するかを決定し、選択理由と実装仕様をこのチケットに記録する  
- 決定後、次チケットとして「v11 学習データ再生成 + 新モデルヘッド追加」を切る

---

### TICKET-030: 赤牌特徴量追加 + visible_counts_vec ブラウザバグ修正

**状態**: 完了  
**優先度**: 高  
**依存**: なし

**概要**  
現行モデルが赤牌をほぼ0%と予測し続ける根本原因が2つある。それぞれを修正してv10学習の前提を整える。

**原因1: 「赤牌が公開情報として見えているか」の特徴量が存在しない**

現行で唯一の赤牌固有入力特徴量は `red_discard_signal(3)` = 「対象プレイヤー自身が赤5を切ったか」のみ。  
他家の捨て牌・副露に m0/p0/s0 が含まれていても「m5が1枚見える」としか認識できず、赤牌か通常5かの情報が失われている。  
赤牌は1ゲームに各1枚しかなく、誰かの捨て牌・副露に見えていれば「対象が持っている可能性はゼロ」という最強の否定シグナルだが、モデルがこれを学習できない。

追加する特徴量 `red_visible(3)`: 全プレイヤーの捨て牌・副露を走査し、m0/p0/s0 が公開されているかを 0/1 でフラグ化。

```javascript
// extract_features.js / ai_phase2.js 両方に追加
function red_visible_flags(discards_l, melds_l) {
    const flags = [0, 0, 0];  // [m0, p0, s0]
    const suits = ['m', 'p', 's'];
    for (let l = 0; l < 4; l++) {
        for (const p of discards_l[l]) {
            const base = p.replace(/[_*+=\-]/g, '');
            for (let i = 0; i < 3; i++) {
                if (base === suits[i] + '0') flags[i] = 1;
            }
        }
        for (const m of (melds_l[l] || [])) {
            if (!m) continue;
            const si = suits.indexOf(m[0]);
            if (si < 0) continue;
            const clean = m.replace(/[+=\-]/g, '');
            for (let j = 1; j < clean.length; j++) {
                if (clean[j] === '0') { flags[si] = 1; break; }
            }
        }
    }
    return flags;
}
```

`make_hand_inference_sample()` の `features` 末尾に `...red_visible_flags(rec.discards_l, rec.melds_l)` を追加。  
これにより **input_dim: 371 → 374**。

**原因2: ブラウザ側 visible_counts_vec の請求牌二重カウント**

`extract_features.js` は副露の請求牌（他家捨て牌から取った牌）を「捨て牌でカウント済み」としてスキップするが、`ai_phase2.js`（L.246〜261）はスキップ処理を省略しており二重カウントになっている。

```javascript
// ai_phase2.js の visible_counts_vec — melds ループを以下に修正
for (const m of state.melds_l[l]) {
    if (!m) continue;
    const s = m[0];
    const dirIdx = m.search(/[+=\-]/);  // 暗槓は -1
    for (let i = 1; i < m.length; i++) {
        if (/[+=\-]/.test(m[i])) continue;
        if (dirIdx >= 0 && i === dirIdx - 1) continue;  // 請求牌スキップ
        const n = parseInt(m[i]);
        if (isNaN(n)) continue;
        const pi = pai_to_idx(s + (n === 0 ? 5 : n));
        if (pi >= 0) counts[pi]++;
    }
}
```

**進め方**  
1. `phase2/scripts/extract_features.js` に `red_visible_flags()` を追加し `make_hand_inference_sample` に組み込む（+3次元）  
2. `phase2/browser/ai_phase2.js` に同関数を追加し `make_hi_features` の末尾に組み込む  
3. `ai_phase2.js` の `visible_counts_vec` の副露ループに請求牌スキップを追加  
4. `tmp_clone/dist/js/ai_phase2.js` にも同内容を反映  
5. `node extract_features.js` を再実行して `hand_inference.ndjson` を再生成（374次元になっていることを確認）  
6. サンプル数行でフラグが正しく立っているかをログで目視確認する

---

### TICKET-031: v10 手牌推測モデル学習（input_dim 374, d_model 256）

**状態**: 実装完了（学習待ち）  
**優先度**: 高  
**依存**: TICKET-030（特徴量再生成完了後）

**概要**  
TICKET-030 で追加した `red_visible(3)` 特徴量を含む 374次元入力で v10 を学習し、赤牌予測精度が改善されることを確認する。

**変更内容**

```
MODEL_DIR : .../v10
input_dim : 371 → 374
d_model   : 256（v9 継承）
GPU_TEMP_THRESHOLD : 65（v9 継承）
epochs    : 100（v9 継承）
```

その他アーキテクチャ・損失関数は v9 と同一。

**判定基準**  
- `test_acc > 0.8654`（v9比改善）: 赤牌特徴量が精度向上に寄与
- `test_acc ≈ 0.8654`（v9と同等）: 赤牌特徴量は中立（悪化ではない）
- red_logits の予測分布を確認: v9では常に0%に近かったが、改善されているか

**進め方**  
1. `train_hand_inference_v9.py` を複製して `train_hand_inference_v10.py` を作成  
2. `CONFIG["input_dim"]` を 374 に変更、`MODEL_DIR` を v10 に変更  
3. 学習を実行（GPU冷却閾値 65°C を確認してから）  
4. `eval_result.json` と `train_log.json` を確認  
5. ONNX エクスポート → `tmp_clone/dist/models/hand_inference/v10/model.onnx` に配置  
6. `phase2/browser/ai_phase2.js` と `tmp_clone/dist/js/ai_phase2.js` のモデルパスを v9 → v10 に更新  
7. `localhost:8080` で赤牌確率が意味ある値を示すことを目視確認する

---

### TICKET-032: フロント赤牌表示改善 — 赤5縦列追加

**状態**: 完了  
**優先度**: 中  
**依存**: TICKET-031（v10 ONNX が配置されて red_logits が有意になってから）

**概要**  
手牌推測テーブルの 5 の列の横に「赤5」縦列を追加し、赤牌所持確率を per-tile テーブルと一体で視認できるようにする。

**現状の表示構造**

```
     1  2  3  4  5  6  7  8  9
M  [枚数分布] ...
P  ...
S  ...
Z  ...
赤牌: m0:XX% p0:XX% s0:XX%  ← 現在はテーブル外に別行
```

**変更後の表示構造**

```
     1  2  3  4  5  5赤  6  7  8  9
M  [枚数分布]  [持:%]  ...
P  ...
S  ...
Z  ...
```

- 5赤列の表示内容: `aka.m0`（モデルの `red_logits` から算出した所持確率）を「持: XX%」形式で表示
- 通常の 5 列（5枚数分布）と隣接させることで、「5mを2枚持ちのうち赤かどうか」を直感的に比較できる
- 字牌行には赤牌列を追加しない
- 現行の「赤牌: m0:XX% ...」セクションは削除して、テーブルに統合する

**実装場所**  
`tmp_clone/dist/js/majiang-2.5.1.js` L.8388〜8443（SUITS_HI 定義 + テーブル描画部分）

主な変更:
1. `SUITS_HI` の m/p/s エントリに `has_aka: true` フラグを追加
2. テーブルのヘッダー生成で、n===5 の後に「5赤」列ヘッダーを挿入
3. テーブルのデータ行で、n===5 の後に `aka[suit+'0']` の確率を表示するセルを追加
4. 赤牌セクション（`.ai-hi-aka-section` L.8428〜8437）を削除して整理

**進め方**  
1. `majiang-2.5.1.js` のテーブル描画ループを修正（m/p/s の 5 列の直後に赤5列を挿入）  
2. CSS（`majiang-2.5.1.css`）で赤5列のスタイルを設定（背景色等で区別）  
3. `src/` 側の pug/stylus ソースも対応させてビルドできる状態にする  
4. `localhost:8080` で表示確認

---

### TICKET-033: 手牌推測 解釈説明機能（ルールベース説明 + 感度分析）

**状態**: 未着手  
**優先度**: 中  
**依存**: なし（既存モデル出力とフロント特徴量を利用）

**概要**  
「なぜこの推測結果になったのか」を人間が理解できるようにする。Transformer はブラックボックスだが、入力特徴量の意味を可視化することで間接的な説明を提供する。

**フェーズ1: ルールベース説明テキスト生成**

モデル推論後、特徴量の値を参照して自然言語の説明を生成するルールベース関数を追加する。

| 条件 | 説明テキスト例 |
|------|----------------|
| `red_discard_signal[s] === 1` | 「赤5sを切り済みのため赤牌所持可能性なし」 |
| `red_visible_flags[s] === 1` | 「赤5sが他家に公開済みのため所持可能性なし」（TICKET-030後） |
| `riichi === 1` | 「リーチ中のため手牌変化なし」 |
| `pass_pon_signal[i] > 0.5` | 「牌Xのポンスルーあり: 2枚未満の可能性」 |
| `visible_counts[i] >= 0.75` | 「牌Xは3枚以上が公開済み: 所持最大1枚」 |
| `remaining < 10 tiles` | 「残り牌少なく手牌が固定化している可能性」 |

実装: `make_explanation_text(features, probs_per_tile, aka, target_l)` 関数を `ai_phase2.js` に追加。  
フロント: 各プレイヤーの手牌推測セクションの下部に「推測根拠」として表示（折りたたみ可）。

**フェーズ2: 感度分析（Stretch Goal）**

ルールベースでは表現しきれない「モデルが何に反応しているか」を確認するため、特定の特徴量をON/OFFして推論結果の変化を計算する。

対象特徴量:
- リーチフラグ (ON→OFF)
- 赤牌捨てシグナル (1→0)
- ポンスルー信号 (全ゼロ化)
- visible_counts の特定牌 (0にリセット)

UI: 「感度分析モード」ボタンをクリックすると、主要特徴量の変化量を差分ヒートマップとして per-tile テーブル上に重ねて表示する。

**注意**: Attention 可視化（Transformer 内部）はブラウザ上の ONNX ランタイムではアクセスが困難なため、このチケットのスコープ外とする。

**進め方（フェーズ1）**  
1. `ai_phase2.js` に `make_explanation_text(state, target_l, probs_per_tile, aka)` を実装する  
2. フロントの hand_inference 描画部分に説明テキストの表示エリアを追加する  
3. 数パターンの牌譜で説明テキストが意味あるものになっているか目視確認する  
4. フェーズ2（感度分析）は別 PR で対応

---

### TICKET-034: 手牌推測モデル 順目別精度評価

**状態**: 実装完了（train_hand_inference_v10.py に組み込み済み、学習待ち）  
**優先度**: 中  
**依存**: なし（既存モデルと学習データで実施可）

**概要**  
現行の評価指標は全サンプルの平均 `test_acc` のみ。「終盤ほど高精度」という仮説を定量的に検証するため、`remaining`（残り牌数）によるバケット別 accuracy を算出する。

**バケット定義**

| バケット | remaining 範囲 | ゲーム段階 |
|---------|---------------|---------|
| 序盤 | > 50 | 1〜7巡目頃 |
| 中盤 | 30 < r ≤ 50 | 8〜14巡目頃 |
| 終盤 | 10 < r ≤ 30 | 15〜21巡目頃 |
| 局終盤 | r ≤ 10 | 22巡目以降 |

**実装内容**

`evaluate()` 関数（`train_hand_inference.py` 系）を拡張:
1. `remaining` をサンプルメタデータから取得（features の index 0 × 70 で逆算）
2. サンプルをバケットに振り分け
3. バケットごとに accuracy・loss を計算
4. `eval_result.json` に `by_remaining_bucket` フィールドとして追記

```json
{
  "test_acc": 0.8654,
  "test_loss": 0.3312,
  "by_remaining_bucket": {
    "early":    { "acc": 0.820, "loss": 0.412, "n": 12345 },
    "mid":      { "acc": 0.851, "loss": 0.371, "n": 18234 },
    "late":     { "acc": 0.887, "loss": 0.298, "n": 11203 },
    "endgame":  { "acc": 0.921, "loss": 0.241, "n": 3456 }
  }
}
```

**将来的な損失重み付けへの活用**  
バケット別精度が確認できれば、将来的に「終盤サンプルを高重みで学習する sample_weight」の導入判断ができる。このチケットは測定フェーズのみで、重み付け導入は別チケットとする。

**進め方**  
1. `train_hand_inference_v10.py` の `evaluate()` 関数にバケット別集計を追加する  
2. v9 の学習済みモデルに対して評価だけ再実行して現状値を確認する（v10 学習前の参照値として記録）  
3. v10 学習後の `eval_result.json` にバケット別精度が記録されることを確認する  
4. 仮説「終盤ほど高精度」が成り立つか判定してコメントに残す

---

### TICKET-035: 王牌・カン情報の特徴量追加（残り牌エンコーディング改善）

**状態**: 未着手  
**優先度**: 低  
**依存**: なし（他チケットと独立）

**概要**  
「王牌の残りを入力に使っているか」という問いに対する現状確認と、未利用の情報（カン回数）を追加する微改善チケット。

**現状の確認（実装済み）**

`remaining = board.shan.paishu`（`shan.js` L.72: `_pai.length - 14`）として `game_state_features` の先頭次元 `vec[0] = remaining / 70` に既に含まれている。

- `paishu` は「ツモ可能な生き牌の残り枚数」= 全牌数 − 王牌固定14枚 − ツモ済み枚数
- 王牌（14枚固定）は常に除外されているため「何巡目か」は正確にエンコードされている
- 生き牌の残り枚数は `visible_counts` と組み合わせることで「誰かが何巡ツモしたか」の情報も間接的に含まれる

**未利用の情報: カン回数**

カン（槓）が発生すると死牌（王牌）からの補充ツモが発生し、生き牌の上限が事実上1枚ずつ減少する。現行の `remaining / 70` には「カンが何回行われたか」が折り込まれているが、モデルが「カン発生により live wall が縮んだのか、ただツモが進んだのか」を区別できない。

追加する特徴量 `kan_count_total(1)`: 局内で発生したカン（暗槓・明槓・加槓）の合計回数を正規化した値。

```javascript
// parse_paipu.js で局ごとに集計してレコードに付与
// event 'gang'/'kaigang'/'angang' の発生回数をカウント
```

期待する効果:
- カンが多い局面では「死牌にアクセスが多く、ドラ表示牌が増えている可能性」「補充ツモ分だけ手牌進度が進んでいる」という情報がより明示的になる
- ただし `remaining` との相関が高く、独立した情報量は小さいため **効果は微小**

**input_dim の変化**: +1次元（v10以降に組み込むなら 374 → 375）

**進め方**  
1. `parse_paipu.js` の `parse_round()` でカン回数（gang / kaigang / angang イベント）を集計し、レコードに `kan_count` フィールドを追加する  
2. `extract_features.js` の `game_state_features` または `make_hand_inference_sample` に `kan_count / 4` を追加する（最大4カンで正規化）  
3. `ai_phase2.js` のブラウザ側でも `melds_l` からカン副露をカウントする同等実装を追加する  
4. 特徴量再生成後、TICKET-031 の v10 または後続モデルで `input_dim` を +1 して効果を測定する  
5. TICKET-034 のバケット別評価と合わせて「カン回数が多い局面での精度変化」も確認する

**補足: 他に追加価値のある王牌関連情報**  
- 現時点で追加価値があると判断できる情報は上記カン回数のみ
- 王牌の具体的な構成（ドラ牌の種類・残り枚数）は原理上は牌譜から計算可能だが、実際の対局では非公開情報のため入力特徴量として使うべきではない（データリーク）
- 残り牌数を turn_bucket（序盤/中盤/終盤）としてバケット化した one-hot 特徴量を追加することも考えられるが、`remaining / 70` の連続値の方が情報量が多いため不要

---

## 特殊牌対応グループ

> 仕様確定（TICKET-002）が他すべての前提。TICKET-003 完了後は TICKET-004/005/006 を並行着手可。

### TICKET-002: 特殊牌の仕様を確定する

**状態**: 未着手  
**優先度**: 低  
**依存**: なし

**概要**  
コード着手前に特殊牌のルール上の振る舞いをすべて決定する。仕様が未確定のままコードを書くと手戻りが大きい。チェックリストは `SPECIAL_PAI_PLAN.md` §5 を参照。

**進め方**  
1. `SPECIAL_PAI_PLAN.md` の仕様確定チェックリストを埋める  
2. スート識別子・種類数・枚数・順子可否・鳴き可否・国士対象・ドラ有無・既存役との関係・新設役の有無・山への混入方法 をすべて決定する  
3. 決定内容を `SPECIAL_PAI_PLAN.md` に反映してチェックリストを完了にする

---

### TICKET-003: shoupai.js を特殊牌対応に拡張する

**状態**: 未着手  
**優先度**: 低  
**依存**: TICKET-002

**概要**  
手牌管理の根幹である `shoupai.js` に特殊牌スートを追加する。ここが通れば他のファイルへの影響範囲が芋づる式に明確になる。

**進め方**  
1. `valid_pai()` の正規表現に特殊スートと番号範囲を追加する  
2. コンストラクタの `_bingpai` に特殊スートのキーと配列を追加する  
3. `fromString()` / `toString()` のパース・直列化ループに特殊スートを追加する  
4. `clone()` のコピー処理に特殊スートを追加する  
5. `get_dapai()` / `get_chi_mianzi()` / `get_peng_mianzi()` / `get_gang_mianzi()` の候補列挙を仕様に合わせて修正する  
6. 既存テストがすべて通ることを確認し、特殊牌用テストケースを追加する

---

### TICKET-004: shan.js の牌山生成に特殊牌を組み込む

**状態**: 未着手  
**優先度**: 低  
**依存**: TICKET-003

**概要**  
牌山（136枚）の生成ループに特殊牌の種類・枚数を追加する。山に混ぜない仕様の場合はこのチケットは別途検討する。

**進め方**  
1. コンストラクタ L.22–29 のループに特殊牌の生成を追加する  
2. ドラになる仕様の場合は `zhenbaopai()` にドラ循環ロジックを追加する（不要なら対象外に設定）  
3. 牌山の総枚数が意図通りになることをテストで確認する

---

### TICKET-005: xiangting.js のシャンテン数計算を特殊牌に対応させる

**状態**: 未着手  
**優先度**: 低  
**依存**: TICKET-003

**概要**  
シャンテン数計算・待ち牌列挙で特殊牌が正しく扱われるようにする。特殊牌を「数牌型（順子あり）」か「字牌型（刻子のみ）」に分類して各関数に追加する。

**進め方**  
1. `mianzi_all()` L.66–99 に特殊牌スートを追加し、数牌型か字牌型かに応じて分岐する  
2. `xiangting_guoshi()` L.121–138 の幺九牌リストに特殊牌を追加するか除外するかを仕様に従って決定する  
3. `tingpai()` L.169–185 の `['m','p','s','z']` ループに特殊スートを追加する  
4. テストケースを追加する

---

### TICKET-006: hule.js の和了・役判定を特殊牌に対応させる

**状態**: 未着手  
**優先度**: 低  
**依存**: TICKET-003

**概要**  
和了形の分解・役判定・点数計算で特殊牌が正しく扱われるようにする。専用役を新設する場合はここに追加する。

**進め方**  
1. `mianzi_all()` L.40–63 に特殊牌スートを追加し、順子可否に応じて分岐する  
2. `hule_mianzi_guoshi()` L.131–159 の幺九牌定義を仕様に従って修正する  
3. `get_hudi()` L.199–297 の役分類正規表現に特殊牌を追加する  
4. 専用役がある場合は `get_hupai()` L.317–552 に追加する  
5. テストケースを追加する

---

### TICKET-007: 架空牌譜（特殊牌含む）のフォーマットを設計する

**状態**: 未着手  
**優先度**: 低  
**依存**: TICKET-002

**概要**  
特殊牌を含む牌譜 JSON の記述方法を定義し、牌譜ビューアに読み込めるサンプルを作成する。

**進め方**  
1. 既存の牌譜 JSON フォーマット（`CODEBASE_KNOWLEDGE.md` §5）を参照する  
2. 特殊牌スートを含む牌文字列を使ったサンプル牌譜 JSON を手書きで作成する  
3. `paipu.html` の「牌譜追加」ボタンでアップロードし、ビューアがクラッシュしないことを確認する（コード改修前の動作確認）

---

### TICKET-008: 特殊牌の牌画像（GIF）を用意する

**状態**: 未着手  
**優先度**: 低  
**依存**: TICKET-002

**概要**  
フロントエンドで特殊牌を表示するための画像ファイルを `tmp_clone/dist/img/` に追加する。

**進め方**  
1. 既存の `dist/img/` 内の牌画像のファイル名規則・サイズ・フォーマットを確認する  
2. 特殊牌のデザインを決定し、同じ規則に合わせた GIF ファイルを作成する  
3. `dist/img/` に配置し、`paiga.html` で表示されることを確認する

---

### TICKET-009: 牌譜ビューアで特殊牌が正しく表示されるか確認する

**状態**: 未着手  
**優先度**: 低  
**依存**: TICKET-006, TICKET-008

**概要**  
コード改修と画像追加が完了した後、特殊牌を含む実際の対局牌譜を牌譜ビューアで再生し、表示・挙動に問題がないことを確認する。

**進め方**  
1. TICKET-006 完了後、特殊牌を含む対局を実際に動かして牌譜 JSON を取得する  
2. `paipu.html` にアップロードして再生する  
3. 特殊牌の画像・鳴き表示・役表示などに崩れがないことを確認する  
4. 問題があれば `majiang-ui` 側（未クローン）の修正が必要か判断する
