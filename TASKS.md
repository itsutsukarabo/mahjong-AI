# タスク一覧

最終更新: 2026-05-24（三麻ビューア対応チケット追加）

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

**状態**: 未着手  
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

**状態**: 未着手  
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

**状態**: 未着手  
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

**状態**: 未着手  
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
