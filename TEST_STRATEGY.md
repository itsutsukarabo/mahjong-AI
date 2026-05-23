# テスト戦略

最終更新: 2026-05-24

---

## 1. 方針

- **仕様はテストで担保する**: 実装前にテストケースを定義し、何が正しい動作かを明文化してからコードを書く
- **CI で自動検証**: `git push` のたびに全テストが自動実行される。green でないブランチはマージしない
- **リアル通信は CI 内でしない**: 天鳳サーバーへのHTTPリクエストはテスト内ではモックに差し替える。実際のAPIはローカル手動確認のみ

---

## 2. リポジトリ構成とテスト対象の対応

```
mahjong-AI/
├── majiang-core/          ← 改修対象①
│   ├── lib/board.js       ← kita() 追加
│   └── test/board.js      ← ユニットテスト（Mocha / TDD）
│
├── tenhou-log-local/      ← 改修対象②
│   ├── lib/convlog.js     ← 三麻変換ロジック
│   ├── server.js          ← HTTP サーバー
│   └── test/
│       ├── convlog.js     ← ユニットテスト
│       ├── server.js      ← 統合テスト
│       └── fixtures/
│           ├── sanma_sample.xml      ← 実際の天鳳三麻 XML
│           └── sanma_expected.json   ← 期待変換結果
│
├── tmp_clone/
│   └── patches/           ← patch-package パッチファイル（CI で自動適用）
│
└── .github/workflows/ci.yml
```

---

## 3. テスト種別と対象ファイル

### 3-1. ユニットテスト: `convlog.js`

**ファイル**: `tenhou-log-local/test/convlog.js`  
**フレームワーク**: Mocha (`-u tdd`)  
**実行**: `npm test` in `tenhou-log-local/`

| テストスイート | 検証内容 |
|---|---|
| `parse_type(185)` | `type.sanma` が truthy、戻り値が `"三"` で始まる |
| 北抜き N タグ変換 | `m=30752/31008/31264` → `{kita:{l, p:'z4'}}`、`gang=true` で次 draw が gangzimo になる |
| `qipai()` 三麻 | `hai3=""` → `shoupai[3]=''`、`ten[3]=0` → `defen[3]=0` |
| `owari` 三麻 | 4番目の rank=4（強制最下位）、defen=0、point=0.0 |
| full conversion（スナップショット） | `sanma_sample.xml` → `sanma_expected.json` と `deepEqual` |

**フィクスチャ作成手順**（初回のみ手動）:
1. `http://tenhou.net/0/log/?2025080617gm-00b9-0000-104adf08` の内容を `sanma_sample.xml` として保存
2. 改修済み `convlog.js` で変換した結果を `sanma_expected.json` として保存
3. 以降はテストコードが「期待値と一致するか」を自動検証する

---

### 3-2. ユニットテスト: `board.js kita()`

**ファイル**: `majiang-core/test/board.js`（既存スイートに追加）  
**フレームワーク**: Mocha (`-u tdd`)  
**実行**: `npm test` in `majiang-core/`

| テストケース | 検証内容 |
|---|---|
| 手番が更新される | `board.lunban === kita.l` |
| 手牌から z4 が 1 枚除去される | `shoupai.toString()` に z4 が 1 枚減っている |
| 複数回連続して呼べる | 2回・3回と呼ぶたびに z4 が減る |
| kita 後に zimo / dapai が正常に呼べる | kita → zimo → dapai の順に呼んでエラーにならない |

---

### 3-3. 統合テスト: `server.js`

**ファイル**: `tenhou-log-local/test/server.js`  
**フレームワーク**: Mocha + supertest + sinon  
**実行**: `npm test` in `tenhou-log-local/`

```
GET /tenhou-log/:id.json
  ├── getlog をフィクスチャ XML で stub したとき
  │     ├── HTTP 200 を返す
  │     ├── Content-Type が application/json である
  │     └── レスポンスボディに kita アクションが含まれる（三麻の場合）
  ├── getlog が 404 エラーを返したとき
  │     └── HTTP 404 を伝播する
  └── getlog が 500 エラーを返したとき
        └── HTTP 500 を伝播する
```

**モック戦略**:
- `sinon.stub(getlog, 'fetch').resolves(fixtureXml)` のように天鳳への通信を差し替える
- CI からは天鳳サーバーに一切アクセスしない

---

### 3-4. ビルドスモークテスト

**実行**: CI の `build` ジョブで `npm run build:js` が通るかを確認  
**目的**: webpack がバンドルエラーなく完走することを担保する  
**ノート**: `patch-package` の postinstall が CI で正しく動くことも間接的に確認される

---

## 4. GitHub Actions 設計

**ファイル**: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: ['**']
  pull_request:

jobs:
  test-majiang-core:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: majiang-core
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: majiang-core/package-lock.json
      - run: npm ci
      - run: npm test

  test-tenhou-log-local:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: tenhou-log-local
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: tenhou-log-local/package-lock.json
      - run: npm ci
      - run: npm test

  build:
    runs-on: ubuntu-latest
    needs: [test-majiang-core, test-tenhou-log-local]
    defaults:
      run:
        working-directory: tmp_clone
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: tmp_clone/package-lock.json
      - run: npm ci        # postinstall で patch-package が自動適用される
      - run: npm run build:js
```

---

## 5. テストを書く順序（チケットとの対応）

```
TICKET-010  CI yml を書いて push → build ジョブだけ先に green にする
TICKET-011  tenhou-log-local の package.json を作る → test ジョブが追加される
TICKET-012  convlog テストを先に書いてから改修する（TDD）
TICKET-013  server テストを先に書いてから実装する（TDD）
TICKET-014  board.js kita テストを先に書いてから実装する（TDD）
TICKET-015  パッチ適用後に build ジョブで確認
TICKET-016  手動動作確認（CI 外）
```

---

## 6. CI が通らない変更はマージしない（ブランチ保護設定）

GitHub リポジトリの Settings → Branches → Branch protection rules で以下を設定:
- `main` への直接 push を禁止
- PR へのマージ条件: `test-majiang-core` + `test-tenhou-log-local` + `build` がすべて green

---

## 7. 将来の拡張（今回スコープ外）

| テスト種別 | 対象 | 備考 |
|---|---|---|
| E2E テスト | paipu.html のブラウザ再生 | Playwright 等で実装可。初期コストが高いため今回は見送り |
| カバレッジ計測 | convlog.js, board.js | nyc は majiang-core に既設。tenhou-log-local にも追加可 |
| 点数計算テスト | hule.js 三麻対応（将来） | 三麻役（北抜きドラ加算等）追加時に必要 |
