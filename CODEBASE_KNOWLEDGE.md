# コードベース調査メモ

調査日: 2026-05-23

---

## 1. リポジトリ構成

| リポジトリ | パス | 役割 |
|---|---|---|
| Majiang (メインアプリ) | `tmp_clone/` | HTML5フロントエンド。ビルド成果物は `dist/` |
| majiang-core | `majiang-core/` | ゲームロジック中核。npm: `@kobalab/majiang-core` v1.4.1 |

majiang-core 以外のサブパッケージ（majiang-ai, majiang-ui, tenhou-url-log）はこのワークスペースには未クローン。

---

## 2. 牌の表現フォーマット

### 文字列表現

牌は `[スート文字][数字]` の2文字で表す。

| スート | 対象 | 数字範囲 | 備考 |
|---|---|---|---|
| `m` | 万子 | 1–9 | `0` = 赤5 |
| `p` | 筒子 | 1–9 | `0` = 赤5 |
| `s` | 索子 | 1–9 | `0` = 赤5 |
| `z` | 字牌 | 1–7 | 1–4=風牌, 5–7=三元牌 |

サフィックス:
- `*` : リーチ宣言
- `+`, `=`, `-` : 副露元の方向（下家・対面・上家）
- `_` : 裏向き（不明牌）

バリデーション正規表現（`shoupai.js:valid_pai` L.9）:
```
/^(?:[mps]\d|z[1-7])_?\*?[\+\=\-]?$/
```

### 内部データ構造（`_bingpai`）

```js
_bingpai = {
    _:  0,                          // 不明牌の枚数
    m: [0,0,0,0,0,0,0,0,0,0],      // index 0=赤5枚数, 1–9=各数字の枚数
    p: [0,0,0,0,0,0,0,0,0,0],
    s: [0,0,0,0,0,0,0,0,0,0],
    z: [0,0,0,0,0,0,0,0],           // index 0未使用, 1–7
}
```

赤5は `bingpai[0]` に赤5の枚数、`bingpai[5]` に通常5+赤5の合計枚数を持つ二重管理。

---

## 3. majiang-core のファイル別責務

| ファイル | クラス/エクスポート | 主な責務 |
|---|---|---|
| `lib/shoupai.js` | `class Shoupai` | 手牌の管理・操作。牌のバリデーション・パース・直列化・打牌/副露/槓の手牌操作・候補列挙 |
| `lib/shan.js` | `class Shan` | 牌山。136枚生成・ツモ・カンツモ・ドラ表示管理 |
| `lib/he.js` | `class He` | 捨て牌管理。フリテン検出用 |
| `lib/hule.js` | `hule()`, `hule_param()`, `hule_mianzi()` | 和了判定・面子分解・役判定・点数計算 |
| `lib/xiangting.js` | `xiangting()`, `tingpai()` など | シャンテン数計算・待ち牌列挙 |
| `lib/board.js` | `class Board` | 卓全体の状態管理（各プレイヤーの手牌・捨て牌・点数など） |
| `lib/game.js` | `class Game` | 局進行の制御。全イベントを順番に処理し牌譜を構築 |
| `lib/player.js` | `class Player` | プレイヤー（思考ルーチン）の基底クラス |
| `lib/rule.js` | `rule()` 関数 | ルール設定オブジェクトのデフォルト値を返す |
| `lib/index.js` | — | 上記を束ねて `module.exports` |

---

## 4. 牌に関わる主要な処理とその所在

### バリデーション
- `Shoupai.valid_pai(p)` — `shoupai.js` L.8–10
- `Shoupai.valid_mianzi(m)` — `shoupai.js` L.12–31

### 牌山（136枚）の生成
- `Shan` コンストラクタ — `shan.js` L.17–41
- スートと枚数のループ: `['m','p','s','z']` × 各数字 × 4枚（`shan.js` L.22–29）

### ドラ表示牌の次牌計算
- `Shan.zhenbaopai(p)` — `shan.js` L.11–14
- `z` は `1→2→3→4→1`（風）/ `5→6→7→5`（三元）、`mps` は `n % 9 + 1` で循環

### シャンテン数計算
- `xiangting.js` の `mianzi_all()` L.66–99: `m/p/s` は順子・刻子計算、`z` は刻子のみ
- `xiangting_guoshi()` L.121–138: 幺九牌を `[1,9]`（数牌）と `[1–7]`（字牌）でハードコード

### 和了判定
- `hule.js` の `mianzi_all()` L.40–63: `['m','p','s']` のみ順子計算
- `hule_mianzi_jiulian()` L.162–183: 九蓮宝燈は `m/p/s` のみ対象

---

## 5. 牌譜（paipu）の構造と流れ

### データ構造（トップレベルは配列。要素1つ＝1ゲーム）

```js
[
  {
    title:  string,         // ゲームタイトル
    player: [名前×4],       // プレイヤー名 [l0, l1, l2, l3]
    qijia:  number,         // 起家 (0–3)
    log:    [               // 局ごとのイベント配列
      [ {qipai}, {zimo}, {dapai}, {fulou},
        {gang}, {gangzimo}, {kaigang},
        {hule} or {pingju} ],
      …
    ],
    defen:  [点数×4],       // 最終点数
    point:  [順位点×4],     // 順位点（文字列）
    rank:   [順位×4]        // 順位
  },
  …
]
```

### 各イベントオブジェクトの詳細

**`qipai`（配牌）**
```js
{ qipai: {
    zhuangfeng: 0,          // 場風 (0=東, 1=南, 2=西, 3=北)
    jushu:      0,          // 局数 (0=一局, 1=二局…)
    changbang:  0,          // 本場数
    lizhibang:  0,          // リーチ棒供託数
    defen:      [25000, 25000, 25000, 25000],  // 開始時点数
    baopai:     "p5",       // ドラ表示牌
    shoupai:    ["m127p234688s457z4", "", "", ""]
    //           ↑自分の配牌(13枚)   ↑他家は空文字
} }
```

**`zimo`（ツモ）**
```js
{ zimo: { l: 0, p: "s1" } }
// l=プレイヤー番号, p=ツモ牌（他家視点では ""）
```

**`dapai`（打牌）**
```js
{ dapai: { l: 0, p: "z4"  } }   // 手出し
{ dapai: { l: 0, "p": "m8_" } } // 自摸切り（牌末尾に _ が付く）
{ dapai: { l: 0, p: "p8*"  } }  // リーチ宣言（末尾 *）
```

> **手出し / 自摸切りの区別**: `dapai.p` の末尾に `_` があれば自摸切り、なければ手出し。
> `get_dapai()` (`shoupai.js` L.280) が自摸切り候補を `this._zimo + '_'` で生成する。

**`fulou`（副露）**
```js
{ fulou: { l: 2, m: "p234-" } }
// l=副露したプレイヤー, m=面子文字列（方向符号付き）
```

**`hule`（和了）**
```js
{ hule: {
    l:        0,                    // 和了プレイヤー
    baojia:   3,                    // 放銃プレイヤー（ツモは null）
    shoupai:  "m777p23468s34577p7*",// 和了時の手牌文字列
    fubaopai: ["s2"],               // 裏ドラ表示牌（リーチなしは null）
    hupai:    [{ name:"立直", fanshu:1 }, …],
    fu:       40,
    fanshu:   4,
    defen:    12000,
    fenpei:   [13000, 0, 0, -12000] // 各プレイヤーの点数増減
} }
```

**`pingju`（流局）**
```js
{ pingju: {
    shoupai: ["", "m123p456s789z1234", "", ""],  // 聴牌者の手牌（テンパイ開示）
    fenpei:  [3000, -1000, -1000, -1000]
} }
```

### 書き込みタイミング（`game.js` の `add_paipu()`）

| イベント | メソッド | 行番号 |
|---|---|---|
| 配牌 | `qipai()` | 258 |
| ツモ | `zimo()` | 282 |
| 打牌 | `dapai()` | 328 |
| 副露 | `fulou()` | 361 |
| 槓 | `gang()` | 379 |
| 槓ツモ | `gangzimo()` | 406 |
| 開槓 | `kaigang()` | 433 |
| 和了 | `hule()` | 510 |
| 流局 | `pingju()` | 597 |
| 終局 | `jieju()` | 670, 676, 687 |

### 保存・出力

- majiang-core 側にファイル出力機能はなく、ゲーム終了時にコールバックで `paipu` オブジェクトを返すのみ
- フロントエンド（`index.js` / `netplay.js`）が `Majiang.UI.PaipuFile` 経由でブラウザのローカルストレージに保存
  - キー: `'Majiang.paipu'`（ビューア共通）、`'Majiang.game'`（ローカル対戦）、`'Majiang.netplay'`（ネット対戦）
- 外部出力したい場合は `JSON.stringify(paipu)` するだけでよい
- 天鳳JSON形式への変換: `@kobalab/tenhou-url-log` の `logconv(paipuObj)` を使う

### 牌譜ビューアへの入力方法（`paipu.html`）

| 方法 | 手順 |
|---|---|
| **JSONファイルアップロード** | 「牌譜追加」ボタンでJSONファイルを選択 |
| **エディタで作成** | 「牌譜作成」ボタンで `Majiang.UI.PaipuEditor` を起動 |
| **URLクエリ渡し** | `paipu.html?...` で `location.search` から読み込み（`paipu.js` L.66–72） |
| **天鳳牌譜ID** | 画面下部の入力欄に天鳳の牌譜IDまたはURLを入力 |

架空牌譜を入力する最も簡単な方法はJSONファイルのアップロード。

### URL フラグメントによる局面共有

ビューア以外の各ツールページは `#paistr/baopai/...` の形式で局面を URL に埋め込める。

| ページ | 用途 |
|---|---|
| `dapai.html` | 何切る問題 |
| `hule.html` | 和了点計算 |
| `drill.html` | 点数計算ドリル |
| `paili.html` | 牌理（手牌効率） |
| `paiga.html` | 牌文字列→牌画像変換 |

---

## 6. 特殊牌を追加する際の改修対象ファイル一覧

詳細は [`SPECIAL_PAI_PLAN.md`](./SPECIAL_PAI_PLAN.md) を参照。

| 優先度 | ファイル | 改修の核心 |
|---|---|---|
| 高 | `shoupai.js` | `valid_pai()` 正規表現・`_bingpai` 構造・パース/直列化 |
| 高 | `shan.js` | 牌山生成ループ・ドラ循環 |
| 中 | `xiangting.js` | 特殊牌を数牌型/字牌型に分類 |
| 中 | `hule.js` | 和了形分解・役判定への追加 |
| 低 | `he.js` | `shoupai.js` 修正後は自動的に対応 |
