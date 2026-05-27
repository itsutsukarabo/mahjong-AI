# majiang-AI 現行仕様調査結果

調査日: 2026-05-27

---

## パッケージ構成

| パッケージ | 場所 | バージョン | 役割 |
|---|---|---|---|
| `@kobalab/majiang-core` | `majiang-core/lib/` (ローカル編集版) | 1.4.1 | ゲームエンジン本体 |
| `@kobalab/majiang-ai` | `tmp_clone/node_modules/@kobalab/majiang-ai/` | 1.1.0 | ルールベースAI |
| `@kobalab/majiang-ui` | `tmp_clone/node_modules/@kobalab/majiang-ui/` | — | ブラウザUI |

参照元リポジトリ:
- `majiang-core`: https://github.com/kobalab/majiang-core
- `majiang-ai`: https://github.com/kobalab/majiang-ai

---

## majiang-core: ゲームエンジン

### ファイル構成

```
majiang-core/lib/
  index.js      — エントリポイント
  game.js       — ゲーム進行管理
  player.js     — プレイヤー基底クラス
  board.js      — プレイヤー側の盤面モデル
  shoupai.js    — 手牌管理
  he.js         — 捨て牌管理
  shan.js       — 山牌管理
  rule.js       — ルール定義
  xiangting.js  — 向聴数計算
  hule.js       — 和了判定・点数計算

majiang-core/dev/
  game.js       — 牌譜再現用Gameサブクラス（テスト用）
  index.js      — devエントリポイント
```

### Game クラス (`game.js`)

ゲーム進行の中心。4人のプレイヤーへのメッセージ配信と応答収集を管理する。

**コンストラクタ**
```js
new Game(players, callback, rule, title)
// players: [Player, Player, Player, Player]
// callback: 対局終了時に paipu を受け取る関数
// rule: rule() で生成したルールオブジェクト
```

**主要メソッド**

| メソッド | 説明 |
|---|---|
| `do_sync()` | 同期実行。ループで即完走。強化学習の高速自己対戦に使う |
| `start()` | 非同期実行（ブラウザ向け） |
| `kaiju(qijia?)` | 開局処理 |
| `qipai(shan?)` | 配牌処理 |
| `zimo()` / `dapai()` / `fulou()` / `gang()` | 各局面の進行 |
| `call_players(type, msg, timeout)` | 全プレイヤーに状態通知し応答待ち |
| `reply(id, reply)` | 4人全員の応答が揃ったら `next()` を呼ぶ |
| `stop(callback)` | ゲーム停止 |

**ゲーム進行フロー**
```
kaiju → qipai → [zimo → dapai → (fulou → dapai)? → (gang → gangzimo)?]* → hule/pingju → ...
```

**対局結果 (paipu)**
```js
paipu = {
  title:  string,
  player: [string, string, string, string],  // プレイヤー名
  qijia:  number,                            // 起家
  log:    [...],                             // 全局の牌譜ログ
  defen:  [number, ...],                     // 最終持ち点
  point:  [number, ...],                     // ウマ後ポイント
  rank:   [number, ...]                      // 順位 (1-4)
}
```

**静的ユーティリティメソッド（合法手判定）**

| メソッド | 説明 |
|---|---|
| `Game.get_dapai(rule, shoupai)` | 打牌可能な牌一覧 |
| `Game.get_chi_mianzi(rule, shoupai, p, paishu)` | チー可能な面子一覧 |
| `Game.get_peng_mianzi(rule, shoupai, p, paishu)` | ポン可能な面子一覧 |
| `Game.get_gang_mianzi(rule, shoupai, p, paishu, n_gang)` | カン可能な面子一覧 |
| `Game.allow_lizhi(rule, shoupai, p, paishu, defen)` | リーチ合法性チェック |
| `Game.allow_hule(rule, shoupai, p, zhuangfeng, menfeng, hupai, neng_rong)` | 和了合法性チェック |
| `Game.allow_pingju(rule, shoupai, diyizimo)` | 流局宣言合法性チェック |
| `Game.allow_no_daopai(rule, shoupai, paishu)` | ノーテン宣言合法性チェック |

### Player クラス (`player.js`)

AIが継承する基底クラス。

**通信インターフェース**
```js
action(msg, callback)
// msg: { kaiju/qipai/zimo/dapai/fulou/gang/kaigang/hule/pingju/jieju: {...} }
// callback: アクション応答関数
//   callback()          → パス
//   callback({dapai: 'p3'})
//   callback({fulou: 'm123+'})
//   callback({gang: 'm1111'})
//   callback({hule: '-'})
//   callback({daopai: '-'})
```

**オーバーライド対象メソッド**
```js
action_kaiju(kaiju)
action_qipai(qipai)
action_zimo(zimo, gangzimo)   // ← ツモ時: dapai/gang/hule/daopai を返す
action_dapai(dapai)           // ← 他家打牌時: hule/fulou/daopai/パス を返す
action_fulou(fulou)           // ← 自家副露後: dapai を返す
action_gang(gang)             // ← 他家カン時: hule/パス を返す
action_hule(hule)
action_pingju(pingju)
action_jieju(jieju)
```

**参照可能なプロパティ**
```js
player.shoupai     // 自分の手牌 (Shoupai)
player.he          // 自分の捨て牌 (He)
player.shan        // 山牌情報 (paishu のみ参照可)
player.hulepai     // 現在のテンパイ牌一覧
player.model       // Board（全員の公開情報）
player.model.defen // 全員の持ち点
player.model.zhuangfeng  // 場風 (0=東, 1=南, ...)
player._menfeng    // 自家の門風
player._rule       // ルール設定
```

### ルール定義 (`rule.js`)

```js
rule({
  // 点数
  '配給原点': 25000,
  '順位点':   ['20.0','10.0','-10.0','-20.0'],
  '連風牌は2符': false,

  // 赤牌/クイタン
  '赤牌':         { m: 1, p: 1, s: 1 },
  'クイタンあり': true,
  '喰い替え許可レベル': 0,   // 0: なし, 1: スジ, 2: 現物

  // 局数
  '場数': 2,                 // 0: 一局, 1: 東風, 2: 東南, 4: 一荘
  '途中流局あり': true,
  '流し満貫あり': true,
  'ノーテン宣言あり': false,
  'ノーテン罰あり': true,
  '最大同時和了数': 2,       // 1: 頭ハネ, 2: ダブロン, 3: トリロン
  '連荘方式': 2,             // 0: なし, 1: 和了, 2: テンパイ, 3: ノーテン
  'トビ終了あり': true,
  'オーラス止めあり': true,
  '延長戦方式': 1,           // 0: なし, 1: サドンデス, 2: 連荘優先, 3: 4局固定

  // リーチ/ドラ
  '一発あり': true,
  '裏ドラあり': true,
  'カンドラあり': true,
  'カン裏あり': true,
  'カンドラ後乗せ': true,
  'ツモ番なしリーチあり': false,
  'リーチ後暗槓許可レベル': 2,  // 0: 不可, 1: 牌姿変化不可, 2: 待ち変化不可

  // 役満
  '役満の複合あり': true,
  'ダブル役満あり': true,
  '数え役満あり': true,
  '役満パオあり': true,
  '切り上げ満貫あり': false,
})
```

### 向聴数・和了計算 (`xiangting.js`, `hule.js`)

```js
Util.xiangting(shoupai)          // 向聴数 (-1=和了, 0=テンパイ, ...)
Util.tingpai(shoupai, xiangting_func?)  // テンパイ牌一覧
Util.hule(shoupai, rongpai, param)      // 和了判定・点数計算
// param = { rule, zhuangfeng, menfeng, hupai, baopai, jicun }
// 返り値: { defen, fu, fanshu, ... }
```

---

## majiang-ai: ルールベースAI

### ファイル構成

```
majiang-ai/lib/
  index.js      — エントリポイント (AI, AI.minipaipu をエクスポート)
  player.js     — AIプレイヤー (Player extends Majiang.Player)
  suanpai.js    — 牌数管理・危険度・価値計算 (SuanPai, Paishu)
  minipaipu.js  — ミニ牌譜再現ユーティリティ
```

### SuanPai クラス (`suanpai.js`)

AIの「見えている情報」を管理する。自分の手牌を引いた残り牌数を追跡。

**内部状態**
```js
_paishu = {
  m: [hongpai_count, 4,4,4,4,4,4,4,4,4],  // index 0: 赤牌枚数
  p: [...], s: [...], z: [0, 4,4,4,4,4,4,4]
}
_dapai   = [{},{},{},{}]   // 各プレイヤーの捨て牌セット
_lizhi   = []              // リーチ状態 (index: 座席)
_n_zimo  = 70              // 残りツモ回数
_baopai  = []              // ドラ表示牌
```

**主要メソッド**

| メソッド | 説明 |
|---|---|
| `get_paishu()` | `Paishu` オブジェクトを返す（ツモ確率付きの残り牌数） |
| `paijia(p)` | 牌の単体価値（ドラ・連携・役牌を考慮したスコア） |
| `make_paijia(shoupai)` | 手牌に応じた牌価値関数を生成（一色・役牌集中時に倍率適用） |
| `suan_weixian(p, l, c?)` | 座席lのリーチに対する牌pの危険度スコア |
| `suan_weixian_all(bingpai)` | 全リーチプレイヤー統合の危険度関数を返す |

**Paishu クラス**
```js
paishu.val(p)       // ツモ確率重み付きの残り枚数
paishu.val(p, 1)    // 実残り枚数 (real mode)
paishu.pop(p)       // 1枚減らす
paishu.push(p)      // 1枚戻す
```

**paijia の倍率ルール**
```
z1-4 (風牌) かつ 風牌合計 >= 9  → ×8
z5-7 (三元牌) かつ 三元牌 >= 6  → ×8
字牌 かつ 最大スート+字牌 >= 10 → ×4
数牌 かつ 同スート+字牌 >= 10   → ×2
それ以外                        → ×1
```

**suan_weixian の危険度スコア基準**
```
残り枚数3 (字牌): +8
残り枚数3 (数牌): +3
残り枚数2:        +3
残り枚数1:        +1
両面跨ぎ:         +10
辺張跨ぎ:         +3
中膨れ:           +3
```

### AI Player クラス (`player.js`)

`Majiang.Player` を継承したルールベースAI。

**意思決定メソッド**

#### `select_dapai(info?)` — 打牌選択

1. `suan_weixian_all()` で危険度関数を取得
2. `make_paijia()` で牌価値順にソート
3. 各打牌候補に `eval_shoupai()` で期待値を計算
4. 危険度が閾値を超える牌を除外（条件付き）
5. 期待値最大の牌を選択、リーチ可能なら `'*'` を付加

**危険度による除外条件**
```
危険度 >= 13.0                   → 無条件除外
向聴数 > 2 または 期待値 < 80   → 危険度 >= 8.0 で除外
                                  安全牌(min) < 3.2 で除外
向聴数 > 0 かつ 期待値 < 750    → 同上
向聴数 == 0 かつ 期待値 < 50    → 同上
```

#### `select_fulou(dapai, info?)` — 副露判断

- 向聴数 < 3: `eval_shoupai()` 比較で副露後の期待値が上回れば副露
  - リーチ者がいる場合: 向聴 > 0 なら期待値 >= 750, テンパイなら >= 250 が必要
- 向聴数 >= 3: `xiangting()` (役志向) ベースで向聴数が下がる副露のみ許可

#### `select_gang(info?)` — カン判断

- 向聴数 < 3: カン後の `eval_shoupai()` が下がらなければカン
- 向聴数 >= 3: カン後も `xiangting()` が同じなら許可

#### `select_hule()` — 和了判断

`allow_hule()` が true なら常に和了する。

#### `select_lizhi(p)` — リーチ判断

`allow_lizhi()` が true なら宣言（`select_dapai()` 内で期待値 >= 350 の場合に `'*'` 付加）。

#### `select_pingju()` — テンパイ流局宣言

`xiangting()` >= 4（かなりの遠手）かつ `allow_pingju()` が true なら宣言。

### `eval_shoupai(shoupai, paishu, back?)` — 手牌評価関数（コア）

再帰的に期待値を計算する。結果は `_eval_cache` にキャッシュ。

```
向聴数 == -1:
  → get_defen() で実際の和了点数を返す

向聴数 >= 0 かつ zimo あり（打牌前）:
  → 各打牌候補を試し、eval_shoupai(打牌後) の最大値

向聴数 0〜2（打牌後、テンパイ〜2向聴）:
  → Σ(残り枚数 × eval_shoupai(ツモ後)) / width[n_xiangting]
  → width = [8, 32, 64]（向聴数 0, 1, 2 に対応）

向聴数 3以上:
  → Σ(tingpai の残り枚数 × 鳴き係数)
  → 鳴き係数: ポン待ち(+)=4, チー待ち(-)=2, 素の待ち=1
```

**`xiangting(shoupai)` — 役志向拡張向聴数**

通常向聴数に加え以下の役形も評価し最小値をとる:
- 門前（メンゼン）
- 役牌（場風・門風・三元）
- 断么九
- 対対和
- 一色（清一色・混一色）

### `minipaipu` ユーティリティ (`minipaipu.js`)

既知の牌譜（または部分情報）から局面を再現してAIに読ませる。  
特定の手牌形からAIの判断を取り出すテストやベンチマークに使用。

```js
const AI = require('@kobalab/majiang-ai');
const player = new AI();
AI.minipaipu(player, baseinfo, heinfo, fix);
// baseinfo: { paistr, zhuangfeng, menfeng, baopai, hongpai, xun }
// heinfo:   他家の捨て牌・副露情報
```

---

## 強化学習に向けた構造整理

### 高速自己対戦の実行方法

```js
const Majiang = require('@kobalab/majiang-core');
const AI = require('@kobalab/majiang-ai');
const rule = Majiang.rule();

const game = new Majiang.Game(
  [new AI(), new AI(), new AI(), new AI()],
  (paipu) => { /* 結果処理 */ },
  rule
);
game.do_sync();  // 同期実行: 1局がミリ秒単位で完了
```

### 観測空間（状態表現）

| 情報 | 取得先 |
|---|---|
| 自分の手牌 | `player.shoupai` |
| 残り牌数（推定） | `player._suanpai._paishu` |
| 残りツモ回数 | `player._suanpai._n_zimo` |
| 各プレイヤーの捨て牌 | `player._suanpai._dapai` |
| リーチ状態 | `player._suanpai._lizhi` |
| 全員の持ち点 | `player.model.defen` |
| 場風・門風 | `player.model.zhuangfeng`, `player._menfeng` |
| ドラ情報 | `player.shan.baopai` |
| 向聴数 | `Util.xiangting(player.shoupai)` |

### 行動空間

| アクション | 生成方法 |
|---|---|
| 打牌 | `player.get_dapai(shoupai)` の出力 |
| チー | `player.get_chi_mianzi(shoupai, p)` の出力 |
| ポン | `player.get_peng_mianzi(shoupai, p)` の出力 |
| カン | `player.get_gang_mianzi(shoupai, p)` の出力 |
| リーチ | 打牌に `'*'` を付加 |
| 和了 | `{ hule: '-' }` |
| パス | `{}` または `callback()` |

### 報酬シグナル候補

| 報酬 | 取得先 | 特徴 |
|---|---|---|
| 最終持ち点 | `paipu.defen[id]` | 絶対値、インフレあり |
| ウマ後ポイント | `paipu.point[id]` | 相対値、比較しやすい |
| 最終順位 | `paipu.rank[id]` | 1〜4の離散値 |
| 局単位の収支 | 各局の `defen` 差分 | 密な報酬、局ごとに計算可 |

### 調整可能なパラメーター（現行ルールベースAIのもの）

| パラメーター | 場所 | 現在値 | 意味 |
|---|---|---|---|
| `width[0]` | `player.js:9` | `8` | テンパイ時の評価正規化幅 |
| `width[1]` | `player.js:9` | `32` | 1向聴時の評価正規化幅 |
| `width[2]` | `player.js:9` | `64` | 2向聴時の評価正規化幅 |
| 危険度上限1 | `player.js:332` | `13.0` | 無条件回避しきい値 |
| 危険度上限2 | `player.js:333,335` | `8.0` | 条件付き回避しきい値 |
| 危険度上限3 | `player.js:335` | `3.2` | 安全牌基準 |
| 副露期待値下限（テンパイ） | `player.js:170` | `750` | リーチ者存在時のポン/チー基準 |
| 副露期待値下限（和了） | `player.js:171` | `250` | 同上、テンパイ取り |
| リーチ期待値下限 | `player.js:392` | `350` | リーチ宣言の期待値基準 |
| 防御期待値下限1 | `player.js:333` | `80` | 防御優先の期待値境界 |
| 防御期待値下限2 | `player.js:337` | `750` | 攻撃継続の期待値境界 |
| `paijia` 倍率 | `suanpai.js:201-207` | `2, 4, 8` | 役志向時の牌価値ブースト |
