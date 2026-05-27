# AI作成 フェーズ1: 牌譜分析モーダル実装計画

作成日: 2026-05-27

---

## 目的

牌譜再生機能を拡張し、切り番プレイヤーの手番ごとに現行ルールベースAI（`@kobalab/majiang-ai`）の分析結果をモーダルで表示する。
フェーズ2（教師あり学習）の出力を差し込めるUIスロットも同時に設計する。

---

## 背景・前提

### 現行AIの `info` 引数（既存実装の活用）

`majiang-ai` の `select_*` メソッドは `info` 引数を渡すと候補評価の配列を返す設計が既にされている。

```js
// 打牌分析
let info = [];
player.select_dapai(info);
// info = [
//   { p: 'm3', n_xiangting: 0, ev: 412.5, n_tingpai: 8, tingpai: [...], weixian: 2.1 },
//   { p: 'p7', n_xiangting: 0, ev: 380.0, n_tingpai: 6, tingpai: [...], weixian: 0.0 },
//   ...
// ]

// 副露分析
let info = [];
player.select_fulou(dapai, info);
// info = [
//   { m: '',       n_xiangting: 1, ev: 320.0 },   // パス
//   { m: 'p456+',  n_xiangting: 0, ev: 410.0 },   // ポン
// ]

// カン分析
let info = [];
player.select_gang(info);
// info = [
//   { p: 'm1', m: 'm1111', n_xiangting: 0, ev: 390.0, tingpai: [...], n_tingpai: 5 },
// ]
```

### `info` フィールドの定義

| フィールド | 出力元 | 意味 |
|---|---|---|
| `p` | 打牌・カン | 候補牌（2文字） |
| `m` | 副露・カン | 候補面子文字列。`''` はパス |
| `n_xiangting` | 全アクション | アクション後の向聴数 |
| `ev` | 全アクション | `eval_shoupai()` による期待値スコア |
| `tingpai` | 打牌・カン | テンパイ牌の一覧配列 |
| `n_tingpai` | 打牌・カン | テンパイ牌の残り枚数合計 |
| `weixian` | 打牌のみ | 危険度スコア（リーチ者がいる場合） |

---

## 実装内容

### 1. 局面再現: `minipaipu()` を使う

牌譜再生は `paipu.js` が JSON を順番に dispatch するだけで `Game` を動かしていない。
特定局面でAIを動かすには「その局面を再現した `Player` インスタンス」を別途作る必要がある。
`majiang-ai` に含まれる `minipaipu()` がこの役割を担う。

```js
const AI = require('@kobalab/majiang-ai');

// 牌譜の任意ターンで局面を再現してAI分析を取得する
function analyze_turn(paipu_log, target_turn) {
    const player = new AI();

    // baseinfo: 現局面の基本情報
    const baseinfo = {
        paistr:    current_shoupai_string,  // 分析対象プレイヤーの手牌
        zhuangfeng: ...,
        menfeng:    ...,
        baopai:     [...],
        hongpai:    true,
        xun:        current_xun            // 現在の巡数
    };

    // heinfo: それまでの他家の捨て牌・副露情報
    AI.minipaipu(player, baseinfo, heinfo);

    // 分析実行
    const info = [];
    player.select_dapai(info);
    return info;
}
```

### 2. 表示するデータの種別

#### 現行ルールベースAIから取れるもの（フェーズ1で表示）

| 表示項目 | データ源 |
|---|---|
| 候補アクションのランキング | `info` 配列（`ev` 降順ソート） |
| 各候補の期待値スコア | `info[i].ev` |
| 各候補の向聴数 | `info[i].n_xiangting` |
| テンパイ牌一覧と残り枚数 | `info[i].tingpai`, `info[i].n_tingpai` |
| 危険度スコア | `info[i].weixian` |
| 残り牌数（`SuanPai._paishu`） | プレイヤー内部状態を可視化 |
| 牌価値スコア（`paijia`） | `SuanPai.paijia(p)` を全牌で計算 |

#### フェーズ2出力のためのUIスロット（フェーズ1で箱だけ作る）

| 表示項目 | フェーズ2で埋まる内容 |
|---|---|
| 各牌の保有確率（他家ごと） | 手牌類推モデルの出力（牌ごとの周辺確率） |
| 他家の聴牌確率 | 聴牌推定モデルの出力 |
| 他家の当たり牌確率 | 条件付き当たり牌確率 |
| 行動確率分布 | 行動クローンモデルの出力 |

UIスロットは初期状態では `「フェーズ2で有効化」` などのプレースホルダーを表示しておく。
モデルが差し込まれたタイミングで自動的に有効になる設計にする。

### 3. UIコンポーネント設計

#### モーダルの構成

```
[分析モーダル]
├── タブ: 打牌分析 / 副露分析 / カン分析
│
├── 候補ランキングテーブル
│   ├── 候補牌/面子
│   ├── 向聴数
│   ├── 期待値 (ev)
│   ├── テンパイ牌枚数
│   └── 危険度
│
├── 特徴量パネル（折りたたみ可）
│   ├── 残り牌数テーブル（mpsz × 0-9）
│   ├── 牌価値スコア（paijia）
│   └── 危険度マップ（各牌の weixian）
│
└── [フェーズ2スロット]
    ├── 他家手牌推定（プレースホルダー）
    └── 聴牌・当たり牌確率（プレースホルダー）
```

#### トリガー

- 牌譜再生の「停止」状態で切り番プレイヤーの手牌エリアをクリック → モーダルを開く
- または再生コントロールバーに「AI分析」ボタンを追加

### 4. パフォーマンス対策

`eval_shoupai()` は再帰的かつキャッシュありの重い計算。
UI のメインスレッドをブロックしないよう Worker（Node.js であれば Worker Threads）での非同期実行を検討する。

```js
// 非同期化の方針
// 分析開始 → モーダルを「計算中」状態で表示 → Worker から結果を受け取り → テーブル描画
```

---

## `eval_shoupai()` の内訳ログ化

現状 `eval_shoupai()` は最終スコアしか返さない。
「どのテンパイ牌が何枚残ってどう加算されたか」の内訳を取るには、計算途中の情報を記録する仕組みが要る。

```js
// eval_shoupai() を改造して途中経過を記録する案
eval_shoupai_with_trace(shoupai, paishu, back, trace = null) {
    // trace 配列に { p, paishu_val, ev_contrib } を push していく
    // フェーズ1の特徴量パネルに表示できる
}
```

---

## フェーズ2との接続インターフェース設計

フェーズ2で学習したモデルを差し込む口を統一しておく。

```js
// フェーズ2のモデルが実装するインターフェース
class HandInferenceModel {
    // 公開情報から他家の手牌を推定する
    predict(player_model, suanpai) {
        // 返り値: { [menfeng]: { [pai]: { 0: prob, 1: prob, 2: prob, 3: prob } } }
        // 例: { 1: { 'm1': { 0: 0.1, 1: 0.6, 2: 0.2, 3: 0.1 }, ... } }
    }
    predict_tenpai(player_model, suanpai) {
        // 返り値: { [menfeng]: { tenpai_prob: 0.72, winning_tiles: { 'p3': 0.45, ... } } }
    }
}

class PolicyModel {
    // 行動クローン / 強化学習後の方策
    predict_action(player_model, suanpai, candidates) {
        // 返り値: candidates に確率を付与した配列
    }
}
```

---

## 実装ステップ

1. `minipaipu()` を使った局面再現モジュールを作成する
2. `select_*` に `info` を渡して結果を収集するラッパーを作成する
3. `eval_shoupai()` の内訳ログ化を実装する（オプションフラグで有効化）
4. モーダルUIのHTMLスケルトンと CSS を作成する
5. 候補ランキングテーブルを実装する
6. 特徴量パネル（残り牌数・paijia・weixian）を実装する
7. フェーズ2スロットのプレースホルダーを実装する
8. トリガー（手牌クリック or ボタン）を実装する
9. Worker による非同期化を実装する
10. フェーズ2モデルのインターフェースを定義し、差し込み口を確認する

---

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `tmp_clone/src/js/paipu.js` | 牌譜再生エンジン（モーダル起動トリガーを追加） |
| `tmp_clone/node_modules/@kobalab/majiang-ai/lib/player.js` | `select_*` メソッド・`info` 引数 |
| `tmp_clone/node_modules/@kobalab/majiang-ai/lib/suanpai.js` | `_paishu`, `paijia`, `suan_weixian` |
| `tmp_clone/node_modules/@kobalab/majiang-ai/lib/minipaipu.js` | 局面再現ユーティリティ |
| `AI_INVESTIGATION.md` | 現行AI仕様の全体調査結果 |
| `AI_CREATION_PHASE2_SUPERVISED_LEARNING.md` | フェーズ2詳細計画 |
