# AI作成 フェーズ1: 牌譜分析モーダル実装計画

作成日: 2026-05-27  
最終更新: 2026-05-27（実装完了後に実装内容を反映）

---

## 目的

牌譜再生機能を拡張し、切り番プレイヤーの手番ごとに現行ルールベースAI（`@kobalab/majiang-ai`）の分析結果をモーダルで表示する。
フェーズ2（教師あり学習）の出力を差し込めるUIスロットも同時に設計する。

---

## 実装状況サマリ

| タスク | 状態 |
|---|---|
| AIエンジンモジュール (`ai-analyzer-engine.js`) | ✅ 完了 |
| モーダルHTML (`ai-modal.pug`) | ✅ 完了 |
| モーダルCSS (`ai-modal.styl`) | ✅ 完了 |
| `Paipu` クラス拡張 (`paipu.js`) | ✅ 完了 |
| 候補ランキングテーブル表示 | ✅ 完了 |
| 残り牌数テーブル (`paishu`) | ✅ 完了 |
| 牌価値スコアテーブル (`paijia_fn`) | ✅ 完了 |
| フェーズ2スロットプレースホルダー | ✅ 完了（静的表示のみ） |
| Jest テストコード | ✅ 完了（27テスト全通過） |
| 危険度マップ（特徴量パネルへの weixian 全牌表示） | 🔲 未実装 |
| `eval_shoupai()` 内訳ログ化 | 🔲 未実装 |
| フェーズ2スロットの実際のフックアップ | 🔲 未実装 |

---

## 背景・前提

### 現行AIの `info` 引数

`majiang-ai` の `select_*` メソッドは `info` 引数を渡すと候補評価の配列を書き込んで返す。

```js
// 打牌分析
const info = [];
player.select_dapai(info);
// info = [
//   { p: 'm3', n_xiangting: 0, ev: 412.5, n_tingpai: 8,
//     tingpai: ['m2','m5','p3',...], weixian: 2.1 },
//   { p: 'p7', n_xiangting: 0, ev: 380.0, n_tingpai: 6,
//     tingpai: [...], weixian: 0.0 },
//   ...
// ]

// 副露分析
const info = [];
player.select_fulou(dapai, info);
// info = [
//   { m: '',       n_xiangting: 1, ev: 320.0 },   // パス
//   { m: 'p456+',  n_xiangting: 0, ev: 410.0 },   // ポン
// ]

// カン分析（select_gang → dapai_info に追記される）
player.select_gang(dapai_info);
// カン候補は { p, m: 'm1111', n_xiangting, ev, tingpai, n_tingpai } の形
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
| `selected` | 打牌・副露 | `true` = AIが選択した最良候補（実装側で付与） |

---

## 設計決定事項（実装調査で判明したこと）

### ログリプレイ方式（`minipaipu()` を使わない）

当初計画では `minipaipu()` で局面を再現する予定だったが、**直接ログをリプレイする方式**を採用した。

**理由:**
- `minipaipu()` の `heinfo` フォーマットが複雑で、牌譜から正確に構築するのが困難
- `player.action(log[i], ()=>{})` でコールバックを空にするだけでAIを副作用なしに任意局面まで進められる
- `eval_shoupai()` のキャッシュはリプレイ中の `action_zimo()` で自動的にウォームアップされるため、最終的な `select_dapai(info)` は高速

```js
// 採用した方式
const player = new AI();
player.action({ kaiju: { id: viewpoint, rule: Majiang.rule(),
    title: paipu.title, player: paipu.player, qijia: paipu.qijia }}, ()=>{});
for (let i = 0; i <= current_idx; i++) {
    player.action(log[i], ()=>{});  // コールバック空 = 副作用なし
}
// この時点で player はターゲット局面の状態を持つ
```

### エンジンモジュール分離

分析ロジックを `ai-analyzer-engine.js` に分離し、DOM に依存しない純粋関数として実装した。
- Jest でテスト可能
- フェーズ2接続時に UI 側を変更せずエンジンだけ差し替えられる

### カン分析の扱い

カン候補は打牌候補テーブルに統合される（`select_gang(dapai_info)` の後に `select_dapai(dapai_info)` を呼ぶ）。
独立した「カン分析」タブは設けず、打牌タブ内に混在表示する。

### `SuanPai` の内部構造

```js
// _paishu: 残り牌数
// { m: [赤牌枚数, 4,4,4,4,4,4,4,4,4], p:[...], s:[...], z:[0,4,4,4,4,4,4,4] }
// インデックス0は赤牌(0)カウント、インデックス1〜9が各牌の残り枚数

// make_paijia(shoupai): 手牌コンテキストを考慮した牌価値関数を返す
// 返り値: (p: string) => number
// 乗数: 字牌役牌(≥9/6枚)→×8, 染め手狙い→×4, 一色多い→×2, 通常→×1
const paijia_fn = player._suanpai.make_paijia(player.shoupai);
const score = paijia_fn('m3');  // 数値を返す
```

### analysis_type の判定ロジック

```js
if (current.zimo && current.zimo.l === mf)
    → 'dapai' (select_gang + select_dapai)
else if (current.gangzimo && current.gangzimo.l === mf)
    → 'dapai' (select_dapai のみ)
else if (current.fulou && current.fulou.l === mf
         && !current.fulou.m.match(/^[mpsz]\d{4}/))
    → 'dapai' (副露後打牌: select_dapai)
else if (current.dapai && current.dapai.l !== mf)
    → 'fulou' (select_fulou)
else
    → null (分析対象外)
```

---

## ファイル構成

### 新規作成ファイル

| ファイル | 役割 |
|---|---|
| `tmp_clone/node_modules/@kobalab/majiang-ui/lib/ai-analyzer-engine.js` | 純粋分析エンジン（DOM不要、testable） |
| `tmp_clone/src/html/inc/ai-modal.pug` | モーダルHTML骨格 |
| `tmp_clone/src/css/ai-modal.styl` | モーダルCSS |
| `tmp_clone/test/ai-analyzer-engine.test.js` | Jest テスト（27テスト） |

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `tmp_clone/node_modules/@kobalab/majiang-ui/lib/paipu.js` | `ai_analyze()`, `_do_ai_analyze()`, `_render_ai_modal()`, `_render_ai_paishu()`, `_render_ai_paijia()` 追加、ボタン・キーバインド追加 |
| `tmp_clone/src/html/inc/board.pug` | `include ai-modal` 追加 |
| `tmp_clone/src/html/inc/controller.pug` | `button.ai-analyze` 追加 |
| `tmp_clone/src/css/index.styl` | `@import "ai-modal"` 追加 |
| `tmp_clone/package.json` | Jest 設定・devDependency 追加 |
| `tmp_clone/patches/@kobalab+majiang-ui+1.6.0.patch` | node_modules 変更を patch-package で管理 |

---

## `analyze()` 関数の仕様

```js
// ファイル: node_modules/@kobalab/majiang-ui/lib/ai-analyzer-engine.js

/**
 * @param {object} paipu       牌譜オブジェクト (title, player, qijia, log)
 * @param {number} log_idx     対象の局インデックス
 * @param {number} current_idx 分析対象の手番インデックス (0-base, inclusive)
 * @param {number} viewpoint   分析視点のプレイヤーID
 * @returns {object}
 */
function analyze(paipu, log_idx, current_idx, viewpoint)
```

### 戻り値

| フィールド | 型 | 内容 |
|---|---|---|
| `analysis_type` | `'dapai' \| 'fulou' \| null` | 分析種別 |
| `dapai_info` | `Array<{p, m?, n_xiangting, ev, tingpai, n_tingpai, weixian, selected?}>` | 打牌候補（カン候補含む） |
| `fulou_info` | `Array<{m, n_xiangting, ev}>` | 副露候補 |
| `best_dapai` | `string \| null` | AIが選んだ最良打牌。リーチ候補は `"s4_*"` 形式 |
| `paishu` | `{m:[],p:[],s:[],z:[]}` | 残り牌数（インデックス0=赤牌、1〜9=各牌） |
| `paijia_fn` | `(p: string) => number` | 手牌コンテキスト込みの牌価値関数 |
| `menfeng` | `number` | 視点プレイヤーの門風（0=東〜3=北） |

### `dapai_info` の `selected` フラグ付与ロジック

```js
// zimo 後の通常打牌: m フィールドなしかつ p が best_dapai の先頭2文字と一致
dapai_info.forEach(i => {
    if (!i.m && i.p === best_dapai.slice(0, 2)) i.selected = true;
});
// ※ best_dapai は "s4_*"（ツモ切りリーチ）のような形もある
```

---

## モーダルUI構成

```
[AI解析モーダル]
├── ヘッダー: "AI解析" タイトル + × 閉じるボタン
├── ローディング: "計算中..." (分析完了で非表示)
│
└── ボディ
    ├── タブナビ: [打牌分析] [副露分析]
    │
    ├── 打牌タブ (#ai-tab-dapai)
    │   └── 候補テーブル: 牌/面子 | 向聴 | 期待値 | テンパイ牌 | 危険度
    │
    ├── 副露タブ (#ai-tab-fulou)
    │   └── 候補テーブル: 鳴き | 向聴 | 期待値
    │
    ├── 特徴量パネル (<details> 折りたたみ)
    │   ├── 残り牌数テーブル (4スーツ × 9牌)
    │   └── 牌価値スコアテーブル (4スーツ × 9牌)
    │
    └── フェーズ2スロット (現在はプレースホルダーのみ)
        ├── "手牌推定モデル" スロット
        └── "方策モデル" スロット
```

### トリガー

- 再生コントロールバーの **「AI解析」ボタン** をクリック
- キーボードショートカット **`u`** キー

### パフォーマンス対策

```js
// モーダルを「計算中」状態で表示してから setTimeout(20ms) 後に分析実行
// UI のペイントが先に走り、フリーズ感を軽減
ai_analyze() {
    modal.removeClass('hide');
    loading.show(); body.hide();
    setTimeout(() => this._do_ai_analyze(modal), 20);
}
```

---

## テスト仕様

ファイル: `test/ai-analyzer-engine.test.js`

### テストフィクスチャ

```js
const HAND_0   = 'm1234567p123s123';      // 好形13枚 (7+3+3)
const HAND_PON = 'm234p123s123z111z2';    // 役牌カン狙い13枚 (z111 + z2)
```

### テストスイート（27テスト）

| describe | テスト数 | 検証内容 |
|---|---|---|
| 打牌分析 (dapai) | 7 | analysis_type, 候補数, フィールド, selected, best_dapai, fulou_info空 |
| 副露分析 (fulou) | 6 | analysis_type, 候補数, パス選択肢, z1への鳴き, フィールド, dapai_info空 |
| 分析対象外ターン | 4 | analysis_type=null, dapai_info空, fulou_info空, best_dapai=null |
| 残り牌数 (paishu) | 4 | スーツ存在, 0〜4の整数, 手牌牌の減少, 字牌インデックス |
| 牌価値スコア (paijia_fn) | 5 | 関数型, 非負スコア, 連続形 > 無価値字牌, 赤五牌 |
| 門風 (menfeng) | 2 | 視点0=東家, 視点1=南家 |

### 重要なハマりポイント（修正済み）

1. **`best_dapai` のフォーマット**: リーチ候補は `"s4_*"` 形式。正規表現は `/^[mpsz][0-9]/` で先頭2文字チェックのみ
2. **手牌枚数**: `HAND_PON` は13枚必須。12枚だと状態再現が狂い `fulou_info` が空になる
3. **ポン vs 大明杠**: `z111` 持ちで相手が `z1` を捨てると大明杠 (`z1111+`) になる。チェックは `i.m.startsWith('z1')`
4. **牌価値比較**: `s2 vs z1` は両方12で等しい（`z1` = 場風 + 門風 = ×4）。`z4`（北）と比較する

---

## 残実装タスク

### 🔲 危険度マップ（特徴量パネル）

特徴量パネルに weixian の全牌マップを追加する。

```js
// SuanPai.suan_weixian_all(bingpai) でクロージャを取得
// 引数: player._suanpai の内部メソッド
// 返り値: (p) => weixian% のクロージャ、またはリーチ者がいない場合は undefined
const weixian_fn = player._suanpai.suan_weixian_all(player._bingpai_open);
// weixian_fn が存在する場合のみ全牌でマップ描画
```

実装箇所: `_render_ai_modal()` の特徴量パネルに `_render_ai_weixian()` を追加

### 🔲 `eval_shoupai()` 内訳ログ化

現状 `eval_shoupai()` は最終スコアしか返さない。「どのテンパイ牌が何枚残ってどう加算されたか」の内訳を取るには以下が必要:

```js
// majiang-ai/lib/player.js の eval_shoupai() を monkey-patch またはサブクラス化
// trace 配列に { tingpai, paishu_remaining, ev_contribution } を記録する
// フェーズ1の特徴量パネルに内訳テーブルとして表示
```

### 🔲 フェーズ2スロットの実際のフックアップ

現在スロットは静的なプレースホルダーのみ。フェーズ2モデルを差し込む口を実装する。

---

## フェーズ2との接続インターフェース設計

```js
// フェーズ2のモデルが実装するインターフェース
class HandInferenceModel {
    // 公開情報から他家の手牌を推定する
    predict(player_model, suanpai) {
        // 返り値: { [menfeng]: { [pai]: { 0: prob, 1: prob, 2: prob, 3: prob } } }
    }
    predict_tenpai(player_model, suanpai) {
        // 返り値: { [menfeng]: { tenpai_prob: 0.72, winning_tiles: { 'p3': 0.45, ... } } }
    }
}

class PolicyModel {
    // 行動クローン / 強化学習後の方策
    predict_action(player_model, suanpai, candidates) {
        // 返り値: candidates に確率フィールドを付与した配列
    }
}

// フェーズ2スロットへの差し込み方法（案）
// _do_ai_analyze() 内で、グローバルまたは注入されたモデルを参照する
if (window.AI_PHASE2?.handModel) {
    result.hand_inference = window.AI_PHASE2.handModel.predict(player, suanpai);
}
```

---

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `tmp_clone/node_modules/@kobalab/majiang-ui/lib/ai-analyzer-engine.js` | 純粋分析エンジン（DOM不要） |
| `tmp_clone/node_modules/@kobalab/majiang-ui/lib/paipu.js` | 牌譜再生エンジン（モーダル起動ロジック含む） |
| `tmp_clone/src/html/inc/ai-modal.pug` | モーダルHTML |
| `tmp_clone/src/css/ai-modal.styl` | モーダルCSS |
| `tmp_clone/test/ai-analyzer-engine.test.js` | Jestテスト |
| `tmp_clone/patches/@kobalab+majiang-ui+1.6.0.patch` | node_modules差分（patch-package管理） |
| `tmp_clone/node_modules/@kobalab/majiang-ai/lib/player.js` | `select_*` メソッド・`info` 引数の実装 |
| `tmp_clone/node_modules/@kobalab/majiang-ai/lib/suanpai.js` | `_paishu`, `make_paijia`, `suan_weixian_all` |
| `AI_INVESTIGATION.md` | 現行AI仕様の全体調査結果 |
| `AI_CREATION_PHASE2_SUPERVISED_LEARNING.md` | フェーズ2詳細計画 |
