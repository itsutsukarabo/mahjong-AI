# 電脳麻将 拡張プロジェクト ロードマップ

最終更新: 2026-05-23

---

## プロジェクト概要

電脳麻将（`@kobalab/majiang-core` ベース）に**特殊牌**を追加し、通常の麻雀牌では実現できない新しいゲーム体験を作る。

---

## やりたいこと一覧

### 1. 特殊牌のシステム拡張（majiang-core 改修）
- [ ] 特殊牌の仕様定義（種類・枚数・スート識別子・ルール上の振る舞い）
- [ ] `shoupai.js` の牌バリデーション・内部構造を特殊牌対応に拡張
- [ ] `shan.js` の牌山生成に特殊牌を組み込む
- [ ] `xiangting.js` のシャンテン数計算を特殊牌に対応させる
- [ ] `hule.js` の和了判定・役判定を特殊牌に対応させる

### 2. 牌譜まわりの活用・整備
- [ ] 架空牌譜（特殊牌を含む）のフォーマット設計
- [ ] 牌譜ビューアで特殊牌が正しく表示されるか検証
- [ ] 特殊牌の牌画像（GIF）を用意して `dist/img/` に追加

### 3. ローカル開発環境の整備
- [x] `tmp_clone/` への `npm install` と `npm run build` の動作確認
- [x] `/build-majiang` スキルの作成（ビルド＋サーバー起動）
- [ ] `majiang-core` をローカルパス参照（`file:../majiang-core`）に切り替えて開発しやすくする

---

## 実装計画

### フェーズ 1: 仕様確定（コード着手前）

特殊牌の以下の仕様を決定する。詳細は [`SPECIAL_PAI_PLAN.md`](./SPECIAL_PAI_PLAN.md) のチェックリストを参照。

| 項目 | 状態 |
|---|---|
| スート識別子（文字） | 未定 |
| 種類数・枚数 | 未定 |
| 順子を組めるか | 未定 |
| 鳴き可否（チー／ポン／カン） | 未定 |
| 国士無双の対象か | 未定 |
| ドラになるか | 未定 |
| 専用役の新設 | 未定 |

---

### フェーズ 2: majiang-core 改修

改修順序と対象ファイルは [`SPECIAL_PAI_PLAN.md`](./SPECIAL_PAI_PLAN.md) の「推奨実装順序」を参照。

```
Step 1: shoupai.js  — valid_pai() 正規表現・_bingpai 構造・パース/直列化
Step 2: shan.js     — 牌山生成ループ・ドラ循環
Step 3: xiangting.js — シャンテン数計算への特殊牌分類追加
Step 4: hule.js     — 和了形分解・役判定への追加
Step 5: テスト追加
```

---

### フェーズ 3: フロントエンド対応

- 特殊牌の牌画像作成・配置（`dist/img/`）
- `majiang-ui` 側の表示処理確認・修正（未クローン）
- 牌譜ビューアでの動作確認

---

## 参照ドキュメント

| ファイル | 内容 |
|---|---|
| [`CODEBASE_KNOWLEDGE.md`](./CODEBASE_KNOWLEDGE.md) | コードベース調査メモ（牌フォーマット・ファイル責務・牌譜構造など） |
| [`SPECIAL_PAI_PLAN.md`](./SPECIAL_PAI_PLAN.md) | 特殊牌拡張の詳細実装計画・仕様チェックリスト |

## リポジトリ構成（このワークスペース）

```
mahjong-AI/
├── tmp_clone/        # 電脳麻将メインアプリ（Majiang）
│   ├── src/          # ソース（pug/stylus/js）
│   ├── dist/         # ビルド成果物（HTML/CSS/JS/img/audio）
│   └── package.json
├── majiang-core/     # majiang-core（直接改修対象）
│   └── lib/          # shoupai.js, shan.js, hule.js, xiangting.js ...
├── ROADMAP.md        # このファイル
├── CODEBASE_KNOWLEDGE.md
└── SPECIAL_PAI_PLAN.md
```
