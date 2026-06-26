# ハーネスエンジニアリング Phase 2: 分析・提案

## 概要

Phase 1 で収集した `feedback_log.jsonl` を CLI で分析し、
弱点パターンをランキング表示して対応策メニューを提示する。

---

## ファイル構成

```
phase2/scripts/
  harness_report.py   ← 新規: ログ分析・ランキング・提案
```

---

## `harness_report.py` の仕様

### 実行方法

```bash
C:\ml\venv\Scripts\python.exe phase2/scripts/harness_report.py \
    --log feedback_log_*.jsonl \
    [--min-records 10]          # 集計に必要な最小サンプル数
```

### 集計軸

各レコードに対して以下の条件でグルーピングする。

| キー | 値の区分 |
|------|---------|
| `turn_bucket` | 序盤(0-9) / 中盤(10-14) / 終盤(15+) |
| `riichi` | true / false |
| `n_melds` | 0 / 1 / 2以上 |
| `checker_type` | A / B1 / B2 / B3 / B4 |

### 出力例

```
============================
  ハーネス弱点レポート
  (n=312局面 / 89ゲーム)
============================

▼ Type A: 枯れ牌ハード制約違反
  終盤(15+)・副露なし:  avg 0.18  n=47  ★★★
  終盤(15+)・副露あり:  avg 0.09  n=21  ★★
  中盤(10-14)・副露なし: avg 0.04  n=68  ★
  序盤(0-9):            avg 0.01  n=89  -

  頻出違反牌:
    字牌(z1-z4): 52件 (44%)
    数牌孤立:   31件 (26%)

▼ Type B2: リーチ後エントロピー未収束
  リーチ後・終盤: avg excess 4.2  n=23  ★★★

▼ Type B3: リーチ後ツモ切り待ち矛盾
  リーチ後: avg 0.41  n=19  ★★

▼ Type B1: 自捨牌ソフト制約違反
  終盤: avg excess 0.06  n=31  ★
  → 序盤よりは良い。許容範囲内。

▼ Type B4: PASSイベント後更新不足
  (データ不足: n=8)

============================
最重要改善ターゲット:
  1位: Type A 終盤・副露なし (score 0.18, n=47)
  2位: Type B2 リーチ後 (excess 4.2, n=23)
  3位: Type B3 リーチ後ツモ切り (score 0.41, n=19)
============================
```

---

## 提案マッピング

各弱点パターンに対して対応策を `REMEDIATION_MAP` で管理する。

```python
REMEDIATION_MAP = {
    ('A', '終盤', False):  ['fix_add_paishu_feature', 'fix_impossible_tile_penalty'],
    ('A', '終盤', True):   ['fix_add_paishu_feature'],
    ('B2', 'riichi', '*'): ['fix_riichi_loss_weight',  'fix_add_riichi_token'],
    ('B3', 'riichi', '*'): ['fix_wait_logit_penalty'],
    ('B1', '*', '*'):      ['fix_late_game_weight'],
}
```

### 提案表示

```
対応策候補:
  [1] fix_add_paishu_feature
      → extract_features.js に「各牌の残り枚数(34次元)」を追加
      → 期待効果: Type A の終盤違反を直接補正

  [2] fix_impossible_tile_penalty
      → train_v39.py に「枯れ牌への確率をペナルティ化」するloss項追加
      → 期待効果: Type A の全体的な改善

  [3] fix_riichi_loss_weight
      → train_v39.py でリーチ局面サンプルのloss_weight を 2.0 → 3.0 に調整
      → 期待効果: Type B2/B3 の改善

実行する対応策を選んでください [1/2/3/all/なし]:
```

---

## チェックリスト

- [ ] `harness_report.py` 実装
- [ ] グルーピング・集計ロジック
- [ ] `REMEDIATION_MAP` の初期定義
- [ ] 対応策の提案表示
- [ ] Phase 3 (`harness_fix.py`) への引き渡し（選択番号のファイル出力）
