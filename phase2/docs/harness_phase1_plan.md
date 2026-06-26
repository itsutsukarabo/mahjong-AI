# ハーネスエンジニアリング Phase 1: 検出・収集

## 概要

推論時にリアルタイムで「モデルが論理的に絞り込めていない箇所」を検出し、
フィードバックログとしてブラウザに蓄積する。

---

## コンポーネント構成

```
ai_phase2.js
  └─ run_phase2() の手牌推定後に追加
       ├─ compute_paishu_per_tile()   ← ゲーム状態から枚数計算
       ├─ run_checkers()              ← A/B1〜B4 を束ねて実行
       │    ├─ check_A()   枯れ牌ハード制約
       │    ├─ check_B1()  自捨牌ソフト制約
       │    ├─ check_B2()  リーチ後エントロピー
       │    ├─ check_B3()  リーチ後ツモ切り待ち割当
       │    └─ check_B4()  PASSイベント後確率変化
       └─ feedback_log に結果を付加

feedback_logger.js（新規）
  ├─ pushLog(record)        ← 閾値超えレコードをlocalStorageに追加
  ├─ exportLogs()           ← JSONL形式でダウンロード
  └─ clearLogs()            ← ログ削除

paipu.js (_render_phase2)
  └─ 手牌推定パネルに信頼度インジケータを追加
```

---

## 1. `compute_paishu_per_tile(state)` → `number[34]`

ゲーム状態から各牌の「まだ見えていない残り枚数」を計算する。

```javascript
function compute_paishu_per_tile(state) {
    // 初期値: 各牌4枚（赤五は別管理のため通常5として扱う）
    const paishu = new Array(34).fill(4);

    // 全プレイヤーの捨牌を引く
    for (let l = 0; l < 4; l++) {
        for (const p of state.discards_l[l] || []) {
            const t = pai_to_idx(p);
            if (t >= 0) paishu[t] = Math.max(0, paishu[t] - 1);
        }
    }

    // 全プレイヤーの副露を引く
    for (let l = 0; l < 4; l++) {
        for (const m of state.melds_l[l] || []) {
            const tiles = parse_meld_tiles(m);
            for (const t of tiles) {
                if (t >= 0) paishu[t] = Math.max(0, paishu[t] - 1);
            }
        }
    }

    // 自分の手牌（既知部分）を引く
    const own_hand = encode_hand(state.hands_l[state.l]);
    for (let t = 0; t < 34; t++) {
        paishu[t] = Math.max(0, paishu[t] - own_hand[t]);
    }

    return paishu;
}
```

---

## 2. checker 関数群

### check_A: 枯れ牌ハード制約

```javascript
function check_A(probs_per_tile, paishu) {
    // paishu[t] === 0 の牌に確率が割り当てられていれば違反
    let score = 0;
    const flagged = [];
    for (let t = 0; t < 34; t++) {
        if (paishu[t] === 0) {
            const p_has = 1.0 - probs_per_tile[t * 5 + 0]; // P(枚数>=1)
            score += p_has;
            if (p_has > 0.02) flagged.push({ tile: PAI_NAMES[t], prob: p_has });
        }
    }
    return { type: 'A', score, flagged };
}
```

### check_B1: 自捨牌ソフト制約

```javascript
function check_B1(probs_per_tile, paishu, state, target_l) {
    // target_l の自捨牌: P(T in hand) の上限 = paishu[T] / unseen_total
    const unseen = paishu.reduce((a, b) => a + b, 0);
    let excess = 0;
    const flagged = [];
    const own_discards = (state.discards_l[target_l] || []);
    const discarded_set = new Set();
    for (const p of own_discards) {
        const t = pai_to_idx(p);
        if (t >= 0) discarded_set.add(t);
    }
    for (const t of discarded_set) {
        const upper = unseen > 0 ? paishu[t] / unseen : 0;
        const actual = 1.0 - probs_per_tile[t * 5 + 0];
        const over = Math.max(0, actual - upper * 2); // 2倍を超えたら過剰
        if (over > 0.05) {
            excess += over;
            flagged.push({ tile: PAI_NAMES[t], excess: over });
        }
    }
    return { type: 'B1', score: excess, flagged };
}
```

### check_B2: リーチ後エントロピー未収束

```javascript
function check_B2(probs_per_tile, state, target_l) {
    if (!state.riichi_l[target_l]) return { type: 'B2', score: 0, flagged: [] };

    // エントロピー計算 (P(枚数=k) の分布から)
    let entropy = 0;
    for (let t = 0; t < 34; t++) {
        for (let k = 0; k < 5; k++) {
            const p = probs_per_tile[t * 5 + k];
            if (p > 1e-9) entropy -= p * Math.log(p);
        }
    }
    // リーチ後は手牌確定 → エントロピーが低いはず（閾値: 経験的に設定）
    const RIICHI_ENTROPY_THRESHOLD = 15.0;
    const score = Math.max(0, entropy - RIICHI_ENTROPY_THRESHOLD);
    return { type: 'B2', score, flagged: score > 0 ? [{ entropy }] : [] };
}
```

### check_B3: リーチ後ツモ切り牌への待ち割当

```javascript
function check_B3(tatsu_probs, state, target_l, round_log, current_idx) {
    if (!state.riichi_l[target_l] || !tatsu_probs) {
        return { type: 'B3', score: 0, flagged: [] };
    }
    // リーチ後にツモ切りした牌 → wait_logits で待ちが高い = 矛盾
    const tsumo_cut_tiles = get_post_riichi_tsumo_tiles(state, target_l, round_log, current_idx);
    let score = 0;
    const flagged = [];
    for (const t of tsumo_cut_tiles) {
        // tatsu_probs の単騎 (idx 45~78) と双碰 (79~112) で該当牌の確率を合算
        const tanki_prob  = tatsu_probs[45 + t] || 0;
        const shanpon_prob = tatsu_probs[79 + t] || 0;
        const p = Math.max(tanki_prob, shanpon_prob);
        if (p > 0.3) {
            score += p;
            flagged.push({ tile: PAI_NAMES[t], prob: p });
        }
    }
    return { type: 'B3', score, flagged };
}
```

### check_B4: PASSイベント後の確率変化不足

```javascript
// 注: 前ターンの probs_per_tile をキャッシュして比較する必要がある。
// 実装: paipu.js の _do_ai_analyze() で「前回の probs」を保持し渡す。
function check_B4(probs_per_tile, prev_probs_per_tile, pass_tiles) {
    if (!prev_probs_per_tile || pass_tiles.length === 0) {
        return { type: 'B4', score: 0, flagged: [] };
    }
    // PASS した牌の確率が PASSイベント後に有意に下がったか確認
    let score = 0;
    const flagged = [];
    for (const t of pass_tiles) {
        const before = 1.0 - prev_probs_per_tile[t * 5 + 0];
        const after  = 1.0 - probs_per_tile[t * 5 + 0];
        const drop   = before - after;
        if (drop < 0.05 && before > 0.15) { // 高確率なのに下がらなかった
            score += (before - drop);
            flagged.push({ tile: PAI_NAMES[t], before, after });
        }
    }
    return { type: 'B4', score, flagged };
}
```

---

## 3. `run_checkers()` 統合

```javascript
function run_checkers(probs_per_tile, tatsu_probs, state, target_l, round_log, current_idx) {
    const paishu = compute_paishu_per_tile(state);
    const turn   = (state.discards_l[0].length + state.discards_l[1].length +
                    state.discards_l[2].length + state.discards_l[3].length);

    return {
        turn,
        A:  check_A(probs_per_tile, paishu),
        B1: check_B1(probs_per_tile, paishu, state, target_l),
        B2: check_B2(probs_per_tile, state, target_l),
        B3: check_B3(tatsu_probs, state, target_l, round_log, current_idx),
        B4: check_B4(probs_per_tile, null, []), // B4は別途前回キャッシュ要
    };
}
```

---

## 4. `feedback_logger.js`（新規ファイル）

```javascript
const STORAGE_KEY = 'majiang_feedback_log';
const MAX_RECORDS = 5000;
const THRESHOLD = { A: 0.05, B1: 0.08, B2: 3.0, B3: 0.3, B4: 0.1 };

function pushLog(record) {
    if (!Object.entries(THRESHOLD).some(([k, v]) => record[k]?.score > v)) return;
    const logs = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    logs.push({ ts: Date.now(), ...record });
    if (logs.length > MAX_RECORDS) logs.splice(0, logs.length - MAX_RECORDS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(logs));
}

function exportLogs() {
    const logs = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    const jsonl = logs.map(r => JSON.stringify(r)).join('\n');
    const blob = new Blob([jsonl], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `feedback_log_${Date.now()}.jsonl`;
    a.click();
}
```

---

## 5. フロントUI追加（`_render_phase2` 内）

既存の `ai-hi-player-section` にインジケータ行を追加する。

```javascript
// checker 結果から信頼度ラベルを生成
function make_confidence_badge(checkers) {
    const issues = [];
    if (checkers.A.score > 0.05)  issues.push(`⚠️A: 枯れ牌割当 ${(checkers.A.score*100).toFixed(0)}%`);
    if (checkers.B2.score > 3.0)  issues.push(`⚠️B2: リーチ後entropy高`);
    if (checkers.B3.score > 0.3)  issues.push(`⚠️B3: 待ち形矛盾`);
    const label = issues.length === 0 ? '✅ 高信頼' : issues.join(' / ');
    return $('<div class="ai-hi-confidence">').text(label);
}
```

---

## チェックリスト

- [ ] `compute_paishu_per_tile()` 実装・テスト
- [ ] `check_A()` 実装
- [ ] `check_B1()` 実装
- [ ] `check_B2()` 実装
- [ ] `check_B3()` 実装（`get_post_riichi_tsumo_tiles()` 補助関数含む）
- [ ] `check_B4()` 実装（前回キャッシュ機構含む）
- [ ] `run_checkers()` 統合
- [ ] `feedback_logger.js` 作成
- [ ] `_render_phase2` に信頼度バッジ追加
- [ ] エクスポートボタン UI 追加
