'use strict';

/**
 * TICKET-027: ブロック期待値計算エンジン のテスト
 */

const path = require('path');

const {
    compute_block_ev,
    get_best_hand_str,
    make_block_display_data,
} = require(path.join(__dirname, '../dist/js/ai_phase2.js'));

// Majiang を global に設定（シャンテン数・面子分解のテスト用）
const Majiang = require('@kobalab/majiang-core');
global.Majiang = Majiang;

// ---- ヘルパー ----

// 指定枚数に確率1を割り当てた1牌分の確率ベクトルを生成 [0]=0枚, [1]=1枚, ...
function certain(count) {
    const p = [0, 0, 0, 0, 0];
    p[count] = 1;
    return p;
}

// 34牌分の確率配列を生成（デフォルト: 全牌が確率0枚）
function make_probs(overrides = {}) {
    const probs = Array.from({ length: 34 }, () => certain(0));
    for (const [idx, cnt] of Object.entries(overrides)) {
        probs[Number(idx)] = certain(Number(cnt));
    }
    return probs;
}

// ---- compute_block_ev ----

describe('compute_block_ev', () => {
    test('全牌が枚数0確定 → 全ブロック: P(0個)=1, その他=0', () => {
        const { triplet_dist, pair_dist, seq_dist } = compute_block_ev(make_probs());
        expect(triplet_dist).toHaveLength(34);
        expect(pair_dist).toHaveLength(34);
        expect(seq_dist).toHaveLength(21);
        expect(triplet_dist.every(d => d[0] === 1 && d[1] === 0)).toBe(true);
        expect(pair_dist.every(d => d[0] === 1 && d[1] === 0 && d[2] === 0)).toBe(true);
        expect(seq_dist.every(d => d[0] === 1 && d[1] === 0 && d[2] === 0 && d[3] === 0)).toBe(true);
    });

    test('triplet_dist: m1が3枚確定 → [P0=0, P1=1]', () => {
        const { triplet_dist } = compute_block_ev(make_probs({ 0: 3 }));
        expect(triplet_dist[0][0]).toBeCloseTo(0);  // P(0刻子)
        expect(triplet_dist[0][1]).toBeCloseTo(1);  // P(1刻子)
    });

    test('triplet_dist: m1が4枚確定 → [P0=0, P1=1]', () => {
        const { triplet_dist } = compute_block_ev(make_probs({ 0: 4 }));
        expect(triplet_dist[0][0]).toBeCloseTo(0);
        expect(triplet_dist[0][1]).toBeCloseTo(1);
    });

    test('triplet_dist: m1が2枚確定 → [P0=1, P1=0]', () => {
        const { triplet_dist } = compute_block_ev(make_probs({ 0: 2 }));
        expect(triplet_dist[0][0]).toBeCloseTo(1);
        expect(triplet_dist[0][1]).toBeCloseTo(0);
    });

    test('triplet_dist: 各要素は長さ2で合計≒1', () => {
        const { triplet_dist } = compute_block_ev(make_probs({ 0: 1 }));
        expect(triplet_dist[0]).toHaveLength(2);
        expect(triplet_dist[0][0] + triplet_dist[0][1]).toBeCloseTo(1);
    });

    test('pair_dist: m1が2枚確定 → [P0=0, P1=1, P2=0]', () => {
        const { pair_dist } = compute_block_ev(make_probs({ 0: 2 }));
        expect(pair_dist[0][0]).toBeCloseTo(0);  // P(0対子)
        expect(pair_dist[0][1]).toBeCloseTo(1);  // P(1対子) = P(count 2 or 3)
        expect(pair_dist[0][2]).toBeCloseTo(0);  // P(2対子) = P(count=4)
    });

    test('pair_dist: m1が4枚確定 → [P0=0, P1=0, P2=1]', () => {
        const { pair_dist } = compute_block_ev(make_probs({ 0: 4 }));
        expect(pair_dist[0][0]).toBeCloseTo(0);
        expect(pair_dist[0][1]).toBeCloseTo(0);
        expect(pair_dist[0][2]).toBeCloseTo(1);
    });

    test('pair_dist: m1が1枚確定 → [P0=1, P1=0, P2=0]', () => {
        const { pair_dist } = compute_block_ev(make_probs({ 0: 1 }));
        expect(pair_dist[0][0]).toBeCloseTo(1);
        expect(pair_dist[0][1]).toBeCloseTo(0);
        expect(pair_dist[0][2]).toBeCloseTo(0);
    });

    test('pair_dist: 各要素は長さ3で合計≒1', () => {
        const { pair_dist } = compute_block_ev(make_probs({ 0: 2 }));
        expect(pair_dist[0]).toHaveLength(3);
        expect(pair_dist[0].reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    });

    test('seq_dist: m1/m2/m3 各1枚確定 → P(0順子)=0, P(1順子)=1', () => {
        const { seq_dist } = compute_block_ev(make_probs({ 0: 1, 1: 1, 2: 1 }));
        expect(seq_dist[0][0]).toBeCloseTo(0);  // P(0順子)
        expect(seq_dist[0][1]).toBeCloseTo(1);  // P(1順子)
        expect(seq_dist[0][2]).toBeCloseTo(0);  // P(2順子)
        expect(seq_dist[0][3]).toBeCloseTo(0);  // P(3順子以上)
    });

    test('seq_dist: m1/m2/m3 各2枚確定 → P(0)=0, P(1)=0, P(2)=1', () => {
        const { seq_dist } = compute_block_ev(make_probs({ 0: 2, 1: 2, 2: 2 }));
        expect(seq_dist[0][0]).toBeCloseTo(0);
        expect(seq_dist[0][1]).toBeCloseTo(0);
        expect(seq_dist[0][2]).toBeCloseTo(1);
        expect(seq_dist[0][3]).toBeCloseTo(0);
    });

    test('seq_dist: m1が0枚確定 → m123 P(0順子)=1', () => {
        const { seq_dist } = compute_block_ev(make_probs({ 1: 1, 2: 1 }));
        expect(seq_dist[0][0]).toBeCloseTo(1);
        expect(seq_dist[0][1]).toBeCloseTo(0);
    });

    test('seq_dist: p1/p2/p3 各1枚確定 → seq_dist[7](p123) P(1順子)=1', () => {
        const { seq_dist } = compute_block_ev(make_probs({ 9: 1, 10: 1, 11: 1 }));
        expect(seq_dist[7][1]).toBeCloseTo(1);  // p123 = suit1*7+0 = index 7
    });

    test('seq_dist: s7/s8/s9 各1枚確定 → seq_dist[20](s789) P(1順子)=1', () => {
        const { seq_dist } = compute_block_ev(make_probs({ 24: 1, 25: 1, 26: 1 }));
        expect(seq_dist[20][1]).toBeCloseTo(1); // s789 = suit2*7+6 = index 20
    });

    test('seq_dist: 各要素は長さ4で合計≒1', () => {
        const { seq_dist } = compute_block_ev(make_probs({ 0: 1, 1: 1, 2: 1 }));
        expect(seq_dist[0]).toHaveLength(4);
        expect(seq_dist[0].reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    });

    test('seq_dist は 21要素 (m7種+p7種+s7種)', () => {
        const { seq_dist } = compute_block_ev(make_probs());
        expect(seq_dist).toHaveLength(21);
    });

    test('全ての確率値は 0≤v≤1（負値なし）', () => {
        const probs = make_probs();
        probs[0] = [0.3, 0.1, 0.1, 0.4, 0.1];
        const { triplet_dist, pair_dist, seq_dist } = compute_block_ev(probs);
        const all = [...triplet_dist.flat(), ...pair_dist.flat(), ...seq_dist.flat()];
        expect(all.every(v => v >= 0 && v <= 1 + 1e-9)).toBe(true);
    });

    test('部分確率: m1 P(3)=0.4, P(4)=0.1 → triplet P(1刻子)=0.5', () => {
        const probs = make_probs();
        probs[0] = [0.3, 0.1, 0.1, 0.4, 0.1];
        const { triplet_dist, pair_dist } = compute_block_ev(probs);
        expect(triplet_dist[0][1]).toBeCloseTo(0.5);  // P(3)+P(4)
        expect(pair_dist[0][1]).toBeCloseTo(0.5);     // P(2)+P(3) = 0.1+0.4
        expect(pair_dist[0][2]).toBeCloseTo(0.1);     // P(4)
    });
});

// ---- get_best_hand_str ----

describe('get_best_hand_str', () => {
    test('全牌が0枚 → 空文字列', () => {
        expect(get_best_hand_str(make_probs(), [])).toBe('');
    });

    test('m1が1枚確定 → "m1"', () => {
        expect(get_best_hand_str(make_probs({ 0: 1 }), [])).toBe('m1');
    });

    test('m1が2枚確定 → "m11"', () => {
        expect(get_best_hand_str(make_probs({ 0: 2 }), [])).toBe('m11');
    });

    test('m1/m2/m3が各1枚確定 → "m123"', () => {
        expect(get_best_hand_str(make_probs({ 0: 1, 1: 1, 2: 1 }), [])).toBe('m123');
    });

    test('z1が2枚確定 → "z11"', () => {
        expect(get_best_hand_str(make_probs({ 27: 2 }), [])).toBe('z11');
    });

    test('m1/p1/s1/z1が各1枚確定 → "m1p1s1z1"', () => {
        expect(get_best_hand_str(make_probs({ 0: 1, 9: 1, 18: 1, 27: 1 }), [])).toBe('m1p1s1z1');
    });

    test('副露あり → カンマ区切りで副露が付与される', () => {
        const result = get_best_hand_str(make_probs({ 0: 1 }), ['m234-']);
        expect(result).toBe('m1,m234-');
    });

    test('副露複数あり → 全副露がカンマ区切りで付与される', () => {
        const result = get_best_hand_str(make_probs({ 0: 1 }), ['m234-', 'p111+']);
        expect(result).toBe('m1,m234-,p111+');
    });

    test('副露にnull/undefinedが混在しても無視される', () => {
        const result = get_best_hand_str(make_probs({ 0: 1 }), [null, 'm234-', null]);
        expect(result).toBe('m1,m234-');
    });

    test('argmax が複数に同値でも クラッシュしない', () => {
        // P(0)=0.5, P(1)=0.5 → indexOf(0.5) = 0 (最初のインデックス)
        const probs = make_probs();
        probs[0] = [0.5, 0.5, 0, 0, 0];
        expect(() => get_best_hand_str(probs, [])).not.toThrow();
    });
});

// ---- make_block_display_data (Majiang あり) ----

describe('make_block_display_data (Majiang グローバルあり)', () => {
    test('返り値に必須フィールドが全て含まれる', () => {
        const result = make_block_display_data(make_probs(), []);
        expect(result).toHaveProperty('triplet_dist');
        expect(result).toHaveProperty('pair_dist');
        expect(result).toHaveProperty('seq_dist');
        expect(result).toHaveProperty('hand_str');
        expect(result).toHaveProperty('shanten');
        expect(result).toHaveProperty('tingpai');
        expect(result).toHaveProperty('decomps');
    });

    test('全牌0枚の空手牌 → shanten は null にならない（Majiang が計算する値）', () => {
        const result = make_block_display_data(make_probs(), []);
        // 空手牌でも Majiang.Util.xiangting は有限の数を返す
        expect(typeof result.shanten).toBe('number');
        expect(result.shanten).toBeGreaterThan(0);
        expect(result.tingpai).toBeNull();
        expect(result.decomps).toBeNull();
    });

    test('テンパイ手牌(m123+m456+m789+p11+p23) → shanten=0, tingpai・decomps が返る', () => {
        // m123(3) + m456(3) + m789(3) + p11(2) + p23(2) = 13枚 → p1かp4待ちのテンパイ
        const probs = make_probs({
            0:1, 1:1, 2:1,   // m1/m2/m3
            3:1, 4:1, 5:1,   // m4/m5/m6
            6:1, 7:1, 8:1,   // m7/m8/m9
            9:2,             // p1×2 (雀頭)
            10:1, 11:1,      // p2/p3 (搭子)
        });
        const result = make_block_display_data(probs, []);
        expect(result.shanten).toBe(0);
        expect(Array.isArray(result.tingpai)).toBe(true);
        expect(result.tingpai.length).toBeGreaterThan(0);
        expect(Array.isArray(result.decomps)).toBe(true);
        expect(result.decomps.length).toBeGreaterThan(0);
    });

    test('decomps の各要素は面子文字列の配列', () => {
        // m1×3 + m2×3 + m3×3 + s1×2 + p1×1 = 12 (不足1) → 別手を試みる
        // m1/2/3各3枚 + z1×4 = 13枚でシャンポン待ち相当にはならないので
        // 簡単な完成手: m1×3 m2×3 m3×3 z1×1 z2×1 z3×1 z1×1 = 13 → 不完全
        // 確実な完成手: m123,m456,m789,p11 待ちp1 → 13枚
        // probs で m1/4/7=1, m2/5/8=1, m3/6/9=1, p1=2 にすると完成形
        const probs = make_probs({
            0:1, 1:1, 2:1,   // m123
            3:1, 4:1, 5:1,   // m456
            6:1, 7:1, 8:1,   // m789
            9:2,             // p11 (対子)
        });
        const result = make_block_display_data(probs, []);
        // シャンテン数が0以下なら decomps を確認
        if (result.decomps !== null) {
            expect(Array.isArray(result.decomps)).toBe(true);
            result.decomps.forEach(d => {
                expect(Array.isArray(d)).toBe(true);
                d.forEach(s => expect(typeof s).toBe('string'));
            });
        }
    });
});

// ---- make_block_display_data (Majiang なし) ----

describe('make_block_display_data (Majiang グローバルなし)', () => {
    let savedMajiang;
    beforeEach(() => { savedMajiang = global.Majiang; delete global.Majiang; });
    afterEach(() => { global.Majiang = savedMajiang; });

    test('Majiang なしでも EV と hand_str は正常に返る', () => {
        const probs = make_probs({ 0: 1, 1: 1, 2: 1 });
        const result = make_block_display_data(probs, []);
        expect(result.hand_str).toBe('m123');
        expect(result.triplet_dist).toHaveLength(34);
        expect(result.pair_dist).toHaveLength(34);
        expect(result.seq_dist).toHaveLength(21);
    });

    test('Majiang なしのとき shanten/tingpai/decomps は null', () => {
        const result = make_block_display_data(make_probs({ 0: 1 }), []);
        expect(result.shanten).toBeNull();
        expect(result.tingpai).toBeNull();
        expect(result.decomps).toBeNull();
    });
});
