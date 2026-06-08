'use strict';

/**
 * TICKET-030: 赤牌特徴量追加 + visible_counts_vecブラウザバグ修正 のテスト
 */

const path = require('path');

// extract_features.js からのインポート
const {
    red_visible_flags: rvf_extract,
    visible_counts_vec: vcv_extract,
    pai_to_idx,
} = require(path.join(__dirname, '../../phase2/scripts/extract_features.js'));

// ai_phase2.js (ブラウザ側) からのインポート
const {
    visible_counts_vec: vcv_browser,
    red_visible_flags: rvf_browser,
} = require(path.join(__dirname, '../dist/js/ai_phase2.js'));

// ---- ヘルパー ----

function emptyState(viewer_l = 0) {
    return {
        l: viewer_l,
        discards_l: [[], [], [], []],
        melds_l:    [[], [], [], []],
        hands_l:    ['', '', '', ''],
    };
}

function emptyRec(viewer_l = 0) {
    return {
        l: viewer_l,
        discards_l: [[], [], [], []],
        melds_l:    [[], [], [], []],
        hands_l:    ['', '', '', ''],
    };
}

// ---- red_visible_flags (extract_features.js) ----

describe('red_visible_flags (extract_features.js)', () => {
    test('初期状態では全て0', () => {
        expect(rvf_extract([[], [], [], []], [[], [], [], []])).toEqual([0, 0, 0]);
    });

    test('捨て牌にm0があればフラグ[0]が1', () => {
        const discards_l = [['m0'], [], [], []];
        expect(rvf_extract(discards_l, [[], [], [], []])).toEqual([1, 0, 0]);
    });

    test('捨て牌にp0があればフラグ[1]が1', () => {
        const discards_l = [[], ['p0'], [], []];
        expect(rvf_extract(discards_l, [[], [], [], []])).toEqual([0, 1, 0]);
    });

    test('捨て牌にs0があればフラグ[2]が1', () => {
        const discards_l = [[], [], [], ['s0']];
        expect(rvf_extract(discards_l, [[], [], [], []])).toEqual([0, 0, 1]);
    });

    test('捨て牌に装飾付きm0（例: m0*）でもフラグが立つ', () => {
        const discards_l = [['m0*'], [], [], []];
        expect(rvf_extract(discards_l, [[], [], [], []])).toEqual([1, 0, 0]);
    });

    test('副露にm0が含まれればフラグ[0]が1（チー形式）', () => {
        // m0 を含むチー: m0は5なので m023- → m[1]='0'
        const melds_l = [['m034-'], [], [], []];
        expect(rvf_extract([[], [], [], []], melds_l)).toEqual([1, 0, 0]);
    });

    test('副露にp0が含まれればフラグ[1]が1（ポン形式）', () => {
        // p000+ = p0のポン（赤牌3枚は実際にはないが副露文字列として有効なテスト）
        const melds_l = [[], ['p500+'], [], []];
        // p500+: clean = 'p500', '5'→ok, '0'→flags[1]=1
        expect(rvf_extract([[], [], [], []], melds_l)).toEqual([0, 1, 0]);
    });

    test('m5（通常の5）はm0として扱わない', () => {
        const discards_l = [['m5'], [], [], []];
        expect(rvf_extract(discards_l, [[], [], [], []])).toEqual([0, 0, 0]);
    });

    test('複数の赤牌が同時に公開されている', () => {
        const discards_l = [['m0'], ['p0'], [], ['s0']];
        expect(rvf_extract(discards_l, [[], [], [], []])).toEqual([1, 1, 1]);
    });
});

// ---- red_visible_flags (ai_phase2.js ブラウザ側) ----

describe('red_visible_flags (ai_phase2.js)', () => {
    test('初期状態では全て0', () => {
        const state = emptyState();
        expect(rvf_browser(state)).toEqual([0, 0, 0]);
    });

    test('捨て牌にm0があればフラグ[0]が1', () => {
        const state = emptyState();
        state.discards_l[0] = ['m0'];
        expect(rvf_browser(state)).toEqual([1, 0, 0]);
    });

    test('副露（チー）にs0が含まれればフラグ[2]が1', () => {
        const state = emptyState();
        // s034- : clean='s034', '0'→flags[2]=1
        state.melds_l[1] = ['s034-'];
        expect(rvf_browser(state)).toEqual([0, 0, 1]);
    });
});

// ---- visible_counts_vec: 副露の請求牌二重カウントバグ修正 ----

describe('visible_counts_vec 二重カウントなし (ai_phase2.js)', () => {
    const M1_IDX = 0;   // m1 = index 0
    const M5_IDX = 4;   // m5 = index 4
    const M2_IDX = 1;
    const M3_IDX = 2;

    test('チー m123-: 請求牌m3は捨て牌のみカウント (合計3枚)', () => {
        // player1がm3を捨て、player0がチー → m123-
        const state = emptyState(0);
        state.discards_l[1] = ['m3'];
        state.melds_l[0] = ['m123-'];
        const vec = vcv_browser(state);
        // m1=1, m2=1, m3=1 (各1枚)
        expect(vec[M1_IDX]).toBeCloseTo(1 / 4);
        expect(vec[M2_IDX]).toBeCloseTo(1 / 4);
        expect(vec[M3_IDX]).toBeCloseTo(1 / 4);
    });

    test('ポン m111+: 請求牌m1は捨て牌のみカウント (合計3枚)', () => {
        // player1がm1を捨て、player0がポン → m111+
        const state = emptyState(0);
        state.discards_l[1] = ['m1'];
        state.melds_l[0] = ['m111+'];
        const vec = vcv_browser(state);
        // m1 = 捨て牌1 + 副露2(請求牌スキップ) = 3
        expect(vec[M1_IDX]).toBeCloseTo(3 / 4);
    });

    test('加カン m111+1: 合計4枚（shouminkan）', () => {
        // player0が最初ポン(m1取得)後、自引きでm1追加 → m111+1
        const state = emptyState(0);
        state.discards_l[1] = ['m1'];       // 元のポンの請求牌（捨て牌扱い）
        state.melds_l[0] = ['m111+1'];      // 加カン後の文字列
        const vec = vcv_browser(state);
        // m1 = 捨て牌1 + 副露3(i=1,2,5 / i=3はスキップ) = 4
        expect(vec[M1_IDX]).toBeCloseTo(4 / 4);
    });

    test('暗カン m1111: 全4枚カウント（捨て牌なし）', () => {
        const state = emptyState(0);
        state.melds_l[0] = ['m1111'];       // 暗カン
        const vec = vcv_browser(state);
        // m1 = 4 (dirIdx=-1なので全てカウント)
        expect(vec[M1_IDX]).toBeCloseTo(4 / 4);
    });

    test('大明カン m1111+: 合計4枚', () => {
        const state = emptyState(0);
        state.discards_l[1] = ['m1'];
        state.melds_l[0] = ['m1111+'];      // 大明カン
        const vec = vcv_browser(state);
        // m1 = 捨て牌1 + 副露3(i=1,2,3 / i=4はスキップ, i=5='+'は方向) = 4
        expect(vec[M1_IDX]).toBeCloseTo(4 / 4);
    });

    test('赤ポン m050+: m5を正しくカウント', () => {
        // m5(=m0)のポン: m050+ → clean後 m050
        const state = emptyState(0);
        state.discards_l[2] = ['m0'];       // player2がm0(=m5)を捨て
        state.melds_l[0] = ['m050+'];       // player0がポン
        const vec = vcv_browser(state);
        // m5 = 捨て牌1 + 副露2(0→5に変換, i=3はスキップ) = 3
        // 注意: m0はm5として扱われるのでM5_IDX=4
        // discards m0 → pai_to_idx('m0') → pai_to_idx('m5') → 4
        expect(vec[M5_IDX]).toBeCloseTo(3 / 4);
    });

    test('結果は34次元', () => {
        const vec = vcv_browser(emptyState(0));
        expect(vec.length).toBe(34);
    });
});

// ---- visible_counts_vec: extract_features.js 側との整合性 ----

describe('visible_counts_vec (extract_features.js)', () => {
    test('ポン m111+: 合計3枚（二重カウントなし）', () => {
        const rec = emptyRec(0);
        rec.discards_l[1] = ['m1'];
        rec.melds_l[0] = ['m111+'];
        const vec = vcv_extract(rec, 0);
        expect(vec[0]).toBeCloseTo(3 / 4);  // m1 index=0
    });

    test('チー m123-: 各1枚', () => {
        const rec = emptyRec(0);
        rec.discards_l[1] = ['m3'];
        rec.melds_l[0] = ['m123-'];
        const vec = vcv_extract(rec, 0);
        expect(vec[0]).toBeCloseTo(1 / 4);  // m1
        expect(vec[1]).toBeCloseTo(1 / 4);  // m2
        expect(vec[2]).toBeCloseTo(1 / 4);  // m3
    });

    test('結果は34次元', () => {
        const vec = vcv_extract(emptyRec(0), 0);
        expect(vec.length).toBe(34);
    });
});
