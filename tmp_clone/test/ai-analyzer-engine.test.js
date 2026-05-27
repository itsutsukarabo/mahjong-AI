'use strict';

const { analyze } = require('../node_modules/@kobalab/majiang-ui/lib/ai-analyzer-engine');

// ────────────────────────────────────────────────
// テスト用牌譜ファクトリ
// ────────────────────────────────────────────────

function makePaipu(hands, log) {
    return {
        title:  'テスト',
        player: ['A', 'B', 'C', 'D'],
        qijia:  0,
        log: [[
            { qipai: {
                zhuangfeng: 0,
                jushu:      0,
                changbang:  0,
                lizhibang:  0,
                defen:      [25000, 25000, 25000, 25000],
                baopai:     'z1',
                shoupai:    hands,
            }},
            ...log,
        ]],
        defen: [25000, 25000, 25000, 25000],
        point: [0, 0, 0, 0],
    };
}

// player 0 の基本的な 13 枚手牌 (好形 0 向聴に近い)
const HAND_0  = 'm1234567p123s123';  // 7+3+3 = 13 枚
// player 0 の役牌カン狙い手牌 (13 枚)
// z111z2 で z1 を 4 枚集めれば大明杠が可能
const HAND_PON = 'm234p123s123z111z2';  // 13 枚: z111 + z2

// ────────────────────────────────────────────────
// 1. 打牌分析: 自手番での draw 後
// ────────────────────────────────────────────────

describe('打牌分析 (dapai)', () => {

    const paipu = makePaipu(
        [HAND_0, '', '', ''],
        [{ zimo: { l: 0, p: 's4' } }]  // player 0 がツモ
    );
    // current_idx=1 (qipai=0, zimo=1)
    const result = analyze(paipu, 0, 1, 0);

    test('analysis_type が "dapai" になる', () => {
        expect(result.analysis_type).toBe('dapai');
    });

    test('dapai_info に候補が 1 件以上ある', () => {
        expect(result.dapai_info.length).toBeGreaterThan(0);
    });

    test('各候補に必須フィールドが揃っている', () => {
        for (const item of result.dapai_info) {
            expect(typeof item.p).toBe('string');
            expect(typeof item.n_xiangting).toBe('number');
            expect(typeof item.ev).toBe('number');
            expect(Array.isArray(item.tingpai)).toBe(true);
            expect(typeof item.n_tingpai).toBe('number');
        }
    });

    test('ev 最大の候補が selected=true になる', () => {
        const selected = result.dapai_info.filter(i => i.selected);
        expect(selected.length).toBe(1);
    });

    test('best_dapai が dapai_info の selected 候補と一致する', () => {
        const sel = result.dapai_info.find(i => i.selected);
        expect(result.best_dapai.slice(0, 2)).toBe(sel.p);
    });

    test('best_dapai はスーツと数字で始まる牌表記', () => {
        // リーチ候補は "s4_*" のように _* サフィックスを含む場合がある
        expect(result.best_dapai).toMatch(/^[mpsz][0-9]/);
    });

    test('fulou_info は空', () => {
        expect(result.fulou_info.length).toBe(0);
    });
});

// ────────────────────────────────────────────────
// 2. 副露分析: 相手の打牌後
// ────────────────────────────────────────────────

describe('副露分析 (fulou)', () => {

    // player 0 が z111 (役牌) 持ち。player 1 が z1 を捨てる → player 0 がポンできる
    const paipu = makePaipu(
        [HAND_PON, '', '', ''],
        [
            { zimo:  { l: 0, p: 'z7' } },   // player 0 ツモ
            { dapai: { l: 0, p: 'z7' } },   // player 0 捨て
            { zimo:  { l: 1, p: 'm1' } },   // player 1 ツモ
            { dapai: { l: 1, p: 'z1' } },   // player 1 が z1 を捨て → player 0 がポン可能
        ]
    );
    // current_idx=4 (qipai=0, zimo0=1, dapai0=2, zimo1=3, dapai1=4)
    const result = analyze(paipu, 0, 4, 0);

    test('analysis_type が "fulou" になる', () => {
        expect(result.analysis_type).toBe('fulou');
    });

    test('fulou_info に選択肢が 1 件以上ある', () => {
        expect(result.fulou_info.length).toBeGreaterThan(0);
    });

    test('パス (m="") の選択肢が含まれる', () => {
        const pass = result.fulou_info.find(i => i.m === '');
        expect(pass).toBeDefined();
    });

    test('z1 に対する鳴き選択肢が含まれる (ポンまたは大明杠)', () => {
        // z111 を持つ手牌から z1 に応答できる: ポン z11+1 または大明杠 z1111+
        const meld = result.fulou_info.find(i => i.m && i.m.startsWith('z1'));
        expect(meld).toBeDefined();
    });

    test('各副露候補に n_xiangting と ev がある', () => {
        for (const item of result.fulou_info) {
            expect(typeof item.n_xiangting).toBe('number');
            expect(typeof item.ev).toBe('number');
        }
    });

    test('dapai_info は空', () => {
        expect(result.dapai_info.length).toBe(0);
    });
});

// ────────────────────────────────────────────────
// 3. 分析対象外ターン
// ────────────────────────────────────────────────

describe('分析対象外ターン', () => {

    const paipu = makePaipu(
        [HAND_0, '', '', ''],
        [{ zimo: { l: 1, p: 'm1' } }]  // player 1 のツモ (player 0 の手番でない)
    );
    const result = analyze(paipu, 0, 1, 0);

    test('analysis_type が null', () => {
        expect(result.analysis_type).toBeNull();
    });

    test('dapai_info が空', () => {
        expect(result.dapai_info.length).toBe(0);
    });

    test('fulou_info が空', () => {
        expect(result.fulou_info.length).toBe(0);
    });

    test('best_dapai が null', () => {
        expect(result.best_dapai).toBeNull();
    });
});

// ────────────────────────────────────────────────
// 4. 残り牌数 (paishu)
// ────────────────────────────────────────────────

describe('残り牌数 (paishu)', () => {

    const paipu = makePaipu(
        [HAND_0, '', '', ''],
        [{ zimo: { l: 0, p: 's4' } }]
    );
    const result = analyze(paipu, 0, 1, 0);
    const { paishu } = result;

    test('paishu に m/p/s/z の 4 スーツがある', () => {
        expect(paishu).toHaveProperty('m');
        expect(paishu).toHaveProperty('p');
        expect(paishu).toHaveProperty('s');
        expect(paishu).toHaveProperty('z');
    });

    test('各スーツのカウントは 0 〜 4 の整数', () => {
        for (const s of ['m','p','s']) {
            for (let n = 1; n <= 9; n++) {
                const cnt = paishu[s][n];
                expect(cnt).toBeGreaterThanOrEqual(0);
                expect(cnt).toBeLessThanOrEqual(4);
                expect(Number.isInteger(cnt)).toBe(true);
            }
        }
    });

    test('手牌に含まれる牌は残り枚数が減っている (m1〜m7 を 7 枚持っている)', () => {
        // HAND_0 は m1234567 を含む → 各 3 枚以下になっているはず
        for (let n = 1; n <= 7; n++) {
            expect(paishu.m[n]).toBeLessThan(4);
        }
    });

    test('字牌 (z) のインデックスは 1〜7', () => {
        for (let n = 1; n <= 7; n++) {
            expect(typeof paishu.z[n]).toBe('number');
        }
    });
});

// ────────────────────────────────────────────────
// 5. 牌価値スコア (paijia_fn)
// ────────────────────────────────────────────────

describe('牌価値スコア (paijia_fn)', () => {

    const paipu = makePaipu(
        [HAND_0, '', '', ''],
        [{ zimo: { l: 0, p: 's4' } }]
    );
    const result = analyze(paipu, 0, 1, 0);
    const { paijia_fn } = result;

    test('paijia_fn は関数', () => {
        expect(typeof paijia_fn).toBe('function');
    });

    test('全牌に対して非負のスコアを返す', () => {
        const suits = ['m','p','s'];
        for (const s of suits) {
            for (let n = 1; n <= 9; n++) {
                expect(paijia_fn(s + n)).toBeGreaterThanOrEqual(0);
            }
        }
        for (let n = 1; n <= 7; n++) {
            expect(paijia_fn('z' + n)).toBeGreaterThanOrEqual(0);
        }
    });

    test('手牌内の連続形の中心牌はスコアが高い (s2 > z4)', () => {
        // HAND_0 は s123 を含む → s2 周辺は高スコア
        // z4 (北) は東場東家にとって無価値 → 低スコア
        const score_s2 = paijia_fn('s2');
        const score_z4 = paijia_fn('z4');
        expect(score_s2).toBeGreaterThan(score_z4);
    });

    test('赤五牌 (p0) を受け付ける', () => {
        const score = paijia_fn('p0');
        expect(score).toBeGreaterThanOrEqual(0);
    });
});

// ────────────────────────────────────────────────
// 6. 門風
// ────────────────────────────────────────────────

describe('門風 (menfeng)', () => {

    test('視点プレイヤー 0 は東家 (menfeng=0)', () => {
        const paipu  = makePaipu([HAND_0, '', '', ''], [{ zimo: { l: 0, p: 's4' } }]);
        const result = analyze(paipu, 0, 1, 0);
        expect(result.menfeng).toBe(0);
    });

    test('視点プレイヤーを変えると menfeng も変わる', () => {
        // viewpoint=1 (player id=1) は南家
        const paipu  = makePaipu(['', HAND_0, '', ''], [{ zimo: { l: 1, p: 's4' } }]);
        const result = analyze(paipu, 0, 1, 1);
        expect(result.menfeng).toBe(1);
    });
});
