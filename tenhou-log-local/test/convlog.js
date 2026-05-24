'use strict';

const assert  = require('assert');
const fs      = require('fs');
const path    = require('path');
const convlog = require('../lib/convlog');
const parse_type = convlog.parse_type;

function minimal_xml(extra_tags) {
    return '<mjloggm ver="2.3">'
        + '<GO type="185" lobby="0"/>'
        + '<UN n0="A" n1="B" n2="C" n3="" dan="0,0,0,0" rate="1500,1500,1500,0" sx="M,M,M,C"/>'
        + '<TAIKYOKU oya="0"/>'
        + '<INIT seed="0,0,0,0,0,135" ten="350,350,350,0" oya="0"'
        + ' hai0="0,4,8,12,16,20,24,28,32,36,40,44,48"'
        + ' hai1="1,5,9,13,17,21,25,29,33,37,41,45,49"'
        + ' hai2="2,6,10,14,18,22,26,30,34,38,42,46,50" hai3=""/>'
        + (extra_tags || '')
        + '<RYUUKYOKU ba="0,0" sc="350,0,350,0,350,0,0,0"/>'
        + '</mjloggm>';
}

suite('parse_type(185)', () => {
    test('三麻フラグが true になること', () => {
        const result = parse_type(185);
        assert.strictEqual(result[0], '三');
    });
    test('戻り値が "三" で始まること', () => {
        assert.ok(parse_type(185).startsWith('三'));
    });
});

suite('北抜き N タグ変換', () => {
    function kita_xml(m_val) {
        return minimal_xml('<T0/><N who="0" m="' + m_val + '"/><T1/>');
    }

    test('m=30752 → {kita:{l:0, p:"z4"}}', () => {
        const result = convlog(kita_xml(30752));
        const hand   = result.log[0];
        const entry  = hand.find(e => e.kita);
        assert.ok(entry, 'kita エントリが存在すること');
        assert.deepStrictEqual(entry.kita, {l: 0, p: 'z4'});
    });

    test('m=31008 → {kita:{l:0, p:"z4"}}', () => {
        const result = convlog(kita_xml(31008));
        const entry  = result.log[0].find(e => e.kita);
        assert.ok(entry);
        assert.deepStrictEqual(entry.kita, {l: 0, p: 'z4'});
    });

    test('m=31264 → {kita:{l:0, p:"z4"}}', () => {
        const result = convlog(kita_xml(31264));
        const entry  = result.log[0].find(e => e.kita);
        assert.ok(entry);
        assert.deepStrictEqual(entry.kita, {l: 0, p: 'z4'});
    });

    test('北抜き後の draw が gangzimo として変換されること', () => {
        const hand      = convlog(kita_xml(30752)).log[0];
        const kita_idx  = hand.findIndex(e => e.kita);
        assert.ok(kita_idx >= 0, 'kita エントリが見つかること');
        const next = hand[kita_idx + 1];
        assert.ok(next && next.gangzimo, '北抜き直後のツモが gangzimo になること');
    });
});

suite('qipai 三麻', () => {
    let qipai;
    setup(() => {
        qipai = convlog(minimal_xml()).log[0][0].qipai;
    });

    test('hai3="" のとき shoupai[3] が "" になること', () => {
        assert.strictEqual(qipai.shoupai[3], '');
    });

    test('ten の4番目が 0 のとき defen[3]=0 になること', () => {
        assert.strictEqual(qipai.defen[3], 0);
    });
});

suite('owari 三麻', () => {
    const owari_xml = minimal_xml('').replace(
        '<RYUUKYOKU ba="0,0" sc="350,0,350,0,350,0,0,0"/>',
        '<RYUUKYOKU ba="0,0" sc="350,0,350,0,350,0,0,0"'
            + ' owari="300,10.0,250,-5.0,450,30.0,0,0.0"/>'
    );
    let result;
    setup(() => {
        result = convlog(owari_xml);
    });

    test('4番目プレイヤーの rank が 4（最下位）になること', () => {
        assert.strictEqual(result.rank[3], 4);
    });

    test('4番目プレイヤーの defen が 0 になること', () => {
        assert.strictEqual(result.defen[3], 0);
    });
});

suite('full conversion（スナップショット）', () => {
    test('sanma_sample.xml の変換結果が sanma_expected.json と一致すること', () => {
        const xml      = fs.readFileSync(
            path.join(__dirname, 'fixtures/sanma_sample.xml'), 'utf-8');
        const expected = JSON.parse(fs.readFileSync(
            path.join(__dirname, 'fixtures/sanma_expected.json'), 'utf-8'));
        assert.deepStrictEqual(convlog(xml), expected);
    });
});
