#!/usr/bin/env node
/**
 * 天鳳牌譜XML → 局面状態 NDJSON
 *
 * ツモ後の打牌決定点を1レコードとして出力する。
 * board.js でリプレイして全プレイヤーの真の手牌を取得。
 *
 * 使い方:
 *   node parse_paipu.js --src data/raw/         --dest data/states/
 *   node parse_paipu.js --src data/raw/foo.xml   --dest data/states/foo.ndjson
 *   node parse_paipu.js --src data/raw/ --dest data/states/ --verbose
 */
'use strict';

const fs      = require('fs');
const path    = require('path');
const convlog = require('../../tenhou-log-local/lib/convlog');
const Majiang = require('../../majiang-core/lib');

// ---- CLI引数パース ----

const args = process.argv.slice(2);
const opts = {};
for (let i = 0; i < args.length; i++) {
    const k = args[i];
    if (k.startsWith('--')) {
        const key = k.slice(2);
        const val = (args[i + 1] && !args[i + 1].startsWith('--')) ? args[++i] : true;
        opts[key] = val;
    }
}

const SRC     = opts['src']  || 'data/raw';
const DEST    = opts['dest'] || 'data/states';
const VERBOSE = !!opts['verbose'];

if (!SRC) {
    console.error('使い方: node parse_paipu.js --src <xml-dir|xml-file> --dest <dir|file>');
    process.exit(1);
}

// ---- ファイル収集 ----

function collect_xml_files(src) {
    const stat = fs.statSync(src);
    if (stat.isFile()) return [src];
    return fs.readdirSync(src)
        .filter(f => f.endsWith('.xml'))
        .map(f => path.join(src, f));
}

// ---- ポンスルー検出用ヘルパー ----

function normalize_tile(p) {
    const base = p.replace(/[_*+=\-]/g, '');
    return base.replace(/^([mps])0$/, '$15');
}

function count_tile_in_shoupai(shoupai, tile_norm) {
    if (!shoupai) return 0;
    const hand_str = shoupai.toString().split(',')[0];
    let count = 0, suit = '';
    for (const c of hand_str) {
        if ('mpsz'.includes(c)) { suit = c; continue; }
        const n = parseInt(c);
        if (isNaN(n)) continue;
        if (`${suit}${n === 0 ? 5 : n}` === tile_norm) count++;
    }
    return count;
}

function is_pon_of(meld_str, tile_norm) {
    if (!meld_str || !meld_str.match(/^[mpsz]\d{3}[\+\=\-]\d$/)) return false;
    const suit = meld_str[0];
    const n    = parseInt(meld_str[1]);
    return `${suit}${n === 0 ? 5 : n}` === tile_norm;
}

function can_chi(shoupai, tile_norm) {
    const suit = tile_norm[0];
    if (!'mps'.includes(suit)) return false;
    const n = parseInt(tile_norm[1]);
    if (n < 1 || n > 9) return false;
    const hasA = n >= 3
        && count_tile_in_shoupai(shoupai, `${suit}${n-2}`) >= 1
        && count_tile_in_shoupai(shoupai, `${suit}${n-1}`) >= 1;
    const hasB = n >= 2 && n <= 8
        && count_tile_in_shoupai(shoupai, `${suit}${n-1}`) >= 1
        && count_tile_in_shoupai(shoupai, `${suit}${n+1}`) >= 1;
    const hasC = n <= 7
        && count_tile_in_shoupai(shoupai, `${suit}${n+1}`) >= 1
        && count_tile_in_shoupai(shoupai, `${suit}${n+2}`) >= 1;
    return hasA || hasB || hasC;
}

function is_chi_of(meld_str, tile_norm) {
    if (!meld_str) return false;
    const match = meld_str.match(/^([mps])(\d*)([\+\=\-])(\d*)$/);
    if (!match) return false;
    if (tile_norm[0] !== match[1]) return false;
    const digits = (match[2] + match[4]).split('').map(c => parseInt(c) === 0 ? 5 : parseInt(c));
    const tile_n = parseInt(tile_norm[1]) === 0 ? 5 : parseInt(tile_norm[1]);
    return digits.includes(tile_n);
}

// ---- 局の役情報・和了スートを収集（バックフィル用） ----

function detect_primary_suit(shoupai_str) {
    // 和了手牌文字列から最多の数牌スートを返す（'m'/'p'/'s'/''）
    if (!shoupai_str) return '';
    const base = shoupai_str.split(',')[0];
    const counts = { m: 0, p: 0, s: 0 };
    let suit = '';
    for (const c of base) {
        if ('mps'.includes(c)) { suit = c; continue; }
        if (suit && !isNaN(parseInt(c))) counts[suit]++;
    }
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return sorted[0][1] > 0 ? sorted[0][0] : '';
}

function detect_sanshoku(shoupai_str) {
    // 和了手牌から三色同順のシーケンス番号を返す（1〜7、なければ0）
    if (!shoupai_str) return 0;
    const cnt = { m: new Array(10).fill(0), p: new Array(10).fill(0), s: new Array(10).fill(0) };
    for (const part of shoupai_str.split(',')) {
        let suit = '';
        for (const c of part.replace(/[+=\-_*]/g, '')) {
            if ('mps'.includes(c)) { suit = c; continue; }
            const n = parseInt(c);
            if (!isNaN(n) && suit) cnt[suit][n === 0 ? 5 : n]++;
        }
    }
    for (let n = 1; n <= 7; n++) {
        if (cnt.m[n] >= 1 && cnt.m[n+1] >= 1 && cnt.m[n+2] >= 1 &&
            cnt.p[n] >= 1 && cnt.p[n+1] >= 1 && cnt.p[n+2] >= 1 &&
            cnt.s[n] >= 1 && cnt.s[n+1] >= 1 && cnt.s[n+2] >= 1) {
            return n;
        }
    }
    return 0;
}

function get_yaku_l(round_log) {
    const yaku_l          = [[], [], [], []];
    const win_suit_l      = ['', '', '', ''];
    const win_sanshoku_l  = [0, 0, 0, 0];
    for (const ev of round_log) {
        if (!ev.hule) continue;
        const l = ev.hule.l;
        yaku_l[l] = (ev.hule.hupai || []).map(h => h.name || h);
        if (ev.hule.shoupai) {
            win_suit_l[l]     = detect_primary_suit(ev.hule.shoupai);
            win_sanshoku_l[l] = detect_sanshoku(ev.hule.shoupai);
        }
    }
    return { yaku_l, win_suit_l, win_sanshoku_l };
}

// ---- 局の最終得失点を収集 ----

function get_round_fenpei(round_log, board) {
    // round_log の最後の hule/pingju から fenpei を収集し player_id 軸に変換
    const fenpei_by_id = [0, 0, 0, 0];
    for (const ev of round_log) {
        const fenpei = ev.hule?.fenpei ?? ev.pingju?.fenpei;
        if (!fenpei) continue;
        // fenpei[l] は局内ポジション順 → player_id に変換
        for (let l = 0; l < 4; l++) {
            fenpei_by_id[board.player_id[l]] += (fenpei[l] || 0);
        }
    }
    return fenpei_by_id;
}

// ---- 1局分をリプレイして全レコードを生成 ----

function parse_round(paipu_id, paipu, round_idx, board) {
    const round_log = paipu.log[round_idx];
    const records   = [];

    // board はすでに前の局の状態。qipai で初期化される。
    // 捨て牌・リーチ・ポンスルーをローカルで追跡
    const discards_l   = [[], [], [], []];
    const riichi_l     = [false, false, false, false];
    const pon_passes_l = [[], [], [], []];
    const chi_passes_l = [[], [], [], []];
    let   total_discards = 0;

    // 局末の得失点・役情報（先に収集してすべてのレコードに付与）
    const round_fenpei          = get_round_fenpei(round_log, board);
    const { yaku_l, win_suit_l, win_sanshoku_l } = get_yaku_l(round_log);

    // 対局終了結果
    const final_points = paipu.point ? paipu.point.map(p => parseFloat(p)) : [0, 0, 0, 0];
    const final_ranks  = paipu.rank  || [0, 0, 0, 0];

    for (let event_idx = 0; event_idx < round_log.length; event_idx++) {
        const ev  = round_log[event_idx];
        const key = Object.keys(ev)[0];
        const val = ev[key];

        // dapai の前に捨て牌リストを更新（リーチフラグも）
        if (key === 'dapai') {
            discards_l[val.l].push(val.p);
            if (val.p.endsWith('*')) riichi_l[val.l] = true;

            // ポンスルー検出: 他家が2枚以上持ちなのにポンしなかったケース
            const tile_norm = normalize_tile(val.p);
            const next_ev   = round_log[event_idx + 1];
            const ponner_l  = (next_ev?.fulou && is_pon_of(next_ev.fulou.m, tile_norm))
                ? next_ev.fulou.l : -1;

            for (let l = 0; l < 4; l++) {
                if (l === val.l)       continue;
                if (riichi_l[l])       continue;
                if (!board.shoupai[l]) continue;
                if (count_tile_in_shoupai(board.shoupai[l], tile_norm) >= 2) {
                    if (l !== ponner_l) {
                        pon_passes_l[l].push({ p: tile_norm, t: total_discards });
                    }
                }
            }

            // チースルー検出: 上家 (val.l+1)%4 のみがチー可能
            const chi_caller_l = (val.l + 1) % 4;
            if (
                !riichi_l[chi_caller_l] &&
                board.shoupai[chi_caller_l] &&
                can_chi(board.shoupai[chi_caller_l], tile_norm)
            ) {
                const is_chi = next_ev?.fulou
                    && is_chi_of(next_ev.fulou.m, tile_norm)
                    && next_ev.fulou.l === chi_caller_l;
                if (!is_chi) {
                    chi_passes_l[chi_caller_l].push({ p: tile_norm, t: total_discards });
                }
            }
            total_discards++;
        }

        // zimo / gangzimo の後 → 打牌決定点としてレコード化
        // Board には gangzimo メソッドがないため zimo として扱う
        if (key === 'zimo' || key === 'gangzimo') {
            board.zimo(val);

            // 次のイベントが dapai であることを確認して action を取得
            const next_ev  = round_log[event_idx + 1];
            const next_key = next_ev ? Object.keys(next_ev)[0] : null;
            if (next_key !== 'dapai' || next_ev.dapai.l !== val.l) {
                // 和了（ツモ）など打牌なしで進む場合はスキップ
                continue;
            }
            const action = next_ev.dapai.p;

            // 全プレイヤーの手牌を文字列で記録（ツモ牌込み）
            const hands_l = board.shoupai.map(s => s.toString());

            // 副露面子をShoupaiの文字列から抽出
            const melds_l = hands_l.map(h => {
                const parts = h.split(',');
                return parts.length > 1 ? parts.slice(1) : [];
            });

            // 現在の点数 (player_id順)
            const scores = [0, 1, 2, 3].map(id => board.defen[id]);

            // 非アクションプレイヤーの向聴数（聴牌推定ラベル用）
            const shanten_l = board.shoupai.map((s, l) => {
                if (l === val.l || !s) return 8;
                try { return Majiang.Util.xiangting(s); } catch(e) { return 8; }
            });

            records.push({
                paipu_id,
                round_idx,
                event_idx,
                event_type:  key,

                zhuangfeng:  board.zhuangfeng,
                jushu:       board.jushu,
                changbang:   board.changbang,
                lizhibang:   board.lizhibang,
                player_ids:  [...board.player_id],

                l:           val.l,
                player_id:   board.player_id[val.l],
                drawn_tile:  val.p,

                hands_l,
                discards_l:  discards_l.map(d => [...d]),
                melds_l,
                riichi_l:    [...riichi_l],

                scores,
                remaining:   board.shan.paishu,
                baopai:      [...board.shan.baopai],

                action,
                round_fenpei,
                final_points,
                final_ranks,
                pon_passes_l: pon_passes_l.map(a => [...a]),
                chi_passes_l: chi_passes_l.map(a => [...a]),
                yaku_l,
                win_suit_l,
                win_sanshoku_l,
                shanten_l,
            });
            continue;
        }

        // その他のイベントを Board に適用
        // gangzimo は zimo として、qipai は parse_file 側で処理済みだが再実行しても無害
        const board_key = key === 'gangzimo' ? 'zimo' : key;
        if (typeof board[board_key] === 'function') {
            board[board_key](val);
        }
    }

    return records;
}

// ---- 1牌譜ファイルを処理 ----

function parse_file(xml_path) {
    const xml    = fs.readFileSync(xml_path, 'utf8');
    const id     = path.basename(xml_path, '.xml');
    let   paipu;
    try {
        paipu = convlog(xml, id);
    } catch (err) {
        if (VERBOSE) console.error(`  変換エラー (${id}): ${err.message}`);
        return [];
    }

    if (!paipu.log || paipu.log.length === 0) return [];

    const board   = new Majiang.Board({ title: paipu.title, player: paipu.player, qijia: paipu.qijia });
    const records = [];

    for (let round_idx = 0; round_idx < paipu.log.length; round_idx++) {
        const round_log = paipu.log[round_idx];
        if (!round_log || round_log.length === 0) continue;

        // qipai でボードを初期化
        const qipai_ev = round_log[0];
        if (!qipai_ev?.qipai) continue;
        board.qipai(qipai_ev.qipai);

        // ただし round_fenpei の計算は board.player_id が確定した後に行う必要があるため
        // parse_round 内で board.player_id を参照している
        // → qipai 後に呼び出す
        const recs = parse_round(id, paipu, round_idx, board);
        records.push(...recs);
    }

    return records;
}

// ---- メイン ----

function main() {
    const files = collect_xml_files(SRC);
    if (files.length === 0) {
        console.error(`XMLファイルが見つかりません: ${SRC}`);
        process.exit(1);
    }

    // 出力先を決定
    let out_path;
    if (DEST.endsWith('.ndjson')) {
        out_path = DEST;
    } else {
        fs.mkdirSync(DEST, { recursive: true });
        const basename = files.length === 1
            ? path.basename(files[0], '.xml') + '.ndjson'
            : 'states.ndjson';
        out_path = path.join(DEST, basename);
    }

    const out = fs.createWriteStream(out_path, { encoding: 'utf8' });
    let total = 0;

    for (const f of files) {
        if (VERBOSE) process.stdout.write(`処理中: ${path.basename(f)} ... `);
        const records = parse_file(f);
        for (const rec of records) {
            out.write(JSON.stringify(rec) + '\n');
        }
        if (VERBOSE) console.log(`${records.length} レコード`);
        total += records.length;
    }

    out.end();
    console.log(`完了: ${files.length} ファイル → ${total} レコード → ${out_path}`);
}

main();
