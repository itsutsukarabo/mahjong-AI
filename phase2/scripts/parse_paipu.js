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

// ---- 局の役情報を収集（バックフィル用） ----

function get_yaku_l(round_log) {
    const yaku_l = [[], [], [], []];
    for (const ev of round_log) {
        if (!ev.hule) continue;
        const l = ev.hule.l;
        yaku_l[l] = (ev.hule.hupai || []).map(h => h.name || h);
    }
    return yaku_l;
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
    let   total_discards = 0;

    // 局末の得失点・役情報（先に収集してすべてのレコードに付与）
    const round_fenpei = get_round_fenpei(round_log, board);
    const yaku_l       = get_yaku_l(round_log);

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

                action,
                round_fenpei,
                final_points,
                final_ranks,
                pon_passes_l: pon_passes_l.map(a => [...a]),
                yaku_l,
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
