#!/usr/bin/env node
/**
 * 局面状態 NDJSON → 特徴量ベクトル + ラベル JSON
 *
 * parse_paipu.js の出力を読み込み、3モデル用の特徴量を生成する。
 *
 * 使い方:
 *   node extract_features.js --src data/states/states.ndjson --dest data/features/
 *   node extract_features.js --src data/states/ --dest data/features/
 *   node extract_features.js --src data/states/states.ndjson --dest data/features/ --split 0.8,0.1,0.1
 */
'use strict';

const fs   = require('fs');
const path = require('path');
const readline = require('readline');

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

const SRC   = opts['src']   || 'data/states';
const DEST  = opts['dest']  || 'data/features';
const SPLIT = (opts['split'] || '0.8,0.1,0.1').split(',').map(Number);

// ---- 牌エンコーディング定数 ----

// 34種の牌インデックス: m1-m9(0-8), p1-p9(9-17), s1-s9(18-26), z1-z7(27-33)
// 赤5 (0牌) は通常5として扱う

const SUITS = ['m', 'p', 's', 'z'];
const PAI_INDEX = {};
for (let i = 1; i <= 9; i++) PAI_INDEX[`m${i}`] = i - 1;
for (let i = 1; i <= 9; i++) PAI_INDEX[`p${i}`] = 9 + i - 1;
for (let i = 1; i <= 9; i++) PAI_INDEX[`s${i}`] = 18 + i - 1;
for (let i = 1; i <= 7; i++) PAI_INDEX[`z${i}`] = 27 + i - 1;
PAI_INDEX['m0'] = PAI_INDEX['m5'];
PAI_INDEX['p0'] = PAI_INDEX['p5'];
PAI_INDEX['s0'] = PAI_INDEX['s5'];

const N_PAI = 34;

function pai_to_idx(p) {
    const base = p.replace(/[_*+=\-]/g, '');
    return PAI_INDEX[base] ?? -1;
}

// ---- レベル1 手作り特徴量 ----

/**
 * 捨て牌から特徴量ベクトルを生成（1プレイヤー分）
 * サイズ: 3*3 + 7 + 3*9 + 1 = 44
 */
function discard_features(discards) {
    const vec = new Array(44).fill(0);
    let idx = 0;

    // 数牌スートごとに番号帯(1-3, 4-6, 7-9)から何枚切られているか
    for (const s of ['m', 'p', 's']) {
        let low = 0, mid = 0, high = 0;
        for (const p of discards) {
            const base = p.replace(/[_*+=\-]/g, '');
            if (!base.startsWith(s)) continue;
            const n = parseInt(base[1]) || 5;  // 0牌→5
            if (n <= 3) low++;
            else if (n <= 6) mid++;
            else high++;
        }
        vec[idx++] = low;
        vec[idx++] = mid;
        vec[idx++] = high;
    }

    // 字牌を何巡目に切ったか（最初の字牌が早い=染め手or対対でない可能性）
    // z1-z7 ごとに「最初に切った巡数」(0=まだ切っていない、1-18=巡数)
    const first_z = new Array(7).fill(0);
    for (let turn = 0; turn < discards.length; turn++) {
        const base = discards[turn].replace(/[_*+=\-]/g, '');
        if (base.startsWith('z')) {
            const zi = parseInt(base[1]) - 1;
            if (zi >= 0 && zi < 7 && first_z[zi] === 0) first_z[zi] = turn + 1;
        }
    }
    for (const v of first_z) vec[idx++] = v;

    // 数牌スートごとに34牌それぞれを何枚切ったか（m1-m9, p1-p9, s1-s9）
    for (const s of ['m', 'p', 's']) {
        const cnt = new Array(9).fill(0);
        for (const p of discards) {
            const base = p.replace(/[_*+=\-]/g, '');
            if (!base.startsWith(s)) continue;
            const n = (parseInt(base[1]) || 5) - 1;
            if (n >= 0 && n < 9) cnt[n]++;
        }
        for (const v of cnt) vec[idx++] = v;
    }

    // 総切り枚数
    vec[idx++] = discards.length;

    return vec;
}

/**
 * 副露から特徴量ベクトルを生成（1プレイヤー分）
 * サイズ: 34 (副露に含まれる牌の有無) + 4 (副露タイプ counts) = 38
 */
function meld_features(melds) {
    const vec = new Array(38).fill(0);
    let n_chi = 0, n_pon = 0, n_kan = 0, n_kita = 0;

    for (const m of melds) {
        if (!m) continue;
        const clean = m.replace(/[+=\-]/g, '');
        // 面子に含まれる牌をマーク
        const s = clean[0];
        for (let i = 1; i < clean.length; i++) {
            const n = parseInt(clean[i]);
            if (isNaN(n)) continue;
            const real_n = n === 0 ? 5 : n;
            const pi = pai_to_idx(`${s}${real_n}`);
            if (pi >= 0 && pi < N_PAI) vec[pi] = 1;
        }
        // タイプ判別
        if (m.match(/^[mpsz]\d{3}[\+\=\-]$/)) n_chi++;
        else if (m.match(/^[mpsz]\d{3}[\+\=\-]\d$/)) n_pon++;
        else if (m.match(/^[mpsz]\d{4}/)) n_kan++;
    }

    vec[34] = n_chi;
    vec[35] = n_pon;
    vec[36] = n_kan;
    vec[37] = n_kita;

    return vec;
}

/**
 * 点数状況から特徴量ベクトルを生成
 * サイズ: 4(点数正規化) + 4(着順ボーダーとの差) + 3(場況) = 11
 */
function score_features(rec) {
    const { scores, player_ids, l, zhuangfeng, jushu, changbang } = rec;
    const player_id = player_ids[l];
    const my_score  = scores[player_id];

    // 全員の点数を25000点基準で正規化（10000点単位）
    const vec = new Array(11).fill(0);
    let idx = 0;
    for (let id = 0; id < 4; id++) {
        vec[idx++] = (scores[id] - 25000) / 10000;
    }

    // 着順ボーダーとの差（1位との差, 2位との差, 3位との差, ラスとの差）
    const sorted = [...scores].sort((a, b) => b - a);
    for (const border of sorted) {
        vec[idx++] = (my_score - border) / 10000;
    }

    // 場況
    vec[idx++] = zhuangfeng;          // 場風 (0=東, 1=南)
    vec[idx++] = jushu / 4;           // 局数正規化
    vec[idx++] = Math.min(changbang, 8) / 8;  // 本場（上限8で正規化）

    return vec;
}

/**
 * 残り牌数・リーチ状況から特徴量ベクトルを生成
 * サイズ: 1(残り) + 4(リーチ有無) + 4(リーチなら捨て牌枚数) = 9
 */
function game_state_features(rec) {
    const { remaining, riichi_l } = rec;
    const vec = new Array(9).fill(0);
    vec[0] = remaining / 70;  // 正規化（最大70枚前後）
    for (let l = 0; l < 4; l++) {
        vec[1 + l] = riichi_l[l] ? 1 : 0;
        vec[5 + l] = riichi_l[l] ? rec.discards_l[l].length / 18 : 0;
    }
    return vec;
}

// ---- 手牌エンコーディング ----

/**
 * 手牌文字列 → 34次元の枚数ベクトル
 */
function encode_hand(hand_str) {
    const vec = new Array(N_PAI).fill(0);
    if (!hand_str) return vec;
    const base = hand_str.split(',')[0];  // 副露部分を除く
    let s = '';
    for (const c of base) {
        if ('mpsz'.includes(c)) { s = c; continue; }
        const n = parseInt(c);
        if (isNaN(n)) continue;
        const real_n = n === 0 ? 5 : n;
        const pi = pai_to_idx(`${s}${real_n}`);
        if (pi >= 0) vec[pi]++;
    }
    return vec;
}

// ---- 3モデル用の特徴量・ラベル生成 ----

/**
 * 手牌類推モデル用
 * 視点プレイヤー l から見た 対象プレイヤー target_l の特徴量 + ラベル
 */
function make_hand_inference_sample(rec, target_l) {
    const features = [
        ...discard_features(rec.discards_l[target_l]),  // 44
        ...meld_features(rec.melds_l[target_l]),         // 38
        riichi_l_val(rec.riichi_l[target_l]),            // 1
        ...score_features(rec),                          // 11
        ...game_state_features(rec),                     // 9
        // 自分の手牌（公開情報として意思決定者の捨て牌・副露のみ）
        ...discard_features(rec.discards_l[rec.l]),      // 44
        ...meld_features(rec.melds_l[rec.l]),            // 38
    ];
    // 合計: 44+38+1+11+9+44+38 = 185次元

    // ラベル: 対象プレイヤーの真の手牌（ツモ牌を除く公開前手牌）
    const hand_vec = encode_hand(rec.hands_l[target_l]);

    // 注意: target_lのhands_lはツモ直後の状態を含む可能性がある（target_l == rec.l以外）
    // 実際の学習時は target_l != rec.l のサンプルのみ使用する

    return { features, label_hand: hand_vec, meta: { paipu_id: rec.paipu_id, round_idx: rec.round_idx, event_idx: rec.event_idx, viewer_l: rec.l, target_l } };
}

function riichi_l_val(v) { return v ? 1 : 0; }

/**
 * 行動クローンモデル用
 * 意思決定者 l の特徴量 + ラベル（打牌した牌のインデックス）
 */
function make_behavior_clone_sample(rec) {
    // 自分の手牌（意思決定者視点 = ツモ直後の真の手牌）
    const hand_vec = encode_hand(rec.hands_l[rec.l]);

    const features = [
        ...hand_vec,                                     // 34
        ...discard_features(rec.discards_l[rec.l]),     // 44
        ...meld_features(rec.melds_l[rec.l]),            // 38
        ...score_features(rec),                          // 11
        ...game_state_features(rec),                     // 9
    ];
    // 合計: 34+44+38+11+9 = 136次元

    // ラベル: 打牌した牌インデックス
    const action_idx = pai_to_idx(rec.action.replace(/[_*]/g, ''));

    return { features, label_action: action_idx, meta: { paipu_id: rec.paipu_id, round_idx: rec.round_idx, event_idx: rec.event_idx } };
}

/**
 * 価値関数モデル用
 * 意思決定者 l の局面特徴量 + ラベル（局の得失点 + 最終着順点）
 */
function make_value_sample(rec) {
    const hand_vec = encode_hand(rec.hands_l[rec.l]);

    const features = [
        ...hand_vec,                                     // 34
        ...score_features(rec),                          // 11
        ...game_state_features(rec),                     // 9
        rec.remaining / 70,                              // 1: 残り枚数
    ];
    // 合計: 34+11+9+1 = 55次元

    const player_id     = rec.player_ids[rec.l];
    const round_fenpei  = rec.round_fenpei[player_id];  // この局の得失点
    const final_point   = rec.final_points[player_id];  // 対局終了時の着順点

    return { features, label_round_fenpei: round_fenpei, label_final_point: final_point, meta: { paipu_id: rec.paipu_id, round_idx: rec.round_idx, event_idx: rec.event_idx } };
}

// ---- ファイル処理 ----

function collect_ndjson_files(src) {
    const stat = fs.statSync(src);
    if (stat.isFile()) return [src];
    return fs.readdirSync(src)
        .filter(f => f.endsWith('.ndjson'))
        .map(f => path.join(src, f));
}

async function process_file(filepath) {
    const rl = readline.createInterface({ input: fs.createReadStream(filepath) });
    const all = [];
    for await (const line of rl) {
        if (!line.trim()) continue;
        all.push(JSON.parse(line));
    }
    return all;
}

function shuffle_split(arr, ratios) {
    // Fisher-Yates shuffle
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    const n = arr.length;
    const n_train = Math.floor(n * ratios[0]);
    const n_val   = Math.floor(n * ratios[1]);
    return {
        train: arr.slice(0, n_train),
        val:   arr.slice(n_train, n_train + n_val),
        test:  arr.slice(n_train + n_val),
    };
}

async function main() {
    const files = collect_ndjson_files(SRC);
    if (files.length === 0) {
        console.error(`NDJSONファイルが見つかりません: ${SRC}`);
        process.exit(1);
    }

    fs.mkdirSync(DEST, { recursive: true });

    let all_records = [];
    for (const f of files) {
        process.stdout.write(`読み込み中: ${path.basename(f)} ... `);
        const recs = await process_file(f);
        console.log(`${recs.length} レコード`);
        all_records.push(...recs);
    }

    console.log(`合計: ${all_records.length} レコード`);

    // 3モデル用のサンプル生成
    const hi_samples  = [];  // hand_inference
    const bc_samples  = [];  // behavior_clone
    const vf_samples  = [];  // value_function

    for (const rec of all_records) {
        // 行動クローン・価値関数: 意思決定者1件
        bc_samples.push(make_behavior_clone_sample(rec));
        vf_samples.push(make_value_sample(rec));

        // 手牌類推: 他家3名分
        for (let target_l = 0; target_l < 4; target_l++) {
            if (target_l === rec.l) continue;  // 自分は除く
            hi_samples.push(make_hand_inference_sample(rec, target_l));
        }
    }

    // train/val/test 分割して保存
    const datasets = {
        hand_inference:  hi_samples,
        behavior_clone:  bc_samples,
        value_function:  vf_samples,
    };

    for (const [name, samples] of Object.entries(datasets)) {
        const splits = shuffle_split(samples, SPLIT);
        for (const [split_name, data] of Object.entries(splits)) {
            const out_path = path.join(DEST, `${name}_${split_name}.json`);
            fs.writeFileSync(out_path, JSON.stringify(data));
            console.log(`保存: ${out_path} (${data.length} サンプル)`);
        }
    }

    console.log('完了');
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
