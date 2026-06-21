#!/usr/bin/env node
/**
 * 局面状態 NDJSON → 特徴量ベクトル + ラベル NDJSON
 *
 * parse_paipu.js の出力を読み込み、3モデル用の特徴量を生成する。
 * train/val/test の分割は Python 側（train_*.py）で行う。
 *
 * 使い方:
 *   node extract_features.js --src data/states/states.ndjson --dest data/features/
 *   node extract_features.js --src data/states/ --dest data/features/
 *
 * 出力:
 *   hand_inference.ndjson  (1行1サンプル)
 *   behavior_clone.ndjson
 *   value_function.ndjson
 */
'use strict';

const fs   = require('fs');
const path = require('path');
const readline = require('readline');
const Majiang  = require('../../majiang-core/lib');

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
 * player_l: 省略時は rec.l (viewer) を使用
 * サイズ: 4(点数正規化) + 4(着順ボーダーとの差) + 3(場況) = 11
 */
function score_features(rec, player_l) {
    const { scores, player_ids, zhuangfeng, jushu, changbang } = rec;
    const l         = (player_l !== undefined) ? player_l : rec.l;
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
 * 自風・場風の特徴量ベクトル
 * サイズ: 4(自風 one-hot: 東南西北) + 1(場風: 東=0/南=1) = 5
 */
function wind_features(rec, player_l) {
    const jikaze = (player_l - rec.jushu + 4) % 4;
    const oh = [0, 0, 0, 0];
    oh[jikaze] = 1;
    return [...oh, rec.zhuangfeng];
}

/**
 * 自風 one-hot のみ（4次元）。場風は wind_features(target) で共有されるため省略。
 */
function jikaze_onehot(rec, player_l) {
    const oh = [0, 0, 0, 0];
    oh[(player_l - rec.jushu + 4) % 4] = 1;
    return oh;
}

/**
 * 全プレイヤーの捨て牌を順序付きトークン列に変換する（42次元/トークン）
 *
 * トークン構造:
 *   tile_onehot(34) + turn_norm(1) + is_tsumogiri(1) + is_riichi_decl(1) + is_red_five(1) + player_role(4)
 *
 * player_role: [is_target, is_self, is_other1, is_other2]
 * 最大トークン数: 4プレイヤー × 18巡 = 72
 */
const DISCARD_TOKEN_DIM = 42;
const MAX_DISCARD_TURNS = 18;

function build_discard_tokens(rec, target_l, other_ls) {
    const tokens = [];
    const players = [
        { l: target_l,    role: [1, 0, 0, 0] },
        { l: rec.l,       role: [0, 1, 0, 0] },
        { l: other_ls[0], role: [0, 0, 1, 0] },
        { l: other_ls[1], role: [0, 0, 0, 1] },
    ];
    for (const { l, role } of players) {
        const discards = rec.discards_l[l] || [];
        for (let t = 0; t < discards.length; t++) {
            const p = discards[t];
            const base = p.replace(/[_*+=\-]/g, '');
            const n_digit = parseInt(base[1]);
            const is_red        = (n_digit === 0) ? 1 : 0;
            const is_tsumogiri  = p.includes('_') ? 1 : 0;
            const is_riichi_decl = p.includes('*') ? 1 : 0;
            const pi = pai_to_idx(base);
            const tile_oh = new Array(N_PAI).fill(0);
            if (pi >= 0) tile_oh[pi] = 1;
            tokens.push([
                ...tile_oh,
                (t + 1) / MAX_DISCARD_TURNS,
                is_tsumogiri,
                is_riichi_decl,
                is_red,
                ...role,
            ]);
        }
    }
    return tokens;
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

/**
 * 手牌文字列 → 枚数ベクトル(34次元) + 赤牌所持フラグ(3次元: 5m/5p/5s)
 */
function encode_hand_red(hand_str) {
    const vec = new Array(N_PAI).fill(0);
    const red = [0, 0, 0];  // [5m, 5p, 5s]
    if (!hand_str) return { counts: vec, red };
    const base = hand_str.split(',')[0];
    let s = '';
    for (const c of base) {
        if ('mpsz'.includes(c)) { s = c; continue; }
        const n = parseInt(c);
        if (isNaN(n)) continue;
        if (n === 0) {
            const suit_idx = ['m', 'p', 's'].indexOf(s);
            if (suit_idx >= 0) red[suit_idx] = 1;
        }
        const pi = pai_to_idx(`${s}${n === 0 ? 5 : n}`);
        if (pi >= 0) vec[pi]++;
    }
    return { counts: vec, red };
}

// ---- 追加特徴量 ----

/**
 * タイルインデックスから suit_onehot(4) + num_onehot(9) + is_jihai(1) を生成（14次元）
 */
function make_tile_identity(tile_idx) {
    const suit_oh = [0, 0, 0, 0];
    const num_oh  = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    let is_jihai  = 0;
    if (tile_idx < 9) {
        suit_oh[0] = 1; num_oh[tile_idx] = 1;
    } else if (tile_idx < 18) {
        suit_oh[1] = 1; num_oh[tile_idx - 9] = 1;
    } else if (tile_idx < 27) {
        suit_oh[2] = 1; num_oh[tile_idx - 18] = 1;
    } else {
        suit_oh[3] = 1; num_oh[tile_idx - 27] = 1; is_jihai = 1;
    }
    return [...suit_oh, ...num_oh, is_jihai];
}

/**
 * タイル i の隣接±1, ±2 の visible_count を返す（4次元; 字牌・範囲外は0）
 */
function get_neighbor_visible(tile_idx, vis_counts) {
    const nb = [0, 0, 0, 0];
    if (tile_idx >= 27) return nb;
    const suit_offset = Math.floor(tile_idx / 9) * 9;
    const pos = tile_idx - suit_offset;
    if (pos >= 2) nb[0] = vis_counts[suit_offset + pos - 2];
    if (pos >= 1) nb[1] = vis_counts[suit_offset + pos - 1];
    if (pos <= 7) nb[2] = vis_counts[suit_offset + pos + 1];
    if (pos <= 6) nb[3] = vis_counts[suit_offset + pos + 2];
    return nb;
}

/**
 * 副露タイプ (4次元): チー枚数/ポン枚数/カン枚数/副露有無
 */
function meld_type_single(melds) {
    let n_chi = 0, n_pon = 0, n_kan = 0;
    for (const m of melds) {
        if (!m) continue;
        if      (m.match(/^[mpsz]\d{3}[\+\=\-]$/))   n_chi++;
        else if (m.match(/^[mpsz]\d{3}[\+\=\-]\d$/))  n_pon++;
        else if (m.match(/^[mpsz]\d{4}/))              n_kan++;
    }
    return [n_chi, n_pon, n_kan, (n_chi + n_pon + n_kan > 0) ? 1 : 0];
}

/**
 * ポンスルー信号: ポンできたのにしなかった牌をタイルインデックスごとに集計（34次元）
 * t / max_discards で早さを重みに使う（最近のスルーほど大きい値）
 */
function pass_pon_signal(pon_passes, max_discards = 70) {
    const signal = new Array(N_PAI).fill(0);
    for (const { p, t } of (pon_passes || [])) {
        const pi = pai_to_idx(p);
        if (pi >= 0) signal[pi] += t / max_discards;
    }
    return signal;
}

/**
 * チースルー信号: 上家の捨て牌にチーできたのにしなかった牌を集計（34次元）
 * t / max_discards で直近のスルーほど大きい値
 */
function pass_chi_signal(chi_passes, max_discards = 70) {
    const signal = new Array(N_PAI).fill(0);
    for (const { p, t } of (chi_passes || [])) {
        const pi = pai_to_idx(p);
        if (pi >= 0) signal[pi] += t / max_discards;
    }
    return signal;
}

/**
 * ドラ表示牌リスト → 実際のドラ牌ごとのカウントベクトル（34次元、/5 正規化）
 * 表示牌の「次の牌」が実際のドラ（zhenbaopai 変換）。初期1枚＋カン最大4枚で計5枚まで。
 */
function zhenbaopai(p) {
    const s = p[0], n = parseInt(p[1]) || 5;
    return s === 'z'
        ? (n < 5 ? s + (n % 4 + 1) : s + ((n - 4) % 3 + 5))
        : s + (n % 9 + 1);
}

function dora_features(baopai) {
    const vec = new Array(N_PAI).fill(0);
    for (const indicator of (baopai || [])) {
        const pi = pai_to_idx(zhenbaopai(indicator));
        if (pi >= 0) vec[pi]++;
    }
    return vec.map(v => v / 5);  // 最大5枚で正規化
}

/**
 * チー受け取り牌信号: ターゲットのチー面子で上家から受け取った牌を返す（34次元バイナリ）
 *
 * direction marker (+/=/-) の直前の数字が鳴いた牌（shoupai.js:284-326 より確認済み）
 *   m3-45 → m3, m23-4 → m3, m234- → m4
 */
function chi_called_tile_signal(melds) {
    const signal = new Array(N_PAI).fill(0);
    for (const m of (melds || [])) {
        const match = m.match(/^([mps])(\d*)([\+\=\-])(\d*)$/);
        if (!match || match[2].length === 0) continue;
        const suit = match[1];
        const raw_digit = parseInt(match[2][match[2].length - 1]);
        const called_n = raw_digit === 0 ? 5 : raw_digit;
        const pi = pai_to_idx(`${suit}${called_n}`);
        if (pi >= 0) signal[pi] = 1;
    }
    return signal;
}

// ---- 役推定用エンコーディング ----

// 21クラス:
//   0:tanyao /
//   1:yakuhai_haku / 2:yakuhai_hatsu / 3:yakuhai_chun /
//   4:yakuhai_ba / 5:yakuhai_ji /
//   6:chanta / 7:riichi / 8:chiitoi / 9:kokushi /
//   10:suit_man / 11:suit_pin / 12:suit_sou  (混一/清一のスート) /
//   13:toitoi /
//   14:sanshoku_123 / 15:sanshoku_234 / 16:sanshoku_345 / 17:sanshoku_456 /
//   18:sanshoku_567 / 19:sanshoku_678 / 20:sanshoku_789
// ※ pinfu は riichi との相関が高すぎるため除外
// ※ honitsu/chinitsu は suit_man/pin/sou に統合（どの色か分かれば十分）
const YAKU_LABELS = [
    (ns, _s, _ss) => ns.includes('断幺九'),
    (ns, _s, _ss) => ns.includes('役牌 白'),
    (ns, _s, _ss) => ns.includes('役牌 發'),
    (ns, _s, _ss) => ns.includes('役牌 中'),
    (ns, _s, _ss) => ns.some(n => n.startsWith('場風')),
    (ns, _s, _ss) => ns.some(n => n.startsWith('自風')),
    (ns, _s, _ss) => ns.includes('混全帯幺九') || ns.includes('純全帯幺九'),
    (ns, _s, _ss) => ns.includes('立直') || ns.includes('両立直'),
    (ns, _s, _ss) => ns.includes('七対子'),
    (ns, _s, _ss) => ns.includes('国士無双') || ns.includes('国士無双１３面'),
    (ns, s,  _ss) => (ns.includes('混一色') || ns.includes('清一色')) && s === 'm',
    (ns, s,  _ss) => (ns.includes('混一色') || ns.includes('清一色')) && s === 'p',
    (ns, s,  _ss) => (ns.includes('混一色') || ns.includes('清一色')) && s === 's',
    (ns, _s, _ss) => ns.includes('対々和'),
    (ns, _s,  ss) => ns.includes('三色同順') && ss === 1,
    (ns, _s,  ss) => ns.includes('三色同順') && ss === 2,
    (ns, _s,  ss) => ns.includes('三色同順') && ss === 3,
    (ns, _s,  ss) => ns.includes('三色同順') && ss === 4,
    (ns, _s,  ss) => ns.includes('三色同順') && ss === 5,
    (ns, _s,  ss) => ns.includes('三色同順') && ss === 6,
    (ns, _s,  ss) => ns.includes('三色同順') && ss === 7,
];

function encode_yaku(hupai_list, win_suit, win_sanshoku) {
    const names = (hupai_list || []).map(h => h.name || h);
    const s  = win_suit     || '';
    const ss = win_sanshoku || 0;
    return YAKU_LABELS.map(fn => fn(names, s, ss) ? 1 : 0);
}

/**
 * ターゲットプレイヤーが立直なしで赤牌を捨てたか（3次元: 5m/5p/5s）
 * 立直なし + 赤5切り = その5を1枚も持っていない可能性が非常に高い
 */
function red_discard_signal(discards, riichi_flag) {
    if (riichi_flag) return [0, 0, 0];
    return ['m0', 'p0', 's0'].map(r =>
        discards.some(p => p.replace(/[_*+=\-]/g, '') === r) ? 1 : 0
    );
}

/**
 * 全プレイヤーの捨て牌・副露に赤牌（m0/p0/s0）が公開されているかをフラグ化（3次元）
 * 赤牌が公開情報として見えている = 対象プレイヤーが所持していない確定情報
 */
function red_visible_flags(discards_l, melds_l) {
    const flags = [0, 0, 0];  // [m0, p0, s0]
    const suits = ['m', 'p', 's'];
    for (let l = 0; l < 4; l++) {
        for (const p of discards_l[l]) {
            const base = p.replace(/[_*+=\-]/g, '');
            for (let i = 0; i < 3; i++) {
                if (base === suits[i] + '0') flags[i] = 1;
            }
        }
        for (const m of (melds_l[l] || [])) {
            if (!m) continue;
            const si = suits.indexOf(m[0]);
            if (si < 0) continue;
            const clean = m.replace(/[+=\-]/g, '');
            for (let j = 1; j < clean.length; j++) {
                if (clean[j] === '0') { flags[si] = 1; break; }
            }
        }
    }
    return flags;
}

/**
 * 全プレイヤーの捨て牌・副露 + viewer自身の手牌から「見え牌枚数」を計算（34次元、/4 正規化）
 * 相手が持てない枚数の上限を示す情報として手牌推定に使用する
 */
function visible_counts_vec(rec, viewer_l) {
    const counts = new Array(N_PAI).fill(0);
    for (let l = 0; l < 4; l++) {
        // 捨て牌：請求済みのものも捨て牌エリアに残って公開されるのでそのままカウント
        for (const p of rec.discards_l[l]) {
            const pi = pai_to_idx(p);
            if (pi >= 0) counts[pi]++;
        }
        // 副露：請求牌（捨て牌から取った牌）は捨て牌で既にカウント済みなのでスキップする
        // 方向マーカー(+/-/=)の直前の digit が請求牌
        // 暗槓（方向マーカーなし）は全4枚を自手牌から出すため全てカウント
        for (const m of rec.melds_l[l]) {
            if (!m) continue;
            const s = m[0];
            const dirIdx = m.search(/[+=\-]/);  // -1 なら暗槓
            for (let i = 1; i < m.length; i++) {
                if (/[+=\-]/.test(m[i])) continue;       // 方向マーカーはスキップ
                if (dirIdx >= 0 && i === dirIdx - 1) continue;  // 請求牌はスキップ
                const n = parseInt(m[i]);
                if (isNaN(n)) continue;
                const pi = pai_to_idx(`${s}${n === 0 ? 5 : n}`);
                if (pi >= 0) counts[pi]++;
            }
        }
    }
    const hand = encode_hand(rec.hands_l[viewer_l]);
    for (let i = 0; i < N_PAI; i++) counts[i] += hand[i];
    return counts.map(c => c / 4);
}

/**
 * viewer_l 以外の3プレイヤーの副露タイプ（チー/ポン/カン/有無）を返す（12次元）
 * 相対順: (l+1)%4, (l+2)%4, (l+3)%4 の順で各4次元
 */
function others_meld_type_features(rec) {
    const vec = [];
    for (let rel = 1; rel <= 3; rel++) {
        const l = (rec.l + rel) % 4;
        let n_chi = 0, n_pon = 0, n_kan = 0;
        for (const m of rec.melds_l[l]) {
            if (!m) continue;
            if (m.match(/^[mpsz]\d{3}[\+\=\-]$/))  n_chi++;
            else if (m.match(/^[mpsz]\d{3}[\+\=\-]\d$/)) n_pon++;
            else if (m.match(/^[mpsz]\d{4}/))             n_kan++;
        }
        vec.push(n_chi, n_pon, n_kan, (n_chi + n_pon + n_kan > 0) ? 1 : 0);
    }
    return vec;
}

// ---- 3モデル用の特徴量・ラベル生成 ----

/**
 * 手牌類推モデル用 (v37: 固定特徴量673次元 + discard_tokens別フィールド)
 * 視点プレイヤー l から見た 対象プレイヤー target_l の特徴量 + ラベル
 *
 * 固定特徴量 (673次元):
 *   target_meld(38) + riichi(1) + score(11) + game(9) +
 *   self_meld(38) + visible_counts(34) + red_discard_sig(3) + red_visible(3) +
 *   pass_pon(target,34) + wind(target,5) + pass_chi(target,34) + chi_called(target,34) +
 *   dora(34) + other1_meld(38) + other2_meld(38) +
 *   self_pon(34) + o1_pon(34) + o2_pon(34) +
 *   self_chi(34) + o1_chi(34) + o2_chi(34) + lizhibang(1) +
 *   self_jikaze(4) + o1_jikaze(4) + o2_jikaze(4) +
 *   self_chi_called(34) + o1_chi_called(34) + o2_chi_called(34)
 *   = 673次元 (+ add_yaku 21 + add_tenpai 1 = 695次元)
 *
 * discard_tokens: 全4プレイヤーの捨て牌を42次元トークン×N列 (別フィールド)
 * target_discard: target捨て牌集計(44次元) → Stage-1 モデル互換用 (別フィールド)
 */
function make_hand_inference_sample(rec, target_l) {
    // viewer でも target でもない2プレイヤーを相対順で取得
    const other_ls = [1, 2, 3]
        .map(rel => (rec.l + rel) % 4)
        .filter(l => l !== target_l);  // 必ず2要素

    const features = [
        ...meld_features(rec.melds_l[target_l]),         // 38
        riichi_l_val(rec.riichi_l[target_l]),            //  1
        ...score_features(rec),                          // 11
        ...game_state_features(rec),                     //  9
        ...meld_features(rec.melds_l[rec.l]),            // 38
        ...visible_counts_vec(rec, rec.l),               // 34
        ...red_discard_signal(rec.discards_l[target_l], rec.riichi_l[target_l]),  //  3
        ...red_visible_flags(rec.discards_l, rec.melds_l),                        //  3
        ...pass_pon_signal(rec.pon_passes_l?.[target_l]),          // 34
        ...wind_features(rec, target_l),                           //  5
        ...pass_chi_signal(rec.chi_passes_l?.[target_l]),          // 34
        ...chi_called_tile_signal(rec.melds_l?.[target_l]),        // 34
        ...dora_features(rec.baopai),                              // 34
        ...meld_features(rec.melds_l[other_ls[0]]),                // 38 other1_meld
        ...meld_features(rec.melds_l[other_ls[1]]),                // 38 other2_meld
        ...pass_pon_signal(rec.pon_passes_l?.[rec.l]),             // 34 self_pon_pass
        ...pass_pon_signal(rec.pon_passes_l?.[other_ls[0]]),       // 34 other1_pon_pass
        ...pass_pon_signal(rec.pon_passes_l?.[other_ls[1]]),       // 34 other2_pon_pass
        ...pass_chi_signal(rec.chi_passes_l?.[rec.l]),             // 34 self_chi_pass
        ...pass_chi_signal(rec.chi_passes_l?.[other_ls[0]]),       // 34 other1_chi_pass
        ...pass_chi_signal(rec.chi_passes_l?.[other_ls[1]]),       // 34 other2_chi_pass
        Math.min(rec.lizhibang || 0, 8) / 8,                       //  1 lizhibang
        ...jikaze_onehot(rec, rec.l),                              //  4 self_jikaze
        ...jikaze_onehot(rec, other_ls[0]),                        //  4 other1_jikaze
        ...jikaze_onehot(rec, other_ls[1]),                        //  4 other2_jikaze
        ...chi_called_tile_signal(rec.melds_l?.[rec.l]),           // 34 self_chi_called
        ...chi_called_tile_signal(rec.melds_l?.[other_ls[0]]),     // 34 other1_chi_called
        ...chi_called_tile_signal(rec.melds_l?.[other_ls[1]]),     // 34 other2_chi_called
    ];  // 673次元 (+ add_yaku 21 + add_tenpai 1 = 695次元)

    const discard_tokens = build_discard_tokens(rec, target_l, other_ls);
    const target_discard = discard_features(rec.discards_l[target_l]);  // Stage-1互換用

    const { counts: hand_vec_target, red: red_target } = encode_hand_red(rec.hands_l[target_l]);

    const label_block = (rec.shanten_l && rec.shanten_l[target_l] === 0)
        ? compute_block_labels(Majiang.Shoupai.fromString(rec.hands_l[target_l]))
        : null;

    return { features, discard_tokens, target_discard, label_hand: hand_vec_target, label_red: red_target, label_block, meta: { paipu_id: rec.paipu_id, round_idx: rec.round_idx, event_idx: rec.event_idx, viewer_l: rec.l, target_l } };
}

/**
 * 役推定モデル用 (Stage 1: 92次元入力)
 * 意思決定者 l の行動情報から最終役を予測するサンプル
 *
 * target_discard(44) + target_meld(38) + riichi(1) + game_state(9) = 92次元
 */
function make_yaku_sample(rec) {
    if (!rec.yaku_l) return null;
    const features = [
        ...discard_features(rec.discards_l[rec.l]),  // 44
        ...meld_features(rec.melds_l[rec.l]),         // 38
        riichi_l_val(rec.riichi_l[rec.l]),            // 1
        ...game_state_features(rec),                  // 9
        ...score_features(rec),                       // 11
        ...wind_features(rec, rec.l),                 // 5
    ];  // 108次元

    const yaku         = rec.yaku_l          ? (rec.yaku_l[rec.l]          || []) : [];
    const win_suit     = rec.win_suit_l      ? (rec.win_suit_l[rec.l]      || '') : '';
    const win_sanshoku = rec.win_sanshoku_l  ? (rec.win_sanshoku_l[rec.l]  || 0)  : 0;
    const label_yaku = encode_yaku(yaku, win_suit, win_sanshoku);
    const won = label_yaku.some(v => v > 0);
    return { features, label_yaku, won, meta: { paipu_id: rec.paipu_id, round_idx: rec.round_idx, event_idx: rec.event_idx } };
}

/**
 * 聴牌推定モデル用 (Stage 1: 108次元入力、yaku_inference と同一構造)
 * target_l の observable 情報から聴牌確率を予測するサンプル
 */
function make_tenpai_sample(rec, target_l) {
    if (!rec.shanten_l || rec.shanten_l[target_l] === 8) return null;
    const features = [
        ...discard_features(rec.discards_l[target_l]),   // 44
        ...meld_features(rec.melds_l[target_l]),          // 38
        riichi_l_val(rec.riichi_l[target_l]),             // 1
        ...game_state_features(rec),                      // 9
        ...score_features(rec),                           // 11
        ...wind_features(rec, target_l),                  // 5
    ];  // 108次元
    const is_tenpai = (rec.shanten_l[target_l] === 0) ? 1 : 0;
    return { features, label: is_tenpai, meta: { paipu_id: rec.paipu_id, round_idx: rec.round_idx, event_idx: rec.event_idx, target_l } };
}

// ---- ブロックラベル生成（v11用） ----

function meld_str_to_block(meld_str, triplet, seq, pair) {
    const suit = meld_str[0];
    const nums = meld_str.slice(1).replace(/[^0-9]/g, '').split('').map(Number);
    const suit_offset = { m: 0, p: 9, s: 18, z: 27 }[suit];
    if (suit_offset === undefined || nums.length === 0) return;

    if (nums.length >= 3 && nums[0] === nums[1] && nums[1] === nums[2]) {
        // 刻子 or 槓 → triplet として扱う
        triplet[suit_offset + nums[0] - 1]++;
    } else if (nums.length === 3) {
        // 順子
        const seq_base = { m: 0, p: 7, s: 14 }[suit];
        if (seq_base !== undefined) seq[seq_base + nums[0] - 1]++;
    } else if (nums.length === 2 && nums[0] === nums[1]) {
        // 対子
        pair[suit_offset + nums[0] - 1]++;
    }
    // 単張(国士用) は無視
}

function compute_block_labels(shoupai) {
    const triplet = new Array(34).fill(0);
    const seq     = new Array(21).fill(0);
    const pair    = new Array(34).fill(0);
    let total = 0;

    try {
        const waiting = Majiang.Util.tingpai(shoupai);
        if (!waiting || waiting.length === 0) return null;
        for (const wp of waiting) {
            const decomps = Majiang.Util.hule_mianzi(shoupai, wp + '+');
            for (const decomp of decomps) {
                total++;
                for (const meld of decomp) {
                    meld_str_to_block(meld, triplet, seq, pair);
                }
            }
        }
    } catch (e) {
        return null;
    }

    if (total === 0) return null;

    return {
        triplet: triplet.map(v => v / total),
        seq:     seq.map(v => v / total),
        pair:    pair.map(v => v / total),
    };
}

function riichi_l_val(v) { return v ? 1 : 0; }

/**
 * 行動クローンモデル用
 * 意思決定者 l の特徴量 + ラベル（打牌した牌のインデックス + 有効アクションマスク）
 */
function make_behavior_clone_sample(rec) {
    const hand_vec = encode_hand(rec.hands_l[rec.l]);

    const features = [
        ...hand_vec,                                                    // 34
        ...discard_features(rec.discards_l[rec.l]),                    // 44
        ...meld_features(rec.melds_l[rec.l]),                          // 38
        ...score_features(rec),                                         // 11
        ...game_state_features(rec),                                    // 9
        ...discard_features(rec.discards_l[(rec.l + 1) % 4]),          // 44
        ...discard_features(rec.discards_l[(rec.l + 2) % 4]),          // 44
        ...discard_features(rec.discards_l[(rec.l + 3) % 4]),          // 44
    ];
    // 合計: 34+44+38+11+9+44+44+44 = 268次元

    const action_idx = pai_to_idx(rec.action.replace(/[_*]/g, ''));
    // 手牌にある牌のみ打牌可能 → 学習時のマスクとして使用
    const label_mask = hand_vec.map(c => c > 0 ? 1 : 0);

    return { features, label_action: action_idx, label_mask, meta: { paipu_id: rec.paipu_id, round_idx: rec.round_idx, event_idx: rec.event_idx } };
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
        rec.remaining / 70,                              // 1
        ...others_meld_type_features(rec),               // 12
    ];
    // 合計: 34+11+9+1+12 = 67次元

    const player_id     = rec.player_ids[rec.l];
    const round_fenpei  = rec.round_fenpei[player_id];
    const final_point   = rec.final_points[player_id];

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

async function main() {
    const files = collect_ndjson_files(SRC);
    if (files.length === 0) {
        console.error(`NDJSONファイルが見つかりません: ${SRC}`);
        process.exit(1);
    }

    fs.mkdirSync(DEST, { recursive: true });

    const streams = {
        hand_inference:   fs.createWriteStream(path.join(DEST, 'hand_inference.ndjson'),   { encoding: 'utf8' }),
        behavior_clone:   fs.createWriteStream(path.join(DEST, 'behavior_clone.ndjson'),   { encoding: 'utf8' }),
        value_function:   fs.createWriteStream(path.join(DEST, 'value_function.ndjson'),   { encoding: 'utf8' }),
        yaku_inference:   fs.createWriteStream(path.join(DEST, 'yaku_inference.ndjson'),   { encoding: 'utf8' }),
        tenpai_inference: fs.createWriteStream(path.join(DEST, 'tenpai_inference.ndjson'), { encoding: 'utf8' }),
    };

    let total_recs = 0, hi_cnt = 0, bc_cnt = 0, vf_cnt = 0, yi_cnt = 0, ti_cnt = 0;

    for (const f of files) {
        process.stdout.write(`処理中: ${path.basename(f)} ... `);
        const recs = await process_file(f);
        for (const rec of recs) {
            streams.behavior_clone.write(JSON.stringify(make_behavior_clone_sample(rec)) + '\n');
            streams.value_function.write(JSON.stringify(make_value_sample(rec)) + '\n');
            bc_cnt++;
            vf_cnt++;

            const yi = make_yaku_sample(rec);
            if (yi) { streams.yaku_inference.write(JSON.stringify(yi) + '\n'); yi_cnt++; }

            for (let target_l = 0; target_l < 4; target_l++) {
                if (target_l === rec.l) continue;
                streams.hand_inference.write(JSON.stringify(make_hand_inference_sample(rec, target_l)) + '\n');
                hi_cnt++;
                const ti = make_tenpai_sample(rec, target_l);
                if (ti) { streams.tenpai_inference.write(JSON.stringify(ti) + '\n'); ti_cnt++; }
            }
        }
        total_recs += recs.length;
        console.log(`${recs.length} レコード`);
    }

    await Promise.all(Object.values(streams).map(s => new Promise(r => s.end(r))));

    console.log(`完了: ${total_recs} レコード処理`);
    console.log(`  hand_inference: ${hi_cnt} サンプル → ${path.join(DEST, 'hand_inference.ndjson')}`);
    console.log(`  behavior_clone: ${bc_cnt} サンプル → ${path.join(DEST, 'behavior_clone.ndjson')}`);
    console.log(`  value_function: ${vf_cnt} サンプル → ${path.join(DEST, 'value_function.ndjson')}`);
    console.log(`  yaku_inference:   ${yi_cnt} サンプル → ${path.join(DEST, 'yaku_inference.ndjson')}`);
    console.log(`  tenpai_inference: ${ti_cnt} サンプル → ${path.join(DEST, 'tenpai_inference.ndjson')}`);
}

if (require.main === module) {
    main().catch(err => {
        console.error(err);
        process.exit(1);
    });
} else {
    module.exports = { red_visible_flags, red_discard_signal, visible_counts_vec, pai_to_idx };
}
