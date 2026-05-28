/**
 * phase2/browser/ai_phase2.js
 * ブラウザ側 Phase2 ONNX推論モジュール
 *
 * 前提: onnxruntime-web (global `ort`) がこのスクリプトより先にロードされていること。
 * 動作: DOMContentLoaded 後に window.AI_PHASE2 を設定する。
 *
 * window.AI_PHASE2.analyze(paipu, log_idx, current_idx, board_model, menfeng, analysis_type)
 *   → Promise<{ behavior_clone?, value_function?, hand_inference? }>
 */
'use strict';

(function () {

    /* ---- 牌インデックス (extract_features.js と完全一致) ---- */

    const PAI_INDEX = {};
    for (let i = 1; i <= 9; i++) PAI_INDEX['m' + i] = i - 1;
    for (let i = 1; i <= 9; i++) PAI_INDEX['p' + i] = 9  + i - 1;
    for (let i = 1; i <= 9; i++) PAI_INDEX['s' + i] = 18 + i - 1;
    for (let i = 1; i <= 7; i++) PAI_INDEX['z' + i] = 27 + i - 1;
    PAI_INDEX['m0'] = PAI_INDEX['m5'];
    PAI_INDEX['p0'] = PAI_INDEX['p5'];
    PAI_INDEX['s0'] = PAI_INDEX['s5'];

    const N_PAI = 34;
    const PAI_NAMES = [];
    for (let i = 1; i <= 9; i++) PAI_NAMES.push('m' + i);
    for (let i = 1; i <= 9; i++) PAI_NAMES.push('p' + i);
    for (let i = 1; i <= 9; i++) PAI_NAMES.push('s' + i);
    for (let i = 1; i <= 7; i++) PAI_NAMES.push('z' + i);

    function pai_to_idx(p) {
        const base = p.replace(/[_*+=\-]/g, '');
        return (PAI_INDEX[base] !== undefined) ? PAI_INDEX[base] : -1;
    }

    /* ---- 特徴量エンコーディング (extract_features.js と完全一致) ---- */

    function discard_features(discards) {
        const vec = new Array(44).fill(0);
        let idx = 0;

        for (const s of ['m', 'p', 's']) {
            let low = 0, mid = 0, high = 0;
            for (const p of discards) {
                const base = p.replace(/[_*+=\-]/g, '');
                if (!base.startsWith(s)) continue;
                const n = parseInt(base[1]) || 5;
                if (n <= 3) low++;
                else if (n <= 6) mid++;
                else high++;
            }
            vec[idx++] = low; vec[idx++] = mid; vec[idx++] = high;
        }

        const first_z = new Array(7).fill(0);
        for (let turn = 0; turn < discards.length; turn++) {
            const base = discards[turn].replace(/[_*+=\-]/g, '');
            if (base.startsWith('z')) {
                const zi = parseInt(base[1]) - 1;
                if (zi >= 0 && zi < 7 && first_z[zi] === 0) first_z[zi] = turn + 1;
            }
        }
        for (const v of first_z) vec[idx++] = v;

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

        vec[idx++] = discards.length;
        return vec;  // 44次元
    }

    function meld_features(melds) {
        const vec = new Array(38).fill(0);
        let n_chi = 0, n_pon = 0, n_kan = 0;

        for (const m of melds) {
            if (!m) continue;
            const clean = m.replace(/[+=\-]/g, '');
            const s = clean[0];
            for (let i = 1; i < clean.length; i++) {
                const n = parseInt(clean[i]);
                if (isNaN(n)) continue;
                const pi = pai_to_idx(s + (n === 0 ? 5 : n));
                if (pi >= 0 && pi < N_PAI) vec[pi] = 1;
            }
            if      (m.match(/^[mpsz]\d{3}[\+\=\-]$/))  n_chi++;
            else if (m.match(/^[mpsz]\d{3}[\+\=\-]\d$/)) n_pon++;
            else if (m.match(/^[mpsz]\d{4}/))             n_kan++;
        }

        vec[34] = n_chi; vec[35] = n_pon; vec[36] = n_kan; vec[37] = 0;
        return vec;  // 38次元
    }

    function score_features(state) {
        const { scores, player_ids, l, zhuangfeng, jushu, changbang } = state;
        const my_score = scores[player_ids[l]];
        const vec = new Array(11).fill(0);
        let idx = 0;

        for (let id = 0; id < 4; id++) vec[idx++] = (scores[id] - 25000) / 10000;

        const sorted = [...scores].sort((a, b) => b - a);
        for (const border of sorted) vec[idx++] = (my_score - border) / 10000;

        vec[idx++] = zhuangfeng;
        vec[idx++] = jushu / 4;
        vec[idx++] = Math.min(changbang, 8) / 8;
        return vec;  // 11次元
    }

    function game_state_features(state) {
        const { remaining, riichi_l, discards_l } = state;
        const vec = new Array(9).fill(0);
        vec[0] = remaining / 70;
        for (let l = 0; l < 4; l++) {
            vec[1 + l] = riichi_l[l] ? 1 : 0;
            vec[5 + l] = riichi_l[l] ? discards_l[l].length / 18 : 0;
        }
        return vec;  // 9次元
    }

    function encode_hand(hand_str) {
        const vec = new Array(N_PAI).fill(0);
        if (!hand_str) return vec;
        const base = hand_str.split(',')[0];
        let s = '';
        for (const c of base) {
            if ('mpsz'.indexOf(c) >= 0) { s = c; continue; }
            const n = parseInt(c);
            if (isNaN(n)) continue;
            const pi = pai_to_idx(s + (n === 0 ? 5 : n));
            if (pi >= 0) vec[pi]++;
        }
        return vec;  // 34次元
    }

    /* ---- 局面状態の抽出 ---- */

    function extract_state(paipu, log_idx, current_idx, board_model, menfeng) {
        const round_log   = paipu.log[log_idx];
        const discards_l  = [[], [], [], []];
        const riichi_l    = [false, false, false, false];

        // current_idx (ツモイベント) までの打牌を収集
        for (let i = 0; i <= current_idx; i++) {
            const ev = round_log[i];
            if (ev && ev.dapai) {
                discards_l[ev.dapai.l].push(ev.dapai.p);
                if (ev.dapai.p.endsWith('*')) riichi_l[ev.dapai.l] = true;
            }
        }

        const hands_l = (board_model.shoupai || []).map(s => s ? s.toString() : '');
        const melds_l = hands_l.map(h => {
            const parts = h.split(',');
            return parts.length > 1 ? parts.slice(1) : [];
        });

        return {
            l:          menfeng,
            player_ids: board_model.player_id ? [...board_model.player_id] : [0, 1, 2, 3],
            discards_l,
            riichi_l,
            melds_l,
            hands_l,
            scores:     [0, 1, 2, 3].map(id => board_model.defen ? (board_model.defen[id] || 0) : 0),
            remaining:  board_model.shan ? board_model.shan.paishu : 0,
            zhuangfeng: board_model.zhuangfeng || 0,
            jushu:      board_model.jushu      || 0,
            changbang:  board_model.changbang  || 0,
        };
    }

    /* ---- 追加特徴量 ---- */

    function make_tile_identity(tile_idx) {
        // suit_onehot(4) + num_onehot(9) + is_jihai(1) = 14次元
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

    function get_neighbor_visible(tile_idx, vis_counts) {
        // 隣接±1, ±2の visible_count (4次元; 字牌・範囲外は0)
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

    function meld_type_single(melds) {
        // 副露タイプ (4次元): チー/ポン/カン/有無
        let n_chi = 0, n_pon = 0, n_kan = 0;
        for (const m of melds) {
            if (!m) continue;
            if      (m.match(/^[mpsz]\d{3}[\+\=\-]$/))   n_chi++;
            else if (m.match(/^[mpsz]\d{3}[\+\=\-]\d$/))  n_pon++;
            else if (m.match(/^[mpsz]\d{4}/))              n_kan++;
        }
        return [n_chi, n_pon, n_kan, (n_chi + n_pon + n_kan > 0) ? 1 : 0];
    }

    function visible_counts_vec(state) {
        // 全プレイヤーの捨て牌・副露 + viewer自身の手牌から見え牌枚数を計算（34次元、/4正規化）
        const counts = new Array(N_PAI).fill(0);
        for (let l = 0; l < 4; l++) {
            for (const p of state.discards_l[l]) {
                const pi = pai_to_idx(p);
                if (pi >= 0) counts[pi]++;
            }
            for (const m of state.melds_l[l]) {
                if (!m) continue;
                const clean = m.replace(/[+=\-]/g, '');
                const s = clean[0];
                for (let i = 1; i < clean.length; i++) {
                    const n = parseInt(clean[i]);
                    if (isNaN(n)) continue;
                    const pi = pai_to_idx(s + (n === 0 ? 5 : n));
                    if (pi >= 0) counts[pi]++;
                }
            }
        }
        const hand = encode_hand(state.hands_l[state.l]);
        for (let i = 0; i < N_PAI; i++) counts[i] += hand[i];
        return counts.map(c => c / 4);
    }

    function meld_type_features_3players(state) {
        // viewer以外の3プレイヤーの副露タイプ: (n_chi, n_pon, n_kan, has_meld) × 3 = 12次元
        const vec = [];
        for (let rel = 1; rel <= 3; rel++) {
            const l = (state.l + rel) % 4;
            let n_chi = 0, n_pon = 0, n_kan = 0;
            for (const m of state.melds_l[l]) {
                if (!m) continue;
                if      (m.match(/^[mpsz]\d{3}[\+\=\-]$/))  n_chi++;
                else if (m.match(/^[mpsz]\d{3}[\+\=\-]\d$/)) n_pon++;
                else if (m.match(/^[mpsz]\d{4}/))             n_kan++;
            }
            vec.push(n_chi, n_pon, n_kan, (n_chi + n_pon + n_kan > 0) ? 1 : 0);
        }
        return vec;
    }

    /* ---- 各モデル用特徴量ベクトル生成 ---- */

    function make_bc_features(state) {
        // 行動クローン: 268次元 = hand(34) + discard(44) + meld(38) + score(11) + game(9) + others_discard(44×3)
        return new Float32Array([
            ...encode_hand(state.hands_l[state.l]),
            ...discard_features(state.discards_l[state.l]),
            ...meld_features(state.melds_l[state.l]),
            ...score_features(state),
            ...game_state_features(state),
            ...discard_features(state.discards_l[(state.l + 1) % 4]),
            ...discard_features(state.discards_l[(state.l + 2) % 4]),
            ...discard_features(state.discards_l[(state.l + 3) % 4]),
        ]);
    }

    function make_vf_features(state) {
        // 価値関数: 67次元 = hand(34) + score(11) + game(9) + remaining(1) + others_meld_type(12)
        return new Float32Array([
            ...encode_hand(state.hands_l[state.l]),
            ...score_features(state),
            ...game_state_features(state),
            state.remaining / 70,
            ...meld_type_features_3players(state),
        ]);
    }

    function make_hi_features(state, target_l) {
        // v2: 219次元 = target_discard(44)+target_meld(38)+riichi(1)+score(11)+game(9)+self_discard(44)+self_meld(38)+visible_counts(34)
        return new Float32Array([
            ...discard_features(state.discards_l[target_l]),
            ...meld_features(state.melds_l[target_l]),
            state.riichi_l[target_l] ? 1 : 0,
            ...score_features(state),
            ...game_state_features(state),
            ...discard_features(state.discards_l[state.l]),
            ...meld_features(state.melds_l[state.l]),
            ...visible_counts_vec(state),
        ]);
    }

    /* ---- ユーティリティ ---- */

    function softmax(arr) {
        const max = Math.max.apply(null, arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum  = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    async function run_session(session, feats) {
        const tensor = new ort.Tensor('float32', feats, [1, feats.length]);
        return session.run({ features: tensor });
    }

    /* ---- Phase2 分析 ---- */

    const SCORE_SCALE = 10000.0;
    const SEAT_NAMES  = ['下家', '対面', '上家'];

    async function run_phase2(paipu, log_idx, current_idx, board_model, menfeng, analysis_type, sessions) {
        const state  = extract_state(paipu, log_idx, current_idx, board_model, menfeng);
        const result = {};

        // 行動クローン: 打牌決定ターンのみ
        if (sessions.behavior_clone && analysis_type === 'dapai') {
            try {
                const out    = await run_session(sessions.behavior_clone, make_bc_features(state));
                const logits = Array.from(out['logits'].data);
                // 手牌にない牌のlogitsを-Infinityにしてsoftmaxから除外
                const hand_vec = encode_hand(state.hands_l[state.l]);
                const masked   = logits.map((v, i) => hand_vec[i] > 0 ? v : -Infinity);
                const probs    = softmax(masked);
                result.behavior_clone = {
                    top_actions: probs
                        .map((p, i) => ({ tile: PAI_NAMES[i], prob: p }))
                        .sort((a, b) => b.prob - a.prob)
                        .slice(0, 5),
                };
            } catch(e) { console.warn('AI Phase2: behavior_clone inference error', e); }
        }

        // 価値関数: 常に計算
        if (sessions.value_function) {
            try {
                const out = await run_session(sessions.value_function, make_vf_features(state));
                result.value_function = {
                    round_score: out['pred_round'].data[0] * SCORE_SCALE,
                    final_score: out['pred_final'].data[0] * SCORE_SCALE,
                };
            } catch(e) { console.warn('AI Phase2: value_function inference error', e); }
        }

        // 手牌類推: 常に計算
        if (sessions.hand_inference) {
            try {
                const players = [];
                for (let rel = 1; rel <= 3; rel++) {
                    const target_l = (menfeng + rel) % 4;
                    const out      = await run_session(sessions.hand_inference, make_hi_features(state, target_l));
                    // out['logits'].data は Float32Array, shape [1, 34, 5] → flat length 170
                    const flat     = out['logits'].data;
                    const probs_per_tile = [];
                    for (let tile = 0; tile < N_PAI; tile++) {
                        probs_per_tile.push(softmax([flat[tile*5], flat[tile*5+1], flat[tile*5+2], flat[tile*5+3], flat[tile*5+4]]));
                    }
                    players.push({
                        l: target_l, rel, seat_name: SEAT_NAMES[rel - 1],
                        probs_per_tile,
                        // 赤牌は現モデル未対応 (m0/p0/s0 は m5/p5/s5 に統合)。将来モデル対応時にここへ確率を格納する。
                        aka: { m0: null, p0: null, s0: null },
                    });
                }
                result.hand_inference = { players };
            } catch(e) { console.warn('AI Phase2: hand_inference error', e); }
        }

        return result;
    }

    /* ---- モデルロード ---- */

    const MODEL_BASE = 'models/';

    async function load_sessions() {
        const s = {};
        const models = [
            ['hand_inference', MODEL_BASE + 'hand_inference/v4/model.onnx'],
            ['behavior_clone', MODEL_BASE + 'behavior_clone/v2/model.onnx'],
            ['value_function', MODEL_BASE + 'value_function/v2/model.onnx'],
        ];
        for (const [name, path] of models) {
            try {
                s[name] = await ort.InferenceSession.create(path);
                console.log('AI Phase2: loaded', name);
            } catch(e) {
                console.warn('AI Phase2: failed to load', name, '-', e.message);
            }
        }
        return s;
    }

    /* ---- 初期化 ---- */

    function init() {
        if (typeof ort === 'undefined') {
            console.warn('AI Phase2: onnxruntime-web not found');
            return;
        }

        // WASM ファイルを CDN から取得（ローカルでの WASM 配信不要）
        try {
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
        } catch(e) {}

        const sessions_promise = load_sessions();

        window.AI_PHASE2 = {
            analyze: function(paipu, log_idx, current_idx, board_model, menfeng, analysis_type) {
                return sessions_promise.then(function(sessions) {
                    return run_phase2(paipu, log_idx, current_idx, board_model, menfeng, analysis_type, sessions);
                });
            },
        };

        console.log('AI Phase2: initialized');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
