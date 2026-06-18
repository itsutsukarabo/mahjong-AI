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

    function score_features(state, player_l) {
        const { scores, player_ids, zhuangfeng, jushu, changbang } = state;
        const l        = (player_l !== undefined) ? player_l : state.l;
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

    function wind_features(state, player_l) {
        // 自風 one-hot [東, 南, 西, 北] (4次元) + 場風(東=0/南=1) (1次元) = 5次元
        const jikaze = (player_l - state.jushu + 4) % 4;
        const oh = [0, 0, 0, 0];
        oh[jikaze] = 1;
        return [...oh, state.zhuangfeng];
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
        // 副露の請求牌（他家捨て牌から取った牌）は捨て牌でカウント済みのためスキップする
        const counts = new Array(N_PAI).fill(0);
        for (let l = 0; l < 4; l++) {
            for (const p of state.discards_l[l]) {
                const pi = pai_to_idx(p);
                if (pi >= 0) counts[pi]++;
            }
            for (const m of state.melds_l[l]) {
                if (!m) continue;
                const s = m[0];
                const dirIdx = m.search(/[+=\-]/);  // 暗槓は -1
                for (let i = 1; i < m.length; i++) {
                    if (/[+=\-]/.test(m[i])) continue;
                    if (dirIdx >= 0 && i === dirIdx - 1) continue;  // 請求牌スキップ
                    const n = parseInt(m[i]);
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

    function red_discard_signal(discards, riichi_flag) {
        // 立直なし + 赤5切り = その5を持っていない可能性大（3次元: 5m/5p/5s）
        if (riichi_flag) return [0, 0, 0];
        return ['m0', 'p0', 's0'].map(r =>
            discards.some(p => p.replace(/[_*+=\-]/g, '') === r) ? 1 : 0
        );
    }

    function red_visible_flags(state) {
        // 全プレイヤーの捨て牌・副露に赤牌（m0/p0/s0）が公開されているかをフラグ化（3次元）
        const flags = [0, 0, 0];  // [m0, p0, s0]
        const suits = ['m', 'p', 's'];
        for (let l = 0; l < 4; l++) {
            for (const p of state.discards_l[l]) {
                const base = p.replace(/[_*+=\-]/g, '');
                for (let i = 0; i < 3; i++) {
                    if (base === suits[i] + '0') flags[i] = 1;
                }
            }
            for (const m of (state.melds_l[l] || [])) {
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

    function pass_pon_signal_from_state(state, target_l, max_discards = 70) {
        // ブラウザ側はターゲットの手牌不明なため全スルーをカウント（訓練側より粗い近似）
        const signal = new Array(N_PAI).fill(0);
        let total_d = 0;
        for (let l = 0; l < 4; l++) {
            if (l === target_l) continue;
            for (const tile of state.discards_l[l]) {
                const tile_norm = tile.replace(/[_*+=\-]/g, '').replace(/^([mps])0$/, '$15');
                const pi = (PAI_INDEX[tile_norm] !== undefined) ? PAI_INDEX[tile_norm] : -1;
                if (pi >= 0) {
                    // ターゲットが後でポンした牌はスルーカウントしない
                    const already_ponned = state.melds_l[target_l].some(m => {
                        if (!m || !m.match(/^[mpsz]\d{3}[\+\=\-]\d$/)) return false;
                        const s = m[0], n = parseInt(m[1]);
                        return `${s}${n === 0 ? 5 : n}` === tile_norm;
                    });
                    if (!already_ponned) signal[pi] += total_d / max_discards;
                }
                total_d++;
            }
        }
        return signal;
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

    function make_yaku_features(state, target_l) {
        // 役推定: 108次元 = target_discard(44)+target_meld(38)+riichi(1)+game_state(9)+score(11)+wind(5)
        return new Float32Array([
            ...discard_features(state.discards_l[target_l]),
            ...meld_features(state.melds_l[target_l]),
            state.riichi_l[target_l] ? 1 : 0,
            ...game_state_features(state),
            ...score_features(state, target_l),
            ...wind_features(state, target_l),
        ]);
    }

    function chi_called_tile_signal_from_melds(melds) {
        // チー副露の受け取り牌を 34次元バイナリで返す
        // direction marker (+/=/-) の直前の数字が鳴いた牌
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

    function make_hi_features_v29(state, target_l, yaku_probs, tenpai_prob) {
        // v26/v29: 442次元 = 420次元ベース + yaku(21) + tenpai(1)
        // 420次元 = target_discard(44)+target_meld(38)+riichi(1)+score(11)+game(9)+
        //   self_discard(44)+self_meld(38)+visible_counts(34)+red_discard_signal(3)+
        //   red_visible(3)+other1_discard(44)+other2_discard(44)+pass_pon_signal(34)+wind(5)+
        //   pass_chi_signal(34)+chi_called_tile_signal(34)
        const other_ls = [1, 2, 3]
            .map(rel => (state.l + rel) % 4)
            .filter(l => l !== target_l);
        return new Float32Array([
            ...discard_features(state.discards_l[target_l]),                              // 44
            ...meld_features(state.melds_l[target_l]),                                    // 38
            state.riichi_l[target_l] ? 1 : 0,                                            // 1
            ...score_features(state),                                                     // 11
            ...game_state_features(state),                                                // 9
            ...discard_features(state.discards_l[state.l]),                               // 44
            ...meld_features(state.melds_l[state.l]),                                     // 38
            ...visible_counts_vec(state),                                                  // 34
            ...red_discard_signal(state.discards_l[target_l], state.riichi_l[target_l]), // 3
            ...red_visible_flags(state),                                                   // 3
            ...discard_features(state.discards_l[other_ls[0]]),                           // 44
            ...discard_features(state.discards_l[other_ls[1]]),                           // 44
            ...pass_pon_signal_from_state(state, target_l),                               // 34
            ...wind_features(state, target_l),                                            // 5
            ...new Array(34).fill(0),                                                     // 34 pass_chi (browser不可のためゼロ)
            ...chi_called_tile_signal_from_melds(state.melds_l[target_l]),                // 34
            ...(yaku_probs || new Array(21).fill(0)),                                     // 21
            tenpai_prob != null ? tenpai_prob : 0,                                        // 1
        ]);
    }

    /* ---- ユーティリティ ---- */

    function softmax(arr) {
        const max = Math.max.apply(null, arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum  = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    // 制約付きソフトマックス: E[Σc_i] = k となる λ を二分探索で求め、
    // 手牌枚数ルール (k = 13 - 3*n_melds) を出力層で構造的に保証する
    function constrained_softmax_js(flat_logits, k) {
        // flat_logits: Float32Array [170] (34×5 row-major, masked logits from ONNX)
        // k: integer total tile count
        // Returns: array[34] of array[5] probabilities with E[Σc_i] = k
        function expected_sum(lam) {
            let total = 0;
            for (let t = 0; t < 34; t++) {
                const base = t * 5;
                let max_adj = -Infinity;
                for (let c = 0; c < 5; c++) {
                    const v = flat_logits[base + c] - lam * c;
                    if (v > max_adj) max_adj = v;
                }
                let sum_exp = 0, weighted = 0;
                for (let c = 0; c < 5; c++) {
                    const e = Math.exp(flat_logits[base + c] - lam * c - max_adj);
                    sum_exp += e;
                    weighted += e * c;
                }
                total += weighted / sum_exp;
            }
            return total;
        }
        let lo = -20, hi = 20;
        for (let i = 0; i < 50; i++) {
            const mid = (lo + hi) / 2;
            if (expected_sum(mid) > k) lo = mid; else hi = mid;
        }
        const lam = (lo + hi) / 2;
        const result = [];
        for (let t = 0; t < 34; t++) {
            const base = t * 5;
            let max_adj = -Infinity;
            for (let c = 0; c < 5; c++) {
                const v = flat_logits[base + c] - lam * c;
                if (v > max_adj) max_adj = v;
            }
            let sum_exp = 0;
            const exps = [];
            for (let c = 0; c < 5; c++) {
                const e = Math.exp(flat_logits[base + c] - lam * c - max_adj);
                exps.push(e);
                sum_exp += e;
            }
            result.push(exps.map(e => e / sum_exp));
        }
        return result;
    }

    async function run_session(session, feats) {
        const tensor = new ort.Tensor('float32', feats, [1, feats.length]);
        return session.run({ features: tensor });
    }

    /* ---- ブロック期待値計算 (TICKET-027) ---- */

    function compute_block_ev(probs) {
        // P(tile の count ≥ k) のヘルパー
        const p_ge = (p, k) => { let s = 0; for (let i = k; i < 5; i++) s += p[i]; return s; };

        // 刻子分布: [P(0刻子), P(1刻子)]
        // 同種牌の刻子は最大1個 (count≥3 で成立)
        const triplet_dist = probs.map(p => [
            p[0] + p[1] + p[2],
            p[3] + p[4],
        ]);

        // 対子分布: [P(0対子), P(1対子), P(2対子)]
        // 4枚持ちのみ2対子が成立
        const pair_dist = probs.map(p => [
            p[0] + p[1],
            p[2] + p[3],
            p[4],
        ]);

        // 順子分布: [P(0順子), P(1順子), P(2順子), P(3順子以上)]
        // 独立近似: P(≥k順子) ≈ P(tile_a≥k) × P(tile_b≥k) × P(tile_c≥k)
        // インデックス: suit*7+n (m123=0, m234=1, ..., s789=20)
        const seq_dist = [];
        for (let suit = 0; suit < 3; suit++) {
            const base = suit * 9;
            for (let n = 0; n < 7; n++) {
                const a = probs[base + n    ];
                const b = probs[base + n + 1];
                const c = probs[base + n + 2];
                const ge1 = p_ge(a,1) * p_ge(b,1) * p_ge(c,1);
                const ge2 = p_ge(a,2) * p_ge(b,2) * p_ge(c,2);
                const ge3 = p_ge(a,3) * p_ge(b,3) * p_ge(c,3);
                seq_dist.push([
                    Math.max(0, 1 - ge1),
                    Math.max(0, ge1 - ge2),
                    Math.max(0, ge2 - ge3),
                    Math.max(0, ge3),
                ]);
            }
        }
        return { triplet_dist, pair_dist, seq_dist };
    }

    function get_best_hand_str(probs, melds) {
        // per-tile 確率の argmax から最尤手牌の Shoupai 形式文字列を構築する
        const suits = ['m', 'p', 's', 'z'];
        const bases  = [0, 9, 18, 27];
        const sizes  = [9, 9, 9, 7];
        let hand = '';
        for (let s = 0; s < 4; s++) {
            let part = '', any = false;
            for (let n = 0; n < sizes[s]; n++) {
                const p   = probs[bases[s] + n];
                const cnt = p.indexOf(Math.max(...p));   // argmax = 最尤枚数
                if (cnt > 0) {
                    part += String(n + 1).repeat(cnt);
                    any = true;
                }
            }
            if (any) hand += suits[s] + part;
        }
        const meld_str = (melds || []).filter(Boolean).join(',');
        return meld_str ? hand + ',' + meld_str : hand;
    }

    function make_block_display_data(probs, melds) {
        // ブロック期待値・最尤手牌・シャンテン数・面子分解をまとめて返す
        // シャンテン数・面子分解は Majiang グローバルが利用できる場合のみ計算する
        const ev       = compute_block_ev(probs);
        const hand_str = get_best_hand_str(probs, melds);
        let shanten = null, tingpai = null, decomps = null;

        if (typeof Majiang !== 'undefined') {
            try {
                const sp = Majiang.Shoupai.fromString(hand_str);
                shanten  = Majiang.Util.xiangting(sp);
                if (shanten <= 0) {
                    tingpai = Majiang.Util.tingpai(sp);
                    if (tingpai.length > 0) {
                        const seen = new Set();
                        decomps = [];
                        for (const p of tingpai) {
                            for (const d of Majiang.Util.hule_mianzi(sp, p + '+')) {
                                const key = d.join('\t');
                                if (!seen.has(key)) { seen.add(key); decomps.push(d); }
                            }
                        }
                    }
                }
            } catch (_) {}
        }
        return { ...ev, hand_str, shanten, tingpai, decomps };
    }

    /* ---- v29: block_logits 134-dim → 表示データ ---- */

    function compute_likely_waits_from_tatsu(ryanmen_probs, kanchan_probs) {
        // ryanmen_probs: Array(24) の sigmoid 確率値 (block_logits の生値ではなく 0-1 の確率)
        // kanchan_probs: Array(21)
        // 返す: [{tatsu, wait_tiles, prob}] を prob 降順でソート
        const SUITS = ['m', 'p', 's'];
        const waits = [];

        // 両面/辺張 (8種 × 3スーツ = 24種)
        // n=0:12両 n=1:23面 ... n=6:78面 n=7:89辺
        for (let s = 0; s < 3; s++) {
            for (let n = 0; n < 8; n++) {
                const p    = ryanmen_probs[s * 8 + n];
                const suit = SUITS[s];
                const lo   = n + 1, hi = n + 2;  // タツを構成する牌番号

                let wait_tiles;
                if (lo === 1) {
                    wait_tiles = [suit + 3];                          // 辺張 12 → 3
                } else if (hi === 9) {
                    wait_tiles = [suit + 7];                          // 辺張 89 → 7
                } else {
                    wait_tiles = [suit + (lo - 1), suit + (hi + 1)]; // 両面
                }
                waits.push({ tatsu: suit + lo + hi, wait_tiles, prob: p });
            }
        }

        // 嵌張 (7種 × 3スーツ = 21種)
        for (let s = 0; s < 3; s++) {
            for (let n = 0; n < 7; n++) {
                const p    = kanchan_probs[s * 7 + n];
                const suit = SUITS[s];
                const lo   = n + 1, hi = n + 3;
                const mid  = n + 2;  // 嵌張の待ち牌
                waits.push({ tatsu: suit + lo + '_' + hi, wait_tiles: [suit + mid], prob: p });
            }
        }

        waits.sort((a, b) => b.prob - a.prob);
        return waits;
    }

    function make_block_display_data_v29(block_logits_134, probs_per_tile, melds, tenpai_prob) {
        // block_logits_134: Float32Array または Array (134要素)
        // ブロックインデックス:
        //   [0..20]    順子 21種: M:0-6, P:7-13, S:14-20
        //   [21..54]   刻子 34種: 牌インデックス順
        //   [55..88]   対子 34種: 牌インデックス順
        //   [89..112]  両面/辺張 24種: M:89-96, P:97-104, S:105-112
        //   [113..133] 嵌張 21種: M:113-119, P:120-126, S:127-133
        const sigmoid = x => 1 / (1 + Math.exp(-x));
        const prob = Array.from(block_logits_134).map(sigmoid);
        const to_dist = p => [1 - p, p];

        const seq_dist      = prob.slice(0,   21).map(to_dist);   // 21種
        const triplet_dist  = prob.slice(21,  55).map(to_dist);   // 34種
        const pair_dist     = prob.slice(55,  89).map(to_dist);   // 34種
        const ryanmen_dist  = prob.slice(89,  113).map(to_dist);  // 24種
        const kanchan_dist  = prob.slice(113, 134).map(to_dist);  // 21種

        // ターツ直接推定: v29のblock_logitsから待ち候補を算出
        const likely_waits = compute_likely_waits_from_tatsu(
            prob.slice(89, 113), prob.slice(113, 134)
        );

        const hand_str = get_best_hand_str(probs_per_tile, melds);
        let shanten = null, tingpai = null, decomps = null;

        if (typeof Majiang !== 'undefined') {
            try {
                const sp = Majiang.Shoupai.fromString(hand_str);
                shanten  = Majiang.Util.xiangting(sp);
                if (shanten <= 0) {
                    tingpai = Majiang.Util.tingpai(sp);
                    if (tingpai && tingpai.length > 0) {
                        const seen = new Set();
                        decomps = [];
                        for (const p of tingpai) {
                            for (const d of Majiang.Util.hule_mianzi(sp, p + '+')) {
                                const key = d.join('\t');
                                if (!seen.has(key)) { seen.add(key); decomps.push(d); }
                            }
                        }
                    }
                }
            } catch (_) {}
        }

        return {
            triplet_dist, seq_dist, pair_dist, ryanmen_dist, kanchan_dist,
            hand_str, shanten, tingpai, decomps,
            tenpai_prob: tenpai_prob != null ? tenpai_prob : undefined,
            likely_waits,
        };
    }

    /* ---- v11: block head → per-tile ブレンド ---- */

    function seq_contribution(tile_idx, seq_probs) {
        const suit = Math.floor(tile_idx / 9);
        if (suit >= 3) return 0;
        const pos      = tile_idx % 9;
        const seq_base = suit * 7;
        let total = 0;
        for (let s = Math.max(0, pos - 2); s <= Math.min(6, pos); s++) {
            total += seq_probs[seq_base + s];
        }
        return Math.min(1, total);
    }

    function block_to_per_tile_dist(block_tenpai) {
        return Array.from({length: 34}, (_, i) => {
            const p_tri  = block_tenpai.triplet[i];
            const p_seq  = seq_contribution(i, block_tenpai.seq) * (1 - p_tri);
            const p_pair = block_tenpai.pair[i] * (1 - p_tri) * (1 - p_seq);
            const p_none = Math.max(0, 1 - p_tri - p_seq - p_pair);
            return [p_none, p_seq, p_pair, p_tri, 0];
        });
    }

    function blend_per_tile(model_probs, block_tenpai, tenpai_p) {
        const block_dists = block_to_per_tile_dist(block_tenpai);
        return model_probs.map((probs, i) =>
            probs.map((p, k) => (1 - tenpai_p) * p + tenpai_p * block_dists[i][k])
        );
    }

    function get_best_tenpai_hand(probs_blended, melds) {
        const hand_str = get_best_hand_str(probs_blended, melds);
        if (typeof Majiang === 'undefined') return { hand_str, shanten: null, decomps: null, tingpai: null };
        try {
            const sp      = Majiang.Shoupai.fromString(hand_str);
            const shanten = Majiang.Util.xiangting(sp);
            const tingpai = shanten <= 0 ? Majiang.Util.tingpai(sp) : null;
            if (!tingpai || tingpai.length === 0) return { hand_str, shanten, decomps: null, tingpai: null };
            const seen = new Set();
            const decomps = [];
            for (const p of tingpai) {
                for (const d of Majiang.Util.hule_mianzi(sp, p + '+')) {
                    const key = d.join('\t');
                    if (!seen.has(key)) { seen.add(key); decomps.push(d); }
                }
            }
            return { hand_str, shanten, decomps, tingpai };
        } catch(_) {
            return { hand_str, shanten: null, decomps: null, tingpai: null };
        }
    }

    function make_block_display_data_v11(block_tenpai, tenpai_prob, probs_blended, melds) {
        const to_dist      = p => [1 - p, p];
        const triplet_dist = block_tenpai.triplet.map(to_dist);
        const seq_dist     = block_tenpai.seq.map(to_dist);
        const pair_dist    = block_tenpai.pair.map(to_dist);
        const best         = get_best_tenpai_hand(probs_blended, melds);
        return {
            triplet_dist, seq_dist, pair_dist,
            tenpai_prob,
            hand_str: best?.hand_str ?? null,
            shanten:  best?.shanten  ?? null,
            decomps:  best?.decomps  ?? null,
            tingpai:  best?.tingpai  ?? null,
        };
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

        // 手牌類推: 常に計算 (v29: 3プレイヤーを1バッチで推論)
        if (sessions.hand_inference) {
            try {
                const target_ls       = [1, 2, 3].map(rel => (menfeng + rel) % 4);
                const yaku_probs_list = [null, null, null];
                const tenpai_prob_list = [null, null, null];

                // Stage 1: 役推定・聴牌推定を各プレイヤーで実行
                for (let i = 0; i < 3; i++) {
                    const target_l = target_ls[i];

                    if (sessions.yaku_inference) {
                        try {
                            const yaku_out = await run_session(sessions.yaku_inference, make_yaku_features(state, target_l));
                            const yaku_logits = Array.from(yaku_out['logits'].data);
                            yaku_probs_list[i] = yaku_logits.map(x => 1 / (1 + Math.exp(-x)));
                        } catch(e) { console.warn('AI Phase2: yaku_inference error', e); }
                    }

                    if (sessions.tenpai_inference) {
                        try {
                            const t_out = await run_session(sessions.tenpai_inference, make_yaku_features(state, target_l));
                            tenpai_prob_list[i] = 1 / (1 + Math.exp(-t_out['logit'].data[0]));
                        } catch(e) { console.warn('AI Phase2: tenpai_inference error', e); }
                    }

                    // 立直者はテンパイ確定: モデル未ロード/失敗時でも 1.0 を保証
                    if (state.riichi_l[target_l] && tenpai_prob_list[i] === null) {
                        tenpai_prob_list[i] = 1.0;
                    }
                }

                // Stage 2: 手牌推定 — (1, 3, 442) バッチ推論
                const flat_3d = new Float32Array(3 * 442);
                for (let i = 0; i < 3; i++) {
                    const feats = make_hi_features_v29(state, target_ls[i], yaku_probs_list[i], tenpai_prob_list[i]);
                    flat_3d.set(feats, i * 442);
                }
                const tensor = new ort.Tensor('float32', flat_3d, [1, 3, 442]);
                const out = await sessions.hand_inference.run({ features: tensor });

                // 出力テンソルをプレイヤーごとに分割
                // logits:      (1, 3, 34, 5) → 各プレイヤー 170要素
                // red_logits:  (1, 3, 3, 2)  → 各プレイヤー 6要素
                // block_logits:(1, 3, 134)   → 各プレイヤー 134要素
                const logits_data = out['logits'].data;
                const red_data    = out['red_logits'] ? out['red_logits'].data : null;
                const block_data  = out['block_logits'] ? out['block_logits'].data : null;

                const players = [];
                for (let i = 0; i < 3; i++) {
                    const target_l  = target_ls[i];
                    const n_melds   = (state.melds_l[target_l] || []).filter(Boolean).length;
                    const k_tiles   = 13 - 3 * n_melds;

                    const tile_logits    = logits_data.slice(i * 170, (i + 1) * 170);
                    const probs_per_tile = constrained_softmax_js(tile_logits, k_tiles);

                    let aka = { m0: null, p0: null, s0: null };
                    if (red_data) {
                        const red_f = red_data.slice(i * 6, (i + 1) * 6);
                        aka = {
                            m0: softmax([red_f[0], red_f[1]])[1],
                            p0: softmax([red_f[2], red_f[3]])[1],
                            s0: softmax([red_f[4], red_f[5]])[1],
                        };
                    }

                    let block_ev;
                    if (block_data) {
                        const block_slice = block_data.slice(i * 134, (i + 1) * 134);
                        block_ev = make_block_display_data_v29(block_slice, probs_per_tile, state.melds_l[target_l], tenpai_prob_list[i]);
                    } else {
                        block_ev = make_block_display_data(probs_per_tile, state.melds_l[target_l]);
                    }

                    players.push({
                        l: target_l, rel: i + 1, seat_name: SEAT_NAMES[i],
                        probs_per_tile, aka, block_ev,
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
            ['hand_inference', MODEL_BASE + 'hand_inference/v29/model.onnx'],
            ['behavior_clone', MODEL_BASE + 'behavior_clone/v2/model.onnx'],
            ['value_function', MODEL_BASE + 'value_function/v2/model.onnx'],
            ['yaku_inference',   MODEL_BASE + 'yaku_inference/v1/model.onnx'],
            ['tenpai_inference', MODEL_BASE + 'tenpai_inference/v1/model.onnx'],
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

    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    }

    if (typeof module !== 'undefined') {
        module.exports = { visible_counts_vec, red_visible_flags, red_discard_signal, pai_to_idx, encode_hand,
                           compute_block_ev, get_best_hand_str, make_block_display_data,
                           block_to_per_tile_dist, blend_per_tile, make_block_display_data_v11,
                           chi_called_tile_signal_from_melds, make_hi_features_v29, make_block_display_data_v29 };
    }

})();
