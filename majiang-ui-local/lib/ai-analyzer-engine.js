/*
 *  AI Analyzer Engine
 *  DOM に依存しない純粋な分析ロジック。テスト可能。
 */
'use strict';

const AI     = require('@kobalab/majiang-ai');
const Majiang = require('@kobalab/majiang-core');

/**
 * 指定した局面まで AI プレイヤーをリプレイし、分析結果を返す。
 *
 * @param {object} paipu      - 牌譜オブジェクト (title, player, qijia, log)
 * @param {number} log_idx    - 対象の局インデックス
 * @param {number} current_idx - 分析対象の手番インデックス (0-base, inclusive)
 * @param {number} viewpoint  - 分析視点のプレイヤー ID
 * @returns {object} 分析結果
 *   - dapai_info  : 打牌候補配列 [{p, n_xiangting, ev, tingpai, n_tingpai, weixian, selected?}]
 *   - fulou_info  : 副露候補配列 [{m, n_xiangting, ev}]
 *   - analysis_type : 'dapai' | 'fulou' | null
 *   - best_dapai  : AI が選んだ最良打牌 (文字列) または null
 *   - paishu      : 残り牌数 {m:[...], p:[...], s:[...], z:[...]}
 *   - paijia_fn   : (p: string) => number  手牌文脈を考慮した牌価値関数
 *   - menfeng     : 視点プレイヤーの門風
 */
function analyze(paipu, log_idx, current_idx, viewpoint) {
    const log = paipu.log[log_idx];

    const player = new AI();
    player.action({ kaiju: {
        id:     viewpoint,
        rule:   Majiang.rule(),
        title:  paipu.title,
        player: paipu.player,
        qijia:  paipu.qijia,
    }}, ()=>{});

    for (let i = 0; i <= current_idx; i++) {
        player.action(log[i], ()=>{});
    }

    const current = log[current_idx];
    const mf = player._menfeng;
    const dapai_info = [];
    const fulou_info = [];
    let analysis_type = null;
    let best_dapai = null;

    if (current.zimo && current.zimo.l === mf) {
        analysis_type = 'dapai';
        player.select_gang(dapai_info);
        best_dapai = player.select_dapai(dapai_info);
        dapai_info.forEach(i => {
            if (!i.m && i.p === best_dapai.slice(0, 2)) i.selected = true;
        });
    } else if (current.gangzimo && current.gangzimo.l === mf) {
        analysis_type = 'dapai';
        best_dapai = player.select_dapai(dapai_info);
        dapai_info.forEach(i => {
            if (i.p === best_dapai.slice(0, 2)) i.selected = true;
        });
    } else if (current.fulou && current.fulou.l === mf
               && !current.fulou.m.match(/^[mpsz]\d{4}/)) {
        analysis_type = 'dapai';
        best_dapai = player.select_dapai(dapai_info);
        dapai_info.forEach(i => {
            if (i.p === best_dapai.slice(0, 2)) i.selected = true;
        });
    } else if (current.dapai && current.dapai.l !== mf) {
        analysis_type = 'fulou';
        player.select_fulou(current.dapai, fulou_info);
    }

    const suanpai = player._suanpai;

    return {
        dapai_info,
        fulou_info,
        analysis_type,
        best_dapai,
        paishu:    suanpai._paishu,
        paijia_fn: player.shoupai ? suanpai.make_paijia(player.shoupai) : null,
        menfeng:   mf,
    };
}

module.exports = { analyze };
