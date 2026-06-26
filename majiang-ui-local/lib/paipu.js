/*
 *  Majiang.UI.Paipu
 */
"use strict";

const Majiang = require('@kobalab/majiang-core');
const Board   = require('./board');

const { hide, show } = require('./fadein');
const logconv = require('@kobalab/tenhou-url-log');
const AI = require('@kobalab/majiang-ai');

const class_name = ['main','xiajia','duimian','shangjia'];

module.exports = class Paipu {

    constructor(root, paipu, pai, audio, pref, callback, analyzer) {
        this._root  = root;
        this._paipu = paipu;
        this._pai   = pai;
        this._model = new Majiang.Board(paipu);
        this._view  = new Board($('.board', root), pai, audio, this._model);
        this._pref  = pref;

        this._log_idx = -1;
        this._idx     =  0;

        this._deny_repeat;
        this._repeat_timer;
        this._autoplay;
        this._autoplay_timer;

        this._view.open_shoupai = true;
        this._view.open_he      = true;

        this._speed = 3;
        this._summary = false;

        this._callback      = callback;
        this._open_analyzer = analyzer;
    }

    get_pref() {
        if (localStorage.getItem(this._pref)) {
            return JSON.parse(localStorage.getItem(this._pref));
        }
        else {
            return {
                sound_on: this._view.sound_on,
                speed:    this._speed
            };
        }
    }

    set_pref() {
        let pref = {
            sound_on: this._view.sound_on,
            speed:    this._speed
        };
        localStorage.setItem(this._pref, JSON.stringify(pref));
    }

    set_handler() {
        this.clear_handler();

        const ctrl = $('.controller', this._root);

        this._root.on('mousedown', (ev)=>{
            if (ev.button) return true;
            this.next();
            return false;
        });
        this._root.on('mouseup mousemove touchend', ()=>{
            this._repeat_timer = clearInterval(this._repeat_timer);
            if (this._repeet) {
                this._repeet = false;
                if (this._analyzer) {
                    if (this._deny_repeat) {
                        this._analyzer.clear();
                    }
                    else {
                        this.seek(this._log_idx, this._idx -1);
                        this._analyzer.active(true);
                    }
                }
            }
        });
        $('.next', ctrl).on('mousedown touchstart', ()=>{
            if (this._repeat_timer) return false;
            this.next();
            this._repeat_timer = setTimeout(()=>{
                this._repeet = true;
                if (this._analyzer) this._analyzer.active(false);
                this._repeat_timer = setInterval(()=>{
                    if (! this._deny_repeat) this.next();
                }, 80);
            }, 200);
            return false;
        });
        $('.prev', ctrl).on('mousedown touchstart', ()=>{
            if (this._repeat_timer) return false;
            this.prev();
            this._repeat_timer = setTimeout(()=>{
                this._repeet = true;
                if (this._analyzer) this._analyzer.active(false);
                this._repeat_timer = setInterval(()=>this.prev(), 80);
            }, 200);
            return false;
        });
        $('.last',     ctrl).on('mousedown', ()=>this.forward());
        $('.first',    ctrl).on('mousedown', ()=>this.backward());
        $('.play',     ctrl).on('mousedown', ()=>this.autoplay());
        $('.summary',  ctrl).on('mousedown', ()=>this.summary());
        $('.analyzer',   ctrl).on('mousedown', ()=>this.analyzer());
        $('.ai-analyze', ctrl).on('mousedown', ()=>this.ai_analyze());
        $('.export',     ctrl).on('mousedown', ()=>this.tenhou_dialog());
        $('.sound',    ctrl).on('mousedown', ()=>this.sound(
                                                    ! this._view.sound_on));
        $('.minus',    ctrl).on('mousedown', ()=>this.speed(this._speed - 1));
        $('.plus',     ctrl).on('mousedown', ()=>this.speed(this._speed + 1));
        $('.exit',     ctrl).on('mousedown', ()=>this.exit());
        for (let i = 0; i < 4; i++) {
            $(`.shoupai.${class_name[i]}`, this._root)
                            .on('mousedown', '.pai', ()=>this.shoupai());
            $(`.he.${class_name[i]}`, this._root)
                            .on('mousedown', '.pai', ()=>this.he());
            $(`.player.${class_name[i]}`, this._root)
                            .addClass('selectable')
                            .on('mousedown', ()=>this.viewpoint(
                                                    this._view.viewpoint + i));
            $(`.player .${class_name[i]}`, this._root)
                            .addClass('selectable')
                            .on('mousedown', ()=>{
                                this.viewpoint(this._view.viewpoint + i);
                            });
        }
        show($('> *',       ctrl));
        hide($('.play.off', ctrl));

        $(window).on('keyup.paipu', (ev)=>{

            if (this._repeet) {
                this._repeet = false;
                if (this._analyzer) {
                    if (this._deny_repeat) {
                        this._analyzer.clear();
                    }
                    else {
                        this.seek(this._log_idx, this._idx -1);
                        this._analyzer.active(true);
                    }
                }
            }

            if      (ev.key == 'ArrowRight') this.forward();
            else if (ev.key == 'ArrowLeft')  this.backward();
            else if (ev.key == ' ') this.autoplay();
            else if (ev.key == 'v') this.viewpoint(this._view.viewpoint + 1);
            else if (ev.key == '?') this.summary();
            else if (ev.key == 'i') this.analyzer();
            else if (ev.key == 'u') this.ai_analyze();
            else if (ev.key == 't') this.tenhou_dialog();
            else if (ev.key == 'a') this.sound(! this._view.sound_on);
            else if (ev.key == '-') this.speed(this._speed - 1);
            else if (ev.key == '+') this.speed(this._speed + 1);
            else if (ev.key == 's') this.shoupai();
            else if (ev.key == 'h') this.he();
            else if (ev.key == 'q' || ev.key == 'Escape')
                                    this.exit();
            else if (ev.key == 'p') {
                if (this._log_idx < 0) this.dummy_name(this._view.viewpoint);
                this.viewpoint(this._view.viewpoint);
            }
        });

        $(window).on('keydown.paipu', (ev)=>{

            if (ev.key.match(/^Arrow/)) ev.preventDefault();

            if (ev.originalEvent.repeat) {
                this._repeet = true;
                if (this._analyzer) this._analyzer.active(false);
                if (this._deny_repeat) return;
            }

            if      (ev.key == 'Enter')     this.next();
            else if (ev.key == 'ArrowDown') this.next();
            else if (ev.key == 'ArrowUp')   this.prev();
        });

        const modal = $('.ai-modal', this._root);
        $('.ai-modal-close', modal).off('click').on('click', ()=>modal.addClass('hide'));
        $('.ai-tab-btn', modal).off('click').on('click', function() {
            const tab = $(this).data('tab');
            $('.ai-tab-btn', modal).removeClass('active');
            $(this).addClass('active');
            $('.ai-tab-pane', modal).addClass('hide');
            $(`#ai-tab-${tab}`, modal).removeClass('hide');
        });

        let pref = this.get_pref();
        this.sound(pref.sound_on);
        this.speed(pref.speed);
    }

    clear_handler() {
        if (this._autoplay) this.autoplay();
        this.set_fragment('');
        const ctrl = $('.controller', this._root);
        this._root.off('mousedown mouseup mousemove touchstart touchend');
        $('.player *').removeClass('selectable').off('mousedown');
        $('.player').removeClass('selectable').off('mousedown');
        $('.shoupai', this._root).off('mousedown', '.pai');
        $('.he', this._root).off('mousedown', '.pai');
        $('*', ctrl).off('mousedown touchstart');
        $(window).off('.paipu');
    }

    start(viewpoint = 0, log_idx = -1, idx = 0) {

        this._html_title = $('title').text();
        let title = this._paipu.title + ' - ' + this._html_title;
        $('title').text(title);
        $('meta[property="og:title"]').attr('content', title);

        $('.tenhou-dialog input[name="limited"]', this._root)
                                    .prop('disabled', false).val([0]);

        this.set_handler();

        this._view.viewpoint = (4 + viewpoint % 4) % 4;
        this.set_fragment();

        if (log_idx < 0) {
            hide($('.controller', this._root));
            this._view.kaiju();
        }
        else {
            this.seek(log_idx, idx);
        }
    }

    suspend() {
        show($('.suspend', this._root));
        this._deny_repeat = true;
    }

    exit() {
        $('title').text(this._html_title);
        $('meta[property="og:title"]').attr('content', this._html_title);

        this.clear_handler();
        this._callback();
    }

    seek(log_idx, idx) {

        this._deny_repeat = false;
        if (this._summary) this.summary();

        log_idx = log_idx < 0   ? 0
                : this._paipu.log.length - 1 < log_idx
                                ? this._paipu.log.length - 1
                : log_idx;
        idx     = idx < 0       ? 0
                : this._paipu.log[log_idx].length - 1 < idx
                                ? this._paipu.log[log_idx].length - 1
                : idx;

        this._log_idx = log_idx;
        this._idx     = 0;
        this._redo    = false;

        let data;

        while (this._idx <= idx) {

            data = this._paipu.log[this._log_idx][this._idx];

            if      (data.qipai)    this._model.qipai(data.qipai);
            else if (data.zimo)     this._model.zimo(data.zimo);
            else if (data.dapai)    this._model.dapai(data.dapai);
            else if (data.fulou)    this._model.fulou(data.fulou);
            else if (data.gang)     this._model.gang(data.gang);
            else if (data.gangzimo) this._model.zimo(data.gangzimo);
            else if (data.kaigang)  this._model.kaigang(data.kaigang);
            else if (data.kita)     this._model.kita(data.kita);
            else if (data.hule)     this._model.hule(data.hule);
            else if (data.pingju)   this._model.pingju(data.pingju);

            if (this._analyzer && ! this._repeet) {
                if (idx == this._idx) this._analyzer.action(data);
                else                  this._analyzer.next(data);
            }

            this._idx++;
        }

        this.set_fragment();

        this._view.redraw();
        hide($('.suspend', this._root));
        show($('.controller', this._root));
        if (data.hule || data.pingju) {
            this._deny_repeat = true;
            this._view.update(data);
        }
    }

    next() {

        hide($('.suspend', this._root));
        show($('.controller', this._root));

        this._autoplay_timer = clearTimeout(this._autoplay_timer);

        if (this._log_idx < 0) {
            this._log_idx = 0;
            this._idx = 0;
        }
        if (this._log_idx == this._paipu.log.length) {
            this.exit();
            return;
        }
        if (this._idx >= this._paipu.log[this._log_idx].length) {
            if (! this._deny_repeat) return this.suspend();
            this._model.last();
            this._log_idx++;
            this._idx = 0;
        }
        if (this._log_idx == this._paipu.log.length) {
            if (this._target_log_idx != null) this.exit();
            this._deny_repeat = false;
            if (this._paipu.defen.length) this._model.jieju(this._paipu);
            this._view.update();
            this.summary();
            return;
        }
        if (this._summary) {
            this.summary();
            return;
        }

        let data = this._paipu.log[this._log_idx][this._idx];

        if      (data.qipai)    this.qipai(data);
        else if (data.zimo)     this.zimo(data);
        else if (data.dapai)    this.dapai(data);
        else if (data.fulou)    this.fulou(data);
        else if (data.gang)     this.gang(data);
        else if (data.gangzimo) this.gangzimo(data);
        else if (data.kaigang)  this.kaigang(data);
        else if (data.kita)     this.kita(data);
        else if (data.hule)     this.hule(data);
        else if (data.pingju)   this.pingju(data);

        if (this._analyzer && ! this._redo
            && ! this._repeet) this._analyzer.action(data);

        if (! this._redo) this._idx++;

        if (this._paipu.log[this._log_idx][this._idx]
            && this._paipu.log[this._log_idx][this._idx].kaigang) this.next();

        if (this._autoplay) {
            let delay = this._redo        ? 500
                      : this._deny_repeat ? 3000 + this._speed * 2000
                      :                     this._speed * 200;
            this._autoplay_timer = setTimeout(()=>this.next(), delay);
        }

        this.set_fragment();
    }

    prev() {
        if (this._summary || this._log_idx < 0 || this._idx == 0) return true;
        if (this._autoplay) this.autoplay();
        let idx = this._idx - 1;
        while (idx > 0) {
            let data = this._paipu.log[this._log_idx][idx];
            if (! data.kaigang) break;
            idx--;
        }
        while (idx > 0) {
            let data = this._paipu.log[this._log_idx][--idx];
            if (data.zimo || data.gangzimo || data.fulou) break;
        }
        while (idx < this._paipu.log[this._log_idx].length) {
            let data = this._paipu.log[this._log_idx][idx + 1];
            if (! data.kaigang) break;
            idx++;
        }
        this.seek(this._log_idx, idx);
    }

    qipai(data) {
        if (this._target_log_idx != null
            && this._log_idx != this._target_log_idx) this.exit();
        this._deny_repeat = false;
        this._model.qipai(data.qipai);
        this._view.redraw();
    }

    zimo(data) {
        this._model.zimo(data.zimo);
        this._view.update(data);
    }

    dapai(data) {
        if (data.dapai.p.slice(-1) == '*' && ! this._redo) {
            this._redo = true;
            this._view.say('lizhi', data.dapai.l);
            return;
        }
        this._redo = false;
        this._model.dapai(data.dapai);
        this._view.update(data);
    }

    fulou(data) {
        if (! this._redo) {
            this._redo = true;
            let m = data.fulou.m.replace(/0/,'5');
            if      (m.match(/^[mpsz](\d)\1\1\1/))
                                        this._view.say('gang', data.fulou.l);
            else if (m.match(/^[mpsz](\d)\1\1/))
                                        this._view.say('peng', data.fulou.l);
            else                        this._view.say('chi',  data.fulou.l);
            return;
        }
        this._redo = false;
        this._model.fulou(data.fulou);
        this._view.update(data);
    }

    gang(data) {
        if (! this._redo) {
            this._redo = true;
            this._view.say('gang', data.gang.l);
            return;
        }
        this._redo = false;
        this._model.gang(data.gang);
        this._view.update(data);
    }

    gangzimo(data) {
        this._model.zimo(data.gangzimo);
        this._view.update(data);
    }

    kaigang(data) {
        this._model.kaigang(data.kaigang);
        this._view.update(data);
    }

    kita(data) {
        this._model.kita(data.kita);
        this._view.update(data);
    }

    hule(data) {
        if (! this._redo
            && ! this._paipu.log[this._log_idx][this._idx - 1].hule)
        {
            this._redo = true;
            if (data.hule.baojia == null) this._view.say('zimo', data.hule.l);
            else                          this._view.say('rong', data.hule.l);
            let i = 1;
            while (this._idx + i < this._paipu.log[this._log_idx].length) {
                let data = this._paipu.log[this._log_idx][this._idx + i];
                this._view.say('rong', data.hule.l)
                i++;
            }
            return;
        }
        this._redo = false;
        this._model.hule(data.hule);
        this._view.update(data);
        this._deny_repeat = true;
    }

    pingju(data) {
        if (! this._redo && data.pingju.name.match(/^三家和/)) {
            this._redo = true;
            for (let i = 1; i < 4; i++) {
                let l = (this._model.lunban + i) % 4;
                this._view.say('rong', l);
            }
            return;
        }
        this._redo = false;
        this._model.pingju(data.pingju);
        this._view.update(data);
        this._deny_repeat = true;
    }

    top(log_idx) {
        if (this._autoplay) this.autoplay();
        if (log_idx < 0 || this._paipu.log.length <= log_idx) return false;
        this.seek(log_idx, 0);
        if (this._target_log_idx != null
            && this._log_idx != this._target_log_idx) this.exit();
        return false;
    }

    last() {
        if (this._autoplay) this.autoplay();
        let idx  = this._paipu.log[this._log_idx].length - 1;
        let data = this._paipu.log[this._log_idx][idx];
        while (idx > 0 && (data.hule || data.pingju)) {
            data = this._paipu.log[this._log_idx][--idx];
        }
        this.seek(this._log_idx, idx);
        data = this._paipu.log[this._log_idx][this._idx];
        if (data.hule || data.pingju) {
            this.next();
            if (this._redo) setTimeout(()=> this.next(), 400);
        }
        return false;
    }

    forward() {
        if (this._summary) return true;
        if (this._log_idx < 0
            || this._paipu.log.length <= this._log_idx) return false;
        if (this._idx < this._paipu.log[this._log_idx].length
            && ! this._deny_repeat)
                return this.last();
        else    this.next();
        return false;
    }

    backward() {
        if (this._summary) return true;
        if (this._paipu.log.length <= this._log_idx) return true;
        if (this._idx > 1)
                return this.top(this._log_idx);
        else    return this.top(this._log_idx - 1);
    }

    preview(log_idx) {
        this._target_log_idx = log_idx;
        let viewpoint = this._paipu.qijia;
        if (log_idx >= 0) {
            let qipai = this._paipu.log[log_idx][0].qipai;
            viewpoint = (this._paipu.qijia + qipai.jushu) % 4;
        }
        this.start(viewpoint, log_idx);

        $('.tenhou-dialog input[name="limited"]', this._root)
                                    .prop('disabled', true).val([1]);
    }

    summary() {
        if (this._log_idx < 0 || this._deny_repeat) return true;
        if (this._summary) {
            this._view.summary();
            if (this._analyzer) this._analyzer.active(true);
            show($('.controller', this._root));
            this._summary = false;
            if (this._autoplay) this.next();
            return false;
        }
        this._autoplay_timer = clearTimeout(this._autoplay_timer);
        hide($('.controller', this._root));
        if (this._analyzer) this._analyzer.active(false);
        this._view.summary(this._paipu);
        for (let i = 0; i < this._paipu.log.length; i++) {
            $('.summary tbody tr', this._root).eq(i)
                .on('mousedown', (ev)=> this.top(i));
        }
        $('.summary', this._root).addClass('paipu')
        this._summary = true;
        return false;
    }

    viewpoint(viewpoint) {
        if (this._summary) return true;
        if (this._autoplay) this.autoplay();
        viewpoint = viewpoint % 4;
        if (viewpoint == this._view.viewpoint) {
            if (this._log_idx >= 0) this.dummy_name(viewpoint);
        }
        else {
            this._view.viewpoint = viewpoint;
            if (this._analyzer) {
                this._analyzer.id(this._view.viewpoint);
                if (this._log_idx >= 0) this.seek(this._log_idx, this._idx - 1);
            }
        }
        this.set_fragment();
        if (this._log_idx < 0) {
            this._view.kaiju();
            return false;
        }
        this._view.redraw();
        let data = this._paipu.log[this._log_idx][this._idx - 1];
        if (data.hule || data.pingju) this._view.update(data);
        return false;
    }

    dummy_name(dummy_name) {
        if (this._view.dummy_name === undefined) return;
        this._view.dummy_name
                = this._view.dummy_name == null ? dummy_name : null;
    }

    speed(speed) {
        if (speed < 1) speed = 1;
        if (speed > 5) speed = 5;
        this._speed = speed;
        const ctrl = $('.controller', this._root);
        $('.speed span', ctrl).each((i, n)=>{
            $(n).css('visibility', i < speed ? 'visible' : 'hidden');
        });
        this.set_pref();
        return false;
    }

    sound(on) {
        this._view.sound_on = on;
        const ctrl = $('.controller', this._root);
        if (on) {
            hide($('.sound.off', ctrl));
            show($('.sound.on', ctrl));
        }
        else {
            hide($('.sound.on', ctrl));
            show($('.sound.off', ctrl));
        }
        this.set_pref();
        return false;
    }

    shoupai() {
        if (this._summary) return true;
        this._view.open_shoupai = ! this._view.open_shoupai;
        this.set_fragment();
        if (this._log_idx < 0) return false;
        this._view.redraw();
        let data = this._paipu.log[this._log_idx][this._idx - 1];
        if (data.hule || data.pingju) this._view.update(data);
        return false;
    }

    he() {
        if (this._summary) return true;
        this._view.open_he = ! this._view.open_he;
        this.set_fragment();
        if (this._log_idx < 0) return false;
        this._view.redraw();
        let data = this._paipu.log[this._log_idx][this._idx - 1];
        if (data.hule || data.pingju) this._view.update(data);
        return false;
    }

    autoplay() {
        if (this._summary && ! this._autoplay) return true;
        this._autoplay_timer = clearTimeout(this._autoplay_timer);
        this._autoplay = ! this._autoplay;
        const ctrl = $('.controller', this._root);
        if (this._autoplay) {
            hide($('.play.on'), ctrl);
            show($('.play.off'), ctrl);
        }
        else {
            hide($('.play.off'), ctrl);
            show($('.play.on'), ctrl);
        }
        if (this._autoplay) this.next();
        return false;
    }

    analyzer() {
        if (this._summary) return true;
        if (! this._analyzer) {
            if (this._autoplay) this.autoplay();
            this._analyzer = this._open_analyzer({ kaiju: {
                id:     this._view.viewpoint,
                rule:   Majiang.rule(),
                title:  this._paipu.title,
                player: this._paipu.player,
                qijia:  this._paipu.qijia
            }});
            if (this._log_idx < 0) {
                this._analyzer.active(false);
            }
            else {
                this.seek(this._log_idx, this._idx - 1);
                this._analyzer.active(true);
            }
        }
        else {
            this._analyzer.close();
            delete this._analyzer;
        }
        this.set_fragment();
        return false;
    }

    tenhou_dialog() {

        const set_data = ()=>{
            let type    = $('[name="type"]:checked', dialog).val();
            let log_idx = $('[name="limited"]', dialog).prop('checked')
                                                    ? this._log_idx : null;
            let data = '';
            if (type == 'JSON') {
                data = JSON.stringify(logconv(this._paipu, log_idx));
                $('textarea', dialog).attr('class','JSON');
            }
            else {
                for (let i = 0; i < this._paipu.log.length; i++) {
                    if (log_idx != null && i != log_idx) continue;
                    data += 'https://tenhou.net/6/#json='
                            + encodeURI(JSON.stringify(logconv(this._paipu, i)))
                            + '\n';
                }
                $('textarea', dialog).attr('class','URL');
            }
            $('textarea', dialog).val(data);
            if (! navigator.clipboard) {
                show($('[type="button"]', dialog));
                hide($('[type="submit"]', dialog));
                setTimeout(()=>
                    $('textarea', dialog).attr('disabled', false).select(),
                    20);
            }
            else {
                show($('[type="button"]', dialog));
                show($('[type="submit"]', dialog));
            }
        };
        const submit = ()=>{
            if (navigator.clipboard) {
                let data = $('textarea', dialog).val();
                navigator.clipboard.writeText(data);
            }
            close_dialog();
            return false;
        };
        const close_dialog = ()=>{
            hide(dialog);
            this.set_handler();
        };

        if (this._log_idx < 0) return;
        const dialog = $('.tenhou-dialog', this._root);
        this.clear_handler();

        $('form', dialog).off('submit').on('submit', submit);
        $('[type="button"]', dialog).off('click').on('click', close_dialog);
        $('input', dialog).off('change').on('change', set_data);

        show(dialog);
        set_data();
    }

    ai_analyze() {
        if (this._summary || this._log_idx < 0) return true;
        if (this._autoplay) this.autoplay();

        const modal = $('.ai-modal', this._root);

        if (!modal.hasClass('hide')) {
            modal.addClass('hide');
            return false;
        }

        const boardEl = this._root[0] || this._root;
        const rect = boardEl.getBoundingClientRect();
        modal.css({ left: rect.right + 10, top: rect.top, height: rect.height });

        modal.removeClass('hide');
        modal.find('.ai-modal-body').hide();
        modal.find('.ai-modal-loading').show().text('計算中...');

        setTimeout(() => {
            this._do_ai_analyze(modal).catch(e => {
                console.error('AI Analysis error:', e);
                modal.find('.ai-modal-loading').text('エラー: ' + e.message);
            });
        }, 20);
        return false;
    }

    async _do_ai_analyze(modal) {
        const { analyze } = require('./ai-analyzer-engine');
        const result = analyze(
            this._paipu,
            this._log_idx,
            this._idx - 1,
            this._view.viewpoint
        );

        if (window.AI_PHASE2 && result.analysis_type !== null) {
            try {
                result.phase2 = await window.AI_PHASE2.analyze(
                    this._paipu,
                    this._log_idx,
                    this._idx - 1,
                    this._model,
                    result.menfeng,
                    result.analysis_type
                );
            } catch(e) {
                console.warn('AI Phase2 error:', e);
            }
        }

        this._render_ai_modal(modal, result);
    }

    _render_ai_modal(modal, result) {
        const { dapai_info, fulou_info, analysis_type, paishu, paijia_fn } = result;
        const pai = this._pai;

        const cmp = (a, b) => {
            if (a.selected) return -1;
            if (b.selected) return  1;
            return (b.ev ?? -Infinity) - (a.ev ?? -Infinity);
        };
        dapai_info.sort(cmp);
        fulou_info.sort(cmp);

        const dapai_body = modal.find('.ai-dapai-body').empty();
        for (const item of dapai_info) {
            const tr = $('<tr>');
            if (item.selected) tr.addClass('ai-selected');

            const td_p = $('<td class="ai-cand-tile">');
            if (item.n_xiangting < 0) {
                td_p.text('和了');
            } else if (item.m) {
                td_p.append(pai(item.p)).append($('<span>').text('カン'));
            } else {
                td_p.append(pai(item.p));
                if (item.p && item.p.slice(-1) == '*') td_p.append($('<span>').text('リーチ'));
            }

            const td_x = $('<td class="ai-xiangting">').text(
                item.n_xiangting < 0  ? '和了形' :
                item.n_xiangting == 0 ? '聴牌'   : `${item.n_xiangting}向聴`
            );

            const weixian_cls = item.weixian >= 13.0 ? 'ai-ev-high'
                              : item.weixian >=  8.0 ? 'ai-ev-middle'
                              : item.weixian >=  3.2 ? 'ai-ev-low'
                              : item.weixian ==  0.0 ? 'ai-ev-none'
                              : '';
            const td_ev = $('<td class="ai-ev">').addClass(weixian_cls);
            if (item.ev == null) {
                td_ev.text('オリ');
            } else if (item.n_xiangting > 2) {
                td_ev.text(`${item.n_tingpai}枚`);
            } else {
                td_ev.text(item.ev.toFixed(2));
            }

            const td_ting = $('<td class="ai-tingpai">');
            for (const p of item.tingpai || []) td_ting.append(pai(p));
            if ((item.n_tingpai ?? 0) > 0 && item.n_xiangting <= 2) {
                td_ting.append($('<span>').text(`(${item.n_tingpai}枚)`));
            }

            const td_w = $('<td class="ai-weixian">').text(
                item.weixian != null ? item.weixian.toFixed(1) : '-'
            );

            tr.append(td_p, td_x, td_ev, td_ting, td_w);
            dapai_body.append(tr);
        }

        if (!dapai_info.length) {
            dapai_body.append($('<tr><td colspan="5" style="color:#666;padding:8px">このターンは打牌分析の対象外です</td></tr>'));
        }

        const fulou_body = modal.find('.ai-fulou-body').empty();
        for (const item of fulou_info) {
            const tr = $('<tr>');
            if (item.selected) tr.addClass('ai-selected');

            const label = !item.m ? 'パス'
                        : item.m.replace(/0/g,'5').match(/[mpsz](\d)\1\1\1/) ? 'カン'
                        : item.m.replace(/0/g,'5').match(/[mpsz](\d)\1\1/)   ? 'ポン'
                        : 'チー';
            const td_m  = $('<td>').text(item.m ? `${label} ${item.m.replace(/0/g,'5')}` : 'パス');
            const td_x  = $('<td>').text(item.n_xiangting == 0 ? '聴牌' : `${item.n_xiangting}向聴`);
            const td_ev = $('<td>').text(item.ev != null ? item.ev.toFixed(2) : '-');

            tr.append(td_m, td_x, td_ev);
            fulou_body.append(tr);
        }

        if (!fulou_info.length) {
            fulou_body.append($('<tr><td colspan="3" style="color:#666;padding:8px">副露の選択肢がありません</td></tr>'));
        }

        this._render_ai_paishu(modal, paishu);
        if (paijia_fn) this._render_ai_paijia(modal, paijia_fn);

        this._render_phase2(modal, result.phase2);

        modal.find('.ai-tab-btn').removeClass('active');
        modal.find('.ai-tab-pane').addClass('hide');
        const active_tab = analysis_type === 'fulou' ? 'fulou' : 'dapai';
        modal.find(`[data-tab="${active_tab}"]`).addClass('active');
        modal.find(`#ai-tab-${active_tab}`).removeClass('hide');

        modal.find('.ai-modal-loading').hide();
        modal.find('.ai-modal-body').show();
    }

    _render_phase2(modal, phase2) {
        const PAI_NAMES_P2 = [];
        for (let i = 1; i <= 9; i++) PAI_NAMES_P2.push('m' + i);
        for (let i = 1; i <= 9; i++) PAI_NAMES_P2.push('p' + i);
        for (let i = 1; i <= 9; i++) PAI_NAMES_P2.push('s' + i);
        for (let i = 1; i <= 7; i++) PAI_NAMES_P2.push('z' + i);
        const pai = this._pai;

        // 打牌予測 (行動クローン)
        const bc_body = modal.find('.ai-bc-body').empty();
        if (phase2 && phase2.behavior_clone) {
            const tbody = $('<tbody>');
            for (const act of phase2.behavior_clone.top_actions) {
                const pct = (act.prob * 100).toFixed(1);
                const tr  = $('<tr>');
                tr.append($('<td class="ai-p2-tile">').append(pai(act.tile)));
                tr.append($('<td class="ai-p2-bar">').html(
                    `<div class="ai-prob-wrap"><div class="ai-prob-fill" style="width:${pct}%"></div><span>${pct}%</span></div>`
                ));
                tbody.append(tr);
            }
            bc_body.append($('<table class="ai-p2-table">').append(tbody));
        } else if (phase2) {
            bc_body.html('<span class="ai-p2-na">打牌分析対象外</span>');
        } else {
            bc_body.html('<span class="ai-p2-na">モデル未ロード</span>');
        }

        // 局面評価 (価値関数)
        const vf_body = modal.find('.ai-vf-body').empty();
        if (phase2 && phase2.value_function) {
            const { round_score, final_score } = phase2.value_function;
            const rs = Math.round(round_score);
            const fs = (final_score / 1000).toFixed(1);
            vf_body.html(
                `<span class="ai-vf-row">この局: <b class="${rs >= 0 ? 'ai-pos' : 'ai-neg'}">${rs >= 0 ? '+' : ''}${rs}点</b></span>` +
                `<span class="ai-vf-row">着順点予測: <b class="${final_score >= 0 ? 'ai-pos' : 'ai-neg'}">${final_score >= 0 ? '+' : ''}${fs}k</b></span>`
            );
        } else {
            vf_body.html('<span class="ai-p2-na">-</span>');
        }

        // 他家手牌推定
        const hi_body = modal.find('.ai-hi-body').empty();
        if (phase2 && phase2.hand_inference) {
            const SUITS_HI = [
                { key: 'm', label: 'M', offset:  0, max: 9, aka_key: 'm0' },
                { key: 'p', label: 'P', offset:  9, max: 9, aka_key: 'p0' },
                { key: 's', label: 'S', offset: 18, max: 9, aka_key: 's0' },
                { key: 'z', label: 'Z', offset: 27, max: 7, aka_key: null  },
            ];

            const make_block_cell = (dist) => {
                const wrap = $('<div class="ai-hi-probs">');
                const last = dist.length - 1;
                for (let k = 0; k < dist.length; k++) {
                    const pct  = Math.round(dist[k] * 100);
                    const label = (k === last && k >= 3) ? k + '個+' : k + '個';
                    wrap.append($('<div>').addClass('ai-hi-b' + k).text(label + ':' + pct + '%'));
                }
                return wrap;
            };

            const PAI_NAMES_HI = [];
            for (let i = 1; i <= 9; i++) PAI_NAMES_HI.push('m' + i);
            for (let i = 1; i <= 9; i++) PAI_NAMES_HI.push('p' + i);
            for (let i = 1; i <= 9; i++) PAI_NAMES_HI.push('s' + i);
            for (let i = 1; i <= 7; i++) PAI_NAMES_HI.push('z' + i);

            // v32: 113クラスターツ形デコードテーブル
            const TATSU_TABLE = (() => {
                const P = PAI_NAMES_HI;
                const SC = ['m','p','s'];
                const tbl = [];
                // 両面 (0..17): suit*6 + (n-1), n=1..6
                for (let suit = 0; suit < 3; suit++) {
                    for (let n = 1; n <= 6; n++) {
                        const t = suit * 9 + n;
                        tbl.push({ type: 'ryanmen', name: '両面' + (n+1) + (n+2) + SC[suit], wait_tiles: [t-1, t+2] });
                    }
                }
                // 辺張 (18..23)
                for (let suit = 0; suit < 3; suit++) {
                    tbl.push({ type: 'penchan', name: '辺張12' + SC[suit], wait_tiles: [suit*9 + 2] });
                    tbl.push({ type: 'penchan', name: '辺張89' + SC[suit], wait_tiles: [suit*9 + 6] });
                }
                // 嵌張 (24..44): suit*7 + n, n=0..6
                for (let suit = 0; suit < 3; suit++) {
                    for (let n = 0; n <= 6; n++) {
                        const t = suit * 9 + n;
                        tbl.push({ type: 'kanchan', name: '嵌張' + (n+1) + (n+3) + SC[suit], wait_tiles: [t+1] });
                    }
                }
                // 単騎 (45..78)
                for (let t = 0; t < 34; t++) {
                    tbl.push({ type: 'tanki', name: '単騎' + P[t], wait_tiles: [t] });
                }
                // 双碰 (79..112): 各牌個別エントリ、表示時にペアリング
                for (let t = 0; t < 34; t++) {
                    tbl.push({ type: 'shanpon', pai_name: P[t], wait_tiles: [t] });
                }
                return tbl;
            })();

            const make_best_hand_section = (block_ev, tatsu_probs) => {
                const div = $('<div class="ai-hi-best-hand">');
                // tenpai_prob がある場合は「聴牌推定形」として表示
                if (block_ev.tenpai_prob !== undefined) {
                    const tp_pct = block_ev.tenpai_prob !== null
                        ? Math.round(block_ev.tenpai_prob * 100) + '%' : '-';
                    div.append($('<span class="ai-hi-shanten-label">').text('聴牌推定形'));
                    div.append($('<span class="ai-hi-tenpai-prob">').text(' テンパイ確率: ' + tp_pct));
                    if (block_ev.decomps && block_ev.decomps.length > 0) {
                        const tp_str = (block_ev.tingpai && block_ev.tingpai.length > 0)
                            ? ' [待: ' + block_ev.tingpai.join(' ') + ']' : '';
                        for (const d of block_ev.decomps.slice(0, 3)) {
                            div.append($('<div class="ai-hi-decomp">').text(d.join(' ') + tp_str));
                        }
                    } else if (block_ev.hand_str) {
                        div.append($('<span class="ai-hi-hand-str">').text(block_ev.hand_str));
                    }
                    // v32: 113クラスターツ形による待ち推定
                    if (tatsu_probs) {
                        const THRESH = 0.3;
                        const parts = [];
                        // 両面/辺張/嵌張/単騎 (idx 0..78)
                        for (let idx = 0; idx < 79; idx++) {
                            if (tatsu_probs[idx] >= THRESH) {
                                parts.push({ label: TATSU_TABLE[idx].name, prob: tatsu_probs[idx] });
                            }
                        }
                        // 双碰 (idx 79..112): prob上位からペアリング
                        const sp = [];
                        for (let idx = 79; idx < 113; idx++) {
                            if (tatsu_probs[idx] >= THRESH) {
                                sp.push({ name: TATSU_TABLE[idx].pai_name, prob: tatsu_probs[idx] });
                            }
                        }
                        sp.sort((a, b) => b.prob - a.prob);
                        for (let i = 0; i + 1 < sp.length; i += 2) {
                            const avg = (sp[i].prob + sp[i+1].prob) / 2;
                            parts.push({ label: '双碰' + sp[i].name + '/' + sp[i+1].name, prob: avg });
                        }
                        if (sp.length % 2 === 1) {
                            const last = sp[sp.length - 1];
                            parts.push({ label: '双碰' + last.name, prob: last.prob });
                        }
                        parts.sort((a, b) => b.prob - a.prob);
                        const top5 = parts.slice(0, 5);
                        if (top5.length > 0) {
                            const text = top5.map(p => p.label + '(' + Math.round(p.prob * 100) + '%)').join(' ');
                            div.append($('<div class="ai-hi-tatsu-waits">').text('待ち推定: ' + text));
                        }
                    }
                    return div;
                }

                // v10 fallback: shanten ラベル表示
                if (block_ev.shanten === null) return div;
                const s_label = block_ev.shanten < 0  ? '和了'
                              : block_ev.shanten === 0 ? 'テンパイ'
                              : block_ev.shanten + '向聴';
                div.append($('<span class="ai-hi-shanten-label">').text(s_label + ': '));
                if (block_ev.decomps && block_ev.decomps.length > 0) {
                    const tp_str = (block_ev.tingpai && block_ev.tingpai.length > 0)
                        ? ' [待: ' + block_ev.tingpai.join(' ') + ']' : '';
                    for (const d of block_ev.decomps.slice(0, 3)) {
                        div.append($('<div class="ai-hi-decomp">').text(d.join(' ') + tp_str));
                    }
                } else {
                    div.append($('<span class="ai-hi-hand-str">').text(block_ev.hand_str));
                }
                return div;
            };

            const make_block_section = (block_ev, tatsu_probs) => {
                const wrap = $('<div class="ai-hi-block-section">');
                const label = block_ev.tenpai_prob !== undefined
                    ? 'ブロック期待値（聴牌形）' : 'ブロック期待値';
                wrap.append($('<div class="ai-hi-block-label">').text(label));

                // base: triplet/pair インデックスの開始位置（34牌順）
                // seq_base: seq_dist インデックスの開始位置（スーツ × 7）
                // ryanmen_base: ryanmen_dist インデックスの開始位置（スーツ × 8）
                // kanchan_base: kanchan_dist インデックスの開始位置（スーツ × 7）
                const SUITS_BLOCK = [
                    { key:'m', label:'M', base:0,  size:9, seq_base:0,  has_seq:true,  ryanmen_base:0,  kanchan_base:0  },
                    { key:'p', label:'P', base:9,  size:9, seq_base:7,  has_seq:true,  ryanmen_base:8,  kanchan_base:7  },
                    { key:'s', label:'S', base:18, size:9, seq_base:14, has_seq:true,  ryanmen_base:16, kanchan_base:14 },
                    { key:'z', label:'Z', base:27, size:7, seq_base:-1, has_seq:false, ryanmen_base:-1, kanchan_base:-1 },
                ];

                for (const suit of SUITS_BLOCK) {
                    const table = $('<table class="ai-hi-table ai-hi-block-table">');
                    const thead_tr = $('<tr>');
                    thead_tr.append($('<th>'));
                    for (let n = 1; n <= suit.size; n++) thead_tr.append($('<th>').text(n));
                    table.append($('<thead>').append(thead_tr));

                    const tbody = $('<tbody>');

                    const tr_tri = $('<tr>').addClass('ai-suit-' + suit.key);
                    tr_tri.append($('<th>').text('刻'));
                    for (let n = 0; n < suit.size; n++) {
                        tr_tri.append($('<td>').append(make_block_cell(block_ev.triplet_dist[suit.base + n])));
                    }
                    tbody.append(tr_tri);

                    if (suit.has_seq) {
                        const tr_seq = $('<tr>').addClass('ai-suit-' + suit.key);
                        tr_seq.append($('<th>').text('順'));
                        for (let n = 0; n < 7; n++) {
                            const seqLabel = suit.key + (n+1) + (n+2) + (n+3);
                            tr_seq.append($('<td>').attr('title', seqLabel)
                                .append(make_block_cell(block_ev.seq_dist[suit.seq_base + n])));
                        }
                        tr_seq.append($('<td colspan="2" class="ai-hi-block-empty">'));
                        tbody.append(tr_seq);
                    }

                    const tr_pair = $('<tr>').addClass('ai-suit-' + suit.key);
                    tr_pair.append($('<th>').text('対'));
                    for (let n = 0; n < suit.size; n++) {
                        tr_pair.append($('<td>').append(make_block_cell(block_ev.pair_dist[suit.base + n])));
                    }
                    tbody.append(tr_pair);

                    // 両面/辺張（v29）
                    if (suit.has_seq && block_ev.ryanmen_dist) {
                        const tr_ryanmen = $('<tr>').addClass('ai-suit-' + suit.key);
                        tr_ryanmen.append($('<th>').text('両'));
                        for (let n = 0; n < 8; n++) {
                            const label = suit.key + (n+1) + (n+2);
                            tr_ryanmen.append($('<td>').attr('title', label)
                                .append(make_block_cell(block_ev.ryanmen_dist[suit.ryanmen_base + n])));
                        }
                        tr_ryanmen.append($('<td class="ai-hi-block-empty">'));
                        tbody.append(tr_ryanmen);
                    }

                    // 嵌張（v29）
                    if (suit.has_seq && block_ev.kanchan_dist) {
                        const tr_kanchan = $('<tr>').addClass('ai-suit-' + suit.key);
                        tr_kanchan.append($('<th>').text('嵌'));
                        for (let n = 0; n < 7; n++) {
                            const label = suit.key + (n+1) + '_' + (n+3);
                            tr_kanchan.append($('<td>').attr('title', label)
                                .append(make_block_cell(block_ev.kanchan_dist[suit.kanchan_base + n])));
                        }
                        tr_kanchan.append($('<td colspan="2" class="ai-hi-block-empty">'));
                        tbody.append(tr_kanchan);
                    }

                    table.append(tbody);
                    wrap.append(table);
                }

                wrap.append(make_best_hand_section(block_ev, tatsu_probs));
                return wrap;
            };

            // ハーネス: 信頼度バッジ生成
            const make_confidence_badge = (checkers) => {
                const issues = [];
                if (checkers.A  && checkers.A.score  > 0.05) issues.push('A: 枯れ牌' + (checkers.A.score * 100).toFixed(0) + '%');
                if (checkers.B1 && checkers.B1.score > 0.08) issues.push('B1: 捨牌超過');
                if (checkers.B2 && checkers.B2.score > 3.0)  issues.push('B2: リーチ後未収束');
                if (checkers.B3 && checkers.B3.score > 0.3)  issues.push('B3: 待ち矛盾');
                const ok = issues.length === 0;
                const label = ok ? '高信頼' : issues.join(' / ');
                const color = ok ? '#2d6a2d' : '#6a2d2d';
                const prefix = ok ? '[OK] ' : '[!] ';
                return $('<div class="ai-hi-confidence">').css({
                    fontSize: '11px', padding: '3px 6px', marginTop: '4px',
                    background: color, borderRadius: '3px', color: '#ccc'
                }).text(prefix + label);
            };

            // ハーネスログエクスポートボタン (プレイヤー行の前に1回だけ挿入)
            if (window.FeedbackLogger) {
                const cnt = window.FeedbackLogger.getCount();
                const btn = $('<button class="ai-hi-export-btn">').text('ハーネスログ出力 (' + cnt + '件)')
                    .css({ fontSize: '11px', margin: '4px 0', padding: '2px 8px', cursor: 'pointer' })
                    .on('click', () => window.FeedbackLogger.exportLogs());
                hi_body.append(btn);
            }

            for (const p of phase2.hand_inference.players) {
                const section = $('<div class="ai-hi-player-section">');
                section.append($('<div class="ai-hi-player-label">').text(p.seat_name + ':'));

                const table = $('<table class="ai-hi-table">');
                const thead = $('<thead><tr>');
                $('tr', thead).append($('<th>'));
                for (let n = 1; n <= 9; n++) {
                    $('tr', thead).append($('<th>').text(n));
                    if (n === 5) $('tr', thead).append($('<th class="ai-hi-aka-th">').text('5赤'));
                }
                table.append(thead);

                const tbody = $('<tbody>');
                for (const suit of SUITS_HI) {
                    const tr = $('<tr>').addClass('ai-suit-' + suit.key);
                    tr.append($('<th>').text(suit.label));
                    for (let n = 1; n <= 9; n++) {
                        const td = $('<td>');
                        if (n > suit.max) {
                            td.css('background', '#1a1a1a');
                        } else {
                            const probs = p.probs_per_tile[suit.offset + (n - 1)];
                            const wrap  = $('<div class="ai-hi-probs">');
                            for (let cnt = 0; cnt < 5; cnt++) {
                                const pct = Math.round(probs[cnt] * 100);
                                wrap.append($('<div>').addClass('ai-hi-p' + cnt).text(cnt + '枚:' + pct + '%'));
                            }
                            td.append(wrap);
                        }
                        tr.append(td);
                        if (n === 5) {
                            const aka_td = $('<td class="ai-hi-aka-td">');
                            if (suit.aka_key) {
                                const v = p.aka ? p.aka[suit.aka_key] : null;
                                if (v !== null && v !== undefined) {
                                    aka_td.append($('<div class="ai-hi-aka-prob">').text('持:' + Math.round(v * 100) + '%'));
                                } else {
                                    aka_td.addClass('ai-hi-aka-na').text('-');
                                }
                            } else {
                                aka_td.css('background', '#1a1a1a');
                            }
                            tr.append(aka_td);
                        }
                    }
                    tbody.append(tr);
                }
                table.append(tbody);
                section.append(table);

                if (p.block_ev) section.append(make_block_section(p.block_ev, p.tatsu_probs));
                if (p.checkers) section.append(make_confidence_badge(p.checkers));
                hi_body.append(section);
            }

            // ---- ユーザーフィードバック入力欄 ----
            if (window.FeedbackLogger) {
                const ctx = phase2._context || {};

                const fb_wrap = $('<div class="ai-user-fb-wrap">').css({
                    marginTop: '12px', padding: '8px',
                    borderTop: '1px solid #444', fontSize: '12px',
                });
                fb_wrap.append(
                    $('<div>').css({ color: '#aaa', marginBottom: '4px' })
                              .text('フィードバック (この局面についてメモ)')
                );

                const textarea = $('<textarea class="ai-user-fb-input">').css({
                    width: '100%', height: '56px', fontSize: '12px',
                    background: '#222', color: '#ddd', border: '1px solid #555',
                    borderRadius: '3px', padding: '4px', boxSizing: 'border-box',
                    resize: 'vertical',
                }).attr('placeholder', 'モデルの挙動で気になった点や仮説を入力…');

                const btn_row = $('<div>').css({ display: 'flex', gap: '6px', marginTop: '4px' });

                const submit_btn = $('<button>').text('記録する').css({
                    fontSize: '11px', padding: '2px 10px', cursor: 'pointer',
                    background: '#2a5a2a', color: '#ccc', border: '1px solid #4a8a4a',
                    borderRadius: '3px',
                });

                const export_btn = $('<button>').text('全件出力').css({
                    fontSize: '11px', padding: '2px 8px', cursor: 'pointer',
                    background: '#333', color: '#aaa', border: '1px solid #555',
                    borderRadius: '3px',
                });

                const status = $('<span class="ai-user-fb-status">').css({
                    fontSize: '11px', color: '#888', marginLeft: '6px', alignSelf: 'center',
                });

                // 最新件数を表示
                const cnt = window.FeedbackLogger.getUserCount();
                status.text(cnt + '件蓄積');

                // 過去フィードバック表示欄
                const history = $('<div class="ai-user-fb-history">').css({
                    marginTop: '6px', maxHeight: '80px', overflowY: 'auto',
                    fontSize: '11px', color: '#999',
                });

                const render_history = () => {
                    history.empty();
                    const recents = window.FeedbackLogger.getUserFeedbacks(5).reverse();
                    if (recents.length === 0) return;
                    for (const r of recents) {
                        const d = new Date(r.ts);
                        const ts_str = `${d.getMonth()+1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2,'0')}`;
                        const turn_str = r.ctx && r.ctx.turn != null ? `巡${r.ctx.turn}` : '';
                        history.append(
                            $('<div>').css({ borderBottom: '1px solid #333', padding: '2px 0' })
                                      .text(`[${ts_str}${turn_str ? ' ' + turn_str : ''}] ${r.text}`)
                        );
                    }
                };
                render_history();

                submit_btn.on('click', () => {
                    const text = textarea.val();
                    if (!text.trim()) return;
                    // チェッカー結果をコンテキストに含める
                    const checker_summary = {};
                    if (phase2.hand_inference) {
                        for (const p of phase2.hand_inference.players) {
                            if (p.checkers) {
                                checker_summary[p.seat_name] = {
                                    A:  p.checkers.A  ? p.checkers.A.score.toFixed(3)  : 0,
                                    B2: p.checkers.B2 ? p.checkers.B2.score.toFixed(3) : 0,
                                    B3: p.checkers.B3 ? p.checkers.B3.score.toFixed(3) : 0,
                                };
                            }
                        }
                    }
                    const full_ctx = { ...ctx, checkers: checker_summary };
                    const ok = window.FeedbackLogger.submitUserFeedback(text, full_ctx);
                    if (ok) {
                        textarea.val('');
                        const new_cnt = window.FeedbackLogger.getUserCount();
                        status.text(new_cnt + '件蓄積');
                        render_history();
                    }
                });

                export_btn.on('click', () => window.FeedbackLogger.exportUserFeedbacks());

                btn_row.append(submit_btn, export_btn, status);
                fb_wrap.append(textarea, btn_row, history);
                hi_body.append(fb_wrap);
            }

        } else {
            hi_body.html('<span class="ai-p2-na">-</span>');
        }
    }

    _render_ai_paishu(modal, paishu) {
        const body   = modal.find('.ai-paishu-body').empty();
        const suits  = ['m','p','s','z'];
        const labels = { m:'M', p:'P', s:'S', z:'Z' };

        for (const s of suits) {
            const tr  = $('<tr>').addClass(`ai-suit-${s}`);
            const max = s === 'z' ? 7 : 9;
            tr.append($('<th>').text(labels[s]));
            for (let n = 1; n <= 9; n++) {
                if (n > max) {
                    tr.append($('<td>').text('').css('background','#1a1a1a'));
                } else {
                    const cnt = paishu[s][n];
                    const td  = $('<td>').text(cnt);
                    if (cnt === 0) td.addClass('ai-zero');
                    tr.append(td);
                }
            }
            body.append(tr);
        }
    }

    _render_ai_paijia(modal, paijia_fn) {
        const body   = modal.find('.ai-paijia-body').empty();
        const suits  = ['m','p','s','z'];
        const labels = { m:'M', p:'P', s:'S', z:'Z' };

        for (const s of suits) {
            const tr  = $('<tr>').addClass(`ai-suit-${s}`);
            const max = s === 'z' ? 7 : 9;
            tr.append($('<th>').text(labels[s]));
            for (let n = 1; n <= 9; n++) {
                if (n > max) {
                    tr.append($('<td>').text('').css('background','#1a1a1a'));
                } else {
                    const score = paijia_fn(s + n);
                    const td    = $('<td>').text(Math.round(score));
                    if (score === 0) td.addClass('ai-zero');
                    tr.append(td);
                }
            }
            body.append(tr);
        }
    }

    set_fragment(hash) {

        if (hash) {
            this._fragment_base = hash.replace(/\/.*$/,'');
        }
        else if (hash == '') {
            history.replaceState('', '', location.href.replace(/#.*$/,''));
        }
        else {
            if (! this._fragment_base) return;

            let state = [ this._view.viewpoint ];
            if (this._log_idx >= 0) {
                state.push(this._log_idx, this._idx - 1);
            }

            let opt = (this._view.open_shoupai ? ''  : 's')
                    + (this._view.open_he      ? ''  : 'h')
                    + (this._analyzer          ? 'i' : '' )
                    + (this._view.dummy_name != null
                                ? 'p' + this._view.dummy_name : '');

            let fragment = this._fragment_base + '/' +  state.join('/')
                         + (opt ? `:${opt}` : '');

            history.replaceState('', '', fragment);
        }
    }
}
