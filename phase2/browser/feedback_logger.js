'use strict';

(function () {
    const STORAGE_KEY      = 'majiang_feedback_log';
    const USER_STORAGE_KEY = 'majiang_user_feedback';
    const SYNC_TS_KEY      = 'majiang_user_feedback_synced_ts'; // 同期済み最終ts
    const MAX_RECORDS      = 5000;
    const MAX_USER_RECORDS = 1000;
    const THRESHOLD = { A: 0.05, B2: 3.0, B3: 0.3, B4: 0.1 };
    const API_URL   = '/api/feedback';

    // ---- 自動ハーネスログ ----

    function pushLog(record) {
        if (!Object.entries(THRESHOLD).some(([k, v]) => record[k] && record[k].score > v)) return;
        try {
            const logs = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
            logs.push({ ts: Date.now(), ...record });
            if (logs.length > MAX_RECORDS) logs.splice(0, logs.length - MAX_RECORDS);
            localStorage.setItem(STORAGE_KEY, JSON.stringify(logs));
        } catch (e) {
            console.warn('FeedbackLogger: localStorage error', e);
        }
    }

    function exportLogs() {
        try {
            const logs = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
            if (logs.length === 0) { alert('ハーネスログがありません'); return; }
            _download(logs, 'harness_log_');
        } catch (e) {
            console.warn('FeedbackLogger: export error', e);
        }
    }

    function clearLogs() { localStorage.removeItem(STORAGE_KEY); }

    function getCount() {
        try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]').length; }
        catch { return 0; }
    }

    // ---- ユーザー手動フィードバック ----

    function submitUserFeedback(text, context) {
        if (!text || !text.trim()) return false;
        const record = { ts: Date.now(), text: text.trim(), ctx: context || {} };
        // localStorage に保存（オフラインフォールバック）
        try {
            const logs = JSON.parse(localStorage.getItem(USER_STORAGE_KEY) || '[]');
            logs.push(record);
            if (logs.length > MAX_USER_RECORDS) logs.splice(0, logs.length - MAX_USER_RECORDS);
            localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(logs));
        } catch (e) {
            console.warn('FeedbackLogger: localStorage error', e);
        }
        // サーバーに POST → phase2/data/user_feedback.jsonl に追記
        _postToServer(record);
        return true;
    }

    function getUserFeedbacks(limit) {
        try {
            const logs = JSON.parse(localStorage.getItem(USER_STORAGE_KEY) || '[]');
            return limit ? logs.slice(-limit) : logs;
        } catch { return []; }
    }

    function getUserCount() {
        try { return JSON.parse(localStorage.getItem(USER_STORAGE_KEY) || '[]').length; }
        catch { return 0; }
    }

    function exportUserFeedbacks() {
        try {
            const logs = JSON.parse(localStorage.getItem(USER_STORAGE_KEY) || '[]');
            if (logs.length === 0) { alert('ユーザーフィードバックがありません'); return; }
            _download(logs, 'user_feedback_');
        } catch (e) {
            console.warn('FeedbackLogger: user export error', e);
        }
    }

    function clearUserFeedbacks() { localStorage.removeItem(USER_STORAGE_KEY); }

    // ---- サーバー同期 ----

    function _postToServer(record) {
        fetch(API_URL, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(record),
        }).then(r => {
            if (r.ok) {
                // 同期済み最終tsを更新
                const prev = parseInt(localStorage.getItem(SYNC_TS_KEY) || '0', 10);
                if (record.ts > prev) localStorage.setItem(SYNC_TS_KEY, String(record.ts));
            }
        }).catch(() => {}); // サーバー未起動時は無視
    }

    // ページ読み込み時に未同期の localStorage データをサーバーへ送る
    function _syncUnsyncedToServer() {
        try {
            const logs      = JSON.parse(localStorage.getItem(USER_STORAGE_KEY) || '[]');
            const syncedTs  = parseInt(localStorage.getItem(SYNC_TS_KEY) || '0', 10);
            const unsynced  = logs.filter(r => (r.ts || 0) > syncedTs);
            if (unsynced.length === 0) return;
            console.info(`FeedbackLogger: 未同期 ${unsynced.length} 件をサーバーへ送信中...`);
            // 順序を保ちつつ逐次送信（並列にしてサーバー負荷をかけない）
            let i = 0;
            function next() {
                if (i >= unsynced.length) return;
                _postToServer(unsynced[i++]);
                // 少し間隔を空けて次を送る
                setTimeout(next, 50);
            }
            next();
        } catch (e) {
            console.warn('FeedbackLogger: sync error', e);
        }
    }

    // ---- 共通ユーティリティ ----

    function _download(logs, prefix) {
        const jsonl = logs.map(r => JSON.stringify(r)).join('\n');
        const blob  = new Blob([jsonl], { type: 'application/x-ndjson' });
        const a     = document.createElement('a');
        a.href      = URL.createObjectURL(blob);
        a.download  = prefix + Date.now() + '.jsonl';
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    }

    // ---- 初期化 ----
    // DOM 構築後にサーバーが起動していれば未同期データを送信
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _syncUnsyncedToServer);
    } else {
        setTimeout(_syncUnsyncedToServer, 500);
    }

    window.FeedbackLogger = {
        // 自動ハーネス
        pushLog, exportLogs, clearLogs, getCount,
        // ユーザー手動
        submitUserFeedback, getUserFeedbacks, getUserCount,
        exportUserFeedbacks, clearUserFeedbacks,
    };
})();
