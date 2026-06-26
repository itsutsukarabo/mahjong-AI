'use strict';

(function () {
    const STORAGE_KEY      = 'majiang_feedback_log';
    const USER_STORAGE_KEY = 'majiang_user_feedback';
    const MAX_RECORDS      = 5000;
    const MAX_USER_RECORDS = 1000;
    const THRESHOLD = { A: 0.05, B2: 3.0, B3: 0.3, B4: 0.1 };

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
        try {
            const logs = JSON.parse(localStorage.getItem(USER_STORAGE_KEY) || '[]');
            logs.push({
                ts:   Date.now(),
                text: text.trim(),
                ctx:  context || {},
            });
            if (logs.length > MAX_USER_RECORDS) logs.splice(0, logs.length - MAX_USER_RECORDS);
            localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(logs));
            return true;
        } catch (e) {
            console.warn('FeedbackLogger: user feedback error', e);
            return false;
        }
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

    window.FeedbackLogger = {
        // 自動ハーネス
        pushLog, exportLogs, clearLogs, getCount,
        // ユーザー手動
        submitUserFeedback, getUserFeedbacks, getUserCount,
        exportUserFeedbacks, clearUserFeedbacks,
    };
})();
