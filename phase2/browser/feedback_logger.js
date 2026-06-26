'use strict';

(function () {
    const STORAGE_KEY = 'majiang_feedback_log';
    const MAX_RECORDS = 5000;
    const THRESHOLD = { A: 0.05, B2: 3.0, B3: 0.3, B4: 0.1 };

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
            if (logs.length === 0) { alert('ログがありません'); return; }
            const jsonl = logs.map(r => JSON.stringify(r)).join('\n');
            const blob = new Blob([jsonl], { type: 'application/x-ndjson' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'feedback_log_' + Date.now() + '.jsonl';
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 10000);
        } catch (e) {
            console.warn('FeedbackLogger: export error', e);
        }
    }

    function clearLogs() {
        localStorage.removeItem(STORAGE_KEY);
    }

    function getCount() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]').length;
        } catch { return 0; }
    }

    window.FeedbackLogger = { pushLog, exportLogs, clearLogs, getCount };
})();
