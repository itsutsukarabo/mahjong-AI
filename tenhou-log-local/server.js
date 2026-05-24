'use strict';

const express        = require('express');
const convlog        = require('./lib/convlog');
const getlog_factory = require('./lib/getlog');

const app = express();
app.locals.getlog = getlog_factory();

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    next();
});

// パス形式: /tenhou-log/2025...json（テスト用）
app.get('/tenhou-log/:id.json', (req, res) => {
    const id = req.params.id;
    req.app.locals.getlog(id)
        .then(xml => res.json(convlog(xml, id)))
        .catch(st  => res.status(typeof st === 'number' ? st : 500).end());
});

// クエリ形式: /tenhou-log/?2025...json（majiang-ui が使用）
app.get('/tenhou-log/', (req, res) => {
    const raw = req.url.split('?')[1] || '';
    const id  = raw.replace(/\.json$/, '');
    if (!id) return res.status(404).end();
    req.app.locals.getlog(id)
        .then(xml => res.json(convlog(xml, id)))
        .catch(st  => res.status(typeof st === 'number' ? st : 500).end());
});

if (require.main === module) {
    const port = process.env.PORT || 8001;
    app.listen(port, () => console.log(`tenhou-log-local listening on port ${port}`));
}

module.exports = app;
