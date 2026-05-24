'use strict';

const express        = require('express');
const convlog        = require('./lib/convlog');
const getlog_factory = require('./lib/getlog');

const app = express();
app.locals.getlog = getlog_factory();

app.get('/tenhou-log/:id.json', (req, res) => {
    req.app.locals.getlog(req.params.id)
        .then(xml => res.json(convlog(xml, req.params.id)))
        .catch(st  => res.status(typeof st === 'number' ? st : 500).end());
});

if (require.main === module) {
    const port = process.env.PORT || 8001;
    app.listen(port, () => console.log(`tenhou-log-local listening on port ${port}`));
}

module.exports = app;
