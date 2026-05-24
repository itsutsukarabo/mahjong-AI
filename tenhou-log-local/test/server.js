'use strict';

const assert   = require('assert');
const sinon    = require('sinon');
const request  = require('supertest');
const fs       = require('fs');
const path     = require('path');
const app      = require('../server');

const FIXTURE_XML = fs.readFileSync(
    path.join(__dirname, 'fixtures/sanma_sample.xml'), 'utf-8');

suite('GET /tenhou-log/:id.json', () => {
    let stub;

    setup(() => {
        stub = sinon.stub();
        app.locals.getlog = stub;
    });

    test('200 と JSON が返ること（getlog をフィクスチャ XML でモック）', done => {
        stub.resolves(FIXTURE_XML);
        request(app)
            .get('/tenhou-log/dummy-id.json')
            .expect(200)
            .expect('Content-Type', /json/)
            .end((err, res) => {
                if (err) return done(err);
                assert.ok(res.body.log, 'log フィールドが存在すること');
                done();
            });
    });

    test('天鳳が 404 を返すとき 404 が伝播すること', done => {
        stub.callsFake(() => Promise.reject(404));
        request(app)
            .get('/tenhou-log/nonexistent.json')
            .expect(404, done);
    });

    test('三麻 XML を変換したとき kita アクションが含まれること', done => {
        stub.resolves(FIXTURE_XML);
        request(app)
            .get('/tenhou-log/sanma-id.json')
            .expect(200)
            .end((err, res) => {
                if (err) return done(err);
                const hasKita = res.body.log.some(hand =>
                    hand.some(entry => entry.kita)
                );
                assert.ok(hasKita, 'kita アクションが含まれること');
                done();
            });
    });
});
