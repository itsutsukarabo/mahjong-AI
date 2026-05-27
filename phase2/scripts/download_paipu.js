#!/usr/bin/env node
/**
 * 天鳳牌譜ダウンロードスクリプト
 *
 * 個別ゲームXML（/0/log/?{id}）を IDリストからダウンロードする。
 * ※ /sc/raw/ 索引ファイルには20分間隔ルールが適用されるが、
 *    個別ゲームXMLはユーザーの観戦URLと同じエンドポイントのため
 *    デフォルトインターバルは5秒（--interval で変更可）。
 *
 * 使い方:
 *   # IDリストファイルから一括ダウンロード（fetch_scc_ids.js の出力を渡す）
 *   node download_paipu.js --ids-file data/raw/ids_scc2025.txt --dest data/raw/xml/
 *
 *   # 単一IDをダウンロード
 *   node download_paipu.js --id 2025080617gm-00b9-0000-104adf08 --dest data/raw/xml/
 *
 *   # 取得件数上限を指定（最新順に--count件だけ）
 *   node download_paipu.js --ids-file ids.txt --count 500 --dest data/raw/xml/
 *
 *   # 実際にダウンロードせず確認だけ
 *   node download_paipu.js --ids-file ids.txt --dry-run
 *
 * IDリストファイルの形式:
 *   1行に1ID。空行・#始まりのコメント行は無視。
 *   例: 2025080617gm-00b9-0000-104adf08
 */
'use strict';

const https    = require('https');
const http     = require('http');
const fs       = require('fs');
const path     = require('path');
const zlib     = require('zlib');

// /sc/raw/ 索引ファイルの20分ルールは個別ゲームXMLには適用されない
// デフォルト5秒（サーバー負荷配慮）、--interval で秒数を変更可
const BASE_URL        = 'http://tenhou.net/0/log/';

// ---- CLI引数パース ----

const args = process.argv.slice(2);
const opts = {};
for (let i = 0; i < args.length; i++) {
    const k = args[i];
    if (k.startsWith('--')) {
        const key = k.slice(2);
        const val = (args[i + 1] && !args[i + 1].startsWith('--')) ? args[++i] : true;
        opts[key] = val;
    }
}

const DRY_RUN    = !!opts['dry-run'];
const DEST       = opts['dest'] || 'data/raw/xml';
const IDS_FILE   = opts['ids-file'];
const SINGLE_ID  = opts['id'];
const COUNT      = opts['count'] ? parseInt(opts['count']) : Infinity;
const INTERVAL_MS = opts['interval'] ? parseInt(opts['interval']) * 1000 : 5000;

if (!IDS_FILE && !SINGLE_ID) {
    console.error('使い方: node download_paipu.js --ids-file <file> [--dest <dir>] [--count N] [--interval 秒]');
    console.error('        node download_paipu.js --id <log-id>     [--dest <dir>]');
    process.exit(1);
}

// ---- ID収集 ----

let ids = [];
if (IDS_FILE) {
    const raw = fs.readFileSync(IDS_FILE, 'utf8');
    ids = raw.split('\n')
        .map(l => l.trim())
        .filter(l => l && !l.startsWith('#'));
} else {
    ids = [SINGLE_ID];
}

if (isFinite(COUNT)) ids = ids.slice(0, COUNT);
console.log(`対象: ${ids.length} 件${DRY_RUN ? ' [DRY RUN]' : ''}  インターバル: ${INTERVAL_MS / 1000}秒`);

// ---- ダウンロード処理 ----

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function fetch_url(url) {
    return new Promise((resolve, reject) => {
        const mod = url.startsWith('https') ? https : http;
        mod.get(url, res => {
            if (res.statusCode === 301 || res.statusCode === 302) {
                return fetch_url(res.headers.location).then(resolve, reject);
            }
            if (res.statusCode !== 200) {
                res.resume();
                return reject(new Error(`HTTP ${res.statusCode}: ${url}`));
            }
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => resolve(Buffer.concat(chunks)));
            res.on('error', reject);
        }).on('error', reject);
    });
}

async function download_one(log_id, dest_dir) {
    const url      = `${BASE_URL}?${log_id}`;
    const filename = `${log_id}.xml`;
    const filepath = path.join(dest_dir, filename);

    if (fs.existsSync(filepath)) {
        console.log(`スキップ (既存): ${log_id}`);
        return;
    }

    if (DRY_RUN) {
        console.log(`[DRY RUN] ダウンロード予定: ${url} → ${filepath}`);
        return;
    }

    process.stdout.write(`ダウンロード中: ${log_id} ... `);
    try {
        const buf = await fetch_url(url);

        // gzip の場合は解凍
        let xml;
        if (buf[0] === 0x1f && buf[1] === 0x8b) {
            xml = zlib.gunzipSync(buf).toString('utf8');
        } else {
            xml = buf.toString('utf8');
        }

        // 天鳳ログでない場合は保存しない
        if (!xml.includes('<mjloggm')) {
            console.log(`スキップ (非麻雀ログ)`);
            return;
        }

        fs.writeFileSync(filepath, xml, 'utf8');
        console.log(`完了 (${(buf.length / 1024).toFixed(1)} KB)`);
    } catch (err) {
        console.log(`失敗: ${err.message}`);
    }
}

async function main() {
    if (!DRY_RUN) fs.mkdirSync(DEST, { recursive: true });

    for (let i = 0; i < ids.length; i++) {
        const id = ids[i];

        if (i > 0) {
            if (!DRY_RUN) await sleep(INTERVAL_MS);
        }

        await download_one(id, DEST);
    }

    console.log('完了');
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
