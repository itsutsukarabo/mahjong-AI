#!/usr/bin/env node
/**
 * 天鳳 scraw年別アーカイブ / scc*.html.gz から鳳凰卓ゲームIDを抽出する
 *
 * 使い方:
 *   # 年別アーカイブZIPから抽出（推奨: 一括）
 *   node fetch_scc_ids.js --zip data/raw/scraw2025.zip --out data/raw/ids_scc2025.txt
 *
 *   # 個別のscc*.html.gzファイルが入ったディレクトリから抽出
 *   node fetch_scc_ids.js --dir data/raw/scc/ --out data/raw/ids_scc.txt
 *
 *   # 直近7日分をtenhou.netから直接取得してIDを抽出
 *   node fetch_scc_ids.js --from-web --out data/raw/ids_scc_recent.txt
 *
 * 出力: 1行1IDのテキストファイル（download_paipu.js の --ids-file 引数に渡す）
 */
'use strict';

const https  = require('https');
const http   = require('http');
const fs     = require('fs');
const path   = require('path');
const zlib   = require('zlib');
const stream = require('stream');

// Node.js 組み込みの stream.pipeline を使用
const { pipeline } = require('stream/promises');

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

const OUT = opts['out'] || 'data/raw/ids_scc.txt';

// ---- HTMLからゲームIDを抽出 ----

// 天鳳の観戦URLパターン: ?log=YYYYMMDD##gm-HHHH-0000-HHHHHHHH
const LOG_ID_RE = /[?&]log=([0-9]{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{8})/g;

function extract_ids_from_html(html) {
    const ids = new Set();
    let m;
    while ((m = LOG_ID_RE.exec(html)) !== null) {
        ids.add(m[1]);
    }
    // 正規表現のlastIndexをリセット（グローバルフラグのため）
    LOG_ID_RE.lastIndex = 0;
    return [...ids];
}

// ---- gzip バッファを解凍してHTML文字列を返す ----

function gunzip_buf(buf) {
    return new Promise((resolve, reject) => {
        zlib.gunzip(buf, (err, result) => {
            if (err) reject(err);
            else resolve(result.toString('utf8'));
        });
    });
}

// ---- HTTP(S)からバッファを取得 ----

function fetch_buf(url) {
    return new Promise((resolve, reject) => {
        const mod = url.startsWith('https') ? https : http;
        mod.get(url, res => {
            if (res.statusCode === 301 || res.statusCode === 302) {
                return fetch_buf(res.headers.location).then(resolve, reject);
            }
            if (res.statusCode !== 200) {
                res.resume();
                return reject(new Error(`HTTP ${res.statusCode}: ${url}`));
            }
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end',  () => resolve(Buffer.concat(chunks)));
            res.on('error', reject);
        }).on('error', reject);
    });
}

// ---- ZIPファイルを解析（組み込みのみ使用、外部依存なし） ----
//
// Node.js に zip ライブラリは標準で含まれないが、
// scraw*.zip は ZIP64 でない通常の ZIPなので
// ローカルファイルヘッダを直接パースする。

function read_u32_le(buf, offset) {
    return buf.readUInt32LE(offset);
}
function read_u16_le(buf, offset) {
    return buf.readUInt16LE(offset);
}

/**
 * ZIP バッファを解析し、フィルタに一致するエントリのデータを返す。
 * @param {Buffer} zip_buf
 * @param {RegExp} file_filter  エントリのファイル名をフィルタする正規表現
 * @returns {{ name: string, data: Buffer }[]}
 */
function parse_zip(zip_buf, file_filter) {
    const entries = [];
    let offset = 0;

    while (offset + 30 <= zip_buf.length) {
        const sig = read_u32_le(zip_buf, offset);

        // Local file header: 0x04034b50
        if (sig !== 0x04034b50) break;

        const compression   = read_u16_le(zip_buf, offset + 8);
        const compressed    = read_u32_le(zip_buf, offset + 18);
        const uncompressed  = read_u32_le(zip_buf, offset + 22);
        const fname_len     = read_u16_le(zip_buf, offset + 26);
        const extra_len     = read_u16_le(zip_buf, offset + 28);
        const fname         = zip_buf.slice(offset + 30, offset + 30 + fname_len).toString('utf8');
        const data_offset   = offset + 30 + fname_len + extra_len;
        const data_buf      = zip_buf.slice(data_offset, data_offset + compressed);

        offset = data_offset + compressed;

        if (!file_filter.test(fname)) continue;

        // compression: 0=stored, 8=deflate
        let raw;
        if (compression === 0) {
            raw = data_buf;
        } else if (compression === 8) {
            try {
                raw = zlib.inflateRawSync(data_buf);
            } catch {
                continue;
            }
        } else {
            continue;
        }

        entries.push({ name: fname, data: raw });
    }

    return entries;
}

// ---- 年別アーカイブZIPから抽出 ----

async function from_zip(zip_path) {
    console.log(`ZIP読み込み中: ${zip_path}`);
    const zip_buf = fs.readFileSync(zip_path);
    console.log(`  ${(zip_buf.length / 1024 / 1024).toFixed(1)} MB`);

    // scc*.html.gz エントリを探す
    const scc_entries = parse_zip(zip_buf, /scc\d{10}\.html\.gz$/);
    console.log(`  scc*.html.gz エントリ: ${scc_entries.length} 件`);

    const all_ids = new Set();
    let processed = 0;
    for (const entry of scc_entries) {
        try {
            const html = await gunzip_buf(entry.data);
            const ids  = extract_ids_from_html(html);
            ids.forEach(id => all_ids.add(id));
        } catch { /* 壊れたファイルはスキップ */ }
        processed++;
        if (processed % 500 === 0) {
            process.stdout.write(`  ${processed}/${scc_entries.length} 処理済 (${all_ids.size} IDs)\r`);
        }
    }
    console.log(`\n  抽出完了: ${all_ids.size} ゲームID`);
    return [...all_ids];
}

// ---- ディレクトリ内のscc*.html.gzから抽出 ----

async function from_dir(dir_path) {
    const files = fs.readdirSync(dir_path).filter(f => /^scc\d{10}\.html\.gz$/.test(f));
    console.log(`scc*.html.gz ファイル: ${files.length} 件`);
    const all_ids = new Set();
    for (const f of files) {
        try {
            const buf  = fs.readFileSync(path.join(dir_path, f));
            const html = await gunzip_buf(buf);
            extract_ids_from_html(html).forEach(id => all_ids.add(id));
        } catch { /* スキップ */ }
    }
    console.log(`抽出完了: ${all_ids.size} ゲームID`);
    return [...all_ids];
}

// ---- tenhou.net/sc/raw/list.cgi から直近分を取得 ----

async function from_web() {
    const LIST_URL = 'http://tenhou.net/sc/raw/list.cgi';
    console.log(`${LIST_URL} を取得中...`);
    const list_buf  = await fetch_buf(LIST_URL);
    const list_text = list_buf.toString('utf8');

    // list.cgi は list([{file:"scc...",size:...},...]); 形式
    const file_re = /"(scc[^"]+\.html\.gz)"/g;
    const scc_files = [];
    let m;
    while ((m = file_re.exec(list_text)) !== null) scc_files.push(m[1]);
    console.log(`  sccファイル数: ${scc_files.length}`);

    const all_ids = new Set();
    for (const fname of scc_files) {
        const url = `http://tenhou.net/sc/raw/dat/${fname}`;
        try {
            const buf  = await fetch_buf(url);
            const html = await gunzip_buf(buf);
            extract_ids_from_html(html).forEach(id => all_ids.add(id));
            process.stdout.write(`  ${all_ids.size} IDs\r`);
            // /sc/raw/ ファイルは20分間隔ルールが適用されるため --from-web は少数取得用
        } catch (e) {
            console.error(`  失敗: ${fname}: ${e.message}`);
        }
    }
    console.log(`\n抽出完了: ${all_ids.size} ゲームID`);
    return [...all_ids];
}

// ---- メイン ----

async function main() {
    let ids = [];

    if (opts['zip']) {
        ids = await from_zip(opts['zip']);
    } else if (opts['dir']) {
        ids = await from_dir(opts['dir']);
    } else if (opts['from-web']) {
        ids = await from_web();
    } else {
        console.error('使い方:');
        console.error('  node fetch_scc_ids.js --zip  <scraw年.zip>  --out <ids.txt>');
        console.error('  node fetch_scc_ids.js --dir  <sccファイルの入ったディレクトリ> --out <ids.txt>');
        console.error('  node fetch_scc_ids.js --from-web             --out <ids.txt>');
        process.exit(1);
    }

    // 最新順にソート（IDの先頭10桁がYYYYMMDDHH）
    ids.sort().reverse();

    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    fs.writeFileSync(OUT, ids.join('\n') + '\n', 'utf8');
    console.log(`保存: ${OUT}  (${ids.length} IDs)`);
}

main().catch(err => { console.error(err); process.exit(1); });
