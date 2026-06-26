"""
ハーネスレポート: ブラウザのフィードバックログを分析し対応策を提案する。

使い方:
  C:\ml\venv\Scripts\python.exe phase2/scripts/harness_report.py feedback_log.jsonl
  C:\ml\venv\Scripts\python.exe phase2/scripts/harness_report.py feedback_log.jsonl --min-records 5

出力:
  弱点ランキング + 対応策の番号メニュー → 選択後に harness_fix.py を起動
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import json
import glob
import subprocess
from pathlib import Path
from collections import defaultdict

# ---- 閾値（feedback_logger.js と同じ値） ----
THRESHOLD = {'A': 0.05, 'B2': 3.0, 'B3': 0.3}

# ---- 対応策マッピング ----
# (checker_type, late_game, riichi) → fix_name のルール
REMEDIATION_MAP = [
    {
        'id':          'fix_impossible_tile_penalty',
        'triggers':    ['A'],
        'description': '学習 loss に「枯れ牌ペナルティ項」を追加 (λ=0.5)',
        'detail':      '枯れ牌への確率をペナルティ化。データ再生成不要。→ Type A 全体改善',
        'pipeline_rerun': False,
    },
    {
        'id':          'fix_add_paishu_feature',
        'triggers':    ['A'],
        'description': '残り枚数 (34次元) を固定特徴量に追加 (695→729次元)',
        'detail':      'extract_features を再実行してデータ再生成が必要。→ Type A の根本対策',
        'pipeline_rerun': True,
    },
    {
        'id':          'fix_riichi_loss_weight',
        'triggers':    ['B2', 'B3'],
        'description': 'リーチ局面サンプルの loss_weight を 2.0 に設定',
        'detail':      'データ再生成不要。リーチ局面の学習を強化。→ B2/B3 改善',
        'pipeline_rerun': False,
    },
    {
        'id':          'fix_late_game_weight',
        'triggers':    ['A'],
        'description': '終盤 (巡目15+) サンプルの loss_weight を 1.5 に設定',
        'detail':      'データ再生成不要。終盤の学習を強化。→ Type A 終盤改善',
        'pipeline_rerun': False,
    },
    {
        'id':          'fix_wait_logit_penalty',
        'triggers':    ['B3'],
        'description': 'リーチ後ツモ切り牌への wait_logits 矛盾ペナルティ追加',
        'detail':      'データ再生成不要。wait head の矛盾を直接ペナルティ化。→ B3 改善',
        'pipeline_rerun': False,
    },
]


def load_logs(paths):
    records = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def bucket_turn(turn):
    if turn <= 9:  return '序盤(0-9)'
    if turn <= 14: return '中盤(10-14)'
    return            '終盤(15+)'


def analyze(records, min_records):
    stats = {}
    active_types = set()

    for checker_key, threshold in THRESHOLD.items():
        violations = [r for r in records if r.get(checker_key, {}).get('score', 0) > threshold]
        if not violations:
            stats[checker_key] = {'count': 0, 'groups': []}
            continue

        active_types.add(checker_key)
        scores = [r[checker_key]['score'] for r in violations]
        avg = sum(scores) / len(scores)

        # グルーピング: (turn_bucket, riichi)
        groups = defaultdict(list)
        for r in violations:
            tb     = bucket_turn(r.get('turn', 0))
            riichi = r.get('B2', {}).get('score', 0) > 0 or any(
                r.get(checker_key, {}).get('flagged', []))
            groups[(tb,)].append(r[checker_key]['score'])

        group_list = []
        for key, sc_list in sorted(groups.items(), key=lambda x: -sum(x[1])/len(x[1])):
            if len(sc_list) < 1:
                continue
            group_list.append({
                'label': key[0],
                'avg':   sum(sc_list) / len(sc_list),
                'n':     len(sc_list),
            })

        stats[checker_key] = {
            'count': len(violations),
            'avg':   avg,
            'groups': group_list,
        }

    return stats, active_types


def stars(avg, thres):
    ratio = avg / thres
    if ratio >= 4: return '★★★'
    if ratio >= 2: return '★★'
    return              '★'


def print_report(records, stats, active_types):
    n = len(records)
    print()
    print('=' * 50)
    print('  ハーネス弱点レポート')
    print(f'  総ログ数: {n} 件')
    print('=' * 50)

    checker_meta = {
        'A':  ('Type A', '枯れ牌ハード制約違反'),
        'B2': ('Type B2', 'リーチ後エントロピー未収束'),
        'B3': ('Type B3', 'リーチ後ツモ切り待ち矛盾'),
    }

    rankings = []
    for ck, (label, desc) in checker_meta.items():
        s = stats.get(ck, {'count': 0})
        if s['count'] == 0:
            print(f'\n▼ {label}: {desc}')
            print(f'  (違反なし)')
            continue
        print(f'\n▼ {label}: {desc}')
        print(f'  違反件数: {s["count"]} / {len(records)} 件  avg_score={s["avg"]:.3f}')
        for g in s['groups'][:4]:
            st = stars(g['avg'], THRESHOLD[ck])
            print(f'    {g["label"]:<14}  avg={g["avg"]:.3f}  n={g["n"]}  {st}')
        rankings.append((ck, s['avg'], s['count']))

    rankings.sort(key=lambda x: -x[1])
    print()
    print('=' * 50)
    print('最重要改善ターゲット:')
    for rank, (ck, avg, cnt) in enumerate(rankings[:3], 1):
        _, desc = checker_meta[ck]
        print(f'  {rank}位: {ck} {desc}  (avg={avg:.3f}, n={cnt})')
    print('=' * 50)


def suggest_fixes(active_types):
    candidates = []
    seen = set()
    for fix in REMEDIATION_MAP:
        if any(t in active_types for t in fix['triggers']):
            if fix['id'] not in seen:
                candidates.append(fix)
                seen.add(fix['id'])
    return candidates


def interactive_menu(candidates):
    if not candidates:
        print('\n対応策の候補がありません。')
        return []

    print()
    print('対応策候補:')
    for i, fix in enumerate(candidates, 1):
        rerun = ' ※データ再生成必要' if fix['pipeline_rerun'] else ''
        print(f'  [{i}] {fix["id"]}{rerun}')
        print(f'      → {fix["description"]}')
        print(f'         {fix["detail"]}')

    print()
    print('実行する対応策を選んでください')
    print('  例: 1,3  または  all  または  なし (Enterでスキップ)')
    choice = input('> ').strip()

    if not choice or choice == 'なし':
        return []
    if choice == 'all':
        return candidates

    selected = []
    for part in choice.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(candidates):
                selected.append(candidates[idx])
            else:
                print(f'  警告: {part} は範囲外です')
    return selected


def run_fixes(selected, base_version=38, target_version=39):
    if not selected:
        print('(対応策なし。終了します。)')
        return

    fix_ids = [f['id'] for f in selected]
    has_rerun = any(f['pipeline_rerun'] for f in selected)

    print()
    print('以下の対応策を適用します:')
    for f in selected:
        print(f'  - {f["id"]}')
    if has_rerun:
        print()
        print('  ※ データパイプライン再実行が必要な fix が含まれています。')
        print('     実行には時間がかかります。続行しますか？ [y/N]')
        if input('> ').strip().lower() != 'y':
            print('キャンセルしました。')
            return

    script = Path(__file__).parent / 'harness_fix.py'
    cmd = [
        sys.executable, str(script),
        '--fixes', *fix_ids,
        '--base-version', str(base_version),
        '--target-version', str(target_version),
    ]
    print()
    print(f'実行: {" ".join(cmd)}')
    print()
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description='ハーネスフィードバックレポート')
    parser.add_argument('logs', nargs='+', help='feedback_log.jsonl (glob可)')
    parser.add_argument('--min-records', type=int, default=3,
                        help='集計に必要な最小違反件数 (デフォルト: 3)')
    parser.add_argument('--base-version',   type=int, default=38)
    parser.add_argument('--target-version', type=int, default=39)
    args = parser.parse_args()

    # glob展開
    paths = []
    for pattern in args.logs:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif Path(pattern).exists():
            paths.append(pattern)
    if not paths:
        print(f'ERROR: ログファイルが見つかりません: {args.logs}')
        sys.exit(1)

    print(f'ログファイル: {paths}')
    records = load_logs(paths)
    if not records:
        print('ERROR: 有効なレコードが0件です。')
        sys.exit(1)

    stats, active_types = analyze(records, args.min_records)
    print_report(records, stats, active_types)
    candidates = suggest_fixes(active_types)
    selected   = interactive_menu(candidates)
    run_fixes(selected, args.base_version, args.target_version)


if __name__ == '__main__':
    main()
