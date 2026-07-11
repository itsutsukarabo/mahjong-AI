"""
ハーネス修正実行: harness_report.py から選択された対応策を適用し
train_hand_inference_v{target}.py を生成する。

使い方:
  C:\ml\venv\Scripts\python.exe phase2/scripts/harness_fix.py \
      --fixes fix_impossible_tile_penalty fix_riichi_loss_weight \
      --base-version 38 --target-version 39
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import importlib.util
import shutil
from pathlib import Path

FIXES_DIR  = Path(__file__).parent / 'fixes'
TRAIN_DIR  = Path(__file__).parent.parent / 'train'
MODELS_DIR = Path(__file__).parent.parent / 'models' / 'hand_inference'

AVAILABLE_FIXES = [
    'fix_impossible_tile_penalty',
    'fix_add_paishu_feature',
    'fix_riichi_loss_weight',
    'fix_late_game_weight',
    'fix_wait_logit_penalty',
    'fix_furiten_penalty',
    'fix_soft_f1_eval',
]


def load_fix(fix_id):
    path = FIXES_DIR / f'{fix_id}.py'
    if not path.exists():
        raise FileNotFoundError(f'Fix スクリプトが見つかりません: {path}')
    spec = importlib.util.spec_from_file_location(fix_id, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_train_script(base_version, target_version, fix_ids):
    """v{base} の学習スクリプトをベースに fix を適用して v{target} を生成する。"""
    src = TRAIN_DIR / f'train_hand_inference_v{base_version}.py'
    dst = TRAIN_DIR / f'train_hand_inference_v{target_version}.py'

    if not src.exists():
        print(f'ERROR: ベーススクリプトが見つかりません: {src}')
        return False

    with open(src, encoding='utf-8') as f:
        code = f.read()

    print(f'ベーススクリプト: {src.name}')

    # 各 fix モジュールの patch_train_script() を順番に適用
    for fix_id in fix_ids:
        try:
            mod = load_fix(fix_id)
            if hasattr(mod, 'patch_train_script'):
                code = mod.patch_train_script(code, base_version, target_version)
                print(f'  [適用] {fix_id}')
            else:
                print(f'  [スキップ] {fix_id}: patch_train_script なし')
        except Exception as e:
            print(f'  [ERROR] {fix_id}: {e}')
            return False

    # バージョン番号を書き換え
    code = code.replace(f'v{base_version}', f'v{target_version}')

    with open(dst, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f'\n生成完了: {dst}')
    return True


def main():
    parser = argparse.ArgumentParser(description='ハーネス修正実行')
    parser.add_argument('--fixes', nargs='+', required=True,
                        choices=AVAILABLE_FIXES, metavar='FIX',
                        help=f'適用する対応策 (選択肢: {", ".join(AVAILABLE_FIXES)})')
    parser.add_argument('--base-version',   type=int, default=38)
    parser.add_argument('--target-version', type=int, default=39)
    args = parser.parse_args()

    print('=' * 50)
    print(f'  harness_fix: v{args.base_version} → v{args.target_version}')
    print(f'  適用 fix: {", ".join(args.fixes)}')
    print('=' * 50)

    # パイプライン再実行が必要かチェック
    pipeline_fixes = {'fix_add_paishu_feature'}
    needs_pipeline = any(f in pipeline_fixes for f in args.fixes)
    if needs_pipeline:
        print()
        print('  ※ fix_add_paishu_feature はデータパイプライン再実行が必要です。')
        print('     特徴量次元が変わるため extract_features を再実行してください:')
        print()
        print('     node phase2/scripts/extract_features.js \\')
        print(f'         --output phase2/data/hand_inference_v{args.target_version}.ndjson')
        print()

    # 学習スクリプト生成
    ok = generate_train_script(args.base_version, args.target_version, args.fixes)
    if not ok:
        sys.exit(1)

    # モデル出力ディレクトリ作成
    model_dir = MODELS_DIR / f'v{args.target_version}'
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f'モデル出力先: {model_dir}')

    print()
    print('次のステップ:')
    if needs_pipeline:
        print('  1. データパイプライン再実行 (上記コマンド参照)')
        print('  2. 以下のコマンドで学習開始:')
        n = 2
    else:
        n = 1

    train_cmd = (
        f'C:\\ml\\venv\\Scripts\\python.exe '
        f'phase2/train/train_hand_inference_v{args.target_version}.py'
    )
    print(f'  {n}. {train_cmd}')
    print()
    print('学習を今すぐ開始しますか？ [y/N]')
    choice = input('> ').strip().lower()
    if choice == 'y':
        import subprocess
        subprocess.run([sys.executable,
                        str(TRAIN_DIR / f'train_hand_inference_v{args.target_version}.py')],
                       check=False)
    else:
        print('(スキップ。準備ができたら上記コマンドを実行してください。)')


if __name__ == '__main__':
    main()
