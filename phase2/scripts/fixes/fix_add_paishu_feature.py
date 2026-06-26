"""
fix_add_paishu_feature: 残り枚数 (34次元) を固定特徴量に追加 (695→729次元)

- データパイプライン再実行が必要 (extract_features.js の修正が別途必要)。
- feature_offsets.py の HI_TOTAL を 695 → 729 に更新する。
- 学習スクリプトの fixed_dim を 695 → 729 に変更する。
"""

OLD_DIM = 695
NEW_DIM = 729  # 695 + 34 (paishu)


def patch_train_script(code: str, base_ver: int, target_ver: int) -> str:
    # fixed_dim の変更
    code = code.replace(
        f'"fixed_dim": {OLD_DIM}',
        f'"fixed_dim": {NEW_DIM}',
    )
    code = code.replace(
        f'fixed_dim={OLD_DIM}',
        f'fixed_dim={NEW_DIM}',
    )

    # コメントに変更理由を記録
    code = code.replace(
        'CONFIG = {',
        f'# fix_add_paishu_feature: fixed_dim {OLD_DIM} -> {NEW_DIM} (残り枚数34次元追加)\nCONFIG = {{',
        1
    )

    return code


def patch_feature_offsets(offsets_path):
    """
    feature_offsets.py の HI_TOTAL を更新し、HI_PAISHU を追加する。
    """
    with open(offsets_path, encoding='utf-8') as f:
        code = f.read()

    if 'HI_PAISHU' in code:
        print('  feature_offsets.py: HI_PAISHU は既に定義済みです。')
        return

    code = code.replace(
        f'HI_TOTAL={OLD_DIM}',
        f'HI_PAISHU={OLD_DIM}\nHI_TOTAL={NEW_DIM}',
    )

    with open(offsets_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f'  feature_offsets.py 更新: HI_TOTAL {OLD_DIM} → {NEW_DIM}, HI_PAISHU={OLD_DIM} 追加')
