#!/bin/bash
# v38 データパイプライン
#
# 変更点 (v37 比):
#   - parse_paipu.js: PASS_PON/CHI を観測事実ベースに変更（手牌不問）
#   - extract_features.js: PASS トークンを時系列に統合（44次元/トークン）
#   - add_intent_labels.py を追加（noise/retreat/push ラベル付与）
#   - prepare_v38_data.py で3行→1行結合
#
# 使い方: bash run_pipeline_v38.sh [--resume N]  (N=再開ステップ番号)
#
# ステップ:
#   1. extract_features.js    (states_v22.ndjson → hand_inference.ndjson)
#   2. add_yaku_features.py   (+21次元 役推定)
#   3. add_tenpai_features.py (+1次元 聴牌推定)
#   4. add_intent_labels.py   (+noise/retreat/push ラベル)
#   5. prepare_v38_data.py    (flat 3行 → combined 1行)
#   6. add_block_labels.py    (+ブロックラベル)
#   7. train_hand_inference_v38.py

set -e
cd "$(dirname "$0")/../.."   # mahjong-AI ルート

LOG=/mnt/c/Users/tetsu-4no/mahjong-AI/phase2/data/features/pipeline_v38.log
FEAT_DIR=/mnt/c/Users/tetsu-4no/mahjong-AI/phase2/data/features
STATES=/mnt/c/Users/tetsu-4no/mahjong-AI/phase2/data/states/states_v22.ndjson

RESUME=${2:-1}
[[ "$1" == "--resume" ]] && RESUME=$2

echo "=== v38 pipeline start $(date) ===" | tee -a "$LOG"

# ---- ステップ1: 特徴量抽出 ----
if [ "$RESUME" -le 1 ]; then
    echo "[1/7] extract_features.js $(date)" | tee -a "$LOG"
    node phase2/scripts/extract_features.js \
        --src "$STATES" \
        --dest "$FEAT_DIR" 2>&1 | tee -a "$LOG"
fi

# ---- ステップ2: 役推定特徴量追加 ----
if [ "$RESUME" -le 2 ]; then
    echo "[2/7] add_yaku_features.py $(date)" | tee -a "$LOG"
    conda run -n mahjong python phase2/scripts/add_yaku_features.py 2>&1 | tee -a "$LOG"
fi

# ---- ステップ3: 聴牌推定特徴量追加 ----
if [ "$RESUME" -le 3 ]; then
    echo "[3/7] add_tenpai_features.py $(date)" | tee -a "$LOG"
    conda run -n mahjong python phase2/scripts/add_tenpai_features.py 2>&1 | tee -a "$LOG"
fi

# ---- ステップ4: 打牌意図ラベル追加 ----
if [ "$RESUME" -le 4 ]; then
    echo "[4/7] add_intent_labels.py $(date)" | tee -a "$LOG"
    conda run -n mahjong python phase2/scripts/add_intent_labels.py \
        --src "$FEAT_DIR/hand_inference.ndjson" \
        --states "$STATES" \
        --dst "$FEAT_DIR/hand_inference.ndjson" 2>&1 | tee -a "$LOG"
fi

# ---- ステップ5: 3プレイヤー結合 ----
if [ "$RESUME" -le 5 ]; then
    echo "[5/7] prepare_v38_data.py $(date)" | tee -a "$LOG"
    conda run -n mahjong python phase2/scripts/prepare_v38_data.py \
        --src "$FEAT_DIR/hand_inference.ndjson" \
        --dst "$FEAT_DIR/hand_inference_v38_raw.ndjson" 2>&1 | tee -a "$LOG"
fi

# ---- ステップ6: ブロックラベル追加 ----
if [ "$RESUME" -le 6 ]; then
    echo "[6/7] add_block_labels.py $(date)" | tee -a "$LOG"
    conda run -n mahjong python phase2/scripts/add_block_labels.py \
        --src "$FEAT_DIR/hand_inference_v38_raw.ndjson" \
        --dst "$FEAT_DIR/hand_inference_v38.ndjson" \
        --v2 --include-tatsu 2>&1 | tee -a "$LOG"
fi

# ---- ステップ7: 学習 ----
if [ "$RESUME" -le 7 ]; then
    echo "[7/7] train_hand_inference_v38.py $(date)" | tee -a "$LOG"
    conda run -n mahjong python phase2/train/train_hand_inference_v38.py 2>&1 | tee -a "$LOG"
fi

echo "=== v38 pipeline complete $(date) ===" | tee -a "$LOG"
