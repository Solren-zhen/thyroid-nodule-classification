#!/usr/bin/env bash
# 单独跑 seed-2024 fusion 训练（电脑冷却后由定时任务触发）
# 用法: nohup bash run_seed2024.sh > logs/run_seed2024.log 2>&1 &
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
ROOT="data/thyroid/thyroidxl"
CLIN="tirads,width_mm,height_mm,age,gender"
SEED=2024
DIR="checkpoints/thyroid/fusion_seed${SEED}"
LOG="logs/train_fusion_seed${SEED}_pw035.log"

echo "[$(date)] === TRAIN fusion seed ${SEED} ==="
rm -f checkpoints/thyroid/fusion/best.pt checkpoints/thyroid/fusion/last.pt checkpoints/thyroid/fusion/metrics.csv
PYTHONIOENCODING=utf-8 "$PY" -u train_thyroid.py \
  --data_root "$ROOT" --ablation fusion \
  --epochs 30 --batch_size 32 --workers 4 \
  --clinical_columns "$CLIN" --seed "$SEED" --pos_weight 0.35 \
  > "$LOG" 2>&1
mkdir -p "$DIR"
cp -f checkpoints/thyroid/fusion/best.pt "$DIR/best.pt"
cp -f checkpoints/thyroid/fusion/last.pt "$DIR/last.pt"
cp -f checkpoints/thyroid/fusion/metrics.csv "$DIR/metrics.csv"
echo "[$(date)] === DONE seed ${SEED} ==="
