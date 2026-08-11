#!/usr/bin/env bash
# nohup 后台串行训练剩余 seed（123, 2024），不依赖 Claude 任务句柄。
# 用法: nohup bash run_remaining_seeds.sh > logs/run_remaining_seeds.log 2>&1 &
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
ROOT="data/thyroid/thyroidxl"
CLIN="tirads,width_mm,height_mm,age,gender"

for seed in 123 2024; do
  DIR="checkpoints/thyroid/fusion_seed${seed}"
  LOG="logs/train_fusion_seed${seed}_pw035.log"
  if [ -f "$DIR/best.pt" ] && grep -q "Done:" "$LOG" 2>/dev/null; then
    echo "[$(date)] SKIP seed ${seed}: already done"
    continue
  fi
  echo "[$(date)] === TRAIN fusion seed ${seed} ==="
  rm -f checkpoints/thyroid/fusion/best.pt checkpoints/thyroid/fusion/last.pt checkpoints/thyroid/fusion/metrics.csv
  PYTHONIOENCODING=utf-8 "$PY" -u train_thyroid.py \
    --data_root "$ROOT" --ablation fusion \
    --epochs 30 --batch_size 32 --workers 4 \
    --clinical_columns "$CLIN" --seed "$seed" --pos_weight 0.35 \
    > "$LOG" 2>&1
  mkdir -p "$DIR"
  cp -f checkpoints/thyroid/fusion/best.pt "$DIR/best.pt"
  cp -f checkpoints/thyroid/fusion/last.pt "$DIR/last.pt"
  cp -f checkpoints/thyroid/fusion/metrics.csv "$DIR/metrics.csv"
  echo "[$(date)] === DONE seed ${seed} ==="
done
echo "[$(date)] ALL REMAINING SEEDS DONE"
