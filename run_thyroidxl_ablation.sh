#!/usr/bin/env bash
# 串行跑 ThyroidXL 三臂消融（fusion / image / clinical）
# 每个消融 30 epochs。fusion 已跑则跳过，只跑 image/clinical。
# 用法: bash run_thyroidxl_ablation.sh
set -e
cd "$(dirname "$0")"
PY="C:/miniconda3/envs/lymph_yolo/python.exe"
ROOT="data/thyroid/thyroidxl"
CLIN="tirads,width_mm,height_mm,age,gender"

# fusion 已在独立后台任务训练；此脚本只补跑 image + clinical
for ablation in image clinical; do
  BEST="checkpoints/thyroid/${ablation}/best.pt"
  LOG="logs/train_thyroidxl_${ablation}.log"
  if [ -f "$BEST" ] && grep -q "Done:" "$LOG" 2>/dev/null; then
    echo "SKIP ${ablation}: already done ($BEST)"
    continue
  fi
  echo "=== TRAIN ${ablation} ==="
  PYTHONIOENCODING=utf-8 "$PY" -u train_thyroid.py \
    --data_root "$ROOT" --ablation "$ablation" \
    --epochs 30 --batch_size 32 --workers 4 \
    --clinical_columns "$CLIN" > "$LOG" 2>&1
  echo "=== DONE ${ablation} ==="
done
echo "ALL ABLATIONS DONE"
