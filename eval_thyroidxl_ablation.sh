#!/usr/bin/env bash
# 批量评估 ThyroidXL 三臂消融：test 集，per-image + per-nodule(mean)
# 依赖: 三臂训练完成 (checkpoints/thyroid/{fusion,image,clinical}/best.pt)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
ROOT="data/thyroid/thyroidxl"
CLIN="tirads,width_mm,height_mm,age,gender"

for ablation in fusion image clinical; do
  BEST="checkpoints/thyroid/${ablation}/best.pt"
  if [ ! -f "$BEST" ]; then
    echo "SKIP ${ablation}: no checkpoint"
    continue
  fi
  echo "=== EVAL ${ablation} ==="
  PYTHONIOENCODING=utf-8 "$PY" -u eval_thyroid.py \
    --weights "$BEST" --data_root "$ROOT" --split test \
    --batch_size 32 --aggregate mean \
    --clinical_columns "$CLIN"
done
echo "ALL EVALS DONE"
