#!/usr/bin/env bash
# 串行训练 fusion multi-seed（pos_weight=0.35）
# seed 42 已在独立任务训练；此脚本补跑 123 和 2024。
# 用法: bash run_fusion_multiseed.sh
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
ROOT="data/thyroid/thyroidxl"
CLIN="tirads,width_mm,height_mm,age,gender"

# 等待 seed 42 训练完成（在独立后台任务），避免争抢 fusion/ 目录与 GPU
echo "Waiting for seed 42 training to finish..."
for i in $(seq 1 60); do
  if grep -q "Done:" logs/train_fusion_seed42_pw035.log 2>/dev/null; then
    echo "seed 42 done at check $i"
    break
  fi
  sleep 60
done
# seed 42 已完成，其 checkpoint 就在 fusion/（训练脚本统一写 fusion/）
cp -f checkpoints/thyroid/fusion/best.pt checkpoints/thyroid/fusion_seed42/best.pt 2>/dev/null || mkdir -p checkpoints/thyroid/fusion_seed42 && cp -f checkpoints/thyroid/fusion/best.pt checkpoints/thyroid/fusion_seed42/best.pt

for seed in 123 2024; do
  DIR="checkpoints/thyroid/fusion_seed${seed}"
  LOG="logs/train_fusion_seed${seed}_pw035.log"
  if [ -f "$DIR/best.pt" ] && grep -q "Done:" "$LOG" 2>/dev/null; then
    echo "SKIP seed ${seed}: already done"
    continue
  fi
  echo "=== TRAIN fusion seed ${seed} ==="
  mkdir -p "$DIR"
  # 保留当前 fusion/best.pt 不被覆盖：把输出写到独立目录
  PYTHONIOENCODING=utf-8 "$PY" -u train_thyroid.py \
    --data_root "$ROOT" --ablation fusion \
    --epochs 30 --batch_size 32 --workers 4 \
    --clinical_columns "$CLIN" --seed "$seed" --pos_weight 0.35 \
    > "$LOG" 2>&1
  # 训练脚本写 checkpoints/thyroid/fusion/，复制到 seed 专属目录
  cp -f checkpoints/thyroid/fusion/best.pt "$DIR/best.pt" 2>/dev/null || true
  cp -f checkpoints/thyroid/fusion/last.pt "$DIR/last.pt" 2>/dev/null || true
  cp -f checkpoints/thyroid/fusion/metrics.csv "$DIR/metrics.csv" 2>/dev/null || true
  echo "=== DONE seed ${seed} (saved to ${DIR}) ==="
done
echo "ALL MULTI-SEED FUSION DONE"
