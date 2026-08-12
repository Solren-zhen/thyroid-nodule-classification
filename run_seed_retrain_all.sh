#!/usr/bin/env bash
# 论文完善方案 §2.1 实验重跑编排
# track A（TN5000-only 与 joint）各补 seed 123/2024（共 4 次）；fusion seed 123/2024 改 pos_weight=1.0 重训（共 2 次）。
# 用法: nohup bash run_seed_retrain_all.sh > logs/run_seed_retrain_all.log 2>&1 &
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-C:/miniconda3/envs/lymph_yolo/python.exe}"
CKPT="checkpoints/thyroid"
mkdir -p "$CKPT/_pre_seedretrain_backup"

# ---- Pre-flight: 快照当前生产 checkpoint（必须保留，训练会覆盖 image/ 与 fusion/）----
echo "[$(date)] 快照生产 checkpoint..."
cp -f "$CKPT/image/best.pt"        "$CKPT/_pre_seedretrain_backup/image_best_thyroidxl_imageonly.pt"
cp -f "$CKPT/image/metrics.csv"     "$CKPT/_pre_seedretrain_backup/image_metrics_thyroidxl_imageonly.csv"
cp -f "$CKPT/image_backup/best_joint_20260810_fixed.pt" "$CKPT/_pre_seedretrain_backup/best_joint_seed42_fixed.pt" 2>/dev/null || true
cp -f "$CKPT/image/best_tn5000.pt"  "$CKPT/_pre_seedretrain_backup/best_tn5000_seed42.pt" 2>/dev/null || true
cp -f "$CKPT/fusion/best.pt"        "$CKPT/_pre_seedretrain_backup/fusion_best_before_retrain.pt" 2>/dev/null || true
echo "[$(date)] 快照完成:"
ls -la "$CKPT/_pre_seedretrain_backup/"

run_trackA () {
  # $1=label  $2=data_root  $3=seed  $4=log
  echo "[$(date)] === TRAIN $1 seed $3 (pos_weight=1.0) ==="
  PYTHONIOENCODING=utf-8 "$PY" -u train_thyroid.py \
    --data_root "$2" --ablation image \
    --epochs 30 --batch_size 64 --workers 10 \
    --seed "$3" --pos_weight 1.0 > "$4" 2>&1
  mkdir -p "$CKPT/${1}_seed$3"
  cp -f "$CKPT/image/best.pt"   "$CKPT/${1}_seed$3/best.pt"
  cp -f "$CKPT/image/metrics.csv" "$CKPT/${1}_seed$3/metrics.csv"
  echo "[$(date)] === DONE $1 seed $3 ==="
}

run_fusion () {
  # $1=seed  $2=log
  echo "[$(date)] === TRAIN fusion seed $1 (pos_weight=1.0) ==="
  PYTHONIOENCODING=utf-8 "$PY" -u train_thyroid.py \
    --data_root data/thyroid/thyroidxl --ablation fusion \
    --epochs 30 --batch_size 32 --workers 4 \
    --clinical_columns "tirads,width_mm,height_mm,age,gender" \
    --seed "$1" --pos_weight 1.0 > "$2" 2>&1
  mkdir -p "$CKPT/fusion_seed${1}_pw1.0"
  cp -f "$CKPT/fusion/best.pt"   "$CKPT/fusion_seed${1}_pw1.0/best.pt"
  cp -f "$CKPT/fusion/metrics.csv" "$CKPT/fusion_seed${1}_pw1.0/metrics.csv"
  echo "[$(date)] === DONE fusion seed $1 ==="
}

# ---- track A ----
run_trackA tn5000 data/thyroid       123  logs/train_tn5000_seed123.log
run_trackA tn5000 data/thyroid       2024 logs/train_tn5000_seed2024.log
run_trackA joint  data/thyroid_multi 123  logs/train_joint_seed123.log
run_trackA joint  data/thyroid_multi 2024 logs/train_joint_seed2024.log

# ---- fusion 重跑（pos_weight=1.0，与 seed 42 锚点同协议）----
run_fusion 123  logs/train_fusion_seed123_pw1.0.log
run_fusion 2024 logs/train_fusion_seed2024_pw1.0.log

# ---- 恢复生产 checkpoint：image/best.pt 必须回到 ThyroidXL image-only（fig5-7/Table3/fig9 依赖）----
echo "[$(date)] 恢复 image/best.pt → ThyroidXL image-only..."
echo "[$(date)] restore fusion/best.pt -> seed42 pw1.0 anchor"
cp -f "$CKPT/fusion_backup/best_posw1.0_seed42_20260809.pt" "$CKPT/fusion/best.pt"
cp -f "$CKPT/_pre_seedretrain_backup/image_best_thyroidxl_imageonly.pt" "$CKPT/image/best.pt"
cp -f "$CKPT/_pre_seedretrain_backup/image_metrics_thyroidxl_imageonly.csv" "$CKPT/image/metrics.csv"
echo "[$(date)] 恢复完成。"

echo "[$(date)] ALL SEED RETRAIN DONE"
