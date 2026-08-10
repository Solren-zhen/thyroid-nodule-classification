#!/usr/bin/env bash
# 联合训练重跑（H1 修复：独立 test 集，无泄漏）
# 训练池 = data/thyroid_multi (TN5000 train+val + TN3K trainval = 7129)
# 内部 test 独立于 training pool（750 张，thyroid_tn5000test）
# 用法: nohup bash run_joint_retrain.sh > logs/run_joint_retrain.log 2>&1 &
set -e
cd "$(dirname "$0")"
PY="C:/miniconda3/envs/lymph_yolo/python.exe"
ROOT="data/thyroid_multi"

PYTHONIOENCODING=utf-8 "$PY" -u train_thyroid.py \
  --data_root "$ROOT" --ablation image \
  --epochs 30 --batch_size 64 --workers 10 \
  --seed 42 --pos_weight 1.0 \
  > logs/run_joint_retrain.log 2>&1

echo "[$(date)] joint retrain DONE"
