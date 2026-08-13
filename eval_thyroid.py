#!/usr/bin/env python3
"""
甲状腺结节良恶性分类 — 评估脚本
================================
在验证/测试集上输出: AUC(95% CI)、ACC、敏感度、特异度、F1、ECE、混淆矩阵、决策曲线。

用法:
    python eval_thyroid.py --weights ./checkpoints/thyroid/fusion/best.pt
    python eval_thyroid.py --weights ./checkpoints/thyroid/fusion/best.pt --split test
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thyroid.models.thyroid import ThyroidClassifier
from thyroid.data.thyroid_dataset import ThyroidDataset
from train_thyroid import compute_metrics, bootstrap_auc, ece_score, get_device


def decision_curve(labels: np.ndarray, probs: np.ndarray,
                   thresholds=None) -> list:
    """决策曲线分析（DCA）：净获益 = TP/n - FP/n * (pt/(1-pt))"""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    n = len(labels)
    rows = []
    for t in thresholds:
        tp = int(((probs >= t) & (labels == 1)).sum())
        fp = int(((probs >= t) & (labels == 0)).sum())
        nb = tp / n - fp / n * (t / (1 - t))
        rows.append({"threshold": float(t), "net_benefit": float(nb)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="甲状腺分类模型评估")
    parser.add_argument("--weights", type=str, required=True, help="best.pt 路径")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "all"])
    parser.add_argument("--mode", type=str, default="real", choices=["real", "mock"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate", type=str, default="none", choices=["none", "mean", "max", "majority"],
                        help="按 patient_id 分组聚合概率（多图/结节）；mean/max/majority 同 eval_thywise")
    parser.add_argument("--tta", action="store_true", default=False,
                        help="测试时增强：5 视图（原图+水平/垂直翻转+旋转±90°）概率平均")
    parser.add_argument("--clinical_columns", type=str, default=None,
                        help="逗号分隔的临床特征列名，默认用 ACR 6 列；ThyroidXL 用 "
                             "tirads,width_mm,height_mm,age,gender")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"权重不存在: {weights}")
        sys.exit(1)

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]

    PROJECT_ROOT = Path(__file__).resolve().parent
    data_root = args.data_root or "./data/thyroid"
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)
    if args.clinical_columns:
        clinical_cols = [c.strip() for c in args.clinical_columns.split(",") if c.strip()]
        num_clin = len(clinical_cols)
    else:
        # 优先用训练时保存的列名；旧 checkpoint 无该字段时对 ThyroidXL 自动识别
        clinical_cols = mc.get("clinical_columns")
        num_clin = mc["clinical_feature_dim"]
        if clinical_cols and len(clinical_cols) != num_clin:
            clinical_cols = None
        if clinical_cols is None:
            _tcol = ["tirads", "width_mm", "height_mm", "age", "gender"]
            _mp = Path(data_root) / "manifest.csv"
            if _mp.exists():
                with open(_mp, "r", encoding="utf-8") as _f:
                    _header = next(csv.reader(_f))
                if num_clin == len(_tcol) and all(c in _header for c in _tcol):
                    clinical_cols = list(_tcol)
        if clinical_cols is None and mc.get("use_clinical", False):
            print("  警告：未指定 --clinical_columns 且 checkpoint 未记录列名；将使用默认 ACR 列（缺失值补 0）")
    if clinical_cols:
        print(f"  [clinical] 使用列: {clinical_cols}")
    ds = ThyroidDataset(data_root, split=args.split, image_size=224,
                        num_clinical_features=num_clin,
                        clinical_columns=clinical_cols,
                        mock=args.mode == "mock", split_by_group=True, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ThyroidClassifier(
        encoder_name=mc["encoder_name"], pretrained=mc["pretrained"],
        vis_dim=mc["vis_dim"], clinical_feature_dim=mc["clinical_feature_dim"],
        clinical_hidden_dim=mc["clinical_hidden_dim"],
        clinical_output_dim=mc["clinical_output_dim"],
        head_hidden=mc["head_hidden"], head_dropout=mc["head_dropout"],
        use_image=mc["use_image"], use_clinical=mc["use_clinical"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = get_device()
    model.to(device)

    all_probs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            if model.use_image:
                images = batch["image"].to(device)
                if args.tta:
                    # 测试时增强：5 视图（原图 + 水平/垂直翻转 + 旋转±90°）概率平均
                    views = [images]
                    if hasattr(torch, "flip"):
                        views.append(torch.flip(images, dims=[3]))          # 水平翻转
                        views.append(torch.flip(images, dims=[2]))          # 垂直翻转
                        views.append(torch.rot90(images, 1, dims=[2, 3]))   # 旋转 +90
                        views.append(torch.rot90(images, -1, dims=[2, 3]))  # 旋转 -90
                    probs_acc = None
                    for v in views:
                        o = model(images=v, clinical=clinical)
                        if probs_acc is None:
                            probs_acc = o["probs"]
                        else:
                            probs_acc = probs_acc + o["probs"]
                    p_batch = (probs_acc / len(views)).cpu().numpy().flatten()
                else:
                    outputs = model(images=images, clinical=clinical)
                    p_batch = outputs["probs"].cpu().numpy().flatten()
            else:
                outputs = model(clinical=clinical)
                p_batch = outputs["probs"].cpu().numpy().flatten()
            all_probs.append(p_batch)
            all_labels.append(batch["label"].numpy().flatten())
            all_pids.append(batch["patient_id"])

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    pids = np.concatenate(all_pids) if all_pids and all_pids[0] is not None else None

    # 可选：按 patient_id（结节）聚合
    if args.aggregate != "none" and pids is not None:
        groups = {}
        for pid, p in zip(pids, probs):
            groups.setdefault(pid, []).append(p)
        agg_probs, agg_labels = [], []
        for pid, ps in groups.items():
            if args.aggregate == "mean":
                ap = float(np.mean(ps))
            elif args.aggregate == "max":
                ap = float(np.max(ps))
            else:  # majority vote
                ap = float((np.array(ps) >= 0.5).mean())
            agg_probs.append(ap)
        probs = np.array(agg_probs)
        # 重新按 patient 取标签（同组标签一致，取首个）
        labels = np.array([int(labels[np.where(pids == pid)[0][0]]) for pid in groups])
        tag = f"per-nodule ({args.aggregate})"
        print(f"  [结节聚合] {tag}: {len(groups)} 结节")
    else:
        tag = "per-image"

    m = compute_metrics(labels, probs)

    print(f"\n{'='*56}")
    print(f"  甲状腺分类评估 | 权重: {weights.name} | 分割: {args.split} | {tag}")
    print(f"{'='*56}")
    print(f"  AUC        : {m['auc']:.4f} (95% CI {m['auc_ci'][0]:.4f}-{m['auc_ci'][1]:.4f})")
    print(f"  AUPRC      : {m['auprc']:.4f}")
    print(f"  ACC        : {m['acc']:.4f}")
    print(f"  Sensitivity: {m['sensitivity']:.4f}  Specificity: {m['specificity']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")
    print(f"  ECE        : {m['ece']:.4f}")
    print(f"  Brier      : {m['brier']:.4f}")
    print(f"  Confusion  : TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"{'='*56}\n")

    dca = decision_curve(labels, probs)
    out = {"metrics": {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()},
           "decision_curve": dca, "n": int(len(labels)),
           "pos_rate": float(labels.mean()), "tag": tag}
    out_path = weights.parent / f"eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()