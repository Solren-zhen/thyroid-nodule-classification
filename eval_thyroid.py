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
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--mode", type=str, default="real", choices=["real", "mock"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
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
        clinical_cols = None
        num_clin = mc["clinical_feature_dim"]
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

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            if model.use_image:
                images = batch["image"].to(device)
                outputs = model(images=images, clinical=clinical)
            else:
                outputs = model(clinical=clinical)
            all_probs.append(outputs["probs"].cpu().numpy().flatten())
            all_labels.append(batch["label"].numpy().flatten())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    m = compute_metrics(labels, probs)

    print(f"\n{'='*56}")
    print(f"  甲状腺分类评估 | 权重: {weights.name} | 分割: {args.split}")
    print(f"{'='*56}")
    print(f"  AUC        : {m['auc']:.4f} (95% CI {m['auc_ci'][0]:.4f}-{m['auc_ci'][1]:.4f})")
    print(f"  AUPRC      : {m['auprc']:.4f}")
    print(f"  ACC        : {m['acc']:.4f}")
    print(f"  Sensitivity: {m['sensitivity']:.4f}  Specificity: {m['specificity']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")
    print(f"  ECE        : {m['ece']:.4f}")
    print(f"  Confusion  : TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"{'='*56}\n")

    dca = decision_curve(labels, probs)
    out = {"metrics": {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()},
           "decision_curve": dca, "n": int(len(labels)),
           "pos_rate": float(labels.mean())}
    out_path = weights.parent / f"eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()