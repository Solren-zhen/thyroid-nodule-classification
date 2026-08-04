#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thy-Wise 外部验证评估（第三外部队列，按结节聚合）。

Thy-Wise 一个结节多张图（均值 ~7 张，最多 49 张），按图像统计会过度加权
多图结节。本脚本同时报告：
  - per-image（与 eval_thyroid 口径一致）
  - per-nodule：组内概率 max / mean 聚合 + 多数投票（threshold=0.5）

用法：
  python eval_thywise.py --weights checkpoints/thyroid/image/best.pt \
      --data_root data/thyroid/thywise --batch_size 64 --workers 6
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

from thyroid.models.thyroid import ThyroidClassifier  # noqa: E402
from thyroid.data.thyroid_dataset import ThyroidDataset  # noqa: E402
from train_thyroid import compute_metrics, get_device  # noqa: E402


def group_by_nodule(probs: np.ndarray, labels: np.ndarray, pids: list):
    """按 patient_id(=结节) 聚合。返回 dict{pid: (probs[], label)}。"""
    groups = {}
    for p, y, pr in zip(pids, labels, probs):
        groups.setdefault(p, []).append((y, pr))
    return groups


def aggregate(groups, method: str):
    """method: max / mean / majority"""
    ys, prs = [], []
    for pid, items in groups.items():
        labs = [it[0] for it in items]
        probs = [it[1] for it in items]
        # 结节级真值：该结节多数图片标签（同结节同标签，保守校验用）
        y = 1 if np.mean(labs) >= 0.5 else 0
        if method == "max":
            pr = max(probs)
        elif method == "mean":
            pr = float(np.mean(probs))
        else:  # majority: 概率投票 + 阈值 0.5
            pr = float(np.mean([p >= 0.5 for p in probs]))
        ys.append(y)
        prs.append(pr)
    return np.array(ys, dtype=int), np.array(prs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_root", default="data/thyroid/thywise")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    weights = Path(args.weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    print(f"[model] encoder={mc['encoder_name']} use_image={mc['use_image']} "
          f"use_clinical={mc['use_clinical']} clinical_dim={mc['clinical_feature_dim']}")

    # 数据集（预处理与训练一致）
    ds = ThyroidDataset(args.data_root, split="test", image_size=224,
                        mock=False, split_by_group=True, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    print(f"[data] {len(ds)} 图像 / 外部队列")

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

    # patient_id 从 manifest 按 loader 顺序读取（split=test 全部保留，顺序一致）
    pids = []
    with open(Path(args.data_root) / "manifest.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("split") or "").strip().lower() == "test":
                pids.append(row["patient_id"])

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            if model.use_image:
                outputs = model(images=batch["image"].to(device), clinical=clinical)
            else:
                outputs = model(clinical=clinical)
            all_probs.append(outputs["probs"].cpu().numpy().flatten())
            all_labels.append(batch["label"].numpy().flatten())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    assert len(probs) == len(pids), f"len mismatch {len(probs)} vs {len(pids)}"

    def show(tag, m, n, pos_rate):
        print(f"\n--- {tag} (n={n}, 恶性率 {pos_rate:.1%}) ---")
        print(f"AUC {m['auc']:.4f} (95%CI {m['auc_ci'][0]:.4f}-{m['auc_ci'][1]:.4f}) | "
              f"Sens {m['sensitivity']:.4f} Spec {m['specificity']:.4f} | "
              f"ACC {m['acc']:.4f} | ECE {m['ece']:.4f}")

    show("per-image", compute_metrics(labels, probs), len(labels), labels.mean())

    groups = group_by_nodule(probs, labels, pids)
    print(f"\n[结节数] {len(groups)}")
    results = {"per_image": len(labels), "per_nodule": len(groups)}
    for method in ("max", "mean", "majority"):
        ys, prs = aggregate(groups, method)
        m = compute_metrics(ys, prs)
        show(f"per-nodule ({method})", m, len(ys), ys.mean())
        results[f"nodule_{method}"] = {
            "auc": m["auc"], "auc_ci": list(m["auc_ci"]),
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "acc": m["acc"], "ece": m["ece"], "n": int(len(ys)),
            "pos_rate": float(ys.mean()),
        }
    out = weights.parent / "eval_thywise.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    sys.exit(main())
