#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patient-level aggregation sensitivity: mean vs max vs attention-weighted.

对 ThyroidXL 每帧预测，比较三种结节级聚合策略的 AUC：
  - mean      : 帧概率平均（主报告口径）
  - max       : 帧概率最大
  - attention : 帧概率的指数加权 softmax（模拟 attention MIL 的"聚焦高置信帧"）

结论：聚合策略显著影响 AUC → 支持与 RCAF（attention MIL + mask）不可直接比较的论证。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
sys.path.insert(0, str(PROJ))
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier
from train_thyroid import bootstrap_auc

CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]
MODELS = ["fusion", "image"]


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    model = ThyroidClassifier(
        encoder_name=mc["encoder_name"], pretrained=mc["pretrained"],
        vis_dim=mc["vis_dim"], clinical_feature_dim=mc["clinical_feature_dim"],
        clinical_hidden_dim=mc["clinical_hidden_dim"],
        clinical_output_dim=mc["clinical_output_dim"],
        head_hidden=mc["head_hidden"], head_dropout=mc["head_dropout"],
        use_image=mc["use_image"], use_clinical=mc["use_clinical"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def frame_predictions(model, ds, device, batch_size=32):
    """返回帧级 (probs, labels, pids)"""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            if model.use_image:
                images = batch["image"].to(device)
                o = model(images=images, clinical=clinical)
            else:
                o = model(clinical=clinical)
            probs.append(o["probs"].cpu().numpy().flatten())
            labels.append(batch["label"].numpy().flatten())
            pids.append(list(batch["patient_id"]))
    return (np.concatenate(probs), np.concatenate(labels).astype(int),
            np.concatenate(pids))


def aggregate(groups, p_all, l_all, method, temp=5.0):
    """groups: {pid: [frame_idx,...]}"""
    agg_p, agg_l = [], []
    for pid, idxs in groups.items():
        ps = p_all[idxs]
        l = l_all[idxs[0]]  # 同结节标签一致
        if method == "mean":
            ap = ps.mean()
        elif method == "max":
            ap = ps.max()
        elif method == "attention":
            w = np.exp(temp * ps)
            w = w / w.sum()
            ap = (w * ps).sum()
        agg_p.append(ap)
        agg_l.append(l)
    return np.array(agg_p), np.array(agg_l)


def main():
    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                        image_size=224, num_clinical_features=5, clinical_columns=CLIN,
                        mock=False, split_by_group=True, seed=42)

    results = {}
    for ab in MODELS:
        wp = PROJ / "checkpoints" / "thyroid" / ab / "best.pt"
        if not wp.exists():
            print(f"SKIP {ab}")
            continue
        model = load_model(wp, device)
        p_all, l_all, pid_all = frame_predictions(model, ds, device)

        groups = {}
        for i, pid in enumerate(pid_all):
            groups.setdefault(pid, []).append(i)

        print(f"\n=== {ab} (ThyroidXL test, {len(groups)} nodules) ===")
        results[ab] = {}
        for method in ["mean", "max", "attention"]:
            p, l = aggregate(groups, p_all, l_all, method)
            auc = roc_auc_score(l, p) if len(np.unique(l)) > 1 else 0
            auprc = average_precision_score(l, p) if len(np.unique(l)) > 1 else 0
            m, lo, hi = bootstrap_auc(l, p)
            print(f"  {method:10s}: AUC {auc:.4f} (95% CI {lo:.4f}-{hi:.4f}) | AUPRC {auprc:.4f}")
            results[ab][method] = {"auc": float(auc), "auc_ci": [float(lo), float(hi)],
                                   "auprc": float(auprc)}

    out = PROJ / "checkpoints" / "thyroid" / "mil_sensitivity.json"
    out.write_text(__import__("json").dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
