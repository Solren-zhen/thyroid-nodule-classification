#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ThyroidXL 三臂消融图：image vs clinical vs fusion 的 ROC / 校准 / DCA。

结节级 mean 聚合（与 manuscript 评估协议一致）。
输入: checkpoints/thyroid/{fusion,image,clinical}/best.pt
输出: paper/figures/thyroidxl_*_{roc,calibration,dca}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 统一出版风格
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc as sk_auc

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier

FIG_DIR = PROJ / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]
ABLATIONS = ["image", "clinical", "fusion"]
COLORS = {"image": "#1f77b4", "clinical": "#2ca02c", "fusion": "#d62728"}
LABELS = {"image": "Image-only", "clinical": "Clinical-only", "fusion": "Fusion"}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(weights_path):
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
    model.eval()
    return model


def predict_nodule_mean(model, ds, device, batch_size=64):
    """返回 (probs, labels)，probs 按 patient_id 取结节内 mean。"""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            if model.use_image:
                images = batch["image"].to(device)
                outputs = model(images=images, clinical=clinical)
            else:
                outputs = model(clinical=clinical)
            probs.append(outputs["probs"].cpu().numpy().flatten())
            labels.append(batch["label"].numpy().flatten())
            pids.append(list(batch["patient_id"]))
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    pids = np.concatenate(pids)

    groups = {}
    for pid, p in zip(pids, probs):
        groups.setdefault(pid, []).append(p)
    agg_p, agg_l = [], []
    for pid, ps in groups.items():
        agg_p.append(np.mean(ps))
        agg_l.append(int(labels[np.where(pids == pid)[0][0]]))
    return np.array(agg_p), np.array(agg_l)


def reliability_curve(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins, right=False) - 1, 0, n_bins - 1)
    mp, my = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mp.append(p[mask].mean())
        my.append(y[mask].mean())
    ece = np.sum(np.abs(np.array(mp) - np.array(my)) *
                 np.array([(idx == b).sum() for b in range(n_bins) if (idx == b).sum() > 0])) / len(y)
    return np.array(mp), np.array(my), ece


def main():
    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                        image_size=224, num_clinical_features=5,
                        clinical_columns=CLIN, mock=False,
                        split_by_group=True, seed=42)
    print(f"test: {len(ds)} 图")

    results = {}
    for ab in ABLATIONS:
        wp = PROJ / "checkpoints" / "thyroid" / ab / "best.pt"
        if not wp.exists():
            print(f"SKIP {ab}: {wp} 不存在")
            continue
        model = load_model(wp).to(device)
        p, y = predict_nodule_mean(model, ds, device)
        fpr, tpr, _ = roc_curve(y, p)
        auc = sk_auc(fpr, tpr)
        mp, my, ece = reliability_curve(y, p)
        results[ab] = {"fpr": fpr, "tpr": tpr, "auc": auc, "mp": mp, "my": my,
                       "ece": ece, "y": y, "p": p}
        print(f"{ab}: nodule AUC {auc:.4f} (n={len(y)}, ECE {ece:.4f})")

    if not results:
        print("无可用模型，退出")
        return

    # ---- ROC ----
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for ab, r in results.items():
        ax.plot(r["fpr"], r["tpr"], lw=2, color=COLORS[ab],
                label=f"{LABELS[ab]} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title("ThyroidXL test: ablation ROC (nodule-level)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thyroidxl_ablation_roc.png", dpi=300)
    plt.close(fig)
    print("saved thyroidxl_ablation_roc.png")

    # ---- Calibration ----
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    for ab, r in results.items():
        ax.plot(r["mp"], r["my"], "o-", color=COLORS[ab], lw=2, ms=5,
                label=f"{LABELS[ab]} (ECE = {r['ece']:.3f})")
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_title("ThyroidXL test: calibration", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thyroidxl_ablation_calibration.png", dpi=300)
    plt.close(fig)
    print("saved thyroidxl_ablation_calibration.png")

    # ---- DCA ----
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for ab, r in results.items():
        y, p = r["y"], r["p"]
        n = len(y)
        pts = np.arange(0.05, 0.96, 0.05)
        nbs = []
        for t in pts:
            tp = int(((p >= t) & (y == 1)).sum())
            fp = int(((p >= t) & (y == 0)).sum())
            nbs.append(tp / n - fp / n * (t / (1 - t)))
        ax.plot(pts, nbs, lw=2, color=COLORS[ab], label=LABELS[ab])
    ax.plot([0, 1], [0, 0], "k:", lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=12)
    ax.set_ylabel("Net benefit", fontsize=12)
    ax.set_title("ThyroidXL test: decision curves", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thyroidxl_ablation_dca.png", dpi=300)
    plt.close(fig)
    print("saved thyroidxl_ablation_dca.png")

    print("ALL THYROIDXL FUSION FIGURES DONE")


if __name__ == "__main__":
    main()
