#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figures: ROC overlay, calibration reliability, DCA, confusion."""

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
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier

FIG_DIR = PROJ / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


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


def predict(model, ds, device, batch_size=64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels = [], []
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
    return np.concatenate(probs), np.concatenate(labels).astype(int)


def reliability_curve(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_p, mean_y, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_p.append(p[mask].mean())
        mean_y.append(y[mask].mean())
        counts.append(mask.sum())
    ece = np.sum(np.abs(np.array(mean_p) - np.array(mean_y)) * np.array(counts)) / len(y)
    return np.array(mean_p), np.array(mean_y), ece


def dca_refs(pt, pos_rate):
    all_treat = pos_rate - (1 - pos_rate) * pt / (1 - pt)
    return np.clip(all_treat, None, None)


def main():
    device = get_device()
    # Single-dataset (TN5000-only) and joint (TN5000 + TN3K train) models, seed 42
    single_weights = PROJ / "checkpoints" / "thyroid" / "single_tn3k_eval" / "best.pt"
    joint_weights = PROJ / "checkpoints" / "thyroid" / "_pre_seedretrain_backup" / "best_joint_seed42_fixed.pt"
    single = load_model(single_weights).to(device)
    joint = load_model(joint_weights).to(device)
    print("models loaded")

    ds_in = ThyroidDataset(str(PROJ / "data" / "thyroid_tn5000test"), split="test",
                           image_size=224, mock=False, split_by_group=True, seed=42)
    ds_t3k_test = ThyroidDataset(str(PROJ / "data" / "thyroid_tn3ktest"), split="test",
                                 image_size=224, mock=False, split_by_group=True, seed=42)
    ds_t3k_full = ThyroidDataset(str(PROJ / "data" / "thyroid" / "tn3k"), split="all",
                                 image_size=224, mock=False, split_by_group=True, seed=42)
    p_s_in, y_s_in = predict(single, ds_in, device)
    p_s_ext, y_s_ext = predict(single, ds_t3k_test, device)
    p_s_full, y_s_full = predict(single, ds_t3k_full, device)
    p_j_in, y_j_in = predict(joint, ds_in, device)
    p_j_ext, y_j_ext = predict(joint, ds_t3k_test, device)
    print(f"single: internal n={len(y_s_in)} (pos {y_s_in.mean():.3f}), "
          f"TN3K official n={len(y_s_ext)} (pos {y_s_ext.mean():.3f}), "
          f"TN3K full n={len(y_s_full)} (pos {y_s_full.mean():.3f})")
    print(f"joint:  internal n={len(y_j_in)} (pos {y_j_in.mean():.3f}), "
          f"TN3K official n={len(y_j_ext)} (pos {y_j_ext.mean():.3f})")

    # ---- Fig 2: ROC overlay (domain-shift drop + joint recovery) ----
    fpr_si, tpr_si, _ = roc_curve(y_s_in, p_s_in)
    fpr_sf, tpr_sf, _ = roc_curve(y_s_full, p_s_full)
    fpr_ji, tpr_ji, _ = roc_curve(y_j_in, p_j_in)
    fpr_je, tpr_je, _ = roc_curve(y_j_ext, p_j_ext)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.plot(fpr_si, tpr_si, lw=2, color="#1f77b4",
            label="Single-dataset: internal test (AUC = 0.915)")
    ax.plot(fpr_sf, tpr_sf, lw=2, ls="--", color="#1f77b4",
            label="Single-dataset: external TN3K full cohort (AUC = 0.712)")
    ax.plot(fpr_ji, tpr_ji, lw=2, color="#2ca02c",
            label="Joint: internal test (AUC = 0.931)")
    ax.plot(fpr_je, tpr_je, lw=2, ls="--", color="#2ca02c",
            label="Joint: external TN3K official test (AUC = 0.813)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title("Receiver operating characteristic curves", fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2.png", dpi=300)
    plt.close(fig)
    print("fig2 saved")

    # ---- Fig 3: calibration (a) + DCA (b) ----
    mp1, my1, ece1 = reliability_curve(y_j_in, p_j_in)
    mp2, my2, ece2 = reliability_curve(y_j_ext, p_j_ext)

    def dca(y, p):
        pts = np.arange(0.05, 0.96, 0.05)
        nbs = []
        for t in pts:
            tp = int(((p >= t) & (y == 1)).sum())
            fp = int(((p >= t) & (y == 0)).sum())
            nbs.append(tp / len(y) - fp / len(y) * (t / (1 - t)))
        return pts, np.array(nbs)

    pt_ji, nb_ji = dca(y_j_in, p_j_in)
    pt_je, nb_je = dca(y_j_ext, p_j_ext)
    pt_se, nb_se = dca(y_s_ext, p_s_ext)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax1.plot(mp1, my1, "o-", color="#1f77b4", lw=2, ms=6,
             label=f"Joint internal (ECE = {ece1:.3f})")
    ax1.plot(mp2, my2, "s-", color="#d62728", lw=2, ms=6,
             label=f"Joint external TN3K (ECE = {ece2:.3f})")
    ax1.set_xlabel("Predicted probability", fontsize=12)
    ax1.set_ylabel("Observed frequency", fontsize=12)
    ax1.set_title("(a) Calibration", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax2.plot(pt_ji, nb_ji, lw=2, color="#1f77b4", label="Joint: internal test")
    ax2.plot(pt_je, nb_je, lw=2, color="#d62728",
             label="Joint: external TN3K official test")
    ax2.plot(pt_se, nb_se, lw=2, ls="--", color="#ff7f0e",
             label="Single-dataset: external TN3K official test")
    ax2.plot([0, 1], [0, 0], "k:", lw=1, label="Treat none")
    ax2.set_xlabel("Threshold probability", fontsize=12)
    ax2.set_ylabel("Net benefit", fontsize=12)
    ax2.set_title("(b) Decision curves", fontsize=13)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3.png", dpi=300)
    plt.close(fig)
    print("fig3 (calibration + DCA) saved")

    # ---- Fig 4: confusion matrices (joint model) ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (y, p, title) in zip(
        axes,
        [(y_j_in, p_j_in, "Internal test"), (y_j_ext, p_j_ext, "External TN3K")],
    ):
        yp = (p >= 0.5).astype(int)
        cm = confusion_matrix(y, yp)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Malignant"])
        ax.set_yticklabels(["Benign", "Malignant"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4.png", dpi=300)
    plt.close(fig)
    print("fig4 saved")
    print("ALL FIGURES DONE")


if __name__ == "__main__":
    main()
