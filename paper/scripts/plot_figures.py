#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figures: ROC overlay, calibration reliability, DCA, confusion."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix
from torch.utils.data import DataLoader

PROJ = Path(r"C:\Users\甄朝晖\Desktop\thyroid")
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
    # Joint 模型（TN5000+TN3K）；image/best.pt 现被 ThyroidXL image-only 占用
    weights = PROJ / "checkpoints" / "thyroid" / "image_backup" / "best_joint_20260809.pt"
    model = load_model(weights).to(device)
    print("model loaded")

    ds_in = ThyroidDataset(str(PROJ / "data" / "thyroid"), split="test", image_size=224,
                           mock=False, split_by_group=True, seed=42)
    ds_ext = ThyroidDataset(str(PROJ / "data" / "thyroid_tn3ktest"), split="test",
                            image_size=224, mock=False, split_by_group=True, seed=42)
    p_in, y_in = predict(model, ds_in, device)
    p_ext, y_ext = predict(model, ds_ext, device)
    print(f"internal n={len(y_in)} (pos {y_in.mean():.3f}), external n={len(y_ext)} (pos {y_ext.mean():.3f})")

    # ---- Fig 2: ROC overlay ----
    fpr1, tpr1, _ = roc_curve(y_in, p_in)
    auc1 = sk_auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(y_ext, p_ext)
    auc2 = sk_auc(fpr2, tpr2)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot(fpr1, tpr1, lw=2, color="#1f77b4",
            label=f"Internal test (AUC = {auc1:.3f}, 95% CI 0.961-0.984)")
    ax.plot(fpr2, tpr2, lw=2, color="#d62728",
            label=f"External TN3K (AUC = {auc2:.3f}, 95% CI 0.782-0.849)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title("Receiver operating characteristic curves", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_roc.png", dpi=300)
    plt.close(fig)
    print("fig2 saved")

    # ---- Fig 3a: calibration ----
    mp1, my1, ece1 = reliability_curve(y_in, p_in)
    mp2, my2, ece2 = reliability_curve(y_ext, p_ext)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.plot(mp1, my1, "o-", color="#1f77b4", lw=2, ms=6,
            label=f"Internal (ECE = {ece1:.3f})")
    ax.plot(mp2, my2, "s-", color="#d62728", lw=2, ms=6,
            label=f"External (ECE = {ece2:.3f})")
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_title("Calibration reliability", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_calibration.png", dpi=300)
    plt.close(fig)
    print("fig3 saved")

    # ---- Fig 3b: DCA ----
    eval_dir = PROJ / "checkpoints" / "thyroid" / "image"
    d1 = json.loads((eval_dir / "eval_tn5000_test.json").read_text(encoding="utf-8"))
    d2 = json.loads((eval_dir / "eval_tn3k_test.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    pt1 = [c["threshold"] for c in d1["decision_curve"]]
    nb1 = [c["net_benefit"] for c in d1["decision_curve"]]
    pt2 = [c["threshold"] for c in d2["decision_curve"]]
    nb2 = [c["net_benefit"] for c in d2["decision_curve"]]
    ax.plot(pt1, nb1, lw=2, color="#1f77b4", label="Internal test")
    ax.plot(pt2, nb2, lw=2, color="#d62728", label="External TN3K")
    # reference: treat none
    ax.plot([0, 1], [0, 0], "k:", lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=12)
    ax.set_ylabel("Net benefit", fontsize=12)
    ax.set_title("Decision curve analysis", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_dca.png", dpi=300)
    plt.close(fig)
    print("fig3 dca saved")

    # ---- Fig 4: confusion matrices ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (y, p, title) in zip(
        axes,
        [(y_in, p_in, "Internal test"), (y_ext, p_ext, "External TN3K")],
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
    fig.savefig(FIG_DIR / "fig4_confusion.png", dpi=300)
    plt.close(fig)
    print("fig4 saved")
    print("ALL FIGURES DONE")


if __name__ == "__main__":
    main()
