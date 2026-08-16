#!/usr/bin/env python3
"""ThyroidXL model-comparison figures: ROC / calibration / DCA.

Models: image-only, clinical-only, image + TI-RADS (feature-selected best),
and full five-feature fusion (prespecified primary). Nodule-level mean
aggregation (same protocol as the manuscript).

Outputs: paper/figures/fig5.png (ROC), fig6.png (calibration), fig7.png (DCA).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
sys.path.insert(0, str(PROJ))
from data.thyroid_dataset import ThyroidDataset
from models.thyroid import ThyroidClassifier

FIG_DIR = PROJ / "paper" / "figures"
CLIN5 = ["tirads", "width_mm", "height_mm", "age", "gender"]
MODELS = [
    {"key": "image",    "ckpt": "checkpoints/thyroid/image/best.pt",                "cols": CLIN5, "label": "Image-only",      "color": "#0072B2", "ls": "-",  "lw": 2},
    {"key": "clinical", "ckpt": "checkpoints/thyroid/clinical/best.pt",             "cols": CLIN5, "label": "Clinical-only",  "color": "#E69F00", "ls": "-",  "lw": 2},
    {"key": "tirads",   "ckpt": "checkpoints/thyroid/ablation_img_tirads/best.pt",  "cols": ["tirads"], "label": "Image + TI-RADS", "color": "#000000", "ls": "-",  "lw": 3},
    {"key": "fusion",   "ckpt": "checkpoints/thyroid/fusion/best.pt",               "cols": CLIN5, "label": "Full five-feature fusion",    "color": "#D55E00", "ls": "--", "lw": 2},
]


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(wp, device):
    ckpt = torch.load(wp, map_location="cpu", weights_only=False)
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


def predict_nodule_mean(model, ds, device, batch_size=64):
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
    agg_p = np.array([np.mean(ps) for ps in groups.values()])
    agg_l = np.array([int(labels[np.where(pids == pid)[0][0]]) for pid in groups])
    return agg_p, agg_l


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
    counts = np.array([(idx == b).sum() for b in range(n_bins)])
    ece = float(np.sum(np.abs(np.array(mp) - np.array(my)) * counts[counts > 0]) / len(y))
    return np.array(mp), np.array(my), ece


def dca(y, p):
    pts = np.arange(0.05, 0.96, 0.05)
    nbs = []
    for t in pts:
        tp = int(((p >= t) & (y == 1)).sum())
        fp = int(((p >= t) & (y == 0)).sum())
        nbs.append(tp / len(y) - fp / len(y) * (t / (1 - t)))
    return pts, np.array(nbs)


def main():
    device = get_device()
    results = {}
    for m in MODELS:
        wp = PROJ / m["ckpt"]
        if not wp.exists():
            print(f"SKIP {m['key']}: {wp}")
            continue
        ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                            image_size=224, num_clinical_features=len(m["cols"]),
                            clinical_columns=m["cols"], mock=False,
                            split_by_group=True, seed=42)
        model = load_model(wp, device)
        p, y = predict_nodule_mean(model, ds, device)
        fpr, tpr, _ = roc_curve(y, p)
        mp, my, ece = reliability_curve(y, p)
        pts, nbs = dca(y, p)
        results[m["key"]] = {"fpr": fpr, "tpr": tpr, "auc": sk_auc(fpr, tpr),
                             "mp": mp, "my": my, "ece": ece,
                             "pts": pts, "nbs": nbs, "y": y, "p": p, **m}
        print(f"{m['label']}: AUC {results[m['key']]['auc']:.4f}, ECE {ece:.4f}")

    # Fig 5: ROC
    fig, ax = plt.subplots(figsize=(6.6, 6))
    for r in results.values():
        ax.plot(r["fpr"], r["tpr"], lw=r["lw"], ls=r["ls"], color=r["color"],
                label=f"{r['label']} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title("ThyroidXL test: ROC curves (nodule-level)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5.png", dpi=300)
    plt.close(fig)
    print("saved fig5.png")

    # Fig 6: calibration
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    for r in results.values():
        ax.plot(r["mp"], r["my"], "o-", color=r["color"], lw=2, ms=4,
                label=f"{r['label']} (ECE = {r['ece']:.3f})")
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_title("ThyroidXL test: calibration", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6.png", dpi=300)
    plt.close(fig)
    print("saved fig6.png")

    # Fig 7: DCA
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for r in results.values():
        ax.plot(r["pts"], r["nbs"], lw=r["lw"], ls=r["ls"], color=r["color"], label=r["label"])
    ax.plot([0, 1], [0, 0], "k:", lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=12)
    ax.set_ylabel("Net benefit", fontsize=12)
    ax.set_title("ThyroidXL test: decision curves", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7.png", dpi=300)
    plt.close(fig)
    print("saved fig7.png")
    print("ALL FIG5-7 DONE")


if __name__ == "__main__":
    main()