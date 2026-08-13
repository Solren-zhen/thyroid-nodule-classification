#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPV/NPV + threshold sensitivity analysis (ThyroidXL test, nodule-level).

Models: image-only, clinical-only, image + TI-RADS (feature-selected best),
and full five-feature fusion (prespecified primary).

Outputs:
  - console table (Youden-optimal operating point)
  - paper/figures/fig8.png (threshold curves, Figure 8)
  - paper/notes/threshold_analysis.md (summary)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
sys.path.insert(0, str(PROJ))
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier

CLIN5 = ["tirads", "width_mm", "height_mm", "age", "gender"]
MODELS = [
    {"key": "image",    "ckpt": "checkpoints/thyroid/image/best.pt",                "cols": CLIN5, "label": "Image-only",      "color": "#0072B2"},
    {"key": "clinical", "ckpt": "checkpoints/thyroid/clinical/best.pt",             "cols": CLIN5, "label": "Clinical-only",  "color": "#E69F00"},
    {"key": "tirads",   "ckpt": "checkpoints/thyroid/ablation_img_tirads/best.pt",  "cols": ["tirads"], "label": "Image + TI-RADS", "color": "#000000"},
    {"key": "fusion",   "ckpt": "checkpoints/thyroid/fusion/best.pt",               "cols": CLIN5, "label": "Full five-feature fusion",    "color": "#D55E00"},
]
FIG_DIR = PROJ / "paper" / "figures"


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


def predict_nodule_mean(model, ds, device, batch_size=32):
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
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    pids = np.concatenate(pids)
    groups = {}
    for pid, p in zip(pids, probs):
        groups.setdefault(pid, []).append(p)
    agg_p = np.array([np.mean(ps) for ps in groups.values()])
    agg_l = np.array([int(labels[np.where(pids == pid)[0][0]]) for pid in groups])
    return agg_p, agg_l


def metrics_at(y, p, t):
    preds = (p >= t).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    return {"t": t,
            "sen": tp / max(tp + fn, 1), "spe": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1)}


def main():
    device = get_device()
    data = {}
    for m in MODELS:
        wp = PROJ / m["ckpt"]
        if not wp.exists():
            print(f"SKIP {m['key']}: no checkpoint")
            continue
        ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                            image_size=224, num_clinical_features=len(m["cols"]),
                            clinical_columns=m["cols"], mock=False,
                            split_by_group=True, seed=42)
        model = load_model(wp, device)
        p, y = predict_nodule_mean(model, ds, device)
        data[m["key"]] = (y, p, m)

    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, (y, p, m) in data.items():
        tgrid = np.arange(0.01, 0.99, 0.01)
        youden = tgrid[np.argmax([metrics_at(y, p, t)["sen"] + metrics_at(y, p, t)["spe"] - 1 for t in tgrid])]
        mt = metrics_at(y, p, youden)
        rows[key] = {**mt, "n": int(len(y))}
        print(f"{m['label']:16s} Youden t={youden:.2f} | Sens {mt['sen']:.3f} Spec {mt['spe']:.3f} "
              f"PPV {mt['ppv']:.3f} NPV {mt['npv']:.3f}")
        sens = [metrics_at(y, p, t)["sen"] for t in thresholds]
        spec = [metrics_at(y, p, t)["spe"] for t in thresholds]
        ppv = [metrics_at(y, p, t)["ppv"] for t in thresholds]
        axes[0].plot(thresholds, sens, color=m["color"], lw=2, label=f"{m['label']} Sens")
        axes[0].plot(thresholds, spec, color=m["color"], lw=2, ls="--", label=f"{m['label']} Spec")
        axes[1].plot(thresholds, ppv, color=m["color"], lw=2, label=f"{m['label']} PPV")

    axes[0].set_xlabel("Threshold", fontsize=12)
    axes[0].set_ylabel("Sensitivity / Specificity", fontsize=12)
    axes[0].set_title("(a) Sens/Spec vs threshold", fontsize=13)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7, loc="center left")
    axes[1].set_xlabel("Threshold", fontsize=12)
    axes[1].set_ylabel("Positive predictive value", fontsize=12)
    axes[1].set_title("(b) PPV vs threshold", fontsize=13)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8.png", dpi=300)
    plt.close(fig)
    print("saved fig8.png")

    md = ["# Threshold Analysis (ThyroidXL test, nodule-level, n=739)",
          "", "## Youden-optimal operating point", "",
          "| Model | Threshold | Sensitivity | Specificity | PPV | NPV |",
          "|---|---|---|---|---|---|"]
    for m in MODELS:
        if m["key"] in rows:
            r = rows[m["key"]]
            md.append(f"| {m['label']} | {r['t']:.2f} | {r['sen']:.3f} | {r['spe']:.3f} | {r['ppv']:.3f} | {r['npv']:.3f} |")
    out = PROJ / "paper" / "notes" / "threshold_analysis.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()