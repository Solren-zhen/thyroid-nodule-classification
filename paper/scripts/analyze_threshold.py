#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPV/NPV + 阈值敏感性分析。

对 ThyroidXL 三臂模型（fusion/image/clinical）的 test 结节级 mean 预测，
计算不同阈值下的 Sens/Spec/PPV/NPV，输出：
  - 控制台表（Youden 阈值处 PPV/NPV）
  - paper/figures/thyroidxl_threshold_sensitivity.png（阈值曲线）
  - paper/notes/threshold_analysis.md（汇总）
"""
import json
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
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier

CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]
ABLATIONS = ["fusion", "image", "clinical"]
COLORS = {"image": "#1f77b4", "clinical": "#2ca02c", "fusion": "#d62728"}
LABELS = {"image": "Image-only", "clinical": "Clinical-only", "fusion": "Fusion"}
FIG_DIR = PROJ / "paper" / "figures"


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
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {"t": t, "sen": sen, "spe": spe, "ppv": ppv, "npv": npv}


def main():
    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                        image_size=224, num_clinical_features=5, clinical_columns=CLIN,
                        mock=False, split_by_group=True, seed=42)

    data = {}
    for ab in ABLATIONS:
        wp = PROJ / "checkpoints" / "thyroid" / ab / "best.pt"
        if not wp.exists():
            print(f"SKIP {ab}: no checkpoint")
            continue
        model = load_model(wp, device)
        p, y = predict_nodule_mean(model, ds, device)
        data[ab] = (y, p)

    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ab, (y, p) in data.items():
        # Youden 阈值在 test 结节级预测上求（论文 §3.5 已注明 operating point 由 test 队列确定，
        # 属探索性分析）
        tgrid = np.arange(0.01, 0.99, 0.01)
        youden = tgrid[np.argmax([m["sen"] + m["spe"] - 1 for m in [metrics_at(y, p, t) for t in tgrid]])]
        m = metrics_at(y, p, youden)
        rows[ab] = {**m, "auc": 0.0, "n": int(len(y))}
        print(f"{LABELS[ab]:14s} Youden t={youden:.2f} | Sens {m['sen']:.3f} Spec {m['spe']:.3f} "
              f"PPV {m['ppv']:.3f} NPV {m['npv']:.3f}")

        sens = [metrics_at(y, p, t)["sen"] for t in thresholds]
        spec = [metrics_at(y, p, t)["spe"] for t in thresholds]
        ppv = [metrics_at(y, p, t)["ppv"] for t in thresholds]
        axes[0].plot(thresholds, sens, color=COLORS[ab], lw=2, label=f"{LABELS[ab]} Sens")
        axes[0].plot(thresholds, spec, color=COLORS[ab], lw=2, ls="--", label=f"{LABELS[ab]} Spec")
        axes[1].plot(thresholds, ppv, color=COLORS[ab], lw=2, label=f"{LABELS[ab]} PPV")

    axes[0].set_xlabel("Threshold", fontsize=12)
    axes[0].set_ylabel("Sensitivity / Specificity", fontsize=12)
    axes[0].set_title("(a) Sens/Spec vs threshold", fontsize=13)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="center left")
    axes[1].set_xlabel("Threshold", fontsize=12)
    axes[1].set_ylabel("Positive predictive value", fontsize=12)
    axes[1].set_title("(b) PPV vs threshold", fontsize=13)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thyroidxl_threshold_sensitivity.png", dpi=300)
    plt.close(fig)
    print("saved thyroidxl_threshold_sensitivity.png")

    # markdown
    md = ["# Threshold Analysis (ThyroidXL test, nodule-level, n=739)",
          "", "## Youden-optimal operating point", "",
          "| Model | Threshold | Sensitivity | Specificity | PPV | NPV |",
          "|---|---|---|---|---|---|"]
    for ab in ABLATIONS:
        if ab in rows:
            r = rows[ab]
            md.append(f"| {LABELS[ab]} | {r['t']:.2f} | {r['sen']:.3f} | {r['spe']:.3f} | {r['ppv']:.3f} | {r['npv']:.3f} |")
    out = PROJ / "paper" / "notes" / "threshold_analysis.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
