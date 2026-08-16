#!/usr/bin/env python3
"""Temperature scaling for the ThyroidXL fusion model (nodule-level).

Fits a temperature T on the held-out validation cohort (logits of mean-
aggregated nodule probabilities) and evaluates AUC / Brier / ECE on the
official test cohort before and after calibration.

Usage: python paper/scripts/temperature_scaling.py
Output: paper/output/repro/temperature_scaling.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
sys.path.insert(0, str(PROJ))

from data.thyroid_dataset import ThyroidDataset
from models.thyroid import ThyroidClassifier
from train_thyroid import compute_metrics, get_device

OUT = PROJ / "paper" / "output" / "repro" / "temperature_scaling.json"
WEIGHTS = PROJ / "checkpoints" / "thyroid" / "fusion" / "best.pt"
CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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


def predict_nodule(model, ds, device, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            images = batch["image"].to(device)
            outputs = model(images=images, clinical=clinical)
            probs.append(outputs["probs"].cpu().numpy().flatten())
            labels.append(batch["label"].numpy().flatten())
            pids.append(batch["patient_id"])
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    pids = np.concatenate(pids) if pids and pids[0] is not None else None
    # mean aggregation per nodule (matches the manuscript protocol)
    agg = {}
    for pid, p, y in zip(pids, probs, labels):
        agg.setdefault(pid, []).append((p, y))
    p_out, y_out = [], []
    for pid, items in agg.items():
        ps = np.array([x[0] for x in items])
        p_out.append(ps.mean())
        y_out.append(items[0][1])
    return np.array(p_out), np.array(y_out).astype(int)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def main():
    device = get_device()
    model = load_model(WEIGHTS).to(device)

    val_ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="val",
                            image_size=224, num_clinical_features=len(CLIN),
                            clinical_columns=CLIN, mock=False, split_by_group=True, seed=42)
    test_ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                             image_size=224, num_clinical_features=len(CLIN),
                             clinical_columns=CLIN, mock=False, split_by_group=True, seed=42)
    p_val, y_val = predict_nodule(model, val_ds, device)
    p_test, y_test = predict_nodule(model, test_ds, device)
    print(f"val n={len(y_val)} (pos {y_val.mean():.3f}); test n={len(y_test)} (pos {y_test.mean():.3f})")

    logit_val = np.log(np.clip(p_val, 1e-7, 1 - 1e-7) / (1 - np.clip(p_val, 1e-7, 1 - 1e-7)))

    def nll(T):
        pT = 1 / (1 + np.exp(-logit_val / T))
        pT = np.clip(pT, 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(pT) + (1 - y_val) * np.log(1 - pT))

    res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    T = float(res.x)
    print(f"fitted temperature T = {T:.4f} (val NLL {res.fun:.4f})")

    logit_test = np.log(np.clip(p_test, 1e-7, 1 - 1e-7) / (1 - np.clip(p_test, 1e-7, 1 - 1e-7)))
    p_cal = 1 / (1 + np.exp(-logit_test / T))

    m_raw = compute_metrics(y_test, p_test)
    m_cal = compute_metrics(y_test, p_cal)
    out = {
        "temperature": T,
        "n_val": len(y_val),
        "n_test": len(y_test),
        "before": {
            "auc": m_raw["auc"], "auc_ci": list(m_raw["auc_ci"]),
            "brier": m_raw["brier"], "ece": m_raw["ece"],
        },
        "after": {
            "auc": m_cal["auc"], "auc_ci": list(m_cal["auc_ci"]),
            "brier": m_cal["brier"], "ece": m_cal["ece"],
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
