#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temperature scaling for an arbitrary ThyroidXL fusion model (nodule-level).

Fits T on the held-out validation cohort (mean-aggregated nodule logits) and
evaluates AUC / Brier / ECE on the official test cohort before/after.

Usage:
  python paper/scripts/temperature_scaling_model.py \
      --ckpt checkpoints/thyroid/ablation_img_tirads/best.pt \
      --cols tirads \
      --out paper/output/repro/temperature_scaling_tirads.json
"""
import argparse
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

from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier
from train_thyroid import compute_metrics, get_device


def load_model(path, device):
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
    model.eval().to(device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cols", type=str, required=True, help="comma-separated clinical columns")
    ap.add_argument("--out", type=str, required=True, help="output json path")
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    device = get_device()
    model = load_model(Path(args.ckpt), device)

    val_ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="val",
                            image_size=224, num_clinical_features=len(cols),
                            clinical_columns=cols, mock=False, split_by_group=True, seed=42)
    test_ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                             image_size=224, num_clinical_features=len(cols),
                             clinical_columns=cols, mock=False, split_by_group=True, seed=42)
    p_val, y_val = predict_nodule(model, val_ds, device)
    p_test, y_test = predict_nodule(model, test_ds, device)
    print(f"val n={len(y_val)} (pos {y_val.mean():.3f}); test n={len(y_test)} (pos {y_test.mean():.3f})")

    logit_val = np.log(np.clip(p_val, 1e-7, 1 - 1e-7) / (1 - np.clip(p_val, 1e-7, 1 - 1e-7)))

    def nll(T):
        pT = np.clip(1 / (1 + np.exp(-logit_val / T)), 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(pT) + (1 - y_val) * np.log(1 - pT))

    res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    T = float(res.x)
    logit_test = np.log(np.clip(p_test, 1e-7, 1 - 1e-7) / (1 - np.clip(p_test, 1e-7, 1 - 1e-7)))
    p_cal = 1 / (1 + np.exp(-logit_test / T))

    m_raw = compute_metrics(y_test, p_test)
    m_cal = compute_metrics(y_test, p_cal)
    out = {
        "ckpt": str(Path(args.ckpt).relative_to(PROJ)),
        "cols": cols,
        "temperature": T,
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "before": {
            "auc": m_raw["auc"], "auc_ci": list(m_raw["auc_ci"]),
            "brier": m_raw["brier"], "ece": m_raw["ece"],
        },
        "after": {
            "auc": m_cal["auc"], "auc_ci": list(m_cal["auc_ci"]),
            "brier": m_cal["brier"], "ece": m_cal["ece"],
        },
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJ / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
