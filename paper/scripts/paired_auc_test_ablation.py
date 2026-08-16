#!/usr/bin/env python3
"""Paired bootstrap ΔAUC test between two ThyroidXL ablation models.

Default comparison: image + TI-RADS vs image-only (seed-42 anchors), the
headline incremental-value comparison after the feature-level ablation.
Same protocol as paired_auc_test.py (paired nodule-level bootstrap, n=2000).

Usage: python paper/scripts/paired_auc_test_ablation.py
Output: paper/output/repro/paired_img_tirads_vs_image.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))  # `import thyroid` (repo root is the package)
sys.path.insert(0, str(PROJ))          # `from train_thyroid import ...`

from data.thyroid_dataset import ThyroidDataset
from models.thyroid import ThyroidClassifier
from train_thyroid import get_device

CLIN5 = ["tirads", "width_mm", "height_mm", "age", "gender"]

MODEL_A = {
    "ckpt": PROJ / "checkpoints" / "thyroid" / "ablation_img_tirads" / "best.pt",
    "cols": ["tirads"],
    "name": "image+TI-RADS",
}
MODEL_B = {
    "ckpt": PROJ / "checkpoints" / "thyroid" / "image" / "best.pt",
    "cols": CLIN5,
    "name": "image-only",
}
OUT = PROJ / "paper" / "output" / "repro" / "paired_img_tirads_vs_image.json"


def load_model(wp: Path, device):
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
            images = batch["image"].to(device)
            o = model(images=images, clinical=clinical)
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


def main():
    device = get_device()
    ds_a = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                          image_size=224, num_clinical_features=len(MODEL_A["cols"]),
                          clinical_columns=MODEL_A["cols"], mock=False,
                          split_by_group=True, seed=42)
    ds_b = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                          image_size=224, num_clinical_features=len(MODEL_B["cols"]),
                          clinical_columns=MODEL_B["cols"], mock=False,
                          split_by_group=True, seed=42)

    print(f"loading {MODEL_A['name']}: {MODEL_A['ckpt'].name}")
    model_a = load_model(MODEL_A["ckpt"], device)
    p_a, y_a = predict_nodule_mean(model_a, ds_a, device)
    print(f"loading {MODEL_B['name']}: {MODEL_B['ckpt'].name}")
    model_b = load_model(MODEL_B["ckpt"], device)
    p_b, y_b = predict_nodule_mean(model_b, ds_b, device)
    assert np.array_equal(y_a, y_b), "label mismatch between datasets"
    y = y_a
    n = len(y)

    auc_a = roc_auc_score(y, p_a)
    auc_b = roc_auc_score(y, p_b)
    d_obs = auc_a - auc_b
    print(f"\nseed-42 anchor | n={n}")
    print(f"  {MODEL_A['name']:14s} AUC: {auc_a:.4f}")
    print(f"  {MODEL_B['name']:14s} AUC: {auc_b:.4f}")
    print(f"  ΔAUC ({MODEL_A['name']} - {MODEL_B['name']}): {d_obs:+.4f}")

    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one = float(np.mean(diffs <= 0))
    p_two = 2 * min(p_one, 1 - p_one)
    print(f"  ΔAUC 95% CI (paired bootstrap): [{lo:+.4f}, {hi:+.4f}]")
    print(f"  p-value (two-sided): {p_two:.4f}")

    out = {
        "n": n,
        "model_a": MODEL_A["name"], "model_b": MODEL_B["name"],
        "auc_a": auc_a, "auc_b": auc_b, "delta_auc": d_obs,
        "delta_ci": [lo, hi], "p_two_sided": p_two, "n_boot": len(diffs),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
