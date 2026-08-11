#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build multi-dataset training manifest (TN5000 + TN3K trainval) and
internal/external test manifests for joint-training evaluation.

FIX (2026-08-10, review H1): TN5000 is split with the SAME numpy RandomState(42)
7:1.5:1.5 patient-grouped split used by thyroid_dataset.py, so the internal
test set (750 images) is EXCLUDED from the training pool. Previously the split
used python random.Random(42) 85/15, which leaked 646/750 test images into
training (data leakage). The training pool is now TN5000 train+val (4,250) +
TN3K trainval (2,879) = 7,129; TN5000 test (750) is held out.

Layout:
  data/thyroid_multi/manifest.csv     -> train/val samples (absolute paths)
  data/thyroid_tn3ktest/manifest.csv  -> TN3K official test 614 (absolute paths)
  data/thyroid_tn5000test/manifest.csv-> TN5000 held-out test 750 (absolute paths)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


CLINICAL_COLS = ["composition", "echogenicity", "shape",
                 "margin", "echogenic_foci", "size_mm"]
PROJ = Path(__file__).resolve().parent


def tn5000_rows():
    root = PROJ / "data" / "thyroid" / "tn5000"
    rows = []
    for img in sorted(root.rglob("*.png")):
        parts = img.relative_to(root).parts
        if len(parts) < 3 or parts[1].lower() not in ("benign", "malignant"):
            continue
        rows.append({
            "image_path": str(img),
            "patient_id": img.stem,
            "label": 1 if parts[1].lower() == "malignant" else 0,
        })
    return rows


def tn3k_trainval_rows():
    root = PROJ / "data" / "thyroid" / "tn3k"
    labels = {}
    with open(root / "label4trainval.csv", newline="", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if "," in line:
                name, lab = line.rsplit(",", 1)
                labels[name.strip()] = int(float(lab.strip()))
    rows = []
    for img in sorted((root / "trainval-image").glob("*.jpg")):
        if img.name in labels:
            rows.append({
                "image_path": str(img),
                "patient_id": img.stem,
                "label": labels[img.name],
            })
    return rows


def tn3k_test_rows():
    root = PROJ / "data" / "thyroid" / "tn3k"
    labels = {}
    with open(root / "label4test.csv", newline="", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if "," in line:
                name, lab = line.rsplit(",", 1)
                labels[name.strip()] = int(float(lab.strip()))
    rows = []
    for img in sorted((root / "test-image").glob("*.jpg")):
        if img.name in labels:
            rows.append({
                "image_path": str(img),
                "patient_id": img.stem,
                "label": labels[img.name],
            })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "patient_id", "label", "split"] + CLINICAL_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    seed = 42
    # numpy RandomState(42) —— 与 thyroid_dataset._apply_split 完全一致
    rng = np.random.RandomState(seed)

    t5 = tn5000_rows()
    t3tr = tn3k_trainval_rows()
    t3te = tn3k_test_rows()
    print(f"TN5000={len(t5)}, TN3K trainval={len(t3tr)}, TN3K test={len(t3te)}")

    # ---- TN5000 split: numpy 7:1.5:1.5 patient-grouped (same as thyroid_dataset) ----
    n = len(t5)
    group_map = defaultdict(list)
    for i, r in enumerate(t5):
        group_map[r["patient_id"]].append(i)
    groups = list(group_map.values())
    rng.shuffle(groups)
    perm = [i for g in groups for i in g]
    n_train = int(n * 0.7)          # 3500
    n_val = int(n * 0.15)           # 750
    train_idx = set(perm[:n_train])
    val_idx = set(perm[n_train:n_train + n_val])
    test_idx = set(perm[n_train + n_val:])
    print(f"TN5000 split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ---- multi training pool: TN5000 train+val + TN3K trainval ----
    multi = []
    for i, r in enumerate(t5):
        if i in train_idx:
            multi.append({**r, "split": "train"})
        elif i in val_idx:
            multi.append({**r, "split": "val"})
        # test_idx images EXCLUDED from training pool
    for r in t3tr:
        multi.append({**r, "split": "train"})

    out_multi = PROJ / "data" / "thyroid_multi" / "manifest.csv"
    write_csv(out_multi, multi)
    n_pos = sum(1 for r in multi if r["label"] == 1)
    print(f"multi manifest: {len(multi)} (train={sum(1 for r in multi if r['split']=='train')}, "
          f"val={sum(1 for r in multi if r['split']=='val')}, malignant={n_pos})")

    # ---- TN3K official test (external validation) ----
    out_test = PROJ / "data" / "thyroid_tn3ktest" / "manifest.csv"
    test_rows = [{**r, "split": "test"} for r in t3te]
    write_csv(out_test, test_rows)
    n_pos_t = sum(1 for r in test_rows if r["label"] == 1)
    print(f"tn3k-test manifest: {len(test_rows)} (malignant={n_pos_t})")

    # ---- TN5000 held-out test (internal validation, excluded from training) ----
    out_t5test = PROJ / "data" / "thyroid_tn5000test" / "manifest.csv"
    t5test_rows = [{**r, "split": "test"} for i, r in enumerate(t5) if i in test_idx]
    write_csv(out_t5test, t5test_rows)
    n_pos_t5 = sum(1 for r in t5test_rows if r["label"] == 1)
    print(f"tn5000-test manifest: {len(t5test_rows)} (malignant={n_pos_t5})")


if __name__ == "__main__":
    main()
