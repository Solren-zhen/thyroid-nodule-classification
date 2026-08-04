#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build multi-dataset training manifest (TN5000 + TN3K trainval) and a
TN3K-official-test manifest for external validation after joint training.

Layout:
  data/thyroid_multi/manifest.csv    -> train/val samples (absolute paths)
  data/thyroid_tn3ktest/manifest.csv -> TN3K official test 614 (absolute paths)
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


CLINICAL_COLS = ["composition", "echogenicity", "shape",
                 "margin", "echogenic_foci", "size_mm"]
PROJ = Path(r"C:\Users\甄朝晖\Desktop\thyroid_ai")


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
    rng = random.Random(seed)

    t5 = tn5000_rows()
    t3tr = tn3k_trainval_rows()
    t3te = tn3k_test_rows()
    print(f"TN5000={len(t5)}, TN3K trainval={len(t3tr)}, TN3K test={len(t3te)}")

    # Split TN5000 by patient group: 85% train / 15% val
    groups = defaultdict(list)
    for i, r in enumerate(t5):
        groups[r["patient_id"]].append(i)
    g = list(groups.values())
    rng.shuffle(g)
    n_train = int(len(t5) * 0.85)
    train_idx = set()
    for grp in g:
        if len(train_idx) >= n_train:
            break
        train_idx.update(grp)

    multi = []
    for i, r in enumerate(t5):
        multi.append({**r, "split": "train" if i in train_idx else "val"})
    for r in t3tr:
        multi.append({**r, "split": "train"})

    out_multi = PROJ / "data" / "thyroid_multi" / "manifest.csv"
    write_csv(out_multi, multi)
    n_pos = sum(1 for r in multi if r["label"] == 1)
    print(f"multi manifest: {len(multi)} (train={sum(1 for r in multi if r['split']=='train')}, "
          f"val={sum(1 for r in multi if r['split']=='val')}, malignant={n_pos})")

    out_test = PROJ / "data" / "thyroid_tn3ktest" / "manifest.csv"
    test_rows = [{**r, "split": "test"} for r in t3te]
    write_csv(out_test, test_rows)
    n_pos_t = sum(1 for r in test_rows if r["label"] == 1)
    print(f"tn3k-test manifest: {len(test_rows)} (malignant={n_pos_t})")


if __name__ == "__main__":
    main()
