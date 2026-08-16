#!/usr/bin/env python3
"""
Build manifest.csv for the TN5000 dataset (HF mirror: Johnyquest7/...).

TN5000 layout:
  tn5000/Train/{Benign,Malignant}/*.png
  tn5000/Valid/{Benign,Malignant}/*.png
  tn5000/Test/{Benign,Malignant}/*.png

The dataset pipeline (ThyroidDataset) performs a grouped 7:1.5:1.5 split on
patient_id, so we keep all images in one manifest and let the pipeline split.
patient_id is the image file stem (each image is an independent nodule sample).

Usage:
  python prepare_tn5000.py --data_root data/thyroid
"""

import argparse
import csv
import sys
from pathlib import Path

CLINICAL_COLS = ["composition", "echogenicity", "shape",
                 "margin", "echogenic_foci", "size_mm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/thyroid")
    ap.add_argument("--dataset_dir", default="tn5000")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    ds_dir = data_root / args.dataset_dir
    if not ds_dir.exists():
        sys.exit(f"dataset dir not found: {ds_dir}")

    imgs = sorted(ds_dir.rglob("*.png"))
    if not imgs:
        sys.exit("no PNG images found - download the dataset first")

    rows = []
    n_skip = 0
    for img in imgs:
        parts = img.relative_to(ds_dir).parts
        if len(parts) < 3 or parts[1].lower() not in ("benign", "malignant"):
            n_skip += 1
            continue
        cls = parts[1].lower()
        label = 1 if cls == "malignant" else 0
        rel = img.relative_to(data_root).as_posix()
        rows.append({
            "image_path": rel,
            "patient_id": img.stem,
            "label": label,
        })

    if not rows:
        sys.exit("no valid rows; unexpected folder layout")

    out = data_root / "manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "patient_id", "label"] + CLINICAL_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    print(f"manifest written: {out}")
    print(f"  images  : {len(rows)}")
    print(f"  benign  : {len(rows) - n_pos}")
    print(f"  malignant: {n_pos} ({n_pos / len(rows) * 100:.1f}%)")
    if n_skip:
        print(f"  skipped (unexpected path): {n_skip}")


if __name__ == "__main__":
    main()
