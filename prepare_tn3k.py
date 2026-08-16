#!/usr/bin/env python3
"""Build manifest for TN3K (trainval labelled split=train; official test labelled split=test)."""

import argparse
import csv
import sys
from pathlib import Path

CLINICAL_COLS = ["composition", "echogenicity", "shape",
                 "margin", "echogenic_foci", "size_mm"]


def load_labels(csv_path):
    """Rows like '0000.jpg,0' (no header). Returns {filename: label}."""
    labels = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            name, lab = line.rsplit(",", 1)
            name = name.strip()
            lab = lab.strip()
            if name:
                labels[name] = int(float(lab)) if lab else -1
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/thyroid/tn3k")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    label_trainval = load_labels(root / "label4trainval.csv")
    label_test = load_labels(root / "label4test.csv")
    if not label_trainval or not label_test:
        sys.exit("label csv files missing or empty - check TN3K layout")

    rows = []
    n_skip = 0
    for folder, labels, split in (("trainval-image", label_trainval, "train"),
                                      ("test-image", label_test, "test")):
        for img in sorted((root / folder).glob("*.jpg")):
            lab = labels.get(img.name, -1)
            if lab < 0:
                n_skip += 1
                continue
            rows.append({
                "image_path": f"{folder}/{img.name}",
                "patient_id": img.stem,
                "label": lab,
                "split": split,
            })

    if not rows:
        sys.exit("no rows generated")

    out = root / "manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image_path", "patient_id", "label", "split"] + CLINICAL_COLS,
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    print(f"manifest written: {out}")
    print(f"  images: {len(rows)} (skipped {n_skip})")
    print(f"  malignant: {n_pos} ({n_pos / len(rows) * 100:.1f}%)")
    print("  split: trainval=train, official test=test")


if __name__ == "__main__":
    main()
