#!/usr/bin/env python3
"""Build an equal-sample-size control (JES) training manifest for a given seed.

The joint pool (data/thyroid_multi) contains 6,379 train + 750 val images.
The control matches the single-dataset development size (4,250 = 3,500
training + 750 validation): it samples 3,500 training images from the joint
pool stratified by dataset source (TN5000 vs TN3K) with the given seed, and
keeps the same 750 TN5000 validation images.

Usage:
    python tools/build_jes_manifest.py --seed 123 --out data/thyroid_jes_s123
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SRC = PROJ / "data" / "thyroid_multi" / "manifest.csv"


def source(r):
    p = r["image_path"].replace("\\", "/")
    return "tn3k" if "/tn3k/" in p else "tn5000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/thyroid_jes")
    args = ap.parse_args()

    with open(SRC, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    print(f"joint pool: train={len(train)} val={len(val)}")

    by_src = defaultdict(list)
    for r in train:
        by_src[source(r)].append(r)
    print({k: len(v) for k, v in by_src.items()})

    rng = random.Random(args.seed)
    target = 3500
    chosen = []
    for src, items in sorted(by_src.items()):
        chosen.extend(rng.sample(items, round(target * len(items) / len(train))))
    if len(chosen) < target:
        chosen.extend(rng.sample([r for r in train if r not in chosen], target - len(chosen)))
    elif len(chosen) > target:
        chosen = rng.sample(chosen, target)
    print(f"JES train: {len(chosen)} "
          f"({ {s: sum(1 for r in chosen if source(r) == s) for s in ('tn5000', 'tn3k')} })")

    jes = [{**r, "split": "train"} for r in chosen] + [{**r, "split": "val"} for r in val]
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJ / out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(jes[0].keys()))
        w.writeheader()
        w.writerows(jes)
    print(f"wrote {out / 'manifest.csv'}: {len(jes)} rows "
          f"(train={sum(1 for r in jes if r['split'] == 'train')}, "
          f"val={sum(1 for r in jes if r['split'] == 'val')})")


if __name__ == "__main__":
    main()
