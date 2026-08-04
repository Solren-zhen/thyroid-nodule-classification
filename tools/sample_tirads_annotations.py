# -*- coding: utf-8 -*-
"""TI-RADS 人工标注取样（方案 B：ThyroidXL 审批不通过时的替代路线）。

从 TN5000（主训练）取样标注子集用于三臂消融（image / clinical / fusion）；
从 TN3K 官方 test（614 张）取样小集用于融合模型的外部验证。

切分逻辑与 data/thyroid_dataset.py 的 _apply_split() 完全一致：
按 manifest 行序 → 按 patient_id 分组 → RandomState(seed=42) shuffle 组 →
70/15/15 切割。保证取样子集与训练切分语义一致（同患者不跨 split）。

用法：
  # TN5000 标注子集（默认比例分层，共 ~900 张）
  python tools/sample_tirads_annotations.py \
      --manifest data/thyroid/manifest.csv \
      --out data/thyroid/tirads_annotations/tn5000_annotate.csv \
      --n 900 --seed 42

  # TN3K 官方 test 标注子集（无切分，直接抽 200 张作外部验证）
  python tools/sample_tirads_annotations.py \
      --manifest data/thyroid_tn3ktest/manifest.csv \
      --images data/thyroid/tn3k/datasets/tn3k \
      --split none --n 200 --seed 42
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\甄朝晖\Desktop\thyroid_ai")
CLINICAL_COLUMNS = ["composition", "echogenicity", "shape", "margin",
                    "echogenic_foci", "size_mm"]


def load_manifest(manifest_path: Path, images_base: Path):
    """按行序读取 manifest，解析 image_path / patient_id / label / split。"""
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = (row.get("image_path") or "").strip()
            patient_id = (row.get("patient_id") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not image_path or not patient_id:
                raise ValueError(f"空 image_path/patient_id: {row}")
            p = Path(image_path)
            if not p.is_absolute():
                p = images_base / p
            if not p.exists():
                raise FileNotFoundError(f"图片不存在: {p}")
            samples.append({
                "image_path": str(p),
                "patient_id": patient_id,
                "label": int(float(label_raw)),
                "split": (row.get("split") or "").strip().lower(),
            })
    return samples


def apply_group_split(samples, seed: int):
    """复刻 ThyroidDataset._apply_split 的 patient 分组 70/15/15。"""
    rng = np.random.RandomState(seed)
    group_map = {}
    for i, s in enumerate(samples):
        group_map.setdefault(s["patient_id"], []).append(i)
    groups = list(group_map.values())
    rng.shuffle(groups)
    perm = [i for g in groups for i in g]
    n = len(samples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    for k, (lo, hi) in enumerate([(0, n_train), (n_train, n_train + n_val),
                                  (n_train + n_val, n)]):
        split_name = ["train", "val", "test"][k]
        for i in perm[lo:hi]:
            samples[i]["split"] = split_name


def stratified_sample(samples, n_total: int, seed: int, balanced: bool):
    """按 (split × label) 分层取样。

    balanced=False（默认）：按分层比例抽，每层保底 min_floor 张。
    balanced=True：每层等量，保证每类样本数均衡（对标注效率友好，但子集
    非比例代表——论文需说明）。
    """
    rng = random.Random(seed)
    strata = {}
    for s in samples:
        key = (s["split"], s["label"])
        strata.setdefault(key, []).append(s)

    if balanced:
        n_strata = len(strata)
        per = max(1, n_total // n_strata)
        picked = []
        for key, pool in sorted(strata.items()):
            picked.extend(rng.sample(pool, min(per, len(pool))))
    else:
        # 比例抽样 + 每层保底 10 张（超过预算时按剩余比例回缩）
        min_floor = 10
        total = len(samples)
        base = {key: max(min_floor, int(n_total * len(pool) / total))
                for key, pool in strata.items()}
        # 回缩到预算
        budget = n_total
        out = {}
        for key, want in sorted(base.items()):
            take = min(want, budget)
            out[key] = take
            budget -= take
        picked = []
        for key, pool in sorted(strata.items()):
            picked.extend(rng.sample(pool, out[key]))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default=None,
                    help="相对路径的基准目录（默认 manifest 所在目录）")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["group", "none"], default="group",
                    help="group=复刻 patient 分组切分；none=全部视为一个池（TN3K test）")
    ap.add_argument("--balanced", action="store_true")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    images_base = Path(args.images) if args.images else manifest.parent
    samples = load_manifest(manifest, images_base)

    if args.split == "group":
        apply_group_split(samples, args.seed)
    else:
        for s in samples:
            s["split"] = "test"

    picked = stratified_sample(samples, args.n, args.seed, args.balanced)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "patient_id", "label", "split"]
                   + CLINICAL_COLUMNS)
        for s in picked:
            w.writerow([s["image_path"], s["patient_id"], s["label"], s["split"]]
                       + [""] * len(CLINICAL_COLUMNS))

    print(f"写出 {len(picked)} 张 → {out_path}")
    from collections import Counter
    c = Counter((s["split"], s["label"]) for s in picked)
    for key in sorted(c):
        print(f"  {key}: {c[key]}")


if __name__ == "__main__":
    sys.exit(main())
