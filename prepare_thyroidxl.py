#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThyroidXL → manifest.csv 构建脚本
==================================
ThyroidXL（hunglc007/ThyroidXL）gated 审核通过后，把官方 annotations.json 转为
本项目统一的 manifest 格式（含 TI-RADS 临床特征列），供 train_thyroid.py 使用。

数据契约（见 decision_log D11）:
  - 每 patient = 一个结节 + 平均 2.8 张切片图，TI-RADS 结节级共享
  - 临床特征: tirads(总分1-5), width_mm, height_mm, age, gender
  - 官方 train/test 按患者隔离（零重叠），直接用
  - train 内部按患者再切 90/10 为 train/val（seed=42），官方 test 为 test

用法:
  PYTHONIOENCODING=utf-8 python prepare_thyroidxl.py [--root data/thyroid/thyroidxl]
"""
import argparse
import csv
import json
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np

CLINICAL_COLS = ["tirads", "width_mm", "height_mm", "age", "gender"]
SEED = 42


def load_annotations(ann_path):
    """annotations.json → {file_name: {tirads, width, height, age, gender, label}}"""
    j = json.loads(ann_path.read_text(encoding="utf-8"))
    meta = j["info"]                      # patient_id -> patient meta
    img2ann = {a["image_id"]: a for a in j["annotations"]}
    img_by_id = {i["id"]: i for i in j["images"]}

    out = {}
    for img_id, ann in img2ann.items():
        img = img_by_id[img_id]
        pid = str(img["patient_id"]).zfill(8)     # e.g. '126' -> '00000126'
        m = meta.get(pid)
        if m is None:
            print(f"  警告：患者 {pid} 无 meta，跳过 {img['file_name']}")
            continue
        nod = m.get("nodule_1") or {}
        out[img["file_name"]] = {
            "patient_id": pid,
            "tirads": nod.get("TIRADS"),
            "width_mm": nod.get("Width"),
            "height_mm": nod.get("Height"),
            "age": m.get("age"),
            "gender": m.get("gender"),
            "label": ann["category_id"],          # 0=benign, 1=malignant
        }
    return out


def split_train_val_by_patient(rows, val_frac=0.1):
    """按 patient_id 分组，train 内部切 90/10 为 train/val（seed 固定）"""
    groups = defaultdict(list)
    for r in rows:
        groups[r["patient_id"]].append(r)
    pids = sorted(groups.keys())
    rng = random.Random(SEED)
    rng.shuffle(pids)
    n_val = max(1, round(len(pids) * val_frac))
    val_pids = set(pids[:n_val])
    train_rows, val_rows = [], []
    for r in rows:
        (val_rows if r["patient_id"] in val_pids else train_rows).append(r)
    return train_rows, val_rows


def write_manifest(rows, dest):
    """写出 manifest.csv（含 split 列）"""
    if not rows:
        return
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "patient_id", "label"] + CLINICAL_COLS + ["split"])
        for r in rows:
            w.writerow([
                r["image_path"], r["patient_id"], r["label"],
                r["tirads"], r["width_mm"], r["height_mm"], r["age"], r["gender"],
                r["split"],
            ])
    print(f"  → {dest}: {len(rows)} 行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/thyroid/thyroidxl")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    anns = {
        "train": load_annotations(root / "train_annotations.json"),
        "test": load_annotations(root / "test_annotations.json"),
    }
    print(f"train anns: {len(anns['train'])} | test anns: {len(anns['test'])}")

    # 校验图片都在磁盘
    missing = 0
    for split in ("train", "test"):
        img_dir = root / split / "images"
        for fn in anns[split]:
            if not (img_dir / fn).exists():
                missing += 1
                if missing <= 5:
                    print(f"  警告：缺图: {split}/images/{fn}")
    if missing:
        print(f"缺图 {missing} 张，先完成下载再构建 manifest")
        sys.exit(1)

    # 组装 train 行（split 列先留空，由切分函数填充）
    train_rows = []
    for fn, a in anns["train"].items():
        r = {**a, "image_path": str(root / "train" / "images" / fn)}
        r["split"] = ""
        train_rows.append(r)

    # 二次切分 train → train/val
    tr_rows, val_rows = split_train_val_by_patient(train_rows, val_frac=0.1)
    for r in tr_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"

    # test 行
    test_rows = []
    for fn, a in anns["test"].items():
        r = {**a, "image_path": str(root / "test" / "images" / fn), "split": "test"}
        test_rows.append(r)

    n_pos = sum(1 for r in tr_rows if r["label"] == 1)
    print(f"train {len(tr_rows)} (阳性 {n_pos}, {n_pos/len(tr_rows)*100:.1f}%) | "
          f"val {len(val_rows)} | test {len(test_rows)} (阳性 "
          f"{sum(1 for r in test_rows if r['label']==1)}/{len(test_rows)})")

    write_manifest(tr_rows + val_rows + test_rows, root / "manifest.csv")


if __name__ == "__main__":
    main()
