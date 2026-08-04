#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download TN5000 thyroid ultrasound dataset from HF mirror.

Why not snapshot_download: hf-mirror rejects HEAD requests, so we list files
via huggingface_hub (GET works) and then GET each file with requests directly.

Usage:
  set HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
  python download_tn5000.py --dest <dest_dir>
"""

import argparse
import concurrent.futures
import os
from pathlib import Path

import requests

REPO = "Johnyquest7/TN5000-thyroid-nodule-classification"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def list_files():
    from huggingface_hub import list_repo_files
    return [f for f in list_repo_files(repo_id=REPO, repo_type="dataset")]


def download_one(item):
    rel, dest_root, retries = item
    dest = dest_root / rel
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + rel
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": UA})
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            return ("ok", rel)
        except Exception as e:
            last_err = e
    return ("fail", f"{rel}: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/thyroid/tn5000")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    dest_root = Path(args.dest).resolve()
    print(f"listing files for {REPO} ...")
    files = list_files()
    print(f"total files: {len(files)}")
    pngs = [f for f in files if f.lower().endswith(".png")]
    print(f"png images: {len(pngs)}")

    tasks = [(f, dest_root, args.retries) for f in pngs]
    ok = skip = 0
    fails = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (status, rel) in enumerate(ex.map(download_one, tasks), 1):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fails.append(rel)
            if i % 500 == 0 or i == len(tasks):
                print(f"  progress {i}/{len(tasks)} (ok={ok}, skip={skip}, fail={len(fails)})")

    total_size = sum(p.stat().st_size for p in dest_root.rglob("*") if p.is_file())
    print(f"\nDONE: ok={ok}, skipped={skip}, failed={len(fails)}")
    print(f"total size on disk: {total_size / 1e6:.1f} MB -> {dest_root}")
    for f in fails[:10]:
        print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
