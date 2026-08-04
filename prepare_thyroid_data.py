#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare manifest.csv from the figshare thyroid ultrasound dataset (RAR archives).

Usage:
  python prepare_thyroid_data.py --raw_dir data/thyroid/raw --data_root data/thyroid

The figshare archives (batch1_image.rar / batch2_image.rar) contain:
  batch*_image.csv           -> patient_name, path
  batch*_image_label.csv     -> patient_name, histo_label (0=benign, 1=malignant)
  batch*_image/dataset/...   -> ultrasound images (Jpg)

This script:
  1. extracts each RAR with Windows built-in tar.exe (libarchive supports RAR),
  2. merges image / label tables,
  3. verifies every image exists,
  4. writes data_root/manifest.csv (image_path, patient_id, label, ...).

The script is idempotent: already-extracted archives are skipped and the output
manifest is overwritten from scratch.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def find_rar_files(raw_dir: Path):
    rars = list(raw_dir.glob("batch*_image.rar"))
    rars += [p for p in sorted(raw_dir.glob("*.rar")) if p not in rars]
    return rars


def extract_rar(rar: Path, dest_root: Path) -> Path:
    """Extract one RAR into dest_root/<stem>/; returns the extraction dir."""
    stem = rar.stem  # e.g. batch1_image
    dest = dest_root / stem
    marker = dest / ".extracted_ok"
    if marker.exists():
        print(f"  [skip] {rar.name} already extracted -> {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    tar = shutil.which("tar")
    if not tar:
        sys.exit("tar.exe not found; Windows 10/11 ships it in System32.")
    print(f"  extracting {rar.name} ({rar.stat().st_size / 1e6:.1f} MB) ...")
    r = subprocess.run([tar, "-xf", str(rar), "-C", str(dest)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"tar failed for {rar.name}:\n{r.stderr}")
    marker.write_text(rar.name, encoding="utf-8")
    print(f"  done -> {dest}")
    return dest


def _find_file(root: Path, pattern: str):
    hits = sorted(root.glob(pattern))
    if not hits:
        return None
    return hits[0]


def parse_batch(extract_dir: Path):
    """Return (patient->label dict, list of (patient, rel_image_path))."""
    image_csv = _find_file(extract_dir, "batch*_image.csv")
    if image_csv is None:
        image_csv = _find_file(extract_dir, "**/batch*_image.csv")
    label_csv = _find_file(extract_dir, "batch*_image_label.csv")
    if label_csv is None:
        label_csv = _find_file(extract_dir, "**/batch*_image_label.csv")
    if image_csv is None or label_csv is None:
        raise RuntimeError(f"cannot find CSV tables under {extract_dir}")

    labels = {}
    with open(label_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = (row.get("patient_name") or "").strip()
            lab_raw = (row.get("histo_label") or "").strip()
            if not pid:
                continue
            labels[pid] = int(float(lab_raw)) if lab_raw else -1

    image_rows = []
    with open(image_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = (row.get("patient_name") or "").strip()
            rel = (row.get("path") or "").strip()
            if not pid or not rel:
                continue
            image_rows.append((pid, rel))
    return labels, image_rows


def build_manifest(raw_dir: Path, data_root: Path, out_manifest: Path):
    rars = find_rar_files(raw_dir)
    if not rars:
        sys.exit(f"No *.rar archives found under {raw_dir}.\n"
                 f"Download batch1_image.rar / batch2_image.rar from "
                 f"https://figshare.com/articles/dataset/27021604 first.")

    extract_root = data_root / "figshare"
    rows = []
    missing_images = 0
    for rar in rars:
        ex_dir = extract_rar(rar, extract_root)
        labels, image_rows = parse_batch(ex_dir)
        if not labels:
            print(f"  [warn] no labels parsed from {ex_dir}")
        for pid, rel in image_rows:
            # Relocate image: rar contains e.g. batch1_image/dataset/0_001.Jpg
            cands = [
                ex_dir / rar.stem / rel,          # batch1_image/dataset/...  inside extract
                ex_dir / rel,
            ]
            img_abs = next((c for c in cands if c.exists()), None)
            if img_abs is None:
                # fall back: search by filename under the extraction dir
                name = Path(rel).name
                found = list(ex_dir.rglob(name))
                img_abs = found[0] if found else None
            if img_abs is None:
                missing_images += 1
                continue
            rel_path = img_abs.relative_to(data_root).as_posix()
            lab = labels.get(pid, -1)
            rows.append({
                "image_path": rel_path,
                "patient_id": pid,
                "label": lab,
            })

    if not rows:
        sys.exit("No usable rows generated - check archive structure.")
    bad = [r for r in rows if r["label"] < 0]
    if bad:
        print(f"  [warn] {len(bad)} rows have no histo label; they will be excluded.")
        rows = [r for r in rows if r["label"] >= 0]

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "patient_id", "label",
                        "composition", "echogenicity", "shape",
                        "margin", "echogenic_foci", "size_mm"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    n_pat = len({r["patient_id"] for r in rows})
    n_pos = sum(1 for r in rows if r["label"] == 1)
    print(f"\nmanifest written: {out_manifest}")
    print(f"  images : {len(rows)}")
    print(f"  patients: {n_pat}")
    print(f"  malignant: {n_pos} ({n_pos / max(len(rows), 1) * 100:.1f}%)")
    print(f"  missing images skipped: {missing_images}")


def main():
    ap = argparse.ArgumentParser(description="Build thyroid manifest from figshare RARs")
    ap.add_argument("--raw_dir", default="data/thyroid/raw")
    ap.add_argument("--data_root", default="data/thyroid")
    ap.add_argument("--out_manifest", default=None,
                    help="default: <data_root>/manifest.csv")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    data_root = Path(args.data_root)
    out_manifest = Path(args.out_manifest) if args.out_manifest else data_root / "manifest.csv"
    build_manifest(raw_dir, data_root, out_manifest)


if __name__ == "__main__":
    main()
