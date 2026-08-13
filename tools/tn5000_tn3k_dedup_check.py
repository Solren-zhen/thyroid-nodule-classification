#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TN5000 <-> TN3K 跨数据集逐字节(MD5)查重（审稿意见 D8 补充）。

比对 TN5000 (5000 张 png) 与 TN3K 全部图像 (trainval-image + test-image,
3493 张 jpg) 的逐字节 MD5，确认两个训练/外部验证数据集之间无图像重复。

用法：python tools/tn5000_tn3k_dedup_check.py
输出：paper/output/repro/tn5000_tn3k_dedup.json
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "thyroid"
TN5000_DIR = ROOT / "tn5000"
TN3K_DIR = ROOT / "tn3k"
OUT = Path(__file__).resolve().parents[1] / "paper" / "output" / "repro" / "tn5000_tn3k_dedup.json"

CHUNK = 1024 * 1024


def md5_of(path: Path) -> tuple:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return str(path), h.hexdigest()


def discover(root: Path, exts=("*.png", "*.jpg", "*.jpeg")):
    files = []
    for ext in exts:
        files.extend(root.rglob(ext))
    return [f for f in files if ".cache" not in f.parts]


def main():
    tn5000_files = discover(TN5000_DIR, ("*.png",))
    tn3k_files = []
    for sub in ("trainval-image", "test-image"):
        p = TN3K_DIR / sub
        if p.exists():
            tn3k_files.extend(discover(p, ("*.jpg",)))
    print(f"[TN5000] {len(tn5000_files)} png", flush=True)
    print(f"[TN3K]   {len(tn3k_files)} jpg (trainval-image + test-image)", flush=True)

    def hash_all(files, label):
        h2p = {}
        with ProcessPoolExecutor() as ex:
            futs = [ex.submit(md5_of, p) for p in files]
            for i, fu in enumerate(as_completed(futs), 1):
                p, h = fu.result()
                h2p.setdefault(h, []).append(p)
                if i % 2000 == 0:
                    print(f"  [{label}] hashed {i}/{len(files)}", flush=True)
        print(f"[{label}] unique md5: {len(h2p)}", flush=True)
        return h2p

    t5 = hash_all(tn5000_files, "TN5000")
    t3 = hash_all(tn3k_files, "TN3K")

    overlap = sorted(set(t5) & set(t3))
    print("\n===== 结果 =====", flush=True)
    print(f"TN5000: {len(tn5000_files)} 文件, {len(t5)} 个唯一 MD5")
    print(f"TN3K:   {len(tn3k_files)} 文件, {len(t3)} 个唯一 MD5")
    print(f"跨数据集重复 MD5 数: {len(overlap)}")
    if overlap:
        print("重复样本示例:")
        for h in overlap[:10]:
            print(f"  {h[:12]}  TN5000: {t5[h][0]}  TN3K: {t3[h][0]}")

    out = {
        "tn5000_files": len(tn5000_files),
        "tn5000_unique_md5": len(t5),
        "tn3k_files": len(tn3k_files),
        "tn3k_unique_md5": len(t3),
        "cross_dataset_duplicate_md5": len(overlap),
        "duplicate_samples": [
            {"md5": h, "tn5000": t5[h][0], "tn3k": t3[h][0]} for h in list(overlap)[:10]
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    sys.exit(main())
