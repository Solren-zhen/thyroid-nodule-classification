#!/usr/bin/env python3
"""Thy-Wise <-> (TN5000 + TN3K) 跨数据集逐字节(MD5)查重。

审稿意见 D8 补充：为手稿中 "no Thy-Wise image is duplicated in the TN5000 or
TN3K datasets" 的声明提供可复现的报告文件。

用法：python tools/thywise_dedup_check.py
输出：paper/output/repro/thywise_dedup.json
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "thyroid"
THYWISE_DIR = ROOT / "thywise" / "images"
REF_DIRS = {
    "tn5000": ROOT / "tn5000",
    "tn3k": ROOT / "tn3k",
}
TN3K_IMAGE_SUBS = ("trainval-image", "test-image")
OUT = Path(__file__).resolve().parents[1] / "paper" / "output" / "repro" / "thywise_dedup.json"

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
    # 参考集：TN5000 全部 + TN3K 图像（trainval-image + test-image）
    ref_files = discover(REF_DIRS["tn5000"], ("*.png",))
    for sub in TN3K_IMAGE_SUBS:
        p = REF_DIRS["tn3k"] / sub
        if p.exists():
            ref_files.extend(discover(p, ("*.jpg",)))
    tw_files = discover(THYWISE_DIR)
    print(f"[ref] TN5000 + TN3K 图像: {len(ref_files)} 文件", flush=True)
    print(f"[Thy-Wise] {len(tw_files)} 文件", flush=True)

    def hash_all(files, label):
        h2p = {}
        with ProcessPoolExecutor() as ex:
            futs = [ex.submit(md5_of, p) for p in files]
            for i, fu in enumerate(as_completed(futs), 1):
                p, h = fu.result()
                h2p.setdefault(h, []).append(p)
                if i % 5000 == 0:
                    print(f"  [{label}] hashed {i}/{len(files)}", flush=True)
        print(f"[{label}] unique md5: {len(h2p)}", flush=True)
        return h2p

    ref = hash_all(ref_files, "REF")
    tw = hash_all(tw_files, "Thy-Wise")

    overlap = sorted(set(ref) & set(tw))
    print("\n===== 结果 =====", flush=True)
    print(f"参考集: {len(ref_files)} 文件, {len(ref)} 个唯一 MD5")
    print(f"Thy-Wise: {len(tw_files)} 文件, {len(tw)} 个唯一 MD5")
    print(f"跨数据集重复 MD5 数: {len(overlap)}")
    if overlap:
        print("重复样本示例:")
        for h in overlap[:10]:
            print(f"  {h[:12]}  ref: {ref[h][0]}  Thy-Wise: {tw[h][0]}")

    out = {
        "reference_files": len(ref_files),
        "reference_unique_md5": len(ref),
        "thywise_files": len(tw_files),
        "thywise_unique_md5": len(tw),
        "cross_dataset_duplicate_md5": len(overlap),
        "duplicate_samples": [
            {"md5": h, "reference": ref[h][0], "thywise": tw[h][0]} for h in list(overlap)[:10]
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    sys.exit(main())
