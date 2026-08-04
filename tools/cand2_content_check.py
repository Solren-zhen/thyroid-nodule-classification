# -*- coding: utf-8 -*-
"""cand2 内容级去重（dHash 感知哈希）。

MD5 只能查逐字节重复；若 cand2 是 TN3K/TN5000 重编码（改尺寸/转 JPG），
dHash（9x8 差分哈希）可命中。对每张 cand2 图算 64bit dHash，
在参考集 dHash 集合中精确匹配（同图重编码后 dHash 通常不变）。
用法：python tools/cand2_content_check.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(r"C:\Users\甄朝晖\Desktop\thyroid\data\thyroid")
REF_PATTERNS = {
    "tn5000": ROOT / "tn5000",
    "tn3k": ROOT / "tn3k" / "datasets" / "tn3k",
}
CAND2_DIR = ROOT / "cand2"


def dhash(path: Path, hash_size=9) -> str:
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size, hash_size + 1), Image.LANCZOS)
        px = list(im.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            diff.append("1" if left > right else "0")
    return "".join(diff)


def hash_one(path: Path):
    try:
        return str(path), dhash(path)
    except Exception as e:  # noqa: BLE001
        return str(path), f"ERR:{e}"


def discover(root: Path):
    return [f for f in root.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg") and ".cache" not in f.parts]


def main():
    ref_files = []
    for name, root in REF_PATTERNS.items():
        fs = discover(root)
        ref_files.extend(fs)
        print(f"[ref] {name}: {len(fs)}", flush=True)
    cand2_files = discover(CAND2_DIR)
    print(f"[cand2] {len(cand2_files)}", flush=True)

    # 参考集 dHash（去重）
    ref_hash = set()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(hash_one, p) for p in ref_files]
        for i, fu in enumerate(as_completed(futs), 1):
            _, h = fu.result()
            if not h.startswith("ERR:"):
                ref_hash.add(h)
            if i % 2000 == 0:
                print(f"  ref {i}/{len(ref_files)}", flush=True)
    print(f"[ref] unique dhash: {len(ref_hash)}", flush=True)

    # cand2 逐张比对
    dup = []
    unique = []
    errs = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(hash_one, p) for p in cand2_files]
        for i, fu in enumerate(as_completed(futs), 1):
            p, h = fu.result()
            if h.startswith("ERR:"):
                errs += 1
                print(f"  warn {p}: {h}", flush=True)
            elif h in ref_hash:
                dup.append(p)
            else:
                unique.append(p)
            if i % 1000 == 0:
                print(f"  cand2 {i}/{len(cand2_files)}", flush=True)

    print("\n===== 内容级结果 =====", flush=True)
    print(f"cand2 总数: {len(cand2_files)}")
    print(f"dHash 命中参考集（疑似重编码重复）: {len(dup)}")
    print(f"独立: {len(unique)}")
    print(f"读取失败: {errs}")

    out = {
        "cand2_total": len(cand2_files),
        "dhash_dup": len(dup),
        "unique": len(unique),
        "errors": errs,
        "sample_dhash_dups": dup[:10],
    }
    (ROOT / "cand2_content_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"报告已写入 {ROOT / 'cand2_content_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
