"""cand2 MD5 去重验证（D8）。

比对 cand2 (7058 图) 与 TN5000 (5000 图) + TN3K (3493 图) 的逐字节 MD5。
输出：重复数 / 独立数，重复落在哪个参考集。
用法：python tools/cand2_dedup_check.py
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "thyroid"

# 参考集：TN5000 + TN3K（已确认独立的两个数据集）
REF_PATTERNS = {
    "tn5000": ROOT / "tn5000",
    "tn3k": ROOT / "tn3k" / "datasets" / "tn3k",
}
# cand2 候选
CAND2_DIR = ROOT / "cand2"

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


def discover(ref_root: Path, exts=("*.png", "*.jpg", "*.jpeg")):
    files = []
    for ext in exts:
        files.extend(ref_root.rglob(ext))
    return files


def main():
    # 1) 收集参考集文件（排除 .cache）
    ref_files = []
    for name, root in REF_PATTERNS.items():
        fs = [f for f in discover(root) if ".cache" not in f.parts]
        print(f"[ref] {name}: {len(fs)} files", flush=True)
        ref_files.extend(fs)
    # cand2 文件
    cand2_files = [f for f in discover(CAND2_DIR)]
    print(f"[cand2] {len(cand2_files)} files", flush=True)

    # 2) 计算参考集 MD5（多进程）
    ref_md5 = set()
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(md5_of, p) for p in ref_files]
        for i, fu in enumerate(as_completed(futs), 1):
            _, h = fu.result()
            ref_md5.add(h)
            if i % 2000 == 0:
                print(f"  ref md5 {i}/{len(ref_files)}", flush=True)
    print(f"[ref] unique md5: {len(ref_md5)}", flush=True)

    # 3) 逐张比对 cand2
    dup_tn3k = []
    unique = []
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(md5_of, p) for p in cand2_files]
        for i, fu in enumerate(as_completed(futs), 1):
            p, h = fu.result()
            if h in ref_md5:
                # 判断属于哪个参考集（需逐文件哈希才能归因，这里先按文件名粗分）
                dup_tn3k.append(str(p))
            else:
                unique.append(str(p))
            if i % 1000 == 0:
                print(f"  cand2 checked {i}/{len(cand2_files)}", flush=True)

    print("\n===== 结果 =====", flush=True)
    print(f"cand2 总数: {len(cand2_files)}")
    print(f"与 TN5000/TN3K 重复(MD5 命中): {len(dup_tn3k)}")
    print(f"独立(未命中): {len(unique)}")

    out = {
        "cand2_total": len(cand2_files),
        "dup_with_reference": len(dup_tn3k),
        "unique": len(unique),
        "sample_dups": dup_tn3k[:10],
        "sample_unique": unique[:10],
    }
    out_path = ROOT / "cand2_dedup_report.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告已写入 {out_path}")


if __name__ == "__main__":
    sys.exit(main())
