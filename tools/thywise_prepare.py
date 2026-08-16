"""Thy-Wise 解压 + 建 manifest + MD5 去重（第三外部验证队列的前提）。

1. 从外层 zip 解出 benign_after.zip / malignant_after.zip → 再解到
   data/thyroid/thywise/images/{benign,malignant}/<nodule folder>/xxx.jpg
2. 建 manifest（patient_id = 结节文件夹名，label 从文件夹名恢复，split=test）
3. 计算全部 Thy-Wise 图像 MD5，与 TN5000+TN3K 参考集比对，确认零重叠

用法：python tools/thywise_prepare.py
"""
import csv
import hashlib
import io
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTER = ROOT / "data" / "thyroid" / "thywise" / "thywise_us_images.zip"
IMG_ROOT = ROOT / "data" / "thyroid" / "thywise" / "images"
MANIFEST = ROOT / "data" / "thyroid" / "thywise" / "manifest.csv"

REF_PATTERNS = {
    "tn5000": ROOT / "data" / "thyroid" / "tn5000",
    "tn3k": ROOT / "data" / "thyroid" / "tn3k" / "datasets" / "tn3k",
}
CLINICAL = ["composition", "echogenicity", "shape", "margin", "echogenic_foci", "size_mm"]
CHUNK = 1024 * 1024


def md5_of(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract_all():
    outer = zipfile.ZipFile(OUTER)
    for inner_name, out_sub in [("benign_after.zip", "benign"),
                                ("malignant_after.zip", "malignant")]:
        dest = IMG_ROOT / out_sub
        if any(dest.rglob("*.jpg")):
            print(f"[extract] {out_sub} 已存在，跳过", flush=True)
            continue
        inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
        members = inner.namelist()
        print(f"[extract] {inner_name}: {len(members)} 条目", flush=True)
        for i, m in enumerate(members, 1):
            if m.endswith("/"):
                continue
            out_path = dest / m.replace(out_sub + "/", "", 1) if m.startswith(out_sub + "/") else dest / m
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with inner.open(m) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            if i % 5000 == 0:
                print(f"  extracted {i}/{len(members)}", flush=True)
        print(f"[extract] {out_sub} 完成", flush=True)


def build_manifest():
    """建 manifest，跳过无法解码的图片（源 zip 本身损坏的坏文件）。"""
    from PIL import Image
    rows = []
    skipped = 0
    for label, folder in [(0, "benign"), (1, "malignant")]:
        base = IMG_ROOT / folder
        for jpg in base.rglob("*.jpg"):
            try:
                with Image.open(jpg) as im:
                    im.load()
            except Exception:  # noqa: BLE001 - 单个坏文件跳过
                skipped += 1
                continue
            pid = jpg.parent.name  # 结节文件夹名（跨良恶性唯一）
            rows.append([str(jpg), pid, label, "test"] + [""] * len(CLINICAL))
    print(f"[manifest] 跳过坏文件 {skipped} 张", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "patient_id", "label", "split"] + CLINICAL)
        w.writerows(rows)
    print(f"[manifest] {len(rows)} 行 → {MANIFEST}", flush=True)
    return rows


def dedup():
    # 参考集 MD5
    ref = []
    for name, root in REF_PATTERNS.items():
        fs = [p for p in root.rglob("*")
              if p.suffix.lower() in (".png", ".jpg", ".jpeg") and ".cache" not in p.parts]
        ref.extend(fs)
        print(f"[ref] {name}: {len(fs)}", flush=True)
    ref_md5 = set()
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(md5_of, p) for p in ref]
        for i, fu in enumerate(as_completed(futs), 1):
            ref_md5.add(fu.result())
            if i % 4000 == 0:
                print(f"  ref md5 {i}/{len(ref)}", flush=True)
    print(f"[ref] unique md5: {len(ref_md5)}", flush=True)

    # Thy-Wise MD5
    tw_files = list(IMG_ROOT.rglob("*.jpg"))
    dup = []
    uniq = []
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(md5_of, p) for p in tw_files]
        for i, fu in enumerate(as_completed(futs), 1):
            if fu.result() in ref_md5:
                dup.append(str(fu))
            else:
                uniq.append(str(fu))
            if i % 5000 == 0:
                print(f"  thywise md5 {i}/{len(tw_files)}", flush=True)

    print("\n===== MD5 去重结果 =====", flush=True)
    print(f"Thy-Wise 图像总数: {len(tw_files)}")
    print(f"与 TN5000/TN3K 重复: {len(dup)}")
    print(f"独立: {len(uniq)}")
    if dup:
        print("重复样例:", dup[:5])
    else:
        print("=> 零重叠，可用作第三外部验证队列 [OK]")


def main():
    extract_all()
    build_manifest()
    dedup()


if __name__ == "__main__":
    sys.exit(main())
