# -*- coding: utf-8 -*-
"""找出 24 张疑似重复 cand2 图片的参考集匹配文件，并输出对照拼图。

用法：python tools/cand2_verify_matches.py
输出：data/thyroid/cand2_matches.txt（配对清单）+ tools/cand2_matches.png（拼图）
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from cand2_content_check import dhash, discover  # 复用

ROOT = Path(__file__).resolve().parents[1] / "data" / "thyroid"
REF_PATTERNS = {
    "tn5000": ROOT / "tn5000",
    "tn3k": ROOT / "tn3k" / "datasets" / "tn3k",
}
CAND2_DIR = ROOT / "cand2"

THUMB = 160


def main():
    # 建立 ref dHash -> [paths] 映射
    ref_map = {}
    ref_files = []
    for root in REF_PATTERNS.values():
        ref_files.extend(discover(root))
    print(f"[ref] {len(ref_files)}", flush=True)
    for i, p in enumerate(ref_files, 1):
        h = dhash(p)
        ref_map.setdefault(h, []).append(p)
        if i % 2000 == 0:
            print(f"  ref {i}", flush=True)

    # 找出 cand2 中命中者
    hits = []
    for p in discover(CAND2_DIR):
        h = dhash(p)
        if h in ref_map:
            hits.append((p, ref_map[h]))
    print(f"[cand2] dHash 命中 {len(hits)}", flush=True)

    # 写配对清单
    lines = []
    for cand2_path, refs in hits:
        for r in refs[:4]:
            lines.append(f"{cand2_path}\t{r}")
    (ROOT / "cand2_matches.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"配对清单: {ROOT / 'cand2_matches.txt'} ({len(lines)} 行)")

    # 生成拼图：每对一行（cand2 | ref1 | ref2），最多 10 对
    ncols = 4
    rows = []
    for cand2_path, refs in hits[:10]:
        row_imgs = [Image.open(cand2_path).convert("RGB")]
        for r in refs[:3]:
            row_imgs.append(Image.open(r).convert("RGB"))
        # 补齐
        while len(row_imgs) < ncols:
            row_imgs.append(Image.new("RGB", row_imgs[0].size, (128, 128, 128)))
        rows.append(row_imgs)

    if rows:
        # 缩放到统一高度
        for i, row in enumerate(rows):
            for j, im in enumerate(row):
                im.thumbnail((THUMB, THUMB))
            rows[i] = [im.convert("RGB") for im in row]
        widths = [sum(im.width for im in row) for row in rows]
        max_w = max(widths)
        pad = 8
        canvas_h = sum(THUMB + 2 * pad for _ in rows) + pad
        canvas = Image.new("RGB", (max_w + pad, canvas_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        y = pad
        for idx, row in enumerate(rows):
            x = pad
            draw.text((x, y - 8), f"#{idx+1}", fill=(0, 0, 0))
            for im in row:
                canvas.paste(im, (x, y))
                x += im.width + pad
            y += THUMB + 2 * pad
        out_png = Path(__file__).parent / "cand2_matches.png"
        canvas.save(out_png)
        print(f"拼图已写入 {out_png}")


if __name__ == "__main__":
    sys.exit(main())
