# -*- coding: utf-8 -*-
"""TI-RADS 人工标注工具（方案 B）。

读取 sample_tirads_annotations.py 生成的标注 CSV，逐张显示超声图，
按 ACR TI-RADS 词典录入 5 个视觉描述符。增量保存：随时退出不丢进度。

设计要点：
- 盲法：界面不显示良性/恶性金标准标签，避免标注偏差
- 5 维（composition/echogenicity/shape/margin/echogenic_foci）；
  size_mm 图中无比例尺无法读出，留空（见 tirads_annotation_protocol.md）
- 命令：q=保存退出，s=跳过本张，r=重看当前图（无图环境用 p 打印路径）

用法：
  python tools/annotate_tirads.py --in data/thyroid/tirads_annotations/tn5000_annotate.csv \
      --out data/thyroid/tirads_annotations/tn5000_annotated.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

CRITERIA = {
    "composition": "0=囊性/海绵 1=混合 2=实性",
    "echogenicity": "0=无回声 1=高/等回声 2=低回声 3=极低回声",
    "shape": "0=宽>高 3=高>宽",
    "margin": "0=光滑 1=模糊 2=分叶/不规则 3=甲状腺外侵犯",
    "echogenic_foci": "0=无/大彗尾 1=粗大钙化 2=周边钙化 3=点状强回声",
}
MAX_CODES = {
    "composition": 2, "echogenicity": 3, "shape": 3,
    "margin": 3, "echogenic_foci": 3,
}
FEATURES = list(CRITERIA)


def show_image(path: str, wait_ms=800):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    max_w = 800
    if img.shape[1] > max_w:
        r = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * r)))
    cv2.imshow("TI-RADS annotate", img)
    cv2.waitKey(wait_ms)
    return True


def prompt_code(feature: str) -> int:
    while True:
        v = input(f"  {feature} [{CRITERIA[feature]}]: ").strip()
        if v == "":
            return None
        try:
            n = int(v)
            if 0 <= n <= MAX_CODES[feature]:
                return n
        except ValueError:
            pass
        print(f"    无效输入，允许 0-{MAX_CODES[feature]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-window", action="store_true",
                    help="无 GUI 环境：只打印图片路径，不弹窗")
    args = ap.parse_args()

    rows = []
    with open(args.inp, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # 续标：已有全部特征的行跳过
    todo = [r for r in rows if not all(r.get(ft) for ft in FEATURES)]
    print(f"总 {len(rows)} 张，待标注 {len(todo)} 张（{len(rows)-len(todo)} 已标）\n")
    if not todo:
        print("全部已标注。")
        return

    for idx, row in enumerate(todo, 1):
        print(f"\n[{idx}/{len(todo)}] {Path(row['image_path']).name}"
              + ("" if args.no_window else "  （窗口显示中，可 r 重看）"))
        if args.no_window:
            print(f"  PATH: {row['image_path']}")
        else:
            if not show_image(row["image_path"], wait_ms=1):
                print(f"  [无法显示，仅路径] {row['image_path']}")
        while True:
            cmd = input("  录入 5 个代码(逗号分隔) 或 q=退出/s=跳过/r=重看: ").strip().lower()
            if cmd in ("q", "quit"):
                save(rows, args.out)
                print(f"已保存到 {args.out}")
                return
            if cmd in ("s", "skip"):
                break
            if cmd in ("r", "re"):
                if args.no_window:
                    print(f"  PATH: {row['image_path']}")
                else:
                    show_image(row["image_path"], wait_ms=1)
                continue
            parts = [p.strip() for p in cmd.replace(",", " ").split()]
            if len(parts) != len(FEATURES):
                print(f"  需要 {len(FEATURES)} 个代码，收到 {len(parts)}")
                continue
            try:
                codes = [int(p) for p in parts]
                if any(c < 0 or c > MAX_CODES[f] for c, f in zip(codes, FEATURES)):
                    print("  有代码超出范围")
                    continue
            except ValueError:
                print("  请输入数字")
                continue
            for ft, c in zip(FEATURES, codes):
                row[ft] = str(c)
            save(rows, args.out)
            print(f"  ✓ 已录 {','.join(parts)} 并保存")
            break
    save(rows, args.out)
    print(f"\n完成，共标注 {len(rows)} 张 → {args.out}")


def save(rows, out: str):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断，已保留上次增量保存结果。")
        sys.exit(130)
