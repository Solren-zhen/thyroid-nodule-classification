#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终轮一致性核对：Table 4 / per-image / supplementary 数字 vs 脚本产出。"""
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]

manu = (PROJ / "paper" / "output" / "doc" / "manuscript.md").read_text(encoding="utf-8")
notes = (PROJ / "paper" / "notes" / "subgroup_analysis.md").read_text(encoding="utf-8")
supp = (PROJ / "paper" / "output" / "supplementary.md").read_text(encoding="utf-8")
pi = json.loads((PROJ / "checkpoints" / "thyroid" / "per_image_stats.json").read_text(encoding="utf-8"))
mil = json.loads((PROJ / "checkpoints" / "thyroid" / "mil_sensitivity.json").read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return s.replace("\u2013", "-").replace("\u2212", "-").replace("\u2014", "-").replace("\uFF0D", "-")


def extract_rows(md: str):
    rows = []
    for line in md.splitlines():
        m = re.match(
            r"\| (TI-RADS[^\|]*|Nodule size[^\|]*|Age \(years\)[^\|]*|Sex[^\|]*) "
            r"\| (\d+) \| ([\d.]+)% \| ([\d.]+ \([^\)]+\)) \| ([\d.]+ \([^\)]+\)) "
            r"\| ([\d.]+ \([^\)]+\)) \| ([+\-−][\d.]+|—) \|", line)
        if m:
            rows.append(tuple(m.groups()))
    return rows


manu_rows = extract_rows(manu)
notes_rows = extract_rows(notes)
print(f"手稿 Table4 行: {len(manu_rows)}, notes 行: {len(notes_rows)}")
ok = True
for a, b in zip(manu_rows, notes_rows):
    same = (a[1] == b[1] and a[2] == b[2] and norm(a[3]) == norm(b[3])
            and norm(a[4]) == norm(b[4]) and norm(a[5]) == norm(b[5])
            and norm(a[6]) == norm(b[6]))
    if not same:
        ok = False
        print("  不一致:")
        print("   manu:", a)
        print("   note:", b)
print("Table 4 核对:", "PASS" if ok else "FAIL")

# ---- per-image stats vs supplementary ----
print("\nper_image_stats.json (ThyroidXL per-image):")
exp = {
    "image": "0.909 (0.896-0.920)",
    "clinical": "0.826 (0.807-0.844)",
    "fusion": "0.917 (0.904-0.928)",
}
for ab, d in pi.items():
    got = f"{d['auc']:.3f} ({d['ci_lo']:.3f}-{d['ci_hi']:.3f})"
    print(f"  {ab}: {got}")
    e = exp[ab]
    if norm(got) != norm(e):
        print(f"    !! supplementary 期望 {e} 与脚本不一致")
        ok = False
for ab, e in exp.items():
    if e.replace("-", "\u2013") not in supp and e not in supp:
        pass  # en-dash 在 supp 中
    en = e.replace("-", "\u2013")
    if en not in supp and e not in supp:
        print(f"  !! supplementary 缺 {ab} per-image 值 {e}")
        ok = False

# ---- MIL sensitivity vs supplementary Table S2 ----
print("\nmil_sensitivity.json vs supplementary S2:")
s2 = {
    "image": {"mean": "0.939 (0.924-0.954)", "max": "0.927 (0.909-0.943)", "attention": "0.928 (0.910-0.944)"},
    "fusion": {"mean": "0.947 (0.932-0.960)", "max": "0.938 (0.921-0.953)", "attention": "0.940 (0.924-0.954)"},
}
for ab, methods in mil.items():
    for meth, d in methods.items():
        got = f"{d['auc']:.3f} ({d['auc_ci'][0]:.3f}-{d['auc_ci'][1]:.3f})"
        e = s2[ab][meth]
        en = e.replace("-", "\u2013")
        flag = "OK" if (norm(got) == norm(e) and (en in supp or e in supp)) else "!! MISMATCH"
        print(f"  {ab}/{meth}: {got} vs 期望 {e} -> {flag}")
        if flag != "OK":
            ok = False

print("\n总核对:", "PASS" if ok else "FAIL")
