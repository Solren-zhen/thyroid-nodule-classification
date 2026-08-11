#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 ThyroidXL 三臂消融 test 评估 → Table 3 数据 + markdown 表格。

读取 checkpoints/thyroid/{fusion,image,clinical}/eval_test.json，
输出三臂对比表（AUC/CI/AUPRC/Sens/Spec/ACC/ECE）到 stdout + paper/notes/thyroidxl_ablation_summary.md
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
ABLATIONS = ["image", "clinical", "fusion"]
LABELS = {"image": "Image-only", "clinical": "Clinical-only", "fusion": "Fusion"}


def load_eval(ab):
    p = PROJ / "checkpoints" / "thyroid" / ab / "eval_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    rows = []
    for ab in ABLATIONS:
        d = load_eval(ab)
        if d is None:
            print(f"⚠ {ab}: eval_test.json 不存在")
            continue
        m = d["metrics"]
        rows.append({
            "ab": ab,
            "auc": m["auc"], "ci": f"{m['auc_ci'][0]:.3f}-{m['auc_ci'][1]:.3f}",
            "auprc": m["auprc"],
            "sens": m["sensitivity"], "spec": m["specificity"],
            "acc": m["acc"], "ece": m["ece"], "n": d.get("n", "?"),
        })
        print(f"{LABELS[ab]:14s} AUC {m['auc']:.4f} ({rows[-1]['ci']}) | "
              f"AUPRC {m['auprc']:.4f} | Sens {m['sensitivity']:.3f} | "
              f"Spec {m['specificity']:.3f} | ACC {m['acc']:.3f} | ECE {m['ece']:.4f}")

    # markdown table
    md = ["| Model | AUC (95% CI) | AUPRC | Sensitivity | Specificity | ACC | ECE | n |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {LABELS[r['ab']]} | {r['auc']:.3f} ({r['ci']}) | {r['auprc']:.3f} | "
                  f"{r['sens']:.3f} | {r['spec']:.3f} | {r['acc']:.3f} | {r['ece']:.3f} | {r['n']} |")
    out = PROJ / "paper" / "notes" / "thyroidxl_ablation_summary.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n已保存: {out}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
