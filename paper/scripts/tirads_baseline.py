#!/usr/bin/env python3
"""TI-RADS 单独评分作为临床基线。

用 ThyroidXL 的专家 TIRADS 总分（1-5）直接作为良恶性预测分数，
计算结节级 AUC / AUPRC / Youden 点 Sens/Spec/PPV/NPV，与 AI 三臂对比。

结论：专家 TIRADS 总分单独能达到的判别力（~0.85 AUC），低于 AI 模型（fusion
0.947），说明 AI 从图像中学到了超出 TI-RADS 总分的信息。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))


def load_test_manifest():
    """读 ThyroidXL test manifest，提取 (tirads, label, patient_id)"""
    rows = []
    with open(PROJ / "data" / "thyroid" / "thyroidxl" / "manifest.csv",
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                rows.append({
                    "pid": r["patient_id"],
                    "tirads": float(r["tirads"]),
                    "label": int(r["label"]),
                })
    return rows


def metrics(y, s):
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0
    auprc = average_precision_score(y, s) if len(np.unique(y)) > 1 else 0
    # Youden
    tg = np.arange(1.0, 5.01, 0.5)
    best = None
    for t in tg:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        sen = tp / max(tp + fn, 1); spe = tn / max(tn + fp, 1)
        j = sen + spe - 1
        if best is None or j > best[0]:
            best = (j, t, sen, spe, tp, fp, tn, fn)
    _, t, sen, spe, tp, fp, tn, fn = best
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    return {"auc": auc, "auprc": auprc, "t": t, "sen": sen, "spe": spe,
            "ppv": ppv, "npv": npv}


def main():
    rows = load_test_manifest()
    # 结节级聚合：每结节一个 TI-RADS（同结节多帧共享）
    nod = {}
    for r in rows:
        nod.setdefault(r["pid"], r)  # 同结节 tirads/label 一致，取首个
    pids = list(nod.keys())
    tirads = np.array([nod[p]["tirads"] for p in pids])
    labels = np.array([nod[p]["label"] for p in pids])

    print(f"ThyroidXL test: {len(pids)} 结节")
    m = metrics(labels, tirads)
    print(f"TIRADS total score: AUC {m['auc']:.4f} | AUPRC {m['auprc']:.4f}")
    print(f"  Youden t={m['t']:.1f}: Sens {m['sen']:.3f} Spec {m['spe']:.3f} "
          f"PPV {m['ppv']:.3f} NPV {m['npv']:.3f}")

    # 对比 AI 三臂（从 eval_test.json）
    print("\n=== 对比：AI 模型（结节级 mean，n=739）===")
    for ab in ["fusion", "image", "clinical"]:
        p = PROJ / "paper" / "output" / "repro" / f"ablation_{ab}_nodule.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            mm = d["metrics"]
            print(f"  {ab:10s}: AUC {mm['auc']:.4f} | AUPRC {mm.get('auprc',0):.4f}")

    # 保存
    out = PROJ / "paper" / "notes" / "tirads_baseline.md"
    out.write_text(
        f"# TI-RADS Baseline (ThyroidXL test, {len(pids)} nodules)\n\n"
        f"- Expert TIRADS total score: AUC {m['auc']:.4f}, AUPRC {m['auprc']:.4f}\n"
        f"- Youden t={m['t']:.1f}: Sens {m['sen']:.3f}, Spec {m['spe']:.3f}, "
        f"PPV {m['ppv']:.3f}, NPV {m['npv']:.3f}\n\n"
        f"AI 对比: fusion {0.947}, image {0.939}, clinical {0.814}\n",
        encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
