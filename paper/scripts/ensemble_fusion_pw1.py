#!/usr/bin/env python3
"""融合三 seed（pos_weight=1.0 统一协议）集成 AUC 重算。

读取 seed42（fusion_backup/best_posw1.0_seed42）、seed123/2024（fusion_seed{123,2024}_pw1.0）
在 ThyroidXL test 的结节级概率，取平均得集成，计算 AUC/AUPRC/ECE。
依赖：完整评估管线跑完后 GPU 空闲。
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
sys.path.insert(0, str(PROJ))
import importlib.util

from data.thyroid_dataset import ThyroidDataset
from train_thyroid import get_device

# paper 不是包，用文件路径加载 paired_auc_test 中的 load_model / predict_nodule_mean
_pat = PROJ / "paper" / "scripts" / "paired_auc_test.py"
_spec = importlib.util.spec_from_file_location("paired_auc_test", _pat)
_pat_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pat_mod)
load_model = _pat_mod.load_model
predict_nodule_mean = _pat_mod.predict_nodule_mean

CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]
CKPTS = {
    42: PROJ / "checkpoints" / "thyroid" / "fusion_backup" / "best_posw1.0_seed42_20260809.pt",
    123: PROJ / "checkpoints" / "thyroid" / "fusion_seed123_pw1.0" / "best.pt",
    2024: PROJ / "checkpoints" / "thyroid" / "fusion_seed2024_pw1.0" / "best.pt",
}


def main():
    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                        image_size=224, num_clinical_features=5, clinical_columns=CLIN,
                        mock=False, split_by_group=True, seed=42)
    from sklearn.metrics import average_precision_score, roc_auc_score

    from train_thyroid import bootstrap_auc

    ensemble_p = None
    labels = None
    print("=== 单 seed 结节级 AUC（pw1.0）===")
    for seed, wp in CKPTS.items():
        print(f"loading seed {seed}: {wp.name}")
        model = load_model(wp, device)
        p, y = predict_nodule_mean(model, ds, device)
        if ensemble_p is None:
            ensemble_p = np.zeros_like(p)
            labels = y
        ensemble_p += p
        auc = roc_auc_score(y, p)
        print(f"  seed {seed}: AUC {auc:.4f}")
    ensemble_p /= len(CKPTS)
    m_auc = roc_auc_score(labels, ensemble_p)
    m_auprc = average_precision_score(labels, ensemble_p)
    _, lo, hi = bootstrap_auc(labels, ensemble_p, n_boot=2000, seed=0)
    print("\n=== ENSEMBLE (3 seeds, pw1.0) ===")
    print(f"  AUC      : {m_auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    print(f"  AUPRC    : {m_auprc:.4f}")

    out = PROJ / "checkpoints" / "thyroid" / "seed_retrain_eval" / "ensemble_fusion_pw1.json"
    out.write_text(json.dumps({
        "ensemble_auc": m_auc, "auc_ci": [lo, hi], "ensemble_auprc": m_auprc,
        "seeds": list(CKPTS.keys()), "n": len(labels),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
