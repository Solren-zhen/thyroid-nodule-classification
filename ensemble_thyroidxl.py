#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multi-seed 集成评估：加载多个 seed 的 fusion 模型，各自推理 test 集
（结节级 mean 聚合），3 个模型的结节概率平均 → 最终 AUC/AUPRC。

用法:
  python ensemble_thyroidxl.py --seeds 42 123 2024 --split test
  # Protocol: pos_weight=1.0 (seed42 = fusion/best.pt; 123/2024 = fusion_seed*_pw1.0)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ.parent))
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier
from train_thyroid import compute_metrics

CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    model = ThyroidClassifier(
        encoder_name=mc["encoder_name"], pretrained=mc["pretrained"],
        vis_dim=mc["vis_dim"], clinical_feature_dim=mc["clinical_feature_dim"],
        clinical_hidden_dim=mc["clinical_hidden_dim"],
        clinical_output_dim=mc["clinical_output_dim"],
        head_hidden=mc["head_hidden"], head_dropout=mc["head_dropout"],
        use_image=mc["use_image"], use_clinical=mc["use_clinical"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def predict_nodule(model, ds, device, batch_size=32, tta=False):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            images = batch["image"].to(device)
            if tta:
                views = [images, torch.flip(images, dims=[3]),
                         torch.flip(images, dims=[2]),
                         torch.rot90(images, 1, dims=[2, 3]),
                         torch.rot90(images, -1, dims=[2, 3])]
                acc = None
                for v in views:
                    o = model(images=v, clinical=clinical)
                    acc = o["probs"] if acc is None else acc + o["probs"]
                pb = (acc / len(views)).cpu().numpy().flatten()
            else:
                o = model(images=images, clinical=clinical)
                pb = o["probs"].cpu().numpy().flatten()
            probs.append(pb)
            labels.append(batch["label"].numpy().flatten())
            pids.append(list(batch["patient_id"]))
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    pids = np.concatenate(pids)
    groups = {}
    for pid, p in zip(pids, probs):
        groups.setdefault(pid, []).append(p)
    agg_p = np.array([np.mean(ps) for ps in groups.values()])
    agg_l = np.array([int(labels[np.where(pids == pid)[0][0]]) for pid in groups])
    return agg_p, agg_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--split", default="test")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="checkpoints/thyroid/ensemble_fusion.json")
    args = ap.parse_args()

    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split=args.split,
                        image_size=224, num_clinical_features=5, clinical_columns=CLIN,
                        mock=False, split_by_group=True, seed=42)

    ensemble_p = None
    for seed in args.seeds:
        if seed == 42:
            wp = PROJ / "checkpoints" / "thyroid" / "fusion" / "best.pt"
        else:
            wp = PROJ / "checkpoints" / "thyroid" / f"fusion_seed{seed}_pw1.0" / "best.pt"
        if not wp.exists():
            raise FileNotFoundError(f"missing final-protocol checkpoint: {wp}")
        print(f"loading {wp} ...")
        model = load_model(wp, device)
        p, y = predict_nodule(model, ds, device, tta=args.tta)
        if ensemble_p is None:
            ensemble_p = np.zeros_like(p)
            labels = y
        ensemble_p += p
        m = compute_metrics(y, p)
        print(f"  seed {seed}: AUC {m['auc']:.4f} | AUPRC {m['auprc']:.4f}")

    ensemble_p /= len(args.seeds)
    m = compute_metrics(labels, ensemble_p)
    print("\n=== ENSEMBLE ===")
    print(f"  AUC      : {m['auc']:.4f} (95% CI {m['auc_ci'][0]:.4f}-{m['auc_ci'][1]:.4f})")
    print(f"  AUPRC    : {m['auprc']:.4f}")
    print(f"  Sens/Spec: {m['sensitivity']:.4f} / {m['specificity']:.4f}")
    print(f"  ACC      : {m['acc']:.4f} | ECE {m['ece']:.4f} | n={len(labels)}")

    out = PROJ / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ensemble": {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()},
        "seeds": args.seeds, "tta": args.tta, "n": int(len(labels)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
