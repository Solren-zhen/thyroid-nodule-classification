#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""错误案例可视化：fusion 模型在 ThyroidXL test 的假阳/假阴病例。

生成 2x2 网格图：2 个假阳 + 2 个假阴，标注预测概率 / 真实标签 / TI-RADS。
输出 paper/figures/fig10_error_cases.png（增大行间距避免标题重叠）
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier

CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]
FIG_DIR = PROJ / "paper" / "figures"


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


def main():
    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                        image_size=224, num_clinical_features=5, clinical_columns=CLIN,
                        mock=False, split_by_group=True, seed=42)
    model = load_model(PROJ / "checkpoints" / "thyroid" / "fusion" / "best.pt", device)

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    # 收集 frame 级预测 + 样本信息
    recs = []
    global_idx = 0
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            images = batch["image"].to(device)
            o = model(images=images, clinical=clinical)
            probs = o["probs"].cpu().numpy().flatten()
            for i in range(len(probs)):
                recs.append({
                    "prob": probs[i],
                    "label": int(batch["label"][i].item()),
                    "pid": batch["patient_id"][i],
                    "image_path": ds.samples[global_idx]["image_path"],
                })
                global_idx += 1

    # 结节级 mean 聚合
    from collections import defaultdict
    groups = defaultdict(list)
    for r in recs:
        groups[r["pid"]].append(r)
    nod = {}
    for pid, rs in groups.items():
        p = np.mean([r["prob"] for r in rs])
        l = rs[0]["label"]
        nod[pid] = {"prob": p, "label": l, "image_path": rs[0]["image_path"],
                    "tirads": ds.samples[0]["clinical"][0]}  # placeholder

    # 找假阳（label 0, prob>=0.5）和假阴（label 1, prob<0.5）
    fp = sorted([v for v in nod.values() if v["label"] == 0 and v["prob"] >= 0.5],
                key=lambda v: -v["prob"])
    fn = sorted([v for v in nod.values() if v["label"] == 1 and v["prob"] < 0.5],
                key=lambda v: v["prob"])
    print(f"假阳 {len(fp)} 个, 假阴 {len(fn)} 个")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    cases = [(fp[0], "False positive (benign\npredicted malignant)"),
             (fp[1], "False positive (benign\npredicted malignant)"),
             (fn[0], "False negative (malignant\npredicted benign)"),
             (fn[1], "False negative (malignant\npredicted benign)")]
    for ax, (case, title) in zip(axes.flat, cases):
        img = cv2.imdecode(np.fromfile(case["image_path"], dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        lab = "Malignant" if case["label"] == 1 else "Benign"
        ax.set_title(f"{title}\nP={case['prob']:.2f} | Truth: {lab}",
                     fontsize=10, pad=12)
        ax.axis("off")
    fig.suptitle("Representative misclassifications of the fused model (ThyroidXL test)",
                 fontsize=12)
    fig.tight_layout(h_pad=3.2, w_pad=1.2, rect=[0, 0, 1, 0.955])
    out = FIG_DIR / "fig10_error_cases.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
