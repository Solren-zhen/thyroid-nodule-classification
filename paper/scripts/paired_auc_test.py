#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fusion vs image-only 的配对 ΔAUC 检验（ThyroidXL test，seed 42 锚点）。

同一 739 个结节上，用 paired bootstrap（重抽样结节索引，两模型同索引）得到
ΔAUC 的 95% CI 与 p 值（Δ ≤ 0 的比例×2，双侧）。

用法: python paper/scripts/paired_auc_test.py
依赖: eval 管线完成后 GPU 空闲时运行。
"""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))  # 使 `import thyroid`（项目根即包）可用
sys.path.insert(0, str(PROJ))          # 使 `from train_thyroid import ...` 可用
from torch.utils.data import DataLoader
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier
from train_thyroid import get_device

CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]

FUSION_CKPT = PROJ / "checkpoints" / "thyroid" / "fusion_backup" / "best_posw1.0_seed42_20260809.pt"
IMAGE_CKPT = PROJ / "checkpoints" / "thyroid" / "image" / "best.pt"


def load_model(wp: Path, device):
    import torch
    ckpt = torch.load(wp, map_location="cpu", weights_only=False)
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


def predict_nodule_mean(model, ds, device, batch_size=32):
    """结节级聚合：帧概率取平均。与 eval_thyroid.py 口径一致。"""
    import numpy as np
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            images = batch["image"].to(device)
            o = model(images=images, clinical=clinical)
            probs.append(o["probs"].cpu().numpy().flatten())
            labels.append(batch["label"].numpy().flatten())
            pids.append(list(batch["patient_id"]))
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    pids = np.concatenate(pids)
    groups = {}
    for pid, p in zip(pids, probs):
        groups.setdefault(pid, []).append(p)
    agg_p = np.array([np.mean(ps) for ps in groups.values()])
    seen = {}
    agg_l = []
    for pid in groups:
        seen.setdefault(pid, int(labels[np.where(pids == pid)[0][0]]))
        agg_l.append(seen[pid])
    return agg_p, np.array(agg_l)


def main():
    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                        image_size=224, num_clinical_features=5, clinical_columns=CLIN,
                        mock=False, split_by_group=True, seed=42)

    print(f"loading fusion: {FUSION_CKPT.name}")
    fusion = load_model(FUSION_CKPT, device)
    p_f, y = predict_nodule_mean(fusion, ds, device)

    print(f"loading image-only: {IMAGE_CKPT.name}")
    image = load_model(IMAGE_CKPT, device)
    p_i, y2 = predict_nodule_mean(image, ds, device)

    assert np.array_equal(y, y2), "label mismatch"
    n = len(y)
    auc_f = roc_auc_score(y, p_f)
    auc_i = roc_auc_score(y, p_i)
    d_obs = auc_f - auc_i
    print(f"\nseed-42 anchor | n={n}")
    print(f"  fusion     AUC: {auc_f:.4f}")
    print(f"  image-only AUC: {auc_i:.4f}")
    print(f"  ΔAUC (fusion-image): {d_obs:+.4f}")

    # paired bootstrap of the difference
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        a_f = roc_auc_score(y[idx], p_f[idx])
        a_i = roc_auc_score(y[idx], p_i[idx])
        diffs.append(a_f - a_i)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one = float(np.mean(diffs <= 0))
    p_two = 2 * min(p_one, 1 - p_one)
    print(f"  ΔAUC 95% CI (bootstrap): [{lo:+.4f}, {hi:+.4f}]")
    print(f"  p-value (two-sided, Δ≤0 proportion): {p_two:.4f}")

    # 保存结果
    import json
    out = PROJ / "checkpoints" / "thyroid" / "seed_retrain_eval" / "paired_fusion_vs_image.json"
    out.write_text(json.dumps({
        "n": n, "fusion_auc": auc_f, "image_auc": auc_i, "delta_auc": d_obs,
        "delta_ci": [lo, hi], "p_two_sided": p_two, "n_boot": len(diffs),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
