#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ThyroidXL test 亚组分析：按 TI-RADS / 结节尺寸 / 年龄 / 性别分层。

对三臂模型（fusion/image/clinical）在 ThyroidXL test 上重跑推理，
一次得到 per-image 概率与结节级 mean 聚合概率，然后：
  - 每亚组计算三臂结节级 AUC + bootstrap 95% CI、恶性率、n
  - 输出 paper/notes/subgroup_analysis.md（Table 4）+ paper/figures/fig9_subgroup_forest.png

分组方案（基于实际样本量）：
  TI-RADS : TR2-3 (score<=3) / TR4 / TR5   （TR2 n=45 过小，并入 TR3 作低中危组）
  尺寸    : <=10mm / 10-20mm / >20mm
  年龄    : <45 / 45-60 / >=60
  性别    : male / female
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ.parent))
sys.path.insert(0, str(PROJ))
from thyroid.data.thyroid_dataset import ThyroidDataset
from thyroid.models.thyroid import ThyroidClassifier
from train_thyroid import bootstrap_auc

FIG_DIR = PROJ / "paper" / "figures"
NOTES_DIR = PROJ / "paper" / "notes"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)

CLIN = ["tirads", "width_mm", "height_mm", "age", "gender"]
ABLATIONS = ["image", "clinical", "fusion"]
COLORS = {"image": "#0072B2", "clinical": "#E69F00", "fusion": "#D55E00"}
LABELS = {"image": "Image-only", "clinical": "Clinical-only", "fusion": "Full five-feature fusion"}
MANIFEST = PROJ / "data" / "thyroid" / "thyroidxl" / "manifest.csv"


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


def predict_per_image(model, ds, device, batch_size=32):
    """返回 per-image (probs, labels, pids)。"""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            if model.use_image:
                images = batch["image"].to(device)
                o = model(images=images, clinical=clinical)
            else:
                o = model(clinical=clinical)
            probs.append(o["probs"].cpu().numpy().flatten())
            labels.append(batch["label"].numpy().flatten())
            pids.append(list(batch["patient_id"]))
    return (np.concatenate(probs),
            np.concatenate(labels).astype(int),
            np.concatenate(pids))


def aggregate_nodule_mean(p, y, pids):
    """结节级 mean 聚合：每 patient 概率取平均。"""
    groups = {}
    for pid, prob in zip(pids, p):
        groups.setdefault(pid, []).append(prob)
    agg_p = np.array([np.mean(v) for v in groups.values()])
    # 每结节取首个 label（同结节 label 一致）
    agg_l = np.array([int(y[np.where(pids == pid)[0][0]]) for pid in groups])
    return agg_p, agg_l


def load_manifest_test():
    """读 ThyroidXL test manifest，构建 patient -> 临床特征/标签 映射。"""
    nod = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            pid = r["patient_id"]
            if pid not in nod:
                nod[pid] = {
                    "tirads": float(r["tirads"]),
                    "width_mm": float(r["width_mm"]),
                    "height_mm": float(r["height_mm"]),
                    "age": float(r["age"]),
                    "gender": r["gender"],
                    "label": int(r["label"]),
                }
    return nod


def subgroup_of(nod, key):
    """按 key 返回每个 patient 的亚组标签。"""
    if key == "tirads":
        def f(r):
            return "TR2-3" if r["tirads"] <= 3 else ("TR4" if r["tirads"] == 4 else "TR5")
    elif key == "size":
        def f(r):
            d = max(r["width_mm"], r["height_mm"])
            if d <= 10:
                return "<=10mm"
            if d <= 20:
                return "10-20mm"
            return ">20mm"
    elif key == "age":
        def f(r):
            if r["age"] < 45:
                return "<45"
            if r["age"] < 60:
                return "45-60"
            return ">=60"
    elif key == "gender":
        def f(r):
            return "male" if r["gender"] == "1" else "female"
    else:
        raise ValueError(key)
    return f


def subgroup_auc(labels, probs):
    """返回 (auc, ci_lo, ci_hi)；若亚组单一类别则返回 None。"""
    if len(np.unique(labels)) < 2:
        return None
    auc = float(roc_auc_score(labels, probs))
    _, lo, hi = bootstrap_auc(np.asarray(labels), np.asarray(probs), n_boot=1000, seed=0)
    return auc, lo, hi


def main():
    device = get_device()
    ds = ThyroidDataset(str(PROJ / "data" / "thyroid" / "thyroidxl"), split="test",
                        image_size=224, num_clinical_features=5, clinical_columns=CLIN,
                        mock=False, split_by_group=True, seed=42)
    print(f"test: {len(ds)} images")
    nod = load_manifest_test()
    print(f"test nodules: {len(nod)}")

    # 三臂推理（一次推理出 per-image + per-nodule 两种口径）
    per_image = {}
    nod_mean = {}
    for ab in ABLATIONS:
        wp = PROJ / "checkpoints" / "thyroid" / ab / "best.pt"
        if not wp.exists():
            print(f"SKIP {ab}: {wp} 不存在")
            continue
        model = load_model(wp, device)
        p_img, y_img, pids = predict_per_image(model, ds, device)
        per_image[ab] = {"probs": p_img, "labels": y_img}
        agg_p, agg_l = aggregate_nodule_mean(p_img, y_img, pids)
        nod_mean[ab] = {"probs": agg_p, "labels": agg_l}
        auc, lo, hi = bootstrap_auc(agg_l, agg_p, n_boot=1000, seed=0)
        print(f"{ab}: per-image n={len(p_img)} | nodule n={len(agg_p)} | nodule AUC {auc:.4f} ({lo:.4f}-{hi:.4f})")

    # 保存 per-image 统计（补充材料 Table S1 用）
    per_image_stats = {}
    for ab in ABLATIONS:
        if ab not in per_image:
            continue
        p, y = per_image[ab]["probs"], per_image[ab]["labels"]
        res = subgroup_auc(y, p)
        per_image_stats[ab] = {"n": int(len(p)), "pos_rate": float(y.mean())}
        if res:
            per_image_stats[ab]["auc"], per_image_stats[ab]["ci_lo"], per_image_stats[ab]["ci_hi"] = res
    (PROJ / "checkpoints" / "thyroid" / "per_image_stats.json").write_text(
        json.dumps(per_image_stats, indent=2), encoding="utf-8")
    print("saved per_image_stats.json")

    # ---- 亚组分析 ----
    keys = ["tirads", "size", "age", "gender"]
    key_labels = {
        "tirads": "TI-RADS",
        "size": "Nodule size",
        "age": "Age (years)",
        "gender": "Sex",
    }
    # 亚组顺序（用于表格和图）
    order = {
        "tirads": ["TR2-3", "TR4", "TR5"],
        "size": ["<=10mm", "10-20mm", ">20mm"],
        "age": ["<45", "45-60", ">=60"],
        "gender": ["male", "female"],
    }

    # 组装 per-patient 数据
    pids = list(nod.keys())
    rows = []  # (key_name, subgroup, n, pos_rate, image_auc, clinical_auc, fusion_auc, delta)
    for key in keys:
        f = subgroup_of(nod, key)
        for sg in order[key]:
            sel = [pid for pid in pids if f(nod[pid]) == sg]
            if len(sel) < 2:
                continue
            y = np.array([nod[pid]["label"] for pid in sel])
            pos = float(y.mean())
            aucs = {}
            for ab in ABLATIONS:
                if ab not in nod_mean:
                    continue
                prob_map = dict(zip(pids, nod_mean[ab]["probs"]))
                p = np.array([prob_map[pid] for pid in sel])
                res = subgroup_auc(y, p)
                aucs[ab] = res
            img = aucs.get("image")
            fus = aucs.get("fusion")
            delta = (fus[0] - img[0]) if (fus and img) else None
            rows.append({
                "key": key, "key_label": key_labels[key], "subgroup": sg, "n": len(sel),
                "pos_rate": pos,
                "image": img, "clinical": aucs.get("clinical"), "fusion": fus,
                "delta": delta,
            })
            fmt = lambda r: f"{r[0]:.3f} ({r[1]:.3f}-{r[2]:.3f})" if r else "single-class"
            print(f"{key_labels[key]:12s} {sg:8s} n={len(sel):4d} pos={pos:.3f} | "
                  f"image {fmt(img)} | clinical {fmt(aucs.get('clinical'))} | fusion {fmt(fus)}")

    # ---- Table 4 (markdown) ----
    md = [
        "**Table 4.** Subgroup analysis on the ThyroidXL test cohort (nodule-level, n=739).",
        "",
        "AUC (95% CI) by model and subgroup. TR = TI-RADS total score;",
        "diameter = maximum nodule diameter (width or height). Δ = Fusion − Image-only AUC.",
        "",
        "| Subgroup | n | Malignant (%) | Image-only | Clinical-only | Fusion | Δ AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cell = lambda x: (f"{x[0]:.3f} ({x[1]:.3f}-{x[2]:.3f})" if x else "—")
        delta = f"{r['delta']:+.3f}" if r["delta"] is not None else "—"
        md.append(f"| {r['key_label']}: {r['subgroup']} | {r['n']} | {r['pos_rate']*100:.1f}% | "
                  f"{cell(r['image'])} | {cell(r['clinical'])} | {cell(r['fusion'])} | {delta} |")
    out = NOTES_DIR / "subgroup_analysis.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"saved {out}")

    # ---- Fig 10: forest plot ----
    # 行顺序：key 分组 + subgroup 顺序
    row_idx = {}
    i = 0
    for key in keys:
        for sg in order[key]:
            row_idx[(key, sg)] = i
            i += 1
    nrows = i

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ypos = {key: [] for key in keys}
    for r in rows:
        key, sg = r["key"], r["subgroup"]
        j = row_idx[(key, sg)]
        # 每亚组画三臂点（用颜色区分），竖直偏移避免重叠
        for k, ab in enumerate(["image", "clinical", "fusion"]):
            res = r[ab]
            if not res:
                continue
            yy = nrows - 1 - j + (k - 1) * 0.12
            ax.errorbar(res[0], yy, xerr=[[res[0] - res[1]], [res[2] - res[0]]],
                        fmt="o", ms=5, color=COLORS[ab], ecolor=COLORS[ab],
                        capsize=2, lw=1.5)
            ypos[key].append(yy)

    # y 轴标签
    ticks, labels = [], []
    for key in keys:
        for sg in order[key]:
            ticks.append(nrows - 1 - row_idx[(key, sg)])
            labels.append(f"{key_labels[key]}: {sg}")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Nodule-level AUC (95% CI)", fontsize=12)
    ax.set_title("Subgroup analysis: thyroid nodule malignancy (ThyroidXL test)", fontsize=13)
    ax.set_xlim(0.3, 1.02)

    # 图例
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=COLORS[ab], label=LABELS[ab])
               for ab in ABLATIONS if ab in nod_mean]
    ax.legend(handles=handles, loc="lower right", fontsize=10)
    fig.tight_layout()
    out_png = FIG_DIR / "fig9_subgroup_forest.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
