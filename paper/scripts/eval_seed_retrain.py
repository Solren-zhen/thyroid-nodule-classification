#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练后评估编排：为 S3 生成 3-seed 均值/范围。

track A（TN5000-only / joint）三 seed（42,123,2024）在
  - TN5000 internal test（data/thyroid_tn5000test, split=test, n=750）
  - TN3K official test（data/thyroid/tn3k, split=test, n=614）
  - Thy-Wise nodule（eval_thywise.py, 按结节聚合）
fusion（pos_weight=1.0）三 seed 在 ThyroidXL test（aggregate=mean, n=739），并重算三 seed 集成。

用法: python paper/scripts/eval_seed_retrain.py [--only trackA|fusion|all]
依赖：训练完成后（tn5000_seed{123,2024}/joint_seed{123,2024}/fusion_seed{123,2024}_pw1.0 存在）。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
PY = sys.executable
CKPT = PROJ / "checkpoints" / "thyroid"
OUT = CKPT / "seed_retrain_eval"
CLIN = "tirads,width_mm,height_mm,age,gender"

SEEDS = [42, 123, 2024]

TRACK_A_CKPTS = {
    "tn5000": {
        42: CKPT / "_pre_seedretrain_backup" / "best_tn5000_seed42.pt",
        123: CKPT / "tn5000_seed123" / "best.pt",
        2024: CKPT / "tn5000_seed2024" / "best.pt",
    },
    "joint": {
        42: CKPT / "_pre_seedretrain_backup" / "best_joint_seed42_fixed.pt",
        123: CKPT / "joint_seed123" / "best.pt",
        2024: CKPT / "joint_seed2024" / "best.pt",
    },
}
FUSION_CKPTS = {
    42: CKPT / "fusion_backup" / "best_posw1.0_seed42_20260809.pt",
    123: CKPT / "fusion_seed123_pw1.0" / "best.pt",
    2024: CKPT / "fusion_seed2024_pw1.0" / "best.pt",
}


def run(cmd: list) -> dict:
    """跑子进程，解析输出的 JSON 文件（由 eval 脚本写到固定位置）。"""
    tag = cmd[cmd.index("--weights") + 1]
    out_json = Path(tag).parent / f"eval_test.json"
    # eval_thyroid.py 把结果写到 --weights 同目录 eval_test.json
    subprocess.run(cmd, check=True, cwd=PROJ)
    if not out_json.exists():
        raise FileNotFoundError(out_json)
    d = json.loads(out_json.read_text(encoding="utf-8"))
    return {"auc": d["metrics"]["auc"], "auc_ci": d["metrics"]["auc_ci"],
            "auprc": d["metrics"]["auprc"], "ece": d["metrics"]["ece"],
            "n": d.get("n")}


def eval_thyroid(weights: Path, data_root: str, aggregate: str = "none") -> dict:
    cmd = [PY, "eval_thyroid.py", "--weights", str(weights),
           "--data_root", data_root, "--split", "test", "--aggregate", aggregate]
    if aggregate == "mean":
        cmd += ["--clinical_columns", CLIN]
    return run(cmd)


def eval_thywise(weights: Path) -> dict:
    """eval_thywise.py 输出到 {weights dir}/eval_thywise.json，取 nodule_mean（结节级，手稿口径）。"""
    cmd = [PY, "eval_thywise.py", "--weights", str(weights),
           "--data_root", "data/thyroid/thywise"]
    out_json = weights.parent / "eval_thywise.json"
    subprocess.run(cmd, check=True, cwd=PROJ)
    d = json.loads(out_json.read_text(encoding="utf-8"))
    m = d["nodule_mean"]
    return {"auc": m["auc"], "auc_ci": m.get("auc_ci"), "ece": m.get("ece"),
            "n": m.get("n")}


def summarize(name: str, rows: list) -> dict:
    aucs = [r["auc"] for r in rows]
    eces = [r.get("ece") for r in rows if r.get("ece") is not None]
    out = {
        "n": rows[0].get("n"),
        "auc_mean": round(sum(aucs) / len(aucs), 4),
        "auc_range": [round(min(aucs), 4), round(max(aucs), 4)],
        "per_seed": {f"seed{s}": r for s, r in zip(SEEDS, rows)},
    }
    if eces:
        out["ece_mean"] = round(sum(eces) / len(eces), 4)
        out["ece_range"] = [round(min(eces), 4), round(max(eces), 4)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["trackA", "fusion", "all"], default="all")
    ap.add_argument("--datasets", default=None,
                    help="逗号分隔子集：tn5000_internal,tn3k_test,thywise_nodule,thyroidxl（只跑指定子集）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only_ds = set(args.datasets.split(",")) if args.datasets else None

    def want(ds):
        return only_ds is None or ds in only_ds

    results = {}

    if args.only in ("trackA", "all"):
        for variant, ckpts in TRACK_A_CKPTS.items():
            for dataset, (dr, agg, fn) in {
                # 注意：TN3K 官方 test 用独立根 data/thyroid_tn3ktest（n=614）；
                # data/thyroid/tn3k --split test 会含训练图（n=3493，泄漏），不可用。
                "tn5000_internal": ("data/thyroid_tn5000test", "none", eval_thyroid),
                "tn3k_test": ("data/thyroid_tn3ktest", "none", eval_thyroid),
                "thywise_nodule": (None, None, eval_thywise),
            }.items():
                if not want(dataset):
                    continue
                rows = []
                for seed in SEEDS:
                    wp = ckpts[seed]
                    if not wp.exists():
                        print(f"!! MISSING {wp} — 跳过 {variant}/{dataset}/{seed}")
                        continue
                    print(f"[{variant}/{dataset}/seed{seed}] {wp.name}")
                    if fn is eval_thywise:
                        rows.append(eval_thywise(wp))
                    else:
                        rows.append(eval_thyroid(wp, dr, agg))
                if len(rows) == 3:
                    results[f"{variant}.{dataset}"] = summarize(dataset, rows)
                else:
                    print(f"!! {variant}.{dataset} 只收集到 {len(rows)}/3 seed，跳过汇总")

    if args.only in ("fusion", "all") and want("thyroidxl"):
        rows = []
        for seed in SEEDS:
            wp = FUSION_CKPTS[seed]
            if not wp.exists():
                print(f"!! MISSING {wp} — 跳过 fusion/seed{seed}")
                continue
            print(f"[fusion/thyroidxl/seed{seed}] {wp.name}")
            rows.append(eval_thyroid(wp, "data/thyroid/thyroidxl", "mean"))
        if len(rows) == 3:
            results["fusion.thyroidxl"] = summarize("fusion.thyroidxl", rows)
        else:
            print(f"!! fusion 只收集到 {len(rows)}/3 seed")

    out = OUT / "seed_retrain_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 汇总已写 ===")
    for k, v in results.items():
        print(f"{k}: n={v.get('n')} AUC {v.get('auc_mean')} (range {v.get('auc_range')})"
              + (f" ECE {v.get('ece_mean')} (range {v.get('ece_range')})" if "ece_mean" in v else ""))
    print("summary:", out)


if __name__ == "__main__":
    main()
