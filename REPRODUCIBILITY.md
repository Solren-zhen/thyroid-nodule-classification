# Reproducibility

This document maps every quantitative result in the manuscript
(`paper/output/doc/manuscript.md`) and its supplementary material to the exact
command that produced it and the result file that records it. All numbers were
regenerated from the **final protocol** (patient-grouped 70/15/15 split,
`pos_weight = 1.0`, no leakage) using the released checkpoints.

## 1. Environment

- Python 3.11, PyTorch 2.5.1 (CUDA 12.1), torchvision 0.20.1, albumentations 1.4.x
- Hardware used: NVIDIA RTX 3060 6 GB (any CUDA GPU with ≥6 GB works)
- Install: `pip install -r requirements.txt`
- Set `PYTHON=/path/to/python` if `python` is not the conda env interpreter
  (all shell scripts honour `$PYTHON`).

## 2. Canonical evaluation protocol

- Image size 224×224; inference in eval mode (no test-time augmentation in the
  reported numbers).
- `eval_thyroid.py` commands use **batch size 32** (matching
  `eval_thyroidxl_ablation.sh`, `subgroup_analysis.py`, `mil_sensitivity.py` and
  `ensemble_fusion_pw1.py`). `eval_thywise.py` uses its default batch size (64)
  and `eval_seed_retrain.py` the eval default (16); floating-point differences
  between batch sizes affect at most the 3rd decimal of ECE / 0.1 pp of
  sensitivity-specificity. The committed result files in `paper/output/repro/`
  are the exact values behind the paper.
- AUC 95% CIs: bootstrap, n = 2,000, seed = 0 (`train_thyroid.bootstrap_auc`).
- Sensitivity/specificity/accuracy at a fixed 0.5 decision threshold.
- ECE: 10 bins. DCA: net benefit over thresholds 0.05–0.95.
- Thy-Wise and ThyroidXL nodule-level results: per-image probabilities averaged
  within each nodule (`--aggregate mean`); per-image statistics are also
  reported (S1a/S1b).

## 3. Data preparation (Table 1)

All four datasets are public. Download and prepare with the repository
scripts; the resulting `manifest.csv` files define the exact splits.

```bash
# TN5000 (Zhang et al., Scientific Data 2025)
python download_tn5000.py --dest data/thyroid/tn5000
python prepare_tn5000.py --data_root data/thyroid

# TN3K (Hugging Face: haifan-gong/TN3K)
python prepare_tn3k.py --data_root data/thyroid/tn3k

# Thy-Wise (figshare DOI 10.6084/m9.figshare.20417895, CC BY 4.0)
#   -> build data/thyroid/thywise/manifest.csv (see tools/thywise_prepare.py)

# ThyroidXL (Hugging Face: hunglc007/ThyroidXL, gated access; do not commit tokens)
export HF_TOKEN=...
python download_hf_dataset.py --repo hunglc007/ThyroidXL --dest data/thyroid/thyroidxl \
    --subdirs train/images train/labels test/images test/labels
python prepare_thyroidxl.py --root data/thyroid/thyroidxl

# Joint training manifest: TN5000 train/val + TN3K trainval (7,129 images),
# internal test held out (data/thyroid_tn5000test, data/thyroid_tn3ktest)
python prepare_multi.py
```

Table 1 counts (5,000 / 3,493 / 29,070 / 11,635 images, splits, class counts)
are read from these manifests by the eval scripts and reproduced by
`paper/scripts/summarize_thyroidxl.py` for ThyroidXL.

## 4. Checkpoints (final protocol, pw1.0)

`checkpoints/` is git-ignored (model weights are not distributed through the
repo). The canonical final checkpoints are:

| Model | Path |
|---|---|
| TN5000-only, seed 42 | `checkpoints/thyroid/single_tn3k_eval/best.pt` |
| Joint (TN5000+TN3K), seed 42 | `checkpoints/thyroid/_pre_seedretrain_backup/best_joint_seed42_fixed.pt` |
| Fusion, seed 42 | `checkpoints/thyroid/fusion/best.pt` |
| Image-only (ThyroidXL), seed 42 | `checkpoints/thyroid/image/best.pt` |
| Clinical-only, seed 42 | `checkpoints/thyroid/clinical/best.pt` |
| Seeds 123 / 2024 | `checkpoints/thyroid/{tn5000,joint}_seed{123,2024}/best.pt`, `checkpoints/thyroid/fusion_seed{123,2024}_pw1.0/best.pt` |

Training commands that produced them:

```bash
# Track A anchors (seed 42)
python train_thyroid.py --data_root data/thyroid       --ablation image --epochs 30 --batch_size 64 --seed 42 --pos_weight 1.0
python train_thyroid.py --data_root data/thyroid_multi --ablation image --epochs 30 --batch_size 64 --seed 42 --pos_weight 1.0

# ThyroidXL three-arm ablation (seed 42)
CLIN=tirads,width_mm,height_mm,age,gender
python train_thyroid.py --data_root data/thyroid/thyroidxl --ablation fusion   --epochs 30 --batch_size 32 --clinical_columns $CLIN --seed 42 --pos_weight 1.0
python train_thyroid.py --data_root data/thyroid/thyroidxl --ablation image    --epochs 30 --batch_size 32 --clinical_columns $CLIN --seed 42 --pos_weight 1.0
python train_thyroid.py --data_root data/thyroid/thyroidxl --ablation clinical --epochs 30 --batch_size 32 --clinical_columns $CLIN --seed 42 --pos_weight 1.0

# Robustness seeds 123 + 2024 (identical protocol) — full pipeline
bash run_seed_retrain_all.sh
```

## 5. Table 2 (Track A) — number → command → result file

Common eval invocation (writes `eval_{split}.json` next to the weights):

```bash
python eval_thyroid.py --weights <weights> --data_root <data_root> --split <split> --batch_size 32
python eval_thywise.py --weights <weights> --data_root data/thyroid/thywise   # nodule-level + per-image
```

| Table 2 row | Weights | Command (data_root / split) | Result file |
|---|---|---|---|
| TN5000-only: internal val | `single_tn3k_eval/best.pt` | `data/thyroid` / `val` | `paper/output/repro/trackA_tn5000_val.json` |
| TN5000-only: internal test | `single_tn3k_eval/best.pt` | `data/thyroid_tn5000test` / `test` | `paper/output/repro/trackA_tn5000_internal_test.json` |
| TN5000-only: TN3K official test | `single_tn3k_eval/best.pt` | `data/thyroid_tn3ktest` / `test` | `paper/output/repro/trackA_tn5000_tn3k_official_test.json` |
| TN5000-only: TN3K full cohort | `single_tn3k_eval/best.pt` | `data/thyroid/tn3k` / `all` | `paper/output/repro/trackA_tn5000_tn3k_full_cohort.json` |
| TN5000-only: Thy-Wise | `single_tn3k_eval/best.pt` | `eval_thywise.py` | `paper/output/repro/trackA_tn5000_thywise.json` |
| Joint: internal val | `_pre_seedretrain_backup/best_joint_seed42_fixed.pt` | `data/thyroid_multi` / `val` | `paper/output/repro/trackA_joint_val.json` |
| Joint: internal test | same | `data/thyroid_tn5000test` / `test` | `paper/output/repro/trackA_joint_internal_test.json` |
| Joint: TN3K official test | same | `data/thyroid_tn3ktest` / `test` | `paper/output/repro/trackA_joint_tn3k_official_test.json` |
| Joint: Thy-Wise | same | `eval_thywise.py` | `paper/output/repro/trackA_joint_thywise.json` |

Δ rows in Table 2 are computed from the unrounded values in the JSON files
(e.g., TN3K ΔAUC = 0.8135 − 0.7285 = +0.085).

## 6. Table 3 (ThyroidXL ablation) + ensemble + paired test

```bash
CLIN=tirads,width_mm,height_mm,age,gender
python eval_thyroid.py --weights checkpoints/thyroid/fusion/best.pt  --data_root data/thyroid/thyroidxl --split test --batch_size 32 --aggregate mean --clinical_columns $CLIN
python eval_thyroid.py --weights checkpoints/thyroid/image/best.pt   --data_root data/thyroid/thyroidxl --split test --batch_size 32 --aggregate mean --clinical_columns $CLIN
python eval_thyroid.py --weights checkpoints/thyroid/clinical/best.pt --data_root data/thyroid/thyroidxl --split test --batch_size 32 --aggregate mean --clinical_columns $CLIN
```

| Table 3 row | Result file |
|---|---|
| Fusion (0.947, 0.939 AUPRC, ECE 0.118) | `paper/output/repro/ablation_fusion_nodule.json` |
| Image-only (0.939, 0.930 AUPRC, ECE 0.087) | `paper/output/repro/ablation_image_nodule.json` |
| Clinical-only (0.814, 0.735 AUPRC, ECE 0.248) | `paper/output/repro/ablation_clinical_nodule.json` |

Additional analyses reported in Sec. 3.5 / Table 3 note:

```bash
# 3-seed ensemble (42/123/2024, pos_weight = 1.0): AUC 0.951 (0.937–0.964), AUPRC 0.944
python paper/scripts/ensemble_fusion_pw1.py        # -> ensemble_fusion_pw1.json

# Paired bootstrap ΔAUC (fusion vs image-only): +0.007 (95% CI 0.002–0.013), p = 0.012
python paper/scripts/paired_auc_test.py            # -> paired_fusion_vs_image.json

# Expert TI-RADS score baseline: AUC 0.929, AUPRC 0.887; Youden t=4.5 (sens 79.3%, spec 95.6%)
python paper/scripts/tirads_baseline.py            # -> paper/output/repro/tirads_baseline.md

# Youden-optimal operating points (fusion PPV 0.856 / NPV 0.898)
python paper/scripts/analyze_threshold.py          # -> paper/output/repro/threshold_analysis_youden.md
```

## 7. Table 4 (subgroups)

```bash
python paper/scripts/subgroup_analysis.py
# -> paper/notes/subgroup_analysis.md  (source of Table 4)
# -> paper/figures/fig9_subgroup_forest.png
```
Committed copy: `paper/output/repro/subgroup_analysis_table4.md`.

## 8. Supplementary tables

### S1a — ThyroidXL per-image (n = 2,094)
```bash
python eval_thyroid.py --weights checkpoints/thyroid/{fusion,image,clinical}/best.pt \
    --data_root data/thyroid/thyroidxl --split test --batch_size 32 --clinical_columns tirads,width_mm,height_mm,age,gender
```
Result files: `paper/output/repro/ablation_{fusion,image,clinical}_perimage.json`;
summary: `paper/output/repro/s1a_per_image_stats.json`.

### S1b — Thy-Wise per-image (n = 29,070)
`paper/output/repro/trackA_{tn5000,joint}_thywise.json` (see §5; `per_image`
block). Nodule-level values in the same files (`nodule_mean` block).

### S1c — TN3K / TN5000 test (per-image = per-nodule)
Same files as Table 2 rows (`trackA_joint_tn3k_official_test.json`,
`trackA_joint_internal_test.json`).

### S2 — aggregation sensitivity (mean / max / attention)
```bash
python paper/scripts/mil_sensitivity.py
# -> checkpoints/thyroid/mil_sensitivity.json  (committed copy: paper/output/repro/s2_mil_sensitivity.json)
```

### S3 — seed robustness (3 seeds)
```bash
bash run_seed_retrain_all.sh                       # retrains seeds 123/2024 (track A + fusion)
python paper/scripts/eval_seed_retrain.py --only all
# -> checkpoints/thyroid/seed_retrain_eval/seed_retrain_summary.json
#    (committed copy: paper/output/repro/s3_seed_retrain_summary.json)
```

## 9. Figures

```bash
python paper/scripts/plot_figures.py                # Figs 2–4 (track A ROC/calibration/DCA/confusion)
python paper/scripts/plot_flow_diagram.py           # Fig 1
python paper/scripts/plot_thyroidxl_fusion.py       # Figs 5–8 (ablation)
python paper/scripts/subgroup_analysis.py           # Fig 9
python paper/scripts/error_cases.py                 # Fig 10
```

## 10. End-to-end run order

1. `download_*.py` + `prepare_*.py` + `prepare_multi.py` (data)
2. `train_thyroid.py ...` (seed-42 anchors) — or `bash run_seed_retrain_all.sh`
   to also produce seeds 123/2024
3. Evaluation commands in §5–§8
4. `python paper/scripts/verify_supplement_consistency.py` — asserts that the
   manuscript tables, `paper/notes/subgroup_analysis.md`, the per-image stats
   and `mil_sensitivity.json` agree with the supplementary material.

## 11. Integrity notes

- The public repository intentionally contains **no** `leaky`-named or
  pw0.35-protocol artifacts; superseded files are kept only under `archive/`
  (git-ignored) for provenance and are excluded from the submission package.
- Batch-to-batch floating-point variation is the only source of run-to-run
  differences at the 3rd decimal; re-running the exact commands above
  reproduces the committed JSON files to the reported precision.
