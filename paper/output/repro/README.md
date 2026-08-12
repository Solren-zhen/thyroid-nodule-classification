# Reproducibility artifacts

This directory contains the canonical evaluation outputs behind every number
reported in the manuscript and supplementary material. Each file was generated
with the final-protocol checkpoints and the exact commands listed in
[`REPRODUCIBILITY.md`](../../../REPRODUCIBILITY.md) at the repository root.

| File | Reports |
|---|---|
| `trackA_tn5000_val.json` | Table 2 — TN5000-only validation AUC (Sec. 3.2) |
| `trackA_tn5000_internal_test.json` | Table 2 — TN5000-only internal test |
| `trackA_tn5000_tn3k_official_test.json` | Table 2 — TN5000-only TN3K official test |
| `trackA_tn5000_tn3k_full_cohort.json` | Table 2 — TN5000-only TN3K full cohort |
| `trackA_tn5000_thywise.json` | Table 2 + S1b — TN5000-only Thy-Wise (per-image & nodule) |
| `trackA_joint_val.json` | Table 2 — joint validation AUC (Sec. 3.4) |
| `trackA_joint_internal_test.json` | Table 2 — joint TN5000 internal test |
| `trackA_joint_tn3k_official_test.json` | Table 2 — joint TN3K official test |
| `trackA_joint_thywise.json` | Table 2 + S1b — joint Thy-Wise (per-image & nodule) |
| `ablation_fusion_nodule.json` | Table 3 — fusion, nodule-level (mean) |
| `ablation_image_nodule.json` | Table 3 — image-only, nodule-level |
| `ablation_clinical_nodule.json` | Table 3 — clinical-only, nodule-level |
| `ablation_fusion_perimage.json` | S1a — fusion, per-image |
| `ablation_image_perimage.json` | S1a — image-only, per-image |
| `ablation_clinical_perimage.json` | S1a — clinical-only, per-image |
| `s1a_per_image_stats.json` | S1a — per-image AUC/CI summary |
| `s2_mil_sensitivity.json` | S2 — mean/max/attention aggregation |
| `s3_seed_retrain_summary.json` | S3 — 3-seed mean/range (seeds 42/123/2024) |
| `ensemble_fusion_pw1.json` | Table 3 note — 3-seed ensemble AUC/CI/AUPRC |
| `paired_fusion_vs_image.json` | Paired bootstrap ΔAUC test (p = 0.012) |
| `subgroup_analysis_table4.md` | Table 4 — subgroup AUCs |
| `threshold_analysis_youden.md` | Sec. 3.5 — Youden-optimal operating points |
| `tirads_baseline.md` | Sec. 3.5 — expert TI-RADS score baseline |

`checkpoints/` and `data/` are git-ignored (large model weights and licensed
data); the result files above are committed so that every reported number can
be inspected without re-running inference.
