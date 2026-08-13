# Supplementary Material

**Supplementary Tables for:** External generalization and domain shift in thyroid ultrasound AI: quantifying cross-dataset performance, multi-dataset training, and the incremental value of structured features.

All metrics in this supplement use the same protocols as the main manuscript:
AUC with bootstrap 95% confidence intervals (n = 2,000 samples); sensitivity
and specificity at a fixed 0.5 decision threshold; ECE = expected calibration
error (10 equal-width bins over [0,1]); the Brier score for the nodule-level
ablation is reported in Table 3 of the main manuscript (image-only 0.110,
clinical-only 0.293, fusion 0.113).

---

## Supplementary Table S1. Per-image classification performance across datasets

Per-image statistics are reported separately from the nodule-level results in the main
manuscript because a nodule may be depicted in multiple frames (Thy-Wise: median 7,
range 1–48; ThyroidXL: mean 2.8). Per-image analysis weights nodules with more frames
more heavily and is provided here for completeness.

### S1a. ThyroidXL held-out test cohort (per-image, n = 2,094 images; 46.5% malignant)

| Model | AUC (95% CI) |
|---|---|
| Image-only | 0.909 (0.896–0.920) |
| Clinical-only | 0.826 (0.807–0.844) |
| Fusion | 0.917 (0.904–0.928) |

Corresponding nodule-level (mean aggregation, n = 739) values are reported in
Table 3 of the main manuscript (0.939 / 0.814 / 0.947). Fusion improves over
image-only at both the per-image (+0.008) and nodule-level (+0.007) level.

### S1b. Thy-Wise external cohort (per-image, n = 29,070 images; 28.9% malignant)

| Model | AUC (95% CI) | AUPRC | Sensitivity | Specificity | ACC | ECE |
|---|---|---|---|---|---|---|
| TN5000-only (single-dataset) | 0.540 (0.533–0.547) | 0.326 | 0.284 | 0.757 | 0.621 | 0.191 |
| Joint (TN5000 + TN3K train) | 0.607 (0.601–0.614) | 0.382 | 0.368 | 0.744 | 0.635 | 0.170 |

Nodule-level values (mean aggregation, n = 3,954) are reported in Table 2 of the main
manuscript: single-dataset 0.608, joint 0.710. The markedly lower per-image AUCs
(0.540 / 0.607 vs. 0.608 / 0.710) reflect the presence of multiple, partly
redundant frames per nodule; nodule-level aggregation was therefore used as the
primary external estimate.

### S1c. TN3K and TN5000 internal/external test sets (per-image; each image is one nodule)

| Dataset | Model | AUC (95% CI) | Sensitivity | Specificity | ACC | ECE | n |
|---|---|---|---|---|---|---|---|
| TN5000 test | Joint | 0.931 (0.910–0.951) | 0.948 | 0.757 | 0.895 | 0.019 | 750 |
| TN3K test | Joint | 0.813 (0.779–0.846) | 0.674 | 0.788 | 0.744 | 0.050 | 614 |

For these two datasets each image corresponds to an independent nodule, so per-image
and nodule-level statistics coincide.

---

## Supplementary Table S2. Nodule-level aggregation sensitivity on ThyroidXL test (n = 739)

Three strategies were compared for aggregating frame-level probabilities within each
nodule: mean (used throughout the main manuscript), max, and attention-weighted
(exponential softmax over frame probabilities, temperature = 5.0). Mean pooling
yielded the highest AUC for both models; attention weighting provided no gain over
mean, supporting the use of the simpler fixed aggregation and the Discussion
argument that the higher patient-level AUC reported by the mask-based RCAF approach
is not explained by attention pooling.

| Model | Aggregation | AUC (95% CI) | AUPRC |
|---|---|---|---|
| Image-only | Mean | 0.939 (0.924–0.954) | 0.930 |
| Image-only | Max | 0.927 (0.909–0.943) | 0.917 |
| Image-only | Attention | 0.928 (0.910–0.944) | 0.915 |
| Fusion | Mean | 0.947 (0.932–0.960) | 0.939 |
| Fusion | Max | 0.938 (0.921–0.953) | 0.927 |
| Fusion | Attention | 0.940 (0.924–0.954) | 0.929 |

---

## Supplementary Table S3. Seed robustness of the main results (three training seeds, 42 / 123 / 2024)

Each model was retrained three times with different random seeds; because the
grouped train/validation split is itself seeded, the three runs vary both the
model initialisation and the training partition. All external test sets
(TN3K official test, Thy-Wise, ThyroidXL) are fixed official partitions held
out from every run. Reported values are the mean and range across the three
seeds; seed-42 values correspond to the main-manuscript anchors (Table 2/3).
All runs for a given model used an identical training protocol
(pos_weight = 1.0).

| Dataset (n) | Model | AUC mean (range) | ECE mean (range) |
|---|---|---|---|
| TN3K official test (614) | TN5000-only | 0.740 (0.729–0.756) | 0.113 (0.090–0.128) |
| TN3K official test (614) | Joint | 0.808 (0.800–0.813) | 0.065 (0.051–0.077) |
| Thy-Wise nodule (3,954) | TN5000-only | 0.610 (0.594–0.628) | 0.070 (0.057–0.081) |
| Thy-Wise nodule (3,954) | Joint | 0.709 (0.706–0.711) | 0.056 (0.023–0.089) |
| ThyroidXL nodule (739) | Fusion | 0.948 (0.947–0.949) | 0.111 (0.089–0.129) |

Seed-to-seed variation was small on all fixed external test sets (AUC ranges:
0.729–0.756 for TN5000-only on TN3K, 0.800–0.813 for joint on TN3K,
0.594–0.628 for TN5000-only on Thy-Wise, 0.706–0.711 for joint on Thy-Wise,
and 0.947–0.949 for fusion on ThyroidXL), indicating that the reported
results are not artefacts of a single training seed.

---

*Reproducibility note: all supplementary statistics were recomputed from the public
datasets described in the main manuscript using the released checkpoints; see
`paper/scripts/subgroup_analysis.py`, `paper/scripts/mil_sensitivity.py` and
`eval_thywise.py` in the accompanying code repository.*
