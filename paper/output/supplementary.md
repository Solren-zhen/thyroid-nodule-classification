# Supplementary Material

**Supplementary Tables for:** Multimodal fusion of ultrasound imaging and ACR TI-RADS
features for thyroid nodule malignancy classification: a multi-dataset validation study.

All metrics in this supplement use the same protocols as the main manuscript:
AUC with bootstrap 95% confidence intervals (n = 1,000 stratified samples);
sensitivity and specificity at the Youden-optimal operating point determined on the
respective validation cohort; ECE = expected calibration error.

---

## Supplementary Table S1. Per-image classification performance across datasets

Per-image statistics are reported separately from the per-nodule results in the main
manuscript because a nodule may be depicted in multiple frames (Thy-Wise: median 7,
range 1–49; ThyroidXL: mean 2.8). Per-image analysis weights nodules with more frames
more heavily and is provided here for completeness.

### S1a. ThyroidXL held-out test cohort (per-image, n = 2,094 images; 46.5% malignant)

| Model | AUC (95% CI) |
|---|---|
| Image-only | 0.909 (0.896–0.920) |
| Clinical-only | 0.826 (0.807–0.844) |
| Fusion | 0.918 (0.906–0.929) |

Corresponding nodule-level (mean aggregation, n = 739) values are reported in
Table 3 of the main manuscript (0.939 / 0.814 / 0.947). Fusion improves over
image-only at both the per-image (+0.009) and per-nodule (+0.007) level.

### S1b. Thy-Wise external cohort (per-image, n = 29,070 images; 28.9% malignant)

| Model | AUC (95% CI) | AUPRC | Sensitivity | Specificity | ACC | ECE |
|---|---|---|---|---|---|---|
| TN5000-only (single-dataset) | 0.540 (0.533–0.547) | 0.326 | 0.284 | 0.757 | 0.621 | 0.191 |
| Joint (TN5000 + TN3K train) | 0.586 (0.579–0.593) | 0.371 | 0.351 | 0.736 | 0.625 | 0.205 |

Per-nodule values (mean aggregation, n = 3,954) are reported in Table 2 of the main
manuscript: single-dataset 0.608, joint 0.667. The markedly lower per-image AUCs
(0.540 / 0.586 vs. 0.608 / 0.667) reflect the presence of multiple, partly
redundant frames per nodule; nodule-level aggregation was therefore used as the
primary external estimate.

### S1c. TN3K and TN5000 internal/external test sets (per-image; each image is one nodule)

| Dataset | Model | AUC (95% CI) | Sensitivity | Specificity | ACC | ECE | n |
|---|---|---|---|---|---|---|---|
| TN5000 test | Joint | 0.974 (0.961–0.984) | 0.961 | 0.867 | 0.935 | 0.034 | 750 |
| TN3K test | Joint | 0.816 (0.782–0.849) | 0.640 | 0.825 | 0.754 | 0.072 | 614 |

For these two datasets each image corresponds to an independent nodule, so per-image
and per-nodule statistics coincide.

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
| Fusion | Max | 0.934 (0.917–0.950) | 0.922 |
| Fusion | Attention | 0.935 (0.919–0.951) | 0.922 |

---

*Reproducibility note: all supplementary statistics were recomputed from the public
datasets described in the main manuscript using the released checkpoints; see
`paper/scripts/subgroup_analysis.py`, `paper/scripts/mil_sensitivity.py` and
`eval_thywise.py` in the accompanying code repository.*
