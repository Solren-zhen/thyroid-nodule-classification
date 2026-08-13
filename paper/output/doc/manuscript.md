# External generalization and domain shift in thyroid ultrasound AI: quantifying cross-dataset performance, multi-dataset training, and the incremental value of structured features

**Chaohui Zhen**¹

¹ Wenzhou Medical University, Wenzhou, China

Correspondence: Chaohui Zhen (chaohui0408.medical@outlook.com)

## Abstract

**Background.** Thyroid nodules are highly prevalent, yet ultrasound-based
risk stratification remains subjective and operator-dependent. Deep learning
models trained on thyroid ultrasound images show promise, but most studies
lack external validation, and few integrate structured patient and nodule
features such as the American College of Radiology (ACR) Thyroid Imaging
Reporting and Data System (TI-RADS) descriptors.

**Methods.** In track A, an image-only EfficientNetV2-S model trained on
TN5000 was evaluated on two external cohorts (TN3K and Thy-Wise), and a joint
model additionally trained on TN3K images was compared with the
single-dataset model to quantify cross-dataset domain shift. In track B, a full five-feature fusion model combining the image encoder with five structured patient and nodule features (ACR TI-RADS total score, nodule width and height, age, sex)
was compared with image-only and clinical-only models on ThyroidXL.
Performance was assessed using the area under the receiver operating
characteristic curve (AUC) with bootstrap confidence intervals (CI),
sensitivity, specificity, expected calibration error (ECE), Brier score and
decision curves.

**Results.** The single-dataset model achieved an internal test AUC of 0.915
but dropped to 0.712 on the full TN3K cohort (0.729 on the official test
split) and to 0.608 (nodule-level) on the completely unseen Thy-Wise cohort,
quantifying a large cross-dataset domain-shift gap. Joint training raised the
TN3K official-test AUC from 0.729 to 0.813 (matched Δ = +0.085) and the
Thy-Wise nodule-level AUC to 0.710 (Δ = +0.102), showing that the benefit of
multi-dataset training transfers beyond the cohort added to training. In
track B, adding the expert ACR TI-RADS score alone produced the largest
incremental improvement (nodule-level AUC 0.960 vs 0.939 for image-only;
paired ΔAUC +0.021, 95% CI 0.008–0.035, p = 0.003), whereas adding nodule
size or demographics provided little additional benefit, and the full
five-feature fusion achieved AUC 0.947; for the prespecified full five-feature fusion model, the modest gain was concentrated in
low-to-intermediate TI-RADS scores and small nodules.

**Conclusions.** Across four public datasets, image-only classifiers achieved
high internal AUC but lost substantial performance externally, quantifying
cross-dataset domain shift in thyroid ultrasound AI; joint multi-dataset
training recovered part of the gap on both a target-domain held-out cohort and
a completely unseen cohort. The incremental value of structured information was driven
mainly by the expert ACR TI-RADS score (nodule-level AUC 0.960 vs 0.939 for
image-only), with demographic and size features adding little; image features remained the primary discriminative signal, while expert TI-RADS provided substantial complementary information.
These findings support the importance of external validation before clinical
deployment. All datasets are publicly accessible (ThyroidXL under gated
access), and analysis code is available at
https://github.com/ojdanajakir848-a11y/thyroid-nodule-classification.

## Keywords

Thyroid nodule; Ultrasound imaging; Deep learning; Domain shift; External validation; Generalization; Multimodal fusion; TI-RADS; Calibration

## 1. Introduction

Thyroid nodules are among the most common endocrine findings: a
general-population meta-analysis estimated a pooled prevalence of 24.8%
(95% CI 21.4–28.6%) [1], yet the large majority are benign. This high
prevalence, combined with rising thyroid cancer incidence, has driven
concerns about overdiagnosis and unnecessary surgery [2]. Thyroid
ultrasound remains the first-line imaging modality for nodule evaluation
because it is non-invasive, radiation-free and widely available; however,
interpretation is subjective and strongly operator-dependent, and the overlap
of sonographic features between benign and malignant nodules limits diagnostic
reproducibility [3].

Deep learning has emerged as a promising tool to support thyroid nodule
classification, segmentation and detection [4,5,6]. Numerous
models based on convolutional neural networks and vision transformers have
reported high diagnostic accuracy on internal test sets [7,8].
Several open-access datasets have accelerated this progress, including DDTI
(637 images) [9], TN3K (3,493 images) [10,11], the TN5000 dataset (5,000
images with biopsy confirmation) [12], the expert-labelled ThyroidXL
benchmark (11,635 images) [13], and the pathologically confirmed
Thy-Wise cohort (3,954 nodules) [14]. Despite this progress, three limitations
persist in the literature. First, most studies evaluate on a single dataset, and models trained and tested on images from the same institution can overestimate real-world performance. Ultrasound acquisition is operator- and device-dependent: scanner vendor, transducer frequency, gain and focus settings and plane selection alter the pixel statistics of the same nodule, and referral populations differ in malignancy prevalence. These factors can erode both discrimination and calibration when a model is applied at another centre. Second, the vast majority of
models are image-only and discard structured patient and nodule information that
radiologists routinely use, such as the American College of Radiology (ACR)
Thyroid Imaging Reporting and Data System (TI-RADS) descriptors. Third,
reported models rarely include calibration metrics or decision curve
analysis, which are essential for clinical utility assessment.

The ACR TI-RADS lexicon provides a standardised, semantically defined
description of nodule composition, echogenicity, shape, margin and echogenic
foci, and its total score is a validated predictor of malignancy risk [3].
Because the descriptors are defined semantically rather than by pixel
statistics, they are in principle less affected by acquisition variability
than raw pixels, although in practice they are derived from ultrasound
assessment and therefore share some dependence on image acquisition. Whether
fusing such structured patient and nodule features with image-derived deep
features improves classification, and whether any gain is robust across
datasets, has not, to our knowledge, been systematically tested on public
multi-cohort data. We therefore evaluated a full five-feature fusion model that combines image features with structured patient and nodule features (ACR TI-RADS score, nodule width and height, age, sex), and quantified both the internal gain and the behaviour of this model relative to image-only and clinical-only configurations.

In this study, we systematically evaluate an image-only EfficientNetV2
classifier across four public datasets to quantify cross-dataset domain
shift, and test whether joint multi-dataset training and the addition of
structured patient and nodule features (expert TI-RADS total score, nodule
dimensions, age and sex) can mitigate the resulting performance gap. Our contributions are threefold. First, we quantify the external-generalization gap of a single-dataset model on two independent cohorts and show that multi-dataset training recovers part of the gap, including on a cohort that never contributed training data. Second, we compare image-only, clinical-only, full five-feature fusion and feature-specific fusion models head-to-head to isolate the incremental value of structured features. Third, we report calibration (expected calibration error and Brier score) and decision curve analysis to assess clinical utility, following the STARD 2015 and TRIPOD+AI reporting guidelines.

## 2. Methods

### 2.1 Datasets

We used four publicly available thyroid ultrasound datasets with benign/
malignant labels (study flow in Figure 1); their characteristics are
summarised in Table 1.

The TN5000 dataset (Zhang et al., Scientific Data, 2025) [12] contains 5,000
B-mode ultrasound images of thyroid nodules with expert-radiologist labels
and biopsy confirmation, explicitly designed for both detection and
classification. In the present study, 1,426 images were benign and 3,574 were
malignant. We used TN5000 for model development, applying a patient-grouped
random split of 70%/15%/15% for training, validation, and internal testing.

The TN3K dataset (Gong et al., ISBI 2021 / CBM 2022) [10,11] contains 3,493 ultrasound images
from 2,421 patients acquired with multiple ultrasound devices, with
segmentation masks and benign/malignant labels (0 = benign, 1 = malignant).
TN3K was used as an external validation cohort to assess cross-dataset
generalization. All 3,493 TN3K images (2,283 benign, 1,210 malignant) were
held out from training for the single-dataset evaluation. For the joint
training experiment, TN3K training images (n = 2,879) were added to the
training pool and the official TN3K test set (n = 614) was used for external
evaluation.

The Thy-Wise dataset (Jin et al., International Journal of Cancer, 2022) [14]
comprises B-mode ultrasound images of 3,954 thyroid nodules from the
First Affiliated Hospital of Jinan University, released under a CC BY 4.0
licence. Each nodule is represented by one or more images (median 7, range 1–48)
acquired in a single examination, and the benign/malignant class is
pathologically confirmed. We used Thy-Wise exclusively as a second external
validation cohort; because a nodule may be depicted in multiple images, all
Thy-Wise evaluations were aggregated at the nodule level (Section 2.5). After
exclusion of 25 images that were corrupted in the public archive and could not
be decoded, 3,954 nodules (2,876 benign, 1,078 malignant) and 29,070 images
(20,680 benign, 8,390 malignant) remained for analysis.

The ThyroidXL dataset (Duong et al., MICCAI 2025) [13] comprises 11,635
B-mode ultrasound images of 4,093 thyroid nodules collected at the Vietnam
National Hospital of Endocrinology on a single ultrasound system (Hitachi
Aloka Arietta V70), released under gated access with a research-permissive
licence, with benign/malignant labels derived from cytological and
pathological examination. Crucially for the fusion
component of this study, ThyroidXL provides, for each nodule, an expert ACR
TI-RADS total score (1–5), nodule dimensions in millimetres, patient age and
sex. The dataset is divided by the authors into development (9,541 images,
3,354 patients) and test (2,094 images, 739 patients) cohorts with no patient
overlap. We used ThyroidXL exclusively for the fusion ablation (Section 2.2,
Section 3.5): the development cohort was split at the patient level into
training (n = 8,615 images) and validation (n = 926 images), and the official
test cohort (739 nodules) was held out for evaluation. Because a nodule may be
represented by multiple images, ThyroidXL evaluations were aggregated to the
nodule level by averaging image-level probabilities (Section 2.5).

No images were shared between training and test splits at the patient level.
Pairwise byte-level (MD5) comparisons confirmed zero cross-dataset image
duplication: all 5,000 TN5000 images and all 3,493 TN3K images had unique MD5
hashes, with no TN5000–TN3K overlap, and no Thy-Wise image matched any TN5000
or TN3K image (the ThyroidXL cohort, released independently, was not included
in this comparison). Because each TN5000 image corresponds to an independent
nodule sample, image filenames were treated as patient identifiers for
grouped splitting.

Reporting notes. The original public releases do not document exact
acquisition dates, so we report the institutions, devices and publication
years stated by the releasing teams rather than precise collection dates.
All four cohorts are archive collections rather than consecutive clinical
series, and enrolment was determined by the releasing institutions rather
than by a prespecified protocol. Benign/malignant labels were assigned from
biopsy, cytology or pathology examinations performed at the time of
acquisition, so the index test and the reference standard were concurrent;
no follow-up interval was involved.

<!-- FIGURE:1 -->

### 2.2 Structured patient and nodule features

Five structured patient and nodule features were used as the structured input to the full five-feature fusion model. The ThyroidXL dataset provides, for each nodule, an expert ACR
TI-RADS total score (1–5), the nodule width and height in millimetres
(recorded at acquisition), the patient age (years), and sex
(1 = male, 2 = female), which were concatenated into a
five-dimensional structured feature vector. The ACR TI-RADS total score is a
semantically defined risk-stratification summary derived from composition,
echogenicity, shape, margin and echogenic-foci descriptors [3]. Age and sex
are demographic variables that are independent of image acquisition, whereas
the TI-RADS score and nodule dimensions are structured measurements derived
from ultrasound assessment; together they therefore provide structured
patient and nodule information that is less directly tied to raw pixel
statistics than the image modality, without being fully device-independent.
Unlike the image modality, these features do not require nodule segmentation
or manual lesion cropping at inference time. No nodule in ThyroidXL lacked any
of the five features.

### 2.3 Model architecture

The multimodal classifier consisted of three components. The image encoder
was an EfficientNetV2-S backbone pre-trained on ImageNet, with the final
classification layer replaced by a projection head producing a 512-dimensional
embedding. EfficientNetV2-S was chosen as the image encoder because its
compound-scaling design offers a favourable accuracy/efficiency trade-off, is
well-suited to the available 6 GB GPU memory, and has been
widely used for medical-image classification including thyroid ultrasound [8];
we did not perform an architecture search, and other backbones may behave
differently (Section 4, Limitations). The structured-feature encoder was a multi-layer
perceptron mapping the five-dimensional structured feature vector
(Section 2.2) to a 64-dimensional embedding. The two
embeddings were concatenated and passed to a prediction head with hidden
dimensions 256 and 128 and a dropout of 0.3, producing a malignancy probability.
We compared image-only (clinical branch disabled), clinical-only (image branch disabled), full five-feature fusion (both branches active), and feature-specific fusion configurations using subsets of the structured features (Section 3.5; Table 5).

### 2.4 Training

Images were resized to 224 x 224 pixels and normalised using ImageNet mean and standard deviation. Training used random
ShiftScaleRotate and flipping augmentations; validation used resizing and normalisation only.
Models were trained with binary cross-entropy with logits (pos_weight = 1.0, i.e., no class reweighting) and optimised with AdamW (initial learning rate 1e-4, backbone 1e-5,
weight decay 1e-4) with a cosine annealing schedule. Training ran for up to 30 epochs (with early stopping if the validation AUC did not improve for 15 consecutive epochs)
with a batch size of 64 (image-only and joint-training experiments) or 32
(fusion ablation, limited by GPU memory for the dual-branch model) using
automatic mixed precision on an NVIDIA RTX 3060 (6 GB). The best model was
selected by validation AUC. The primary analyses used a fixed random seed (42) for both the grouped splits and model initialisation; in the seed-robustness runs (seeds 123 and 2024), the grouped split and model initialisation were varied together (Supplementary Table S3). For the
domain-shift and joint-training experiments, results are reported from the
seed-42 run, with robustness across training seeds assessed using two
additional runs (seeds 123 and 2024; Supplementary Table S3); for the fusion
ablation, two additional independent runs (seeds 123, 2024) were trained and
test predictions were averaged across the three models (seeds 42, 123 and 2024)
to assess robustness to training-seed variation (Section 3.5). All experiments used
Python 3.11, PyTorch 2.5.1 (CUDA 12.1), torchvision 0.20.1, and albumentations
1.4.x.

### 2.5 Evaluation and statistical analysis

The reference standard against which all predictions were scored was the
benign/malignant label accompanying each image: biopsy-confirmed pathological
labels in TN5000, the published ground-truth labels of the TN3K dataset, the
pathological diagnosis of the Thy-Wise dataset, and the cytological or
pathological diagnosis of the ThyroidXL dataset. Because a Thy-Wise or
ThyroidXL nodule may be depicted in multiple images, per-image predictions were
aggregated to the nodule level by averaging image-level probabilities before
computing metrics (per-image statistics are reported separately and are
provided in Supplementary Table S1).

Discrimination was quantified by the AUC with 95% CIs obtained by seeded bootstrap resampling (n = 2,000 samples); for nodule-level analyses, resampling was performed at the nodule level rather than at the image level. Sensitivity, specificity and accuracy were
reported at a fixed decision threshold of 0.5 applied uniformly to internal
and external test sets. Calibration was quantified by the expected
calibration error (ECE) computed with 10 equal-width bins over [0, 1] (bins
with no observations were excluded) and by the Brier score. Clinical utility
was assessed by decision curve analysis (DCA), reporting net benefit over
threshold probabilities from 0.05 to 0.95 in steps of 0.05 and comparing each
model with the treat-none and treat-all reference strategies. Reporting
follows the STARD 2015 [15] and TRIPOD+AI [16] checklists.

## 3. Results

Two evaluation tracks structure the results: track A (Sections 3.2–3.4) evaluates image-only single- and multi-dataset models across the four cohorts, and track B (Section 3.5) evaluates the structured-feature fusion ablation on ThyroidXL.

### 3.1 Dataset characteristics

**Table 1.** Summary of the four thyroid ultrasound datasets used in this study.

| Characteristic | TN5000 | TN3K | Thy-Wise | ThyroidXL |
|---|---|---|---|---|
| Total images | 5,000 | 3,493 | 29,070 | 11,635 |
| Patients / nodules | 5,000^a^ | 2,421 | 3,954 | 4,093 |
| Benign images | 1,426 (28.5%) | 2,283 (65.4%) | 20,680 (71.1%) | 7,052 (73.9% of dev.) |
| Malignant images | 3,574 (71.5%) | 1,210 (34.6%) | 8,390 (28.9%) | 2,489 (26.1% of dev.) |
| Malignancy prevalence (test cohort) | 72.0%^b^ | 38.4%^c^ | 27.3%^e^ | 47.8%^d^ |
| Train / Val / Test | 3,500 / 750 / 750^b^ | — / — / 614^c^ | — / — / 3,954 | 8,615 / 926 / 2,094^d^ |
| Ground truth | Biopsy-confirmed | Published labels [11] | Pathological diagnosis [14] | Cytological/pathological [13] |
| Acquisition | Single institution, GE Logiq E9/S7 | Multi-device, multi-institution | Single institution, Jinan University Hospital | Single institution, Hitachi Aloka Arietta V70 |
| License | Open access [12] | Open access [10,11] | CC BY 4.0 [14] | Gated (research use) [13] |
| Role in this study | Primary training + internal test | External validation #1 | External validation #2 | Fusion training/test (Section 3.5) |

^a^ Each TN5000 image corresponds to an independent nodule; filenames serve as patient identifiers for grouped splitting.
^b^ Patient-level grouped random split (70/15/15, seed = 42); the held-out internal TN5000 test cohort (n = 750) was 72.0% malignant.
^c^ Official test split of the TN3K dataset (n = 614; 38.4% malignant); joint training used TN3K train (n = 2,879) for training and the official test set for evaluation.
^d^ Official ThyroidXL test cohort (739 nodules; 47.8% malignant); the development cohort was split at the patient level into train/val (90/10, seed = 42).
^e^ Nodule-level malignancy prevalence of Thy-Wise; all 3,954 nodules were used for external evaluation.

### 3.2 Internal test performance (image-only, single-dataset model)

The image-only model trained on TN5000 achieved a validation AUC of 0.922
(95% CI 0.896–0.943) and a held-out test AUC of 0.915 (95% CI
0.893–0.936). At the fixed 0.5 decision threshold, test sensitivity was 90.6%,
specificity 75.2%, accuracy 86.3%, and F1 score 0.905. The expected
calibration error was 0.042. The internal-test value is the seed-42 estimate;
seed-to-seed variation in the held-out test AUC was 0.915–0.954
(Supplementary Table S3).

### 3.3 External validation reveals substantial domain shift

Applying the single-dataset model to the full TN3K cohort for descriptive external evaluation (3,493 images from
a different institution and device setting) yielded an AUC of 0.712 (95% CI
0.695–0.729) with sensitivity 51.4% and specificity 76.5% (expected
calibration error 0.120). On the official TN3K test split (n = 614), the same
model achieved an AUC of 0.729 (95% CI 0.685–0.767), which we use as the
matched baseline for the joint-training comparison below. The 0.20 decrease
in AUC relative to internal testing quantifies the cross-dataset domain shift
that is frequently underestimated in single-centre evaluations.

### 3.4 Joint multi-dataset training improves internal and external performance

Joint training on TN5000 together with the TN3K training subset (a combined
manifest of 7,129 images: 6,379 training plus 750 validation) modestly
improved internal discrimination while substantially improving external
performance. The joint model reached a validation AUC of 0.932 (95% CI
0.910–0.953) and a held-out TN5000 test AUC of 0.931 (95% CI 0.910–0.951)
with sensitivity 94.8% and specificity 75.7% (expected calibration error
0.020). On the official TN3K test set (n = 614), external AUC improved from
0.729 to 0.813 (95% CI 0.779–0.846; matched Δ = +0.085), with sensitivity
67.4%, specificity 78.8% and expected calibration error 0.050 (Table 2;
Figures 2–4). Joint training therefore recovered approximately half of the
observed AUC loss on the external TN3K cohort. The official TN3K
test set is a target-domain held-out evaluation, because TN3K training images
contributed to the joint model; Thy-Wise, by contrast, never contributed
training data and is the completely unseen cohort. At the fixed 0.5 decision threshold, net benefit on the TN3K official test increased from 0.062 (single-dataset) to 0.129 (joint), compared with 0.000 for treating none (Figure 3b), and was associated with a higher net benefit at the fixed 0.5 threshold, suggesting potential clinical utility.

The gain transferred to a second, fully independent cohort. Applied to
Thy-Wise (3,954 pathology-confirmed nodules from a third institution), the
single-dataset (TN5000-only) model achieved a nodule-level AUC of 0.608
(95% CI 0.589–0.628), and joint training improved this to 0.710 (95% CI
0.693–0.727; Δ = +0.102), with an expected calibration error of 0.057.
Per-image probabilities were averaged within each nodule before metric
computation. Per-image statistics were markedly lower (single-dataset AUC
0.540; joint AUC 0.607; Supplementary Table S1b), reflecting the presence of
multiple, partly-redundant frames per nodule; nodule-level aggregation was
therefore used as the primary external estimate. The joint-training gain on
Thy-Wise (+0.102) was comparable to that on TN3K (+0.085), showing that the
benefit of multi-dataset training transfers to unseen distributions.

**Table 2.** Classification performance of image-only models across datasets.

| Model / Evaluation | AUC (95% CI) | Sensitivity | Specificity | ACC | ECE | n |
|---|---|---|---|---|---|---|
| **TN5000-only (image)** | | | | | | |
| TN5000 internal test | 0.915 (0.893–0.936) | 90.6% | 75.2% | 86.3% | 0.042 | 750 |
| TN3K external (official test) | 0.729 (0.685–0.767) | 51.3% | 78.0% | 67.8% | 0.120 | 614 |
| TN3K external (full cohort)^a^ | 0.712 (0.695–0.729) | 51.4% | 76.5% | 67.8% | 0.120 | 3,493 |
| Thy-Wise external (nodule-level mean) | 0.608 (0.589–0.628) | 18.6% | 90.0% | 70.5% | 0.057 | 3,954 |
| **Joint (TN5000 + TN3K train, image)** | | | | | | |
| TN5000 internal test | 0.931 (0.910–0.951) | 94.8% | 75.7% | 89.5% | 0.019 | 750 |
| TN3K external (official test) | 0.813 (0.779–0.846) | 67.4% | 78.8% | 74.4% | 0.050 | 614 |
| Thy-Wise external (nodule-level mean) | 0.710 (0.693–0.727) | 25.7% | 90.4% | 72.7% | 0.057 | 3,954 |
| **Δ (Joint − TN5000-only, official test)** | | | | | | |
| TN5000 internal | +0.016 | +4.3 pp | +0.5 pp | +3.2 pp | −0.022 | |
| TN3K external | +0.085 | +16.1 pp | +0.8 pp | +6.7 pp | −0.069 | |
| Thy-Wise external | +0.102 | +7.1 pp | +0.4 pp | +2.2 pp | 0.000 | |

^a^ TN3K full-cohort AUC shown for domain-shift illustration only; the matched Δ uses the official test split (n = 614) for both models.

All metrics computed at a fixed 0.5 decision threshold. Sensitivity and specificity for Thy-Wise use mean probability aggregation within each nodule. Δ = Joint minus TN5000-only, computed from unrounded values; displayed Δ may differ by ±0.001 (AUC/ECE) or ±0.1 pp from the difference of the rounded values shown; pp = percentage points.

**Equal-sample-size control.** Because joint training also increased the
number of training images (7,129 vs 4,250), we tested whether the improvement
was simply a sample-size effect. We trained equal-sample-size control models on 3,500 training images sampled
from the combined pool (stratified as 1,920 TN5000 and 1,580 TN3K images),
matching the single-dataset training size while holding the 750-image
validation set constant (total development set matched at 4,250 images); three
independent subsamples (seeds 42, 123 and 2024) were trained with the same
protocol (Table 3). At matched sample size, the multi-dataset control already exceeded
the single-dataset model on the TN3K official test (AUC 0.792 vs 0.729) and on
the completely unseen Thy-Wise cohort (0.651 vs 0.608), although its internal
TN5000 test AUC was lower (0.889 vs 0.915) because fewer TN5000 images were
available. Enlarging the pool to the full joint set added further external
gains (TN3K 0.813; Thy-Wise 0.710). The multi-dataset benefit is therefore not
attributable to sample size alone: data diversity contributed substantially at
matched sample size, and scale added further improvement. Across three independent subsamples (seeds 42, 123 and 2024), the
equal-size control achieved a TN3K official-test AUC of 0.795 ± 0.003
(range 0.792–0.798), consistently above the single-dataset model (0.729),
and a Thy-Wise nodule-level AUC of 0.627 ± 0.028 (range 0.597–0.651), with
a mean above the single-dataset model (0.608) but greater subsample-to-
subsample variability.

**Table 3.** Equal-sample-size control. The development-set size was matched at
4,250 images (3,500 training + 750 validation) for the single-dataset and
equal-size models, with the validation set held constant; the joint model used
7,129 development images (6,379 training + 750 validation). Seed-42 anchor
values are shown; multi-seed means are reported in Section 3.4.

| Evaluation (AUC) | Single (3,500 TN5000) | Equal-size (1,920 TN5000 + 1,580 TN3K) | Joint (3,500 TN5000 + 2,879 TN3K) |
|---|---|---|---|
| TN5000 internal test | 0.915 | 0.889 | 0.931 |
| TN3K official test | 0.729 | 0.792 | 0.813 |
| Thy-Wise (nodule-level) | 0.608 | 0.651 | 0.710 |

<!-- FIGURE:2 -->

<!-- FIGURE:3 -->

<!-- FIGURE:4 -->

### 3.5 Fusion ablation (ThyroidXL)

To test whether structured patient and nodule features improve over image-only
classification, we evaluated image-only, clinical-only, full five-feature
fusion, and feature-specific fusion configurations on the ThyroidXL training
cohort (Section 2.2), and evaluated them on the held-out ThyroidXL test cohort
(2,094 images, 739 nodules), aggregating image-level probabilities to the
nodule level by averaging. Results are summarised in Tables 4 and 5 and Figures
5–8.

The image + TI-RADS model achieved the highest nodule-level AUC (0.960, 95% CI
0.948–0.973), followed by the five-feature fusion model (0.947, 95% CI
0.932–0.960) and the image-only model (0.939, 95% CI 0.924–0.954) (Table 4).
The prespecified five-feature fusion gain over image-only was statistically
significant (ΔAUC +0.007; paired bootstrap 95% CI 0.002–0.013, two-sided
p = 0.012), and the feature-level ablation (Table 5) showed a larger gain for
image + TI-RADS (ΔAUC +0.021). The five-feature fusion also showed higher
AUPRC (0.939 vs. 0.930) and specificity (0.940 vs. 0.909) than image-only, at
a modest cost in sensitivity (0.725 vs. 0.779). In an exploratory threshold analysis, predictive values were examined at the Youden-optimal operating point determined on the test cohort. The image + TI-RADS model achieved balanced positive and negative predictive values (PPV 0.860, NPV 0.943), above the full five-feature fusion (PPV 0.856, NPV 0.898), image-only (PPV 0.815, NPV 0.920) and clinical-only (PPV 0.688, NPV 0.835) models. The gain in discrimination of the best-performing configuration therefore did not come at the cost of poorer predictive values (Figure 8). Because this operating point was derived from the test cohort, these predictive values illustrate achievable performance at a Youden-optimal threshold rather than an unbiased prospective estimate. The ThyroidXL test cohort was 47.8% malignant, compared with 26.1% in the development set; this difference in prevalence is relevant to interpreting the fixed-threshold and Youden-threshold operating points reported here. The clinical-only model, using only the five structured features, was markedly inferior (AUPRC 0.735) and, at the fixed 0.5 threshold, degenerated to predicting all nodules as benign (sensitivity 0.00, specificity 1.00). This reflects both the modest predictive information in these five features alone and the low malignancy prevalence of the development set (26.1%), which compresses predicted probabilities below 0.5. The clinical-only model's value therefore lies in its ranking ability (nodule-level AUC 0.814) rather than in absolute probabilities at a fixed operating point. The image-only model was best calibrated
(ECE 0.087), while the clinical-only model was substantially miscalibrated
(ECE 0.248) and the full five-feature fusion model intermediate (ECE 0.118); the fusion gain in
discrimination therefore came with a small cost in calibration relative to
image-only classification. Brier scores followed the same ordering
(image-only 0.110, fusion 0.113, clinical-only 0.293).

As a clinical reference point, the expert ACR TI-RADS total score alone (a
widely used risk-stratification tool) achieved a nodule-level AUC of 0.929
(AUPRC 0.887) on the same test cohort, with 79.3% sensitivity and 95.6%
specificity at a TI-RADS-4.5 threshold. The image-only model (0.939) and the full five-feature fusion model (0.947) both exceeded this expert-scored baseline in terms of discrimination, indicating that
the deep models extract discriminative information beyond the TI-RADS total
score, although the incremental gain over the clinical reference is modest (Δ AUC +0.018 for the full five-feature fusion model). The clinical-only MLP model (0.814) fell
below the direct TI-RADS score (0.929), reflecting information loss in
regressing the raw five features to a single risk estimate.

These results indicate that image features remained the primary discriminative signal:
the prespecified full five-feature fusion yielded a small overall improvement
over image-only (Δ AUC +0.007, AUPRC +0.009, specificity +0.031), while the
feature-level ablation (Table 5) showed that the incremental value of
structured information was mainly attributable to the expert TI-RADS score
(image + TI-RADS Δ AUC +0.021). The gain is directionally consistent with
the hypothesis that structured patient and nodule priors complement image features, but
the magnitude is modest in this single-device cohort (Figure 5).

**Table 4.** Core model comparison on the ThyroidXL held-out test cohort (nodule-level, n = 739). Rows are ordered to reflect the feature-level finding that the expert TI-RADS score was the main contributor to the structured-feature gain (full ablation in Table 5). ΔAUC is computed from unrounded values relative to image-only.

| Model | AUC (95% CI) | ΔAUC vs image-only | AUPRC | Sensitivity | Specificity | ACC | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| Expert TI-RADS only | 0.929 | — | 0.887 | 0.793^e^ | 0.956^e^ | — | — | — |
| Image-only | 0.939 (0.924–0.954) | — | 0.930 | 0.779 | 0.909 | 0.847 | 0.087 | 0.110 |
| Image + TI-RADS | 0.960 (0.948–0.973) | +0.021 | 0.954 | 0.813 | 0.953 | 0.886 | 0.064 | 0.084 |
| Full five-feature fusion | 0.947 (0.932–0.960) | +0.007 | 0.939 | 0.725 | 0.940 | 0.838 | 0.118 | 0.113 |
| Clinical-only | 0.814 (0.782–0.847) | −0.125 | 0.735 | 0.00 | 1.00 | 0.522 | 0.248 | 0.293 |

Sensitivity and specificity at a fixed 0.5 decision threshold; for the expert
TI-RADS row, ^e^ sensitivity and specificity are reported at a TI-RADS-4.5
threshold, and ACC, ECE and Brier are not defined because the expert score is
an ordinal 1–5 scale rather than a continuous probability. AUPRC = area
under the precision-recall curve. Table 4 reports seed-42 single-run estimates.

As a robustness check, two additional fusion models were trained with the same protocol (pos_weight = 1.0) and different seeds (123 and 2024). Individual seed AUCs ranged 0.947–0.949, and averaging the test predictions across the three models improved the nodule-level AUC to 0.951 (95% CI 0.937–0.964) and AUPRC to 0.944 (Supplementary Table S3). The ablation result was therefore robust across training seeds. The fusion
gain over image-only was statistically significant (ΔAUC +0.007; paired
bootstrap 95% CI 0.002–0.013, two-sided p = 0.012).

**Feature-level ablation.** To locate the source of the structured-feature
gain, we trained variants of the fusion model with image features plus each
structured-feature subset under the same protocol (Table 5). Adding the expert
TI-RADS total score alone yielded the largest improvement (nodule-level AUC
0.960, 95% CI 0.948–0.973), above image-only (0.939) and above the full
five-feature fusion (0.947); adding nodule size (0.948) or demographics (0.944)
provided little or no gain over image-only, and the full five-feature fusion did not exceed
image + TI-RADS alone. The incremental value of structured information
therefore derives mainly from the expert TI-RADS score in this cohort. The difference between image + TI-RADS and image-only was statistically significant in a paired bootstrap analysis (ΔAUC +0.021, 95% CI 0.008–0.035, two-sided p = 0.003).

**Temperature scaling.** As a calibration robustness analysis, temperature
scaling was fitted on the validation cohort and applied to the prespecified
full five-feature fusion model and to the best-performing image + TI-RADS model. For the
full five-feature fusion model (T = 0.686), it produced a modest improvement in calibration
without affecting discrimination (ECE 0.118 → 0.109; Brier 0.113 → 0.112; AUC
unchanged at 0.947). The image + TI-RADS model was already well calibrated
(ECE 0.064; Brier 0.084) and temperature scaling did not improve it further
(ECE 0.079; Brier 0.085; AUC unchanged at 0.960), indicating that the
best-performing configuration does not require post-hoc recalibration.

**Table 5.** Feature-level ablation on the ThyroidXL held-out test cohort
(nodule-level, n = 739).

| Structured input (added to image) | AUC (95% CI) | AUPRC | ECE | Brier |
|---|---|---|---|---|
| None (image-only) | 0.939 (0.924–0.954) | 0.930 | 0.087 | 0.110 |
| TI-RADS | 0.960 (0.948–0.973) | 0.954 | 0.064 | 0.084 |
| Size (width, height) | 0.948 (0.934–0.962) | 0.942 | 0.077 | 0.098 |
| Demographics (age, sex) | 0.944 (0.929–0.958) | 0.935 | 0.106 | 0.112 |
| Full five-feature fusion | 0.947 (0.932–0.960) | 0.939 | 0.118 | 0.113 |

<!-- FIGURE:5 -->

<!-- FIGURE:6 -->

<!-- FIGURE:7 -->

<!-- FIGURE:8 -->

### 3.6 Subgroup analysis

To examine whether the fusion gain was consistent across clinically relevant
patient and nodule subgroups, we stratified the ThyroidXL test cohort
(n = 739 nodules) by TI-RADS total score, maximum nodule diameter, age and sex
(Table 6, Figure 9). Subgroup analyses used the prespecified full five-feature fusion model as
the primary multimodal configuration. Given the small number of TI-RADS 2 nodules (n = 45),
scores 2–3 were combined into a single low-to-intermediate risk group
(TR2-3, n = 227; 1.8% malignant), and scores 4 and 5 were analysed separately
(TR4, n = 215; 32.1% malignant; TR5, n = 297; 94.3% malignant).

The fusion gain over image-only was largest in the low-to-intermediate risk
group (TR2-3: Δ AUC +0.046; fusion 0.929 vs. image-only 0.883) and smaller in
TR4 (Δ +0.015; 0.838 vs. 0.823) and TR5 (Δ +0.013; 0.773 vs. 0.760), where
the high malignancy prevalence (94.3%) may reduce the available
discrimination range. By
nodule size, fusion improved over image-only for nodules ≤10 mm (0.882 vs.
0.868, Δ +0.014) and 10–20 mm (0.976 vs. 0.970, Δ +0.006), but not for nodules
>20 mm (0.949 vs. 0.966, Δ −0.017), a subgroup with low malignancy prevalence
(16.8%). Fusion gains were small across age strata
(<45: +0.004; 45–60: +0.011; ≥60: +0.001) and in the male subgroup (n = 86,
Δ +0.006), whereas female patients (n = 653) showed a gain of +0.008. Overall, the observed pattern is consistent with the hypothesis that structured risk information may be more useful when image-based discrimination is less certain (low-to-intermediate TI-RADS risk, small nodules), and the modest average gain of the prespecified full five-feature fusion model in Table 4 is not driven by any single subgroup. Because the TR2-3
subgroup contained only four malignant nodules, its confidence intervals are
very wide, and the largest point estimate (Δ AUC +0.046) should be regarded as
hypothesis-generating rather than as a definitive effect. Subgroup analyses were
exploratory; the reported 95% confidence intervals are not adjusted for
multiple comparisons. These analyses were exploratory subgroup performance
analyses across patient sex, age, nodule size and TI-RADS risk strata; sex-stratified estimates should be interpreted cautiously given the
small male subgroup (n = 86).

**Table 6.** Subgroup analysis on the ThyroidXL test cohort (nodule-level, n = 739).

AUC (95% CI) by model and subgroup. TR = TI-RADS total score;
diameter = maximum nodule diameter (width or height). Δ = Full five-feature fusion − Image-only AUC.

| Subgroup | n | Malignant (%) | Image-only | Clinical-only | Full five-feature fusion | Δ AUC |
|---|---|---|---|---|---|---|
| TI-RADS: TR2-3 | 227 | 1.8% | 0.883 (0.743–1.000) | 0.740 (0.544–0.872) | 0.929 (0.863–1.000) | +0.046 |
| TI-RADS: TR4 | 215 | 32.1% | 0.823 (0.763–0.875) | 0.607 (0.530–0.686) | 0.838 (0.781–0.887) | +0.015 |
| TI-RADS: TR5 | 297 | 94.3% | 0.760 (0.675–0.844) | 0.540 (0.411–0.671) | 0.773 (0.687–0.855) | +0.013 |
| Nodule size: ≤10 mm | 364 | 61.8% | 0.868 (0.829–0.906) | 0.784 (0.732–0.837) | 0.882 (0.843–0.919) | +0.014 |
| Nodule size: 10–20 mm | 238 | 44.1% | 0.970 (0.952–0.984) | 0.854 (0.807–0.899) | 0.976 (0.961–0.988) | +0.006 |
| Nodule size: >20 mm | 137 | 16.8% | 0.966 (0.911–0.997) | 0.693 (0.564–0.808) | 0.949 (0.850–0.999) | −0.017 |
| Age (years): <45 | 309 | 55.7% | 0.925 (0.896–0.951) | 0.766 (0.701–0.821) | 0.929 (0.901–0.954) | +0.004 |
| Age (years): 45–60 | 274 | 50.0% | 0.945 (0.921–0.968) | 0.846 (0.798–0.891) | 0.956 (0.934–0.976) | +0.011 |
| Age (years): ≥60 | 156 | 28.2% | 0.945 (0.899–0.981) | 0.814 (0.723–0.893) | 0.946 (0.890–0.985) | +0.001 |
| Sex: male | 86 | 61.6% | 0.957 (0.909–0.991) | 0.816 (0.707–0.918) | 0.963 (0.914–0.994) | +0.006 |
| Sex: female | 653 | 45.9% | 0.938 (0.922–0.954) | 0.813 (0.782–0.847) | 0.946 (0.931–0.961) | +0.008 |

^a^ Δ = Full five-feature fusion − Image-only AUC, computed from unrounded values; displayed Δ may differ by ±0.001 from the difference of the rounded AUCs shown in the table.

<!-- FIGURE:9 -->

## 4. Discussion

We systematically evaluated the external generalization of a thyroid nodule
malignancy classifier across four public datasets and tested two potential
mitigations: multi-dataset training and structured patient and nodule
features. Three principal findings emerged. First, an image-only model trained
on a single dataset achieved high internal discrimination (test AUC 0.915)
but lost substantial performance externally (AUC 0.712 on TN3K; 0.608
nodule-level on Thy-Wise), a cross-dataset gap that is invisible to
single-cohort evaluation. Second, joint training on two datasets recovered
approximately half of the observed AUC loss on the cohort added to training (TN3K official
test AUC 0.813 vs. 0.729) and provided a comparable gain on a completely
unseen cohort (Thy-Wise, 0.608 → 0.710). Third, the incremental value of structured information was mainly
attributable to the expert ACR TI-RADS score: adding TI-RADS to the image
model improved the nodule-level AUC from 0.939 to 0.960 (ΔAUC +0.021),
whereas nodule size and demographic features contributed little additional
discrimination. Across these analyses we report calibration (ECE and
Brier score) and decision-curve metrics alongside discrimination, and on the
ThyroidXL cohort we report model comparisons among image-only, clinical-only, full five-feature fusion and feature-specific fusion configurations (Section 3.5), addressing reporting gaps that remain common
in the field.

**Domain shift is the central barrier to translation.** The external drop
deserves emphasis because it reflects factors that internal testing cannot
expose: TN5000 and TN3K were acquired with different devices at different
institutions, and their label prevalence differs substantially (71.5% vs
34.6% malignant), which partly explains the concurrent fall in sensitivity at
a fixed operating threshold. A second external cohort (Thy-Wise) confirmed that the magnitude of this shift is cohort-specific. Unlike CT or MRI, where acquisition protocols are largely standardised, ultrasound depends on the operator and the device: transducer frequency, gain and focus settings and plane selection alter the image statistics of the same nodule, and referral populations differ in malignancy prevalence. Prevalence shift affects calibration as well as discrimination, because a model trained on a cohort with 71.5% malignancy will behave differently in lower-prevalence populations (34.6% in TN3K, 27.3% in Thy-Wise). Comparable degradation across devices has been reported in the domain-adaptation literature [17,18].
Our results align with the emerging consensus that single-dataset AUCs
substantially overestimate real-world performance, and support the importance
of external validation before clinical deployment
[16].

**Joint multi-dataset training helps the cohort added to training, and the
gain transfers to unseen cohorts.** Adding TN3K training images to the TN5000
pool improved external AUC on the official TN3K test set by +0.085 and
restored approximately half of the observed AUC loss; this is a pragmatic
robustness strategy when target-domain images can be included in training.
The key transfer result is the comparable gain on the completely independent
Thy-Wise cohort (+0.102), which never contributed training data. The residual
gaps on both cohorts (absolute AUCs of 0.813 and 0.710, well below the
internal 0.931) indicate that multi-dataset pooling alone does not eliminate
domain shift, and that additional robustness strategies, including structured
patient and nodule features, merit evaluation. An equal-sample-size control (Section 3.4) indicated that this gain was not attributable to sample size alone: at a matched training-set size, a multi-dataset control already improved the target-domain external test (TN3K AUC 0.729 → 0.792).

**Structured patient and nodule features complement, but do not dominate, image
features.** The hypothesis that structured patient and nodule priors (an expert ACR
TI-RADS score, nodule width and height, age and sex) could stabilise
predictions under domain shift was tested directly in the ThyroidXL ablation:
age and sex are independent of image acquisition, whereas the TI-RADS score
and nodule dimensions are structured measurements derived from ultrasound
assessment that may remain informative when pixel statistics degrade. The incremental value of structured information was mainly attributable to the expert TI-RADS score rather than demographic or size features (Section 3.5). Adding all five structured features did not improve upon TI-RADS-only augmentation: the full five-feature fusion (AUC 0.947) did not exceed image + TI-RADS (AUC 0.960). This may reflect redundancy among structured variables, limited incremental information from age, sex and nodule dimensions, or optimization effects in the multimodal head; importantly, the feature-level ablation indicates that the gain was not achieved by indiscriminate feature accumulation. The clinical-only model was markedly inferior (AUC 0.814) and poorly calibrated (ECE 0.248). The full five-feature fusion model was slightly less well calibrated than the image-only model (ECE 0.118 vs. 0.087),
indicating that the added structured features improved ranking without
proportionally improving the reliability of the predicted probabilities at the
fixed decision threshold. Two implications follow. First, image features remained the primary discriminative signal, yet the expert TI-RADS score added a substantial incremental gain (nodule-level AUC 0.939 → 0.960), whereas demographic and size features contributed little (Section 3.5). Second, structured features alone (clinical-only model, AUC 0.814) were insufficient for reliable discrimination. The large gain of image + TI-RADS over image-only is consistent with TI-RADS encoding structured expert assessment that is related to, but not identical with, the information extractable from raw ultrasound pixels. This tempers the
"device-independent prior" framing: only a subset of the features (age and
sex) is truly independent of image acquisition, and structured features
appear to be a useful complementary signal rather than a substitute for image
analysis.  Unlike multimodal methods that depend on free-text radiology reports [19,20] or contrast-enhanced sequences [21], our framework uses only routine B-mode ultrasound with structured values, requiring no additional acquisitions.

The subgroup analysis (Section 3.6) adds two clinically relevant nuances to
this finding. First, the fusion gain was concentrated in the subgroups where
image-derived discrimination is least certain: the low-to-intermediate
TI-RADS group (TR2-3, Δ AUC +0.046) and small nodules (≤20 mm). It was
smaller in the highly malignant TR5 group (+0.013) and negative for nodules >20 mm (−0.017). This pattern is consistent with the hypothesis that structured risk information may be more useful where image features are ambiguous, and suggests that the clinical branch
may be most valuable in the triage of indeterminate nodules. Second, no
meaningful sex- or age-dependent asymmetry in the fusion benefit was observed.
The wide confidence intervals in the TR2-3 subgroup (which contained only four
malignant nodules) caution against over-interpreting the largest point
estimate, and the subgroup results should be confirmed in larger,
multi-device cohorts.

**A practical caveat.** The best-performing configuration still depends on an expert TI-RADS score and is therefore not fully automated. That does not remove the need for AI: image analysis alone matched the TI-RADS total score (AUC 0.939 vs. 0.929), and the combination improved on both. The most plausible clinical role is as a second opinion that reduces subjectivity in low-to-intermediate risk nodules, where the fusion gain was largest, rather than as a replacement for the reporting radiologist. Routine deployment would require systematic expert scoring in the reading workflow or automated extraction of TI-RADS descriptors, which is an active area of work [4].
In terms of intended use, the model is aimed at sonographers and clinicians
with limited thyroid ultrasound experience, who would consult it as a
second-opinion aid when triaging low-to-intermediate risk nodules; it is not
intended for autonomous use or as a replacement for expert radiologist
review.

**Comparison with recent ThyroidXL-based studies.** Since ThyroidXL became
available, at least one framework has reported very high patient-level
performance on this benchmark using image-only input: a region- and
context-aware fusion with attention-based multiple-instance learning reported
a patient-level test AUC of 0.993 on ThyroidXL [22]. That approach differs from
ours in three respects relevant to clinical translation. First, it requires
nodule segmentation masks at inference (a lesion-focused branch is multiplied
by the mask), whereas our framework uses only the B-mode image and is
therefore applicable where automated or manual masks are unavailable. Second,
it aggregates frames with learned attention within each patient, whereas we
use fixed mean aggregation, which is simpler and avoids additional learned
parameters. Third, it does not use structured patient and nodule features. Our results
are not directly comparable in absolute terms because of these methodological
differences, and in particular the lower AUC of our image-only model on
ThyroidXL reflects the absence of mask guidance. Mean pooling also outperformed max- and attention-weighted pooling for both models (Supplementary Table S2), indicating that the higher patient-level AUC of the mask-based approach is not explained by its attention pooling. The present study instead
focuses on the questions that the mask-based approach does not address,
namely cross-dataset domain-shift quantification and the incremental value of
structured patient and nodule features.

**Calibration and decision-curve reporting.** Discrimination alone is
insufficient for clinical use. The single-dataset model was reasonably well
calibrated internally (ECE 0.042) but substantially miscalibrated externally
(ECE 0.120 on the full TN3K cohort; ECE 0.120 on the official test set), and
joint training improved calibration on the official TN3K test set (ECE 0.050;
Table 2);
calibration therefore degrades under domain shift in parallel with
discrimination. Decision curves (Figure 7) further show that the models provide net benefit across a range of threshold probabilities; at the fixed 0.5 threshold, net benefit was highest for the image + TI-RADS model (0.364), followed by image-only (0.325) and the full five-feature fusion (0.315), versus 0 for treating none. At the fixed
0.5 decision threshold, sensitivity (90.6% internally) exceeded
specificity (75.2%), reflecting both the class imbalance of the training data
(71.5% malignant) and the clinical trade-off in nodule triage, where
over-referral is preferable to a missed malignancy; the operating point should be selected to match the clinical setting, and the decision curves provide the information to do so explicitly. These results also illustrate that calibration is prevalence-dependent: the clinical-only model, trained on a development set with 26.1% malignancy, was systematically under-confident in the higher-prevalence test cohort (47.8% malignant), which is why its ECE reached 0.248; the slightly higher ECE of the full fusion model (0.118 vs. 0.087 for image-only) may reflect the contribution of this poorly calibrated clinical branch. Temperature scaling improved the fusion model modestly (ECE 0.118 → 0.109), and probability post-processing should be re-evaluated whenever a model is deployed at a site whose prevalence differs from the development cohort.

**Error patterns.** At a fixed 0.5 threshold, the image + TI-RADS model on the ThyroidXL test cohort produced more false negatives than false positives (66 vs. 18 of 739 nodules), i.e. a tendency to under-call malignancy (Figure 10). This
asymmetry is consistent with the low prevalence of malignancy in the ThyroidXL
development set (26.1%) and the high specificity attained by the model; in
clinical triage, where missing a malignancy is the more serious error, a lower
decision threshold or a sensitivity-constrained operating point would be
preferred. This shows that the decision threshold should be chosen explicitly rather than left at a fixed default.

**Limitations.** Several limitations must be acknowledged. First, labels in the
datasets arise from heterogeneous ground-truth sources (biopsy-confirmed
pathological labels in TN5000 and Thy-Wise, published ground-truth labels in
TN3K [11], and cytological/pathological labels in ThyroidXL [13]) and prevalence
differs across datasets (71.5%, 34.6%, 27.3% and 26.1% malignant at the
image/nodule level), complicating direct comparison of absolute metrics. Second,
TN5000 was analysed at the image level on the assumption that
each image corresponds to an independent nodule; although grouped splitting
prevents leakage, within-patient correlation is not modelled, and Thy-Wise was
aggregated to the nodule level precisely because a nodule is represented by
multiple frames. Third, the fusion ablation was conducted on a single-device
cohort (ThyroidXL, acquired on one ultrasound system), so the incremental value
of clinical features under genuine cross-device domain shift could not be
quantified; the fusion gain observed here (Δ AUC +0.007) may not generalise to
multi-device settings where image statistics degrade more substantially.
Fourth, we evaluated a single architecture (EfficientNetV2-S), and other
backbones may behave differently. Fifth, subgroup analyses were conducted using the prespecified full five-feature fusion model and were not repeated for the feature-selected image + TI-RADS model. Sixth, the best-performing image + TI-RADS configuration still depends on an expert-provided TI-RADS score and is therefore not fully automated; practical deployment would require either routine expert scoring or reliable automated extraction of TI-RADS descriptors. Finally, external validation, while performed on multiple public datasets, remains retrospective; prospective,
institution-specific evaluation would be required before clinical deployment.

**Future work.** The fusion gain observed on the single-device ThyroidXL cohort
should be re-tested on multi-device, TI-RADS-annotated data to determine whether
structured patient and nodule priors confer larger robustness under genuine cross-device
domain shift. Extending deep learning to the automated extraction of
TI-RADS descriptors from ultrasound images [4] could reduce the reliance on
expert scoring while preserving the structured-feature prior, and would
strengthen the clinical utility of the framework.

<!-- FIGURE:10 -->

## 5. Conclusion

Using four public datasets, we showed that a thyroid nodule
classification model trained on a single dataset achieves high internal AUC
but loses substantial performance externally (0.91 to 0.71 on TN3K); that joint
multi-dataset training recovers approximately half of the observed AUC loss on the cohort
added to training (external AUC 0.813 vs. 0.729 on the matched official test,
Δ = +0.085); and that on a second, completely unseen external
cohort (Thy-Wise), the same joint training provides a comparable gain
(nodule-level AUC 0.608 → 0.710, +0.102). External
performance is therefore strongly cohort-dependent, yet the benefit of
multi-dataset pooling transfers to unseen distributions; image-only models showed limited robustness across the evaluated cross-dataset settings. These results support the importance of external validation before clinical deployment. On a fourth cohort (ThyroidXL), the incremental value of structured information was mainly attributable to the expert ACR TI-RADS score. Image + TI-RADS achieved a nodule-level AUC of 0.960 (paired ΔAUC +0.021, 95% CI 0.008–0.035, p = 0.003 vs. image-only 0.939), whereas demographic and size features added little and the full five-feature fusion achieved 0.947. The clinical-only model was markedly inferior (AUC 0.814) and poorly calibrated. Together, these findings support the importance of external validation before clinical deployment of ultrasound-based thyroid nodule classifiers, and indicate that structured expert assessment (TI-RADS) provides a valuable complement to image analysis, with the largest incremental gain obtained when combined with image features; image features remained the primary discriminative signal, while expert TI-RADS provided substantial complementary information.

## Declarations

- Data availability: TN5000 (publicly available, [12]), TN3K (Hugging Face,
  haifan-gong/TN3K, [10,11]), Thy-Wise (figshare, DOI
  10.6084/m9.figshare.20417895, CC BY 4.0, [14]). ThyroidXL [13] is available
  at https://huggingface.co/datasets/hunglc007/ThyroidXL under gated access;
  access was granted and all 11,635 images were used in this study.
- Code availability: The analysis code is publicly available at https://github.com/ojdanajakir848-a11y/thyroid-nodule-classification. Evaluation outputs — all reported metric JSON files, calibration and decision-curve data, and subgroup and threshold summaries — are stored alongside the code in the same repository under paper/output/repro/.
- Ethics approval and consent: This study used only fully de-identified
  images from publicly available datasets, each released with its own
  research-use permission; no new patient data were collected and no
  institutional review board approval was required.
- Funding: This work received no specific grant from any funding agency.
- Competing interests: The authors declare no competing interests.
- Registration and protocol: This diagnostic accuracy study was not
  registered and no study protocol was prepared in advance.
- Patient and public involvement: No patients or members of the public were
  involved in the design, conduct, reporting, or dissemination of this study.
- Sample size: All available images from the four public datasets were used;
  no a priori sample size calculation was performed because the study
  analysed complete public cohorts.
- Use of artificial intelligence: During the preparation of this work, the authors used AI-assisted tools (including Claude and Codex) for drafting and revising manuscript text, writing and debugging analysis code, and supporting experimental design and statistical analyses. The authors reviewed and verified all AI-assisted outputs, take full responsibility for the content of the manuscript, and confirm that all reported results were produced and independently checked using the described public datasets and code.
- Consent for publication: Not applicable (no individual person's data are
  presented; all images are de-identified and from public repositories).
- Acknowledgements: The authors thank the developers of the TN5000, TN3K,
  Thy-Wise and ThyroidXL datasets for making their data publicly available.
- Authors' contributions: Chaohui Zhen designed the study, developed the methodology, implemented the models and analysis scripts, conducted all experiments and statistical analyses, prepared the figures and tables, and drafted the manuscript. The author read and approved the final manuscript.

## Abbreviations

ACR: American College of Radiology; AUC: area under the receiver operating
characteristic curve; AUPRC: area under the precision-recall curve; CI:
confidence interval; DCA: decision curve analysis; ECE: expected calibration
error; MLP: multi-layer perceptron; NPV: negative predictive value; PPV:
positive predictive value; ROC: receiver operating characteristic; TI-RADS:
Thyroid Imaging Reporting and Data System.

## Figure Legends

**Figure 1.** Study flow diagram. Data acquisition, patient-level split,
model training and evaluation pipeline for the four public datasets.

**Figure 2.** Receiver operating characteristic (ROC) curves of the
single-dataset (TN5000-only) and joint multi-dataset (TN5000 + TN3K training)
models. Internal TN5000 test set (single 0.915, joint 0.931) and external TN3K
cohort (single, full cohort 0.712; joint, official test 0.813) illustrate both
the cross-dataset drop and its partial recovery from multi-dataset training.

**Figure 3.** (a) Calibration reliability plots of the joint model on the
internal TN5000 test set and the external TN3K official test set. (b) Decision
curves of the joint model on the internal TN5000 test set and the external TN3K
official test set, and of the single-dataset model on the external TN3K
official test set, compared with the treat-none strategy. ECE = expected
calibration error.

**Figure 4.** Confusion matrices of the joint model on the internal TN5000
test set (left) and external TN3K test set (right) at the fixed 0.5 decision
threshold.

**Figure 5.** Model comparison on the ThyroidXL test cohort (n = 739
nodules, mean frame aggregation): ROC curves of the image-only, clinical-only, image + TI-RADS and full five-feature fusion models.

**Figure 6.** Calibration reliability plots of the image-only, clinical-only, image + TI-RADS and full five-feature fusion models on the ThyroidXL test cohort.

**Figure 7.** Decision curves of the image-only, clinical-only, image + TI-RADS and full five-feature fusion models on the ThyroidXL test cohort, reporting net benefit across threshold probabilities.

**Figure 8.** Threshold sensitivity analysis on the ThyroidXL test cohort:
(a) sensitivity and specificity as a function of the decision threshold, and
(b) positive predictive value as a function of the threshold, for the image-only, clinical-only, image + TI-RADS and full five-feature fusion models.

**Figure 9.** Subgroup analysis on the ThyroidXL test cohort (nodule-level,
n = 739): nodule-level AUC (95% CI) of the image-only, clinical-only and prespecified full five-feature fusion models stratified by TI-RADS total score (TR2-3, TR4, TR5), maximum nodule
diameter (≤10, 10–20, >20 mm), age (<45, 45–60, ≥60 years) and sex. The
dashed vertical line marks an AUC of 0.5.

**Figure 10.** Representative misclassifications of the image + TI-RADS model on the ThyroidXL test cohort. Top row: false positives (benign predicted malignant);
bottom row: false negatives (malignant predicted benign). Predicted
probabilities and ground-truth labels are shown for each case.

## References

1. Mu C, Ming X, Tian Y, Liu Y, Yao M, Ni Y, et al. Mapping global epidemiology of thyroid nodules among general population: a systematic review and meta-analysis. *Frontiers in Oncology*. 2022;12:1029926. doi:10.3389/fonc.2022.1029926.

2. Acosta GJ, Singh Ospina N, Brito JP. Overuse of thyroid ultrasound. *Current Opinion in Endocrinology, Diabetes and Obesity*. 2023;30(5):225–230. doi:10.1097/MED.0000000000000814.

3. Tessler FN, Middleton WD, Grant EG, Hoang JK, Berland LL, Teefey SA, et al. ACR Thyroid Imaging, Reporting and Data System (TI-RADS): White Paper of the ACR TI-RADS Committee. *Journal of the American College of Radiology*. 2017;14(5):587–595. doi:10.1016/j.jacr.2017.01.046.

4. Sharifi Y, Danay Ashgzari M, Shafiei S, Zakavi SR, Eslami S. Using deep learning for thyroid nodule risk stratification from ultrasound images. *WFUMB Ultrasound Open*. 2025;3(1):100082. doi:10.1016/j.wfumbo.2025.100082.

5. Savelonas M. An overview of AI-guided thyroid ultrasound image segmentation and classification for nodule assessment. *Big Data and Cognitive Computing*. 2025;9(10):255. doi:10.3390/bdcc9100255.

6. Gong H, Chen J, Chen G, Li H, Li G, Chen F. Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules. *Computers in Biology and Medicine*. 2023;155:106389. doi:10.1016/j.compbiomed.2022.106389.

7. Ni J, You Y, Wu X, Chen X, Wang J, Li Y. Performance evaluation of deep learning for the detection and segmentation of thyroid nodules: systematic review and meta-analysis. *Journal of Medical Internet Research*. 2025;27:e73516. doi:10.2196/73516.

8. Zhou X, Qiu M, Shang J, Wei G. Research progress on deep learning-based computer-aided diagnosis of thyroid nodules using ultrasound imaging. *Journal of Biomedical Engineering*. 2025;42(5):1069–1075. doi:10.7507/1001-5515.202412047.

9. Pedraza L, Vargas C, Narváez F, Durán O, Muñoz E, Romero E. An open access thyroid ultrasound image database. *Proceedings of SPIE, 10th International Symposium on Medical Information Processing and Analysis*. 2015;9287:188–193. doi:10.1117/12.2073532.

10. Gong H, Chen G, Wang R, Xie X, Mao M, Yu Y, et al. Multi-task learning for thyroid nodule segmentation with thyroid region prior. *IEEE ISBI 2021*. doi:10.1109/ISBI48211.2021.9434087.

11. Gong H, Cheng H, Xie Y, Tan S, Chen G, Chen F, et al. Less is more: adaptive curriculum learning for thyroid nodule diagnosis. *MICCAI 2022*. LNCS Vol. 13434, pp. 248–257. doi:10.1007/978-3-031-16440-8_24.

12. Zhang H, Liu Q, Han X, Niu L, Sun W. TN5000: an ultrasound image dataset for thyroid nodule detection and classification. *Scientific Data*. 2025;12:1437. doi:10.1038/s41597-025-05757-4.

13. Duong VH, Vu H, Phan HD, Nguyen DQ, Pham DH, Le QT, et al. ThyroidXL: advancing thyroid nodule diagnosis with an expert-labeled, pathology-validated dataset. *MICCAI 2025*. LNCS Vol. 15974, pp. 616–626. doi:10.1007/978-3-032-05182-0_60.

14. Jin Z, Pei S, Ouyang L, Zhang L, Mo X, Chen Q, et al. Thy-Wise: an interpretable machine learning model for the evaluation of thyroid nodules. *International Journal of Cancer*. 2022;151(12):2229–2243. doi:10.1002/ijc.34248.

15. Bossuyt PM, Reitsma JB, Bruns DE, Gatsonis CA, Glasziou PP, Irwig L, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. *BMJ*. 2015;351:h5527. doi:10.1136/bmj.h5527.

16. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378.

17. Duan Y, Huang Y, Yang X, Han L, Xie X, Zhu Z, et al. ADAptation: reconstruction-based unsupervised active learning for breast ultrasound diagnosis. *MICCAI 2025*. LNCS Vol. 15975, pp. 35–45. doi:10.1007/978-3-032-05325-1_4.

18. Yu R, Wei Z, Zhu J, Li X, Fu X, Zhang Z, et al. DRSGen: diagnostic-region-guided single-domain generalization for thyroid nodule segmentation. *Pattern Recognition Letters*. 2026;206:120–127. doi:10.1016/j.patrec.2026.05.010.

19. Yu M, Yan Y, Yan T, Xi Z, Zeng J, Huang T, et al. A two-stage multimodal learning framework based on text-driven vision pretraining and cross-modal feature fusion for thyroid ultrasound diagnosis. *Expert Systems with Applications*. 2026;312:131440. doi:10.1016/j.eswa.2026.131440.

20. Xiang T, Hu Z. ThyroFusion: a multi-modal deep learning framework integrating vision and language for thyroid nodule malignancy risk assessment. *Journal of Imaging Informatics in Medicine*. 2026. doi:10.1007/s10278-026-01964-6.

21. Li Y, Yan X, Li J, Chen M, Chen S, Xu Y, et al. Differentiating mummified thyroid nodules from papillary thyroid carcinoma: a machine learning approach using multi-modal ultrasound radiomics. *Ultrasound in Medicine and Biology*. 2026;52(8):1763–1774. doi:10.1016/j.ultrasmedbio.2026.05.007.

22. Sherif M, Elsayed EK, Deif MA. RCAF for patient-level thyroid ultrasound malignancy prediction under leakage-free evaluation and calibration. *Scientific Reports*. 2026;16:22408. doi:10.1038/s41598-026-61342-8.

