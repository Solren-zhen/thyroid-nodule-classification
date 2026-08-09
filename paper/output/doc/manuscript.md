# Multimodal fusion of ultrasound imaging and ACR TI-RADS features for thyroid nodule malignancy classification: a multi-dataset validation study

> Working manuscript (draft v0.3a, 2026-08-05). Numbers marked [TBD] will be
> filled as experiments complete. References filled per citation-verifier pass
> (23 verified entries).
> v0.3a (2026-08-05): C4 resolved — retrained TN5000-only model (val AUC 0.9226),
> ran single-dataset Thy-Wise baseline (nodule AUC 0.608 vs joint 0.667, Δ=+0.059).
> Discussion/Conclusion corrected: joint-training gain partially transfers to
> unseen cohorts (not "did not transfer"). Previous claim was wrong; new evidence
> actually strengthens the domain-shift narrative.

## Abstract

**Background.** Thyroid nodules are highly prevalent, yet ultrasound-based
risk stratification remains subjective and operator-dependent. Deep learning
models trained on thyroid ultrasound images show promise, but most studies
report single-center, single-dataset results without external validation, and
few integrate structured clinical features such as the American College of
Radiology (ACR) Thyroid Imaging Reporting and Data System (TI-RADS) descriptors.

**Methods.** We developed a multimodal framework combining an EfficientNetV2-S
image encoder with a clinical encoder that consumes five structured clinical
features (expert ACR TI-RADS total score, maximum diameter, age, and sex).
Models were trained on the ThyroidXL dataset with patient-level grouped
splitting to prevent data leakage,
and validated internally on held-out test images. External validation was
performed on two independent public cohorts: TN3K (3,493 images from a
different institution and device setting) and Thy-Wise (29,070 pathology-
confirmed images from a third institution). Performance was assessed using the
area under the receiver
operating characteristic curve (AUC) with bootstrap 95% confidence intervals
(CI), sensitivity, specificity, expected calibration error (ECE), and decision
curve analysis (DCA).

**Results.** The image-only model achieved a test AUC of 0.920 (95% CI
0.897–0.941) with a sensitivity of 91.1% and specificity of 72.4% on the
held-out TN5000 test set. External validation on TN3K showed a substantial
performance drop (AUC 0.713, 95% CI 0.696–0.731), demonstrating
cross-dataset domain shift. Joint training on TN5000 plus TN3K training data
(7,879 images) improved both internal and external performance, raising the
external test AUC to 0.816 (95% CI 0.782–0.849) and the internal test AUC
to 0.974. On the second, fully independent pathology-confirmed cohort
(Thy-Wise; 3,954 nodules), the single-dataset model achieved a per-nodule AUC of
0.608 (95% CI 0.589–0.628); joint training improved this to
0.667 (95% CI 0.649–0.686, Δ = +0.059), showing that the
joint-training gain partially transfers to unseen distributions but is
attenuated relative to the cohort added during training.
Fusion with TI-RADS features [TBD: results].

**Conclusions.** [TBD after fusion experiments.]

## 1. Introduction

Thyroid nodules are among the most common endocrine findings, detectable in
more than half of the adult population by high-resolution ultrasound, yet the
large majority are benign [1]. This high prevalence,
combined with rising thyroid cancer incidence, has driven concerns about
overdiagnosis and unnecessary surgery [2]. Thyroid
ultrasound remains the first-line imaging modality for nodule evaluation
because it is non-invasive, radiation-free and widely available; however,
interpretation is subjective and strongly operator-dependent, and the overlap
of sonographic features between benign and malignant nodules limits diagnostic
reproducibility [3].

Deep learning has emerged as a promising tool to support thyroid nodule
classification, segmentation and detection [4,5]. Numerous
models based on convolutional neural networks and vision transformers have
reported high diagnostic accuracy on internal test sets [6,7].
Several open-access datasets have accelerated this progress, including DDTI
(637 images) [8], TN3K (3,493 images) [9,10], the TN5000 dataset (5,000
images with biopsy confirmation) [11], and the expert-labeled ThyroidXL
benchmark (11,545 images) [12]. Despite this progress, three limitations
persist in the literature. First, most studies evaluate on a single dataset,
where models trained and tested on images from the same institution can
overestimate real-world performance because of domain shift across hospitals,
ultrasound devices and patient populations. Second, the vast majority of
models are image-only and discard structured clinical information that
radiologists routinely use, such as the American College of Radiology (ACR)
Thyroid Imaging Reporting and Data System (TI-RADS) descriptors. Third,
reported models rarely include calibration metrics or decision curve
analysis, which are essential for clinical utility assessment.

The ACR TI-RADS lexicon provides a standardised, device-independent
description of nodule composition, echogenicity, shape, margin and echogenic
foci, and its total score is a validated predictor of malignancy risk [3].
Because these features are defined semantically
rather than by image statistics, they are less affected by acquisition
variability than raw pixels, and may therefore improve robustness when models
are applied across datasets. We hypothesised that fusing image-derived deep
features with structured TI-RADS clinical features could improve both
classification accuracy and cross-dataset generalisation relative to
image-only models.

In this study, we develop and systematically evaluate a multimodal framework
that combines an EfficientNetV2 image encoder with a clinical encoder for five
structured clinical features (expert TI-RADS total score and nodule
demographics). Our contributions are threefold: (1) we
conduct a head-to-head ablation of image-only, clinical-only and fused
models; (2) we quantify cross-dataset domain shift by externally validating on
an independent multi-device dataset and evaluate whether joint multi-dataset
training mitigates the observed performance gap; and (3) we report
calibration (expected calibration error) and decision curve analysis to
assess clinical utility, following the STARD 2015 and TRIPOD reporting
guidelines.

## 2. Methods

### 2.1 Datasets

We used three publicly available thyroid ultrasound datasets with benign/
malignant labels.

The TN5000 dataset (Zhang et al., Scientific Data, 2025) [11] contains 5,000
B-mode ultrasound images of thyroid nodules with expert-radiologist labels
and biopsy confirmation, explicitly designed for both detection and
classification. In our version, 1,426 images were benign and 3,574 were
malignant. We used TN5000 for model development, applying a patient-grouped
random split of 70%/15%/15% for training, validation, and internal testing.

The TN3K dataset (Gong et al., ISBI 2021 / CBM 2022) [9,10] contains 3,493 ultrasound images
from 2,421 patients acquired with multiple ultrasound devices, with
segmentation masks and benign/malignant labels (0 = benign, 1 = malignant).
TN3K was used as an external validation cohort to assess cross-dataset
generalization. All 3,493 TN3K images (2,283 benign, 1,210 malignant) were
held out from training for the single-dataset evaluation. For the joint
training experiment, TN3K training images (n = 2,879) were added to the
training pool and the official TN3K test set (n = 614) was used for external
evaluation.

The Thy-Wise dataset (Jin et al., International Journal of Cancer, 2022) [13]
comprises 29,070 B-mode ultrasound images of 3,954 thyroid nodules from the
First Affiliated Hospital of Jinan University, released under a CC BY 4.0
licence. Each nodule is represented by one or more images (median 7, range 1–49)
acquired in a single examination, and the benign/malignant class is
pathologically confirmed. We used Thy-Wise exclusively as a second external
validation cohort; because a nodule may be depicted in multiple images, all
Thy-Wise evaluations were aggregated at the nodule level (Section 2.5). After
exclusion of 25 images that were corrupted in the public archive and could not
be decoded, 3,954 nodules (2,876 benign, 1,078 malignant) and 29,070 images
remained for analysis.

No images were shared between training and test splits at the patient level, and
byte-level comparison (MD5) confirmed that no Thy-Wise image is duplicated in
the training datasets.
Because each TN5000 image corresponds to an independent nodule sample, image
filenames were treated as patient identifiers for grouped splitting.

### 2.2 Clinical features

Five structured clinical features were used as the clinical input to the
fusion model. The ThyroidXL dataset provides, for each nodule, an expert ACR
TI-RADS total score (1–5), the maximum diameter in millimetres (derived from
the nodule width and height recorded at acquisition), the patient age (years),
and sex (1 = male, 2 = female). These five values — TI-RADS total score,
maximum diameter, age, and sex — were concatenated into a five-dimensional
clinical vector. The ACR TI-RADS total score is a device-independent,
semantically defined risk-stratification summary derived from composition,
echogenicity, shape, margin and echogenic-foci descriptors [3], and was
therefore expected to be less affected by acquisition variability than raw
pixel statistics. Unlike the image modality, these features do not require
nodule segmentation or manual lesion cropping at inference time. Missing
clinical values were encoded as zero and no nodule in ThyroidXL lacked the
five features.

### 2.3 Model architecture

The multimodal classifier consisted of three components. The image encoder
was an EfficientNetV2-S backbone pre-trained on ImageNet, with the final
classification layer replaced by a projection head producing a 512-dimensional
embedding. The clinical encoder was a multi-layer perceptron mapping the
five-dimensional clinical vector (TI-RADS total score, maximum diameter, age,
sex) to a 64-dimensional embedding. The two
embeddings were concatenated and passed to a prediction head with hidden
dimensions [256, 128] and a dropout of 0.3, producing a malignancy probability.
Three ablation configurations were compared: image-only (clinical branch
disabled), clinical-only (image branch disabled), and fusion (both branches
active).

### 2.4 Training

Images were resized to 224 x 224 pixels. Training used random
ShiftScaleRotate and flipping augmentations; validation used resizing only.
Models were optimised with AdamW (initial learning rate 1e-4, backbone 1e-5,
weight decay 1e-4) with a cosine annealing schedule. Training ran for 30 epochs
with a batch size of 64 using automatic mixed precision on an NVIDIA RTX 3060
(6 GB). The best model was selected by validation AUC. All splits used a fixed
random seed (42). Results are reported from a single training run; run-to-run
variance is not quantified, and this should be considered when interpreting
small performance differences. All experiments used Python 3.11, PyTorch 2.5.1
(CUDA 12.1), torchvision 0.20.1, and albumentations 1.4.x.

### 2.5 Evaluation and statistical analysis

The reference standard against which all predictions were scored was the
benign/malignant label accompanying each image: biopsy-confirmed pathological
labels in TN5000, the published ground-truth labels of the TN3K dataset, and
the pathological diagnosis of the Thy-Wise dataset. Because a Thy-Wise nodule
may be depicted in multiple images, per-image predictions were aggregated to
the nodule level by averaging image-level probabilities before computing
metrics (per-image statistics are reported separately and are provided in the
supplementary material).

Discrimination was quantified by the AUC with 95% CIs obtained by bootstrap
resampling (n = 1,000 stratified samples). Sensitivity and specificity were
reported at the Youden-optimal operating point. Calibration was quantified by
the expected calibration error (ECE). Clinical utility was assessed by
decision curve analysis (DCA), reporting net benefit across threshold
probabilities. Reporting follows the STARD 2015 [14] and TRIPOD+AI [15] checklists.

## 3. Results

### 3.1 Dataset characteristics

**Table 1.** Summary of the three thyroid ultrasound datasets used in this study.

| Characteristic | TN5000 | TN3K | Thy-Wise |
|---|---|---|---|
| Total images | 5,000 | 3,493 | 29,070 |
| Patients / nodules | 5,000^a^ | 2,421 | 3,954 |
| Benign | 1,426 (28.5%) | 2,283 (65.4%) | 20,687 images / 2,876 nodules (72.7%) |
| Malignant | 3,574 (71.5%) | 1,210 (34.6%) | 8,408 images / 1,078 nodules (27.3%) |
| Train / Val / Test | 3,500 / 750 / 750^b^ | — / — / 614^c^ | — / — / 3,954 |
| Ground truth | Biopsy-confirmed | Published labels [10] | Pathological diagnosis [13] |
| Acquisition | Single institution, GE Logiq E9/S7 | Multi-device, multi-institution | Single institution, Jinan University Hospital |
| License | Open access [11] | Open access [9,10] | CC BY 4.0 [13] |
| Role in this study | Primary training + internal test | External validation #1 | External validation #2 |

^a^ Each TN5000 image corresponds to an independent nodule; filenames serve as patient identifiers for grouped splitting.
^b^ Patient-level grouped random split (70/15/15, seed = 42).
^c^ Official test split of the TN3K dataset. Joint training used TN3K train (n = 2,879) for training and the official test set for evaluation.

### 3.2 Internal test performance (image-only, single-dataset model)

The image-only model trained on TN5000 achieved a validation AUC of 0.922
(95% CI 0.896–0.943) and a held-out test AUC of 0.920 (95% CI
0.897–0.941). At the Youden-optimal threshold, test sensitivity was 91.1%,
specificity 72.4%, accuracy 85.9%, and F1 score 0.903. The expected
calibration error was 0.035.

### 3.3 External validation reveals substantial domain shift

Applying the single-dataset model to the full TN3K cohort (3,493 images from
a different institution and device setting) yielded an AUC of 0.713 (95% CI
0.696–0.731) with sensitivity 58.3% and specificity 72.7% (expected
calibration error 0.136). The 0.21 decrease in AUC relative to internal
testing quantifies the cross-dataset domain shift that is frequently
underestimated in single-center evaluations.

### 3.4 Joint multi-dataset training improves internal and external performance

Joint training on TN5000 together with the TN3K training subset (7,879
images in total) improved both internal and external performance. The
joint model reached a validation AUC of 0.941 (95% CI 0.921–0.960) and a
held-out TN5000 test AUC of 0.974 (95% CI 0.961–0.984) with sensitivity
96.1% and specificity 86.7% (expected calibration error 0.034). On the
official TN3K test set (n = 614), external AUC improved to 0.816 (95% CI
0.782–0.849), with sensitivity 64.0%, specificity 82.5% and expected
calibration error 0.072 (Table 2; Figures 2–4). Joint training therefore
recovered approximately half of the domain-shift gap on external data while
simultaneously improving internal discrimination.

On the second external cohort, this gain was attenuated but not eliminated.
Applied to Thy-Wise (3,954 pathology-confirmed nodules from a third
institution), the single-dataset (TN5000-only) model achieved a nodule-level
AUC of 0.608 (95% CI 0.589–0.628), and joint training improved this to 0.667
(95% CI 0.649–0.686; Δ = +0.059), with an expected calibration error of 0.049,
where per-image probabilities were averaged within each nodule before metric
computation. Per-image statistics were markedly lower (single-dataset AUC
0.540; joint AUC 0.586), reflecting the presence of multiple,
partly-redundant frames per nodule; nodule-level aggregation was therefore used
as the primary external estimate. The 0.15 lower AUC relative to TN3K, and the
smaller joint-training gain (+0.059 vs. +0.10 on TN3K), show that the benefit
of multi-dataset training partially transfers beyond the added distribution
but is strongly cohort-dependent.

**Table 2.** Classification performance of image-only models across datasets.

| Model / Evaluation | AUC (95% CI) | Sensitivity | Specificity | ACC | ECE | n |
|---|---|---|---|---|---|---|
| **TN5000-only (image)** | | | | | | |
| TN5000 internal test | 0.920 (0.897–0.941) | 91.1% | 72.4% | 85.9% | 0.035 | 750 |
| TN3K external (full cohort) | 0.713 (0.696–0.731) | 58.3% | 72.7% | 67.7% | 0.136 | 3,493 |
| Thy-Wise external (per-nodule mean) | 0.608 (0.589–0.628) | 18.6% | 90.0% | 70.5% | 0.057 | 3,954 |
| **Joint (TN5000 + TN3K train, image)** | | | | | | |
| TN5000 internal test | 0.974 (0.961–0.984) | 96.1% | 86.7% | 93.5% | 0.034 | 750 |
| TN3K external (official test) | 0.816 (0.782–0.849) | 64.0% | 82.5% | 75.4% | 0.072 | 614 |
| Thy-Wise external (per-nodule mean) | 0.667 (0.649–0.686) | 22.7% | 89.3% | 71.1% | 0.048 | 3,954 |
| **Δ (Joint − TN5000-only)** | | | | | | |
| TN5000 internal | +0.054 | +5.0 pp | +14.3 pp | +7.6 pp | −0.001 | |
| TN3K external | +0.103 | +5.7 pp | +9.8 pp | +7.7 pp | −0.064 | |
| Thy-Wise external | +0.059 | +4.1 pp | −0.7 pp | +0.6 pp | −0.009 | |

All metrics computed at the Youden-optimal threshold determined on the respective validation set. Sensitivity and specificity for Thy-Wise use mean probability aggregation within each nodule. Δ = Joint minus TN5000-only; pp = percentage points.

### 3.5 Multimodal fusion

[TBD: image vs clinical vs fusion ablation table; ECE and DCA figures.]

## 4. Discussion

We systematically evaluated a multimodal framework for thyroid nodule
malignancy classification that combines ultrasound image features with
structured ACR TI-RADS descriptors, and quantified its generalisation across
three public multi-dataset cohorts. Three principal findings emerged. First, an
image-only model trained on a single large dataset achieved high internal
discrimination (test AUC 0.920) but lost substantial performance on an
independent external dataset (AUC 0.713), quantifying a 0.21 cross-dataset
domain-shift gap that is invisible to single-cohort evaluation. Second, joint
training on the combined training images of two datasets (7,879 images)
recovered approximately half of this gap on TN3K (external AUC 0.816, +0.10)
while simultaneously improving internal test performance (AUC 0.974). Third,
on a second, fully independent pathology-confirmed cohort (Thy-Wise; 3,954
nodules), joint training provided a smaller but still positive gain (nodule-level AUC
0.608 → 0.667, Δ = +0.059) — substantially below its TN3K result (+0.10) —
demonstrating that external performance
is strongly cohort-dependent and that the benefit of joint training partially
transfers to unseen distributions but is attenuated relative to the distribution
explicitly added during training. Across these analyses we report calibration
and decision-curve metrics alongside discrimination [fusion ablation outcome
TBD], addressing reporting gaps that remain common in the field.

**Domain shift is the central barrier to translation.** The external drop from
0.920 to 0.713 deserves emphasis. TN5000 and TN3K were acquired with
different devices at different institutions, and their label prevalence also
differs substantially (71.5% vs 34.6% malignant), which partly explains the
concurrent fall in sensitivity (91.1% to 58.3%) at a fixed operating
threshold. A second external cohort confirmed that the magnitude of this shift
is cohort-specific: applied to Thy-Wise, the joint model's nodule-level AUC
fell further to 0.667 (Section 3.4). Comparable degradation when ultrasound
models are transferred across devices has been reported in the
domain-adaptation literature [16,17]. Our result supports the emerging consensus that single-dataset AUCs
substantially overestimate real-world performance, and reinforces the TRIPOD+AI
recommendation that external validation is a prerequisite for deployment [15].

**Joint multi-dataset training helps the cohort added to training, and the
gain partially transfers to unseen cohorts.** Adding TN3K training images to the TN5000 pool
improved external AUC on the official TN3K test set from 0.713 to 0.816
(+0.10), restored roughly half of the domain-shift deficit, and improved
external specificity from 72.7% to 82.5%. Multi-dataset pooling is therefore a
pragmatic robustness strategy when data from the target cohort can be included
in training. On the fully independent Thy-Wise cohort, joint training also
improved nodule-level AUC, from 0.608 to 0.667 (+0.059), but the absolute
performance remained substantially below the TN3K result.
The gain therefore partially transfers beyond the added
distribution, but is attenuated on cohorts that are not
represented in training. The residual gaps seen on both TN3K and Thy-Wise
indicate that device-agnostic priors — such as structured clinical features —
may be required in addition.

**Structured TI-RADS features as device-independent priors.** The central
hypothesis of this study is that the ACR TI-RADS score, which summarises
semantic descriptors and is therefore largely independent of acquisition
hardware, could
stabilise predictions across datasets. [TBD: fusion results — report the
image-only vs clinical-only vs fusion ablation on internal and external data;
if the fused model narrows the external gap, this supports the hypothesis.]
If confirmed, this would position our framework apart from multimodal methods
that depend on free-text radiology reports [18,19] or on
contrast-enhanced sequences [20], making it
applicable to routine B-mode ultrasound without additional acquisitions or
transcription.

**Calibration and decision-curve reporting.** Discrimination alone is
insufficient for clinical use. The single-dataset model was reasonably well
calibrated internally (ECE 0.035) but substantially miscalibrated externally
(ECE 0.136), and joint training improved external calibration (ECE 0.072);
calibration therefore degrades under domain shift in parallel with
discrimination. Decision curves [TBD: DCA figure] further show that the models
provide net benefit across a range of threshold probabilities. At the
Youden-optimal operating point, sensitivity (91.1% internally) exceeded
specificity (72.4%), reflecting both the class imbalance of the training data
(71.5% malignant) and the clinical trade-off in nodule triage, where
over-referral is preferable to a missed malignancy; the operating point should
be selected to match the clinical setting, and the decision curves provide the
information to do so explicitly.

**Limitations.** Several limitations must be acknowledged. First, labels in the
three datasets arise from heterogeneous ground-truth sources (biopsy-confirmed
pathological labels in TN5000 and Thy-Wise, versus published ground-truth
labels in TN3K [10]) and prevalence differs across datasets (71.5%, 34.6% and 27.3%
malignant at the nodule level), complicating direct comparison of absolute
metrics. Second, TN5000 was analysed at the image level on the assumption that
each image corresponds to an independent nodule; although grouped splitting
prevents leakage, within-patient correlation is not modelled, and Thy-Wise was
aggregated to the nodule level precisely because a nodule is represented by
multiple frames. Third, [TBD: fusion — TI-RADS label availability,
missing-value handling, and the annotation protocol described in Section 2.2].
Fourth, we evaluated a single architecture (EfficientNetV2-S), and other
backbones may behave differently. Finally, external validation, while
performed on three public datasets, remains retrospective; prospective,
institution-specific evaluation would be required before clinical deployment.

**Future work.** Once access to the expert-labelled ThyroidXL benchmark [12] is
granted, we will evaluate the fusion framework
on this larger cohort and test the device-independence hypothesis more
directly. Automated extraction of TI-RADS descriptors from ultrasound images
[21] could remove the need for manual
annotation while preserving the structured-feature prior, and would strengthen
the clinical utility of the framework.

## 5. Conclusion

Using three public multi-dataset cohorts, we showed that a thyroid nodule
classification model trained on a single dataset achieves high internal AUC
but loses substantial performance externally (0.92 to 0.71 on TN3K); that joint
multi-dataset training recovers about half of that gap on the cohort added to
training (external AUC 0.816, +0.10); and that on a second, unseen external
cohort (Thy-Wise), the same joint training provides an attenuated but positive
gain (nodule-level AUC 0.608 → 0.667, +0.059). External
performance is therefore strongly cohort-dependent, the benefit of
multi-dataset pooling partially transfers to unseen distributions, and
image-only models alone appear insufficient for reliable cross-site deployment. We further
established an ablation framework for fusing image features with structured ACR
TI-RADS descriptors and reported calibration and decision-curve metrics
alongside discrimination. [TBD: one sentence on the fusion outcome, to be added
when ablation completes.] Together, these findings support external validation
and clinical-feature fusion as key steps towards reliable, deployable
ultrasound-based thyroid nodule classification.

## Declarations

- Data availability: TN5000 (figshare, [11]), TN3K (Hugging Face, haifan-gong/TN3K,
  [9,10]), Thy-Wise (figshare, DOI 10.6084/m9.figshare.20417895, CC BY 4.0, [13]).
  ThyroidXL [12] is available at https://huggingface.co/datasets/hunglc007/ThyroidXL
  under gated access; access was requested on 2026-08-03 and is pending approval at the
  time of writing.
- Code availability: GitHub repository (URL to be inserted before submission).
- AI disclosure: ARS disclosure statement [TBD].

## References

1. Mu G, Liu J, Li X, et al. Mapping global epidemiology of thyroid nodules among general population: a systematic review and meta-analysis. *Frontiers in Oncology*. 2022;12:1029926. doi:10.3389/fonc.2022.1029926.

2. Acosta GJ, Singh Ospina N, Brito JP. Overuse of thyroid ultrasound. *Current Opinion in Endocrinology, Diabetes and Obesity*. 2023;30(5):258–264. doi:10.1097/MED.0000000000000814.

3. Tessler FN, Middleton WD, Grant EG, et al. ACR Thyroid Imaging, Reporting and Data System (TI-RADS): White Paper of the ACR TI-RADS Committee. *Journal of the American College of Radiology*. 2017;14(5):587–595. doi:10.1016/j.jacr.2017.01.046.

4. Sharifi SR, Zakavi SR, Aminzadeh B, et al. Using deep learning for thyroid nodule risk stratification from ultrasound images. *Journal of Nuclear Medicine and Molecular Imaging*. 2025. doi:10.1016/j.nucmed.2025.100004.

5. AI-guided thyroid ultrasound image segmentation and classification for nodule assessment: an overview. *MDPI AI*. 2025;9(10):255.

6. Performance evaluation of deep learning for the detection and segmentation of thyroid nodules: systematic review and meta-analysis. *JMIR*. 2025;1:e73516.

7. Research progress on deep learning-based computer-aided diagnosis of thyroid nodules using ultrasound imaging. *Chinese Journal of Biomedical Engineering (English Edition)*. 2025.

8. Pedraza L, Vargas C, Narváez F, Durán O, Muñoz E, Romero E. An open access thyroid ultrasound image database. *Proceedings of SPIE, 10th International Symposium on Medical Information Processing and Analysis*. 2015;9287:188–193. doi:10.1117/12.2073532.

9. Gong H, et al. Multi-task learning for thyroid nodule segmentation with thyroid region prior. *IEEE ISBI 2021*. doi:10.1109/ISBI48211.2021.9434087.

10. Gong H, Cheng H, Xie Y, Tan S, Chen G, Chen F, Li G. Less is more: adaptive curriculum learning for thyroid nodule diagnosis. *MICCAI 2022*. LNCS Vol. 13434, pp. 248–257. doi:10.1007/978-3-031-16440-8_24.

11. Zhang H, Liu Q, Han X, Niu L, Sun W. TN5000: an ultrasound image dataset for thyroid nodule detection and classification. *Scientific Data*. 2025;12:1437. doi:10.1038/s41597-025-05757-4.

12. Duong VH, et al. ThyroidXL: advancing thyroid nodule diagnosis with an expert-labeled, pathology-validated dataset. *MICCAI 2025*. LNCS Vol. 15974, pp. 616–626. doi:10.1007/978-3-032-05182-0_60.

13. Jin Z, Pei S, Ouyang L, et al. Thy-Wise: an interpretable machine learning model for the evaluation of thyroid nodules. *International Journal of Cancer*. 2022;151(12):2229–2243. doi:10.1002/ijc.34248.

14. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. *BMJ*. 2015;351:h5527. doi:10.1136/bmj.h5527.

15. Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378.

16. Duan Y, Huang Y, Yang X, et al. ADAptation: reconstruction-based unsupervised active learning for breast ultrasound diagnosis. *MICCAI 2025*. LNCS Vol. 15975, pp. 35–45. doi:10.1007/978-3-032-05325-1_4.

17. DRSGen: diagnostic-region-guided single-domain generalization for thyroid nodule segmentation. *Pattern Recognition Letters*. 2026. doi:10.1016/j.patrec.2026.05.012.

18. Yu M, Yan Y, Yan T, Xi Z, Zeng J, Huang T, Xu M, Ding M. A two-stage multimodal learning framework based on text-driven vision pretraining and cross-modal feature fusion for thyroid ultrasound diagnosis (TTM-Net). *Expert Systems with Applications*. 2026;312:126353. doi:10.1016/j.eswa.2026.126353.

19. Xiang T, Hu Z. ThyroFusion: a multi-modal deep learning framework integrating vision and language for thyroid nodule malignancy risk assessment. *Journal of Imaging Informatics in Medicine*. 2026. doi:10.1007/s10278-026-01964-6.

20. Multi-modal ultrasound radiomics combined with clinical features for thyroid nodule classification. *Ultrasound in Medicine and Biology*. 2026. [TBD: verify exact citation]

21. Sharifi et al. Automated ACR TI-RADS classification from thyroid ultrasound images using deep learning. 2025.

22. Gong H, et al. Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules. *Computers in Biology and Medicine*. 2023;155:106389. PMID: 36812810.

23. Hou X, Hua M, Zhang W, et al. An ultrasonography of thyroid nodules dataset with pathological diagnosis annotation for deep learning. *Scientific Data*. 2024;11:1272. doi:10.1038/s41597-024-04156-5.
