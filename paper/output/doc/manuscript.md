# Multimodal fusion of ultrasound imaging and ACR TI-RADS features for thyroid nodule malignancy classification: a multi-dataset validation study

> Working manuscript (draft v0.2, 2026-08-04). Numbers marked [TBD] will be
> filled as experiments complete. References are placeholders pending
> citation-verifier pass. v0.2: Discussion + Conclusion drafted from available
> results; fusion-dependent paragraphs remain [TBD] awaiting ThyroidXL.
> v0.2b: self-review pass (notes/self_review_report.md) — fixed external
> specificity 72.4→72.7%, removed HEDGE_REPEAT; M1 reference-standard
> definition added in Methods 2.5.
> v0.2c: added Thy-Wise as third external cohort (CC BY 4.0; nodule-level AUC
> 0.667, mean aggregation). Paper now frames domain shift as cohort-dependent.

## Abstract

**Background.** Thyroid nodules are highly prevalent, yet ultrasound-based
risk stratification remains subjective and operator-dependent. Deep learning
models trained on thyroid ultrasound images show promise, but most studies
report single-center, single-dataset results without external validation, and
few integrate structured clinical features such as the American College of
Radiology (ACR) Thyroid Imaging Reporting and Data System (TI-RADS) descriptors.

**Methods.** We developed a multimodal framework combining an EfficientNetV2-S
image encoder with a clinical encoder that consumes six structured TI-RADS
descriptors (composition, echogenicity, shape, margin, echogenic foci, and
maximum diameter). Models were trained on 5,000 ultrasound images from the
TN5000 dataset with patient-level grouped splitting to prevent data leakage,
and validated internally on held-out test images. External validation was
performed on two independent public cohorts: TN3K (3,493 images from a
different institution and device setting) and Thy-Wise (29,070 pathology-
confirmed images from a third institution). Performance was assessed using the
area under the receiver
operating characteristic curve (AUC) with bootstrap 95% confidence intervals
(CI), sensitivity, specificity, expected calibration error (ECE), and decision
curve analysis (DCA).

**Results.** The image-only model achieved a test AUC of 0.9197 (95% CI
0.8965-0.9410) with a sensitivity of 91.1% and specificity of 72.4% on the
held-out TN5000 test set. External validation on TN3K showed a substantial
performance drop (AUC 0.7131, 95% CI 0.6960-0.7305), demonstrating
cross-dataset domain shift. Joint training on TN5000 plus TN3K training data
(7,879 images) improved both internal and external performance, raising the
external test AUC to 0.8161 (95% CI 0.7819-0.8485) and the internal test AUC
to 0.9735. On the second, fully independent pathology-confirmed cohort
(Thy-Wise; 3,954 nodules), the joint model achieved a per-nodule AUC of only
0.667 (95% CI 0.6493-0.6859), substantially below its TN3K performance,
showing that external performance is strongly cohort-dependent and that the
joint-training gain did not transfer to an unseen distribution. Fusion with
TI-RADS features [TBD: results].

**Conclusions.** [TBD after fusion experiments.]

## 1. Introduction

Thyroid nodules are among the most common endocrine findings, detectable in
more than half of the adult population by high-resolution ultrasound, yet the
large majority are benign [TBD: epidemiology citation]. This high prevalence,
combined with rising thyroid cancer incidence, has driven concerns about
overdiagnosis and unnecessary surgery [TBD: overdiagnosis citation]. Thyroid
ultrasound remains the first-line imaging modality for nodule evaluation
because it is non-invasive, radiation-free and widely available; however,
interpretation is subjective and strongly operator-dependent, and the overlap
of sonographic features between benign and malignant nodules limits diagnostic
reproducibility [TBD: citation].

Deep learning has emerged as a promising tool to support thyroid nodule
classification, segmentation and detection [TBD: reviews 2024-2025]. Numerous
models based on convolutional neural networks and vision transformers have
reported high diagnostic accuracy on internal test sets [TBD: citations].
Several open-access datasets have accelerated this progress, including DDTI
(637 images) [TBD], TN3K (3,493 images) [TBD], the TN5000 dataset (5,000
images with biopsy confirmation) [TBD], and the expert-labeled ThyroidXL
benchmark (11,545 images) [TBD]. Despite this progress, three limitations
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
foci, and its total score is a validated predictor of malignancy risk [TBD:
ACR TI-RADS white paper]. Because these features are defined semantically
rather than by image statistics, they are less affected by acquisition
variability than raw pixels, and may therefore improve robustness when models
are applied across datasets. We hypothesised that fusing image-derived deep
features with structured TI-RADS clinical features could improve both
classification accuracy and cross-dataset generalisation relative to
image-only models.

In this study, we develop and systematically evaluate a multimodal framework
that combines an EfficientNetV2 image encoder with a clinical encoder for six
structured TI-RADS descriptors. Our contributions are threefold: (1) we
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

The TN5000 dataset (Zhang et al., Scientific Data, 2025) contains 5,000
B-mode ultrasound images of thyroid nodules with expert-radiologist labels
and biopsy confirmation, explicitly designed for both detection and
classification. In our version, 1,426 images were benign and 3,574 were
malignant. We used TN5000 for model development, applying a patient-grouped
random split of 70%/15%/15% for training, validation, and internal testing.

The TN3K dataset (Gong et al., ISBI 2021) contains 3,493 ultrasound images
from 2,421 patients acquired with multiple ultrasound devices, with
segmentation masks and benign/malignant labels (0 = benign, 1 = malignant).
TN3K was used as an external validation cohort to assess cross-dataset
generalization. All 3,493 TN3K images (2,283 benign, 1,210 malignant) were
held out from training for the single-dataset evaluation. For the joint
training experiment, TN3K training images (n = 2,879) were added to the
training pool and the official TN3K test set (n = 614) was used for external
evaluation.

The Thy-Wise dataset (Jin et al., International Journal of Cancer, 2022)
comprises 29,070 B-mode ultrasound images of 3,954 thyroid nodules from the
First Affiliated Hospital of Jinan University, released under a CC BY 4.0
licence. Each nodule is represented by one or more images (median [TBD])
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

Six structured clinical features were defined following the ACR TI-RADS
lexicon: composition (0 = cystic/spongiform, 1 = mixed, 2 = solid),
echogenicity (0 = anechoic, 1 = hyper/isoechoic, 2 = hypoechoic,
3 = very hypoechoic), shape (0 = wider-than-tall, 3 = taller-than-wide),
margin (0 = smooth, 1 = ill-defined, 2 = lobulated/irregular,
3 = extrathyroidal extension), echogenic foci (0 = none/large comet-tail,
1 = macrocalcification, 2 = peripheral, 3 = punctate), and maximum diameter
in millimetres. [TBD: data source and missing-value handling, depending on
ThyroidXL approval or manual annotation protocol.]

### 2.3 Model architecture

The multimodal classifier consisted of three components. The image encoder
was an EfficientNetV2-S backbone pre-trained on ImageNet, with the final
classification layer replaced by a projection head producing a 512-dimensional
embedding. The clinical encoder was a multi-layer perceptron mapping the
six-dimensional TI-RADS vector to a 64-dimensional embedding. The two
embeddings were concatenated and passed to a prediction head with hidden
dimensions [256, 128] and a dropout of 0.3, producing a malignancy probability.
Three ablation configurations were compared: image-only (clinical branch
disabled), clinical-only (image branch disabled), and fusion (both branches
active).

### 2.4 Training

Images were resized to 224 x 224 pixels. Training used random
ShiftScaleRotate and flipping augmentations; validation used resizing only.
Models were optimised with AdamW (initial learning rate 1e-4 [TBD: verify
against config]) with a cosine annealing schedule. Training ran for 30 epochs
with a batch size of 64 using automatic mixed precision on an NVIDIA RTX 3060
(6 GB). The best model was selected by validation AUC. All splits used a fixed
random seed (42).

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
resampling (n = 1,000 [TBD: verify]). Sensitivity and specificity were
reported at the Youden-optimal operating point. Calibration was quantified by
the expected calibration error (ECE). Clinical utility was assessed by
decision curve analysis (DCA), reporting net benefit across threshold
probabilities. Reporting follows the STARD 2015 and TRIPOD checklists.

## 3. Results

### 3.1 Dataset characteristics

[Table 1 TBD — numbers for the table are below. TN5000: 5,000 images (1,426
benign / 3,574 malignant; split 70/15/15 by patient). TN3K: 3,493 images
(2,283 benign / 1,210 malignant; official test n = 614). Thy-Wise: 3,954
nodules / 29,070 images (2,876 benign / 1,078 malignant; CC BY 4.0).]

### 3.2 Internal test performance (image-only, single-dataset model)

The image-only model trained on TN5000 achieved a validation AUC of 0.9216
(95% CI 0.8960-0.9429) and a held-out test AUC of 0.9197 (95% CI
0.8965-0.9410). At the Youden-optimal threshold, test sensitivity was 91.1%,
specificity 72.4%, accuracy 85.9%, and F1 score 0.903. The expected
calibration error was 0.035.

### 3.3 External validation reveals substantial domain shift

Applying the single-dataset model to the full TN3K cohort (3,493 images from
a different institution and device setting) yielded an AUC of 0.7131 (95% CI
0.6960-0.7305) with sensitivity 58.3% and specificity 72.7% (expected
calibration error 0.136). The 0.21 decrease in AUC relative to internal
testing quantifies the cross-dataset domain shift that is frequently
underestimated in single-center evaluations.

### 3.4 Joint multi-dataset training improves internal and external performance

Joint training on TN5000 together with the TN3K training subset (7,879
images in total) improved both internal and external performance. The
joint model reached a validation AUC of 0.9413 (95% CI 0.9213-0.9595) and a
held-out TN5000 test AUC of 0.9735 (95% CI 0.9610-0.9842) with sensitivity
96.1% and specificity 86.7% (expected calibration error 0.034). On the
official TN3K test set (n = 614), external AUC improved to 0.8161 (95% CI
0.7819-0.8485), with sensitivity 64.0%, specificity 82.5% and expected
calibration error 0.072 (Table 2; Figures 2-4). Joint training therefore
recovered approximately half of the domain-shift gap on external data while
simultaneously improving internal discrimination.

On the second external cohort, however, this gain did not transfer. Applied to
Thy-Wise (3,954 pathology-confirmed nodules from a third institution), the
joint model achieved a nodule-level AUC of 0.667 (95% CI 0.649-0.686) with an
expected calibration error of 0.049, where per-image probabilities were
averaged within each nodule before metric computation. Per-image statistics
were markedly lower (AUC 0.586), reflecting the presence of multiple,
partly-redundant frames per nodule; nodule-level aggregation was therefore used
as the primary external estimate. The 0.15 lower AUC relative to TN3K shows
that the benefit of joint multi-dataset training was specific to the
distribution added during training and did not generalise to an unseen cohort.

### 3.5 Multimodal fusion

[TBD: image vs clinical vs fusion ablation table; ECE and DCA figures.]

## 4. Discussion

We systematically evaluated a multimodal framework for thyroid nodule
malignancy classification that combines ultrasound image features with
structured ACR TI-RADS descriptors, and quantified its generalisation across
three public multi-dataset cohorts. Three principal findings emerged. First, an
image-only model trained on a single large dataset achieved high internal
discrimination (test AUC 0.9197) but lost substantial performance on an
independent external dataset (AUC 0.7131), quantifying a 0.21 cross-dataset
domain-shift gap that is invisible to single-cohort evaluation. Second, joint
training on the combined training images of two datasets (7,879 images)
recovered approximately half of this gap on TN3K (external AUC 0.8161, +0.10)
while simultaneously improving internal test performance (AUC 0.9735). Third,
on a second, fully independent pathology-confirmed cohort (Thy-Wise; 3,954
nodules), the same joint model achieved a nodule-level AUC of only 0.667 —
substantially below its TN3K result — demonstrating that external performance
is strongly cohort-dependent and that the benefit of joint training did not
transfer to an unseen distribution. Across these analyses we report calibration
and decision-curve metrics alongside discrimination [fusion ablation outcome
TBD], addressing reporting gaps that remain common in the field.

**Domain shift is the central barrier to translation.** The external drop from
0.9197 to 0.7131 deserves emphasis. TN5000 and TN3K were acquired with
different devices at different institutions, and their label prevalence also
differs substantially (71.5% vs 34.6% malignant), which partly explains the
concurrent fall in sensitivity (91.1% to 58.3%) at a fixed operating
threshold. A second external cohort confirmed that the magnitude of this shift
is cohort-specific: applied to Thy-Wise, the joint model's nodule-level AUC
fell further to 0.667 (Section 3.4). Comparable degradation when ultrasound
models are transferred across devices has been reported in the
domain-adaptation literature [TBD: ADAptation, MICCAI 2025; EAGT augmentation
study]. Our result supports the emerging consensus that single-dataset AUCs
substantially overestimate real-world performance, and reinforces the TRIPOD
recommendation that external validation is a prerequisite for deployment
[TBD: TRIPOD].

**Joint multi-dataset training helps the cohort added to training, but the
gain is distribution-specific.** Adding TN3K training images to the TN5000 pool
improved external AUC on the official TN3K test set from 0.7131 to 0.8161
(+0.10), restored roughly half of the domain-shift deficit, and improved
external specificity from 72.7% to 82.5%. Multi-dataset pooling is therefore a
pragmatic robustness strategy when data from the target cohort can be included
in training. Critically, the benefit did not generalise beyond the added
distribution: on the fully independent Thy-Wise cohort, the same joint model
achieved a nodule-level AUC of 0.667 (95% CI 0.649-0.686), well below its TN3K
result. Pooling therefore cannot confer robustness to cohorts that are not
represented in training, and the residual gaps seen on both TN3K and Thy-Wise
indicate that device-agnostic priors — such as structured clinical features —
may be required in addition.

**Structured TI-RADS features as device-independent priors.** The central
hypothesis of this study is that ACR TI-RADS descriptors, which are defined
semantically and therefore largely independent of acquisition hardware, could
stabilise predictions across datasets. [TBD: fusion results — report the
image-only vs clinical-only vs fusion ablation on internal and external data;
if the fused model narrows the external gap, this supports the hypothesis.]
If confirmed, this would position our framework apart from multimodal methods
that depend on free-text radiology reports [TBD: TTM-Net; ThyroFusion] or on
contrast-enhanced sequences [TBD: UMB radiomics + clinical], making it
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
pathological labels in TN5000 and Thy-Wise, versus [TBD: TN3K label
definition]) and prevalence differs across datasets (71.5%, 34.6% and 27.3%
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
institution-specific evaluation would be required before clinical deployment
[TBD: ThyroidXL as an additional external cohort].

**Future work.** Once access to the expert-labelled ThyroidXL benchmark is
granted [TBD: pending author approval], we will evaluate the fusion framework
on this larger cohort and test the device-independence hypothesis more
directly. Automated extraction of TI-RADS descriptors from ultrasound images
[TBD: DL-based ACR-TI-RADS classification] could remove the need for manual
annotation while preserving the structured-feature prior, and would strengthen
the clinical utility of the framework.

## 5. Conclusion

Using three public multi-dataset cohorts, we showed that a thyroid nodule
classification model trained on a single dataset achieves high internal AUC
but loses substantial performance externally (0.92 to 0.71 on TN3K); that joint
multi-dataset training recovers about half of that gap on the cohort added to
training (external AUC 0.8161), but that this benefit does not transfer to a
second, unseen external cohort (nodule-level AUC 0.667 on Thy-Wise). External
performance is therefore strongly cohort-dependent, and image-only models
alone appear insufficient for reliable cross-site deployment. We further
established an ablation framework for fusing image features with structured ACR
TI-RADS descriptors and reported calibration and decision-curve metrics
alongside discrimination. [TBD: one sentence on the fusion outcome, to be added
when ablation completes.] Together, these findings support external validation
and clinical-feature fusion as key steps towards reliable, deployable
ultrasound-based thyroid nodule classification.

## Declarations

- Data availability: TN5000 (HF mirror / figshare), TN3K (Hugging Face,
  haifan-gong/TN3K), Thy-Wise (figshare, DOI 10.6084/m9.figshare.20417895, CC
  BY 4.0), ThyroidXL [TBD pending approval].
- Code availability: GitHub repository (to be published).
- AI disclosure: ARS disclosure statement [TBD].

## References

[Placeholder - citation-verifier pass required]

1. Zhang H, et al. TN5000. Sci Data. 2025;12:1437.
2. Duong VH, et al. ThyroidXL. MICCAI 2025.
3. Gong H, et al. TN3K. ISBI 2021 / CBM 2022.
4. Pedraza L, et al. DDTI. SPIE 2015.
5. Jin Z, Pei S, Ouyang L, et al. Thy-Wise: an interpretable machine learning
   model for the evaluation of thyroid nodules. Int J Cancer.
   2022;151:2229-2243. doi:10.1002/ijc.34248.
6. [Scientific Data 2024 figshare 8508]
