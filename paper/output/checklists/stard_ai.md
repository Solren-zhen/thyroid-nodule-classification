# STARD-AI Checklist (2025)

**Standards for Reporting of Diagnostic Accuracy Studies — Artificial Intelligence**

Manuscript title: External generalization and domain shift in thyroid ultrasound AI: quantifying cross-dataset performance, multi-dataset training, and the incremental value of structured features
Date: 2026-08-12
Status: Author self-assessment (fill page numbers after final formatting)

Reference: Sounderajah V, et al. The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. Nat Med. 2025;31:3283-3289. doi:10.1038/s41591-025-03953-8

| # | Item | Reported in Section | Status |
|---|------|--------------------|--------|
| 1 | Title — identify as AI diagnostic accuracy study, report accuracy measure | Title | ✓ PRESENT |
| 2 | Abstract — structured summary | Abstract | ✓ PRESENT |
| 3 | Scientific background, intended use of index test, workflow integration | §1 Introduction | ✓ PRESENT |
| 4 | Objectives / hypotheses | §1 Introduction | ✓ PRESENT |
| 5 | Study design (prospective vs retrospective) | §2.1 (retrospective, public datasets) | ✓ PRESENT |
| 6 | Ethics approval (or justification) | Declarations (de-identified public data; no IRB required) | ✓ PRESENT |
| 7 | Eligibility criteria (participant + data level) | §2.1 (4 datasets; 25 corrupted Thy-Wise images excluded) | ✓ PRESENT |
| 8 | Participant identification basis | §2.1 (open repositories) | ✓ PRESENT |
| 9 | Setting and dates | §2.1 (reporting note: exact acquisition dates not published; institutions, devices and publication years reported) | ● PRESENT |
| 10 | Participant series (consecutive/random/convenience) | §2.1 (reporting note: public archive cohorts, not consecutive clinical series) | ● PRESENT |
| 11 | Data source (routinely collected / open-source) | §2.1 + Declarations | ✓ PRESENT |
| 12 | Dataset annotation (who, experience, how) | §2.1 (expert-radiologist labels; pathology/cytology confirmation) | ✓ PRESENT |
| 13 | Data capture devices and software | §2.1 (device listed) + §2.4 (software versions) | ✓ PRESENT |
| 14 | Data acquisition and preprocessing | §2.4 | ✓ PRESENT |
| 15a | Index test detail | §2.3 Model architecture | ✓ PRESENT |
| 15b | Index test development (train/val/test) | §2.4 + §3.2/3.4 | ✓ PRESENT |
| 15c | Index test cutoffs (prespecified vs exploratory) | §2.5 (fixed 0.5 threshold) | ✓ PRESENT |
| 15d | End-user specification and expertise | §4 Discussion (intended users: sonographers and clinicians with limited thyroid ultrasound experience, second-opinion aid) | ● PRESENT |
| 16a | Reference standard detail | §2.5 | ✓ PRESENT |
| 16b | Reference standard rationale | §2.5 | ✓ PRESENT |
| 16c | Reference standard cutoffs | §2.5 | ✓ PRESENT |
| 17a | Blinding (index test readers) | Not applicable (automated model) | N/A — model has no human reader |
| 17b | Blinding (reference standard assessors) | §2.1 (reference labels from published datasets, independent of model) | ✓ PRESENT |
| 18 | Statistical methods for accuracy | §2.5 | ✓ PRESENT |
| 19 | Indeterminate results handling | §2.1 (25 Thy-Wise images excluded) | ✓ PRESENT |
| 20 | Missing data handling | §2.2 (no missing clinical features) + §2.1 (25 corrupted images excluded) | ✓ PRESENT |
| 21 | Variability analyses | §3.5 (multi-seed) | ✓ PRESENT |
| 22 | Sample size justification | Declarations (all available public data used) | ✓ PRESENT |
| 23 | Fairness assessment (methods) | §3.6 (exploratory fairness: sex/age/size/TI-RADS) | ✓ PRESENT |
| 24 | Flow diagram | Fig 1 | ✓ PRESENT |
| 25 | Baseline characteristics (train/val/test) | Table 1 + §3.1 | ✓ PRESENT |
| 26a | Severity distribution | Table 1 (malignant %) | ✓ PRESENT |
| 26b | Alternative diagnoses | Not applicable (binary) | N/A |
| 27 | Time interval (index vs reference) | §2.1 (reporting note: labels assigned at acquisition; no follow-up interval) | ● PRESENT |
| 28 | Test set representativeness | §4 (domain shift) | ✓ PRESENT |
| 29 | External evaluation differences | §3.3/3.4 | ✓ PRESENT |
| 30 | Cross-tabulation (index × reference) | Table 2/3 + Fig 4 | ✓ PRESENT |
| 31 | Accuracy estimates with CIs | Table 2/3 + text | ✓ PRESENT |
| 32 | Adverse events | Not applicable | N/A — non-invasive ultrasound |
| 33 | Study limitations | §4 Limitations | ✓ PRESENT |
| 34 | Clinical applicability | §4 Discussion | ✓ PRESENT |
| 35 | Ethical considerations and fairness | §3.6 (exploratory fairness) + Declarations (ethics) | ✓ PRESENT |
| 36 | Registration | Declarations (not registered) | ✓ PRESENT |
| 37 | Protocol | Declarations (no protocol prepared) | ✓ PRESENT |
| 38 | Funding | Declarations (no specific grant) | ✓ PRESENT |
| 39 | Commercial interests | Declarations (no competing interests) | ✓ PRESENT |
| 40a | Data and code availability | Declarations | ● PRESENT — Declarations; public GitHub repository |
| 40b | Audit and evaluation of outputs | Declarations (evaluation outputs stored in repo under paper/output/repro) | ● PRESENT |

**Summary**: 42 PRESENT, 0 PARTIAL, 0 MISSING, 2 N/A. Updated 2026-08-13: reporting notes for acquisition dates/participant series/time interval (§2.1); end-user specification (§4); evaluation-outputs statement (Declarations). Page numbers to be filled after final docx formatting.
