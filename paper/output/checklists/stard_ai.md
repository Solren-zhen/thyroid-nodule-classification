# STARD-AI Checklist (2025)

**Standards for Reporting of Diagnostic Accuracy Studies — Artificial Intelligence**

Manuscript title: Multimodal fusion of ultrasound imaging and ACR TI-RADS features for thyroid nodule malignancy classification: a multi-dataset validation study
Date: 2026-08-10
Status: Author self-assessment (fill page numbers after final formatting)

Reference: Sounderajah V, et al. The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. Nat Med. 2025;31:3283-3289. doi:10.1038/s41591-025-03953-8

| # | Item | Reported in Section | Status |
|---|------|--------------------|--------|
| 1 | Title — identify as AI diagnostic accuracy study, report accuracy measure | Title | ✓ PRESENT |
| 2 | Abstract — structured summary | Abstract | ✓ PRESENT |
| 3 | Scientific background, intended use of index test, workflow integration | §1 Introduction | ✓ PRESENT |
| 4 | Objectives / hypotheses | §1 Introduction | ✓ PRESENT |
| 5 | Study design (prospective vs retrospective) | §2.1 (retrospective, public datasets) | ✓ PRESENT |
| 6 | Ethics approval (or justification) | Declarations | ✗ MISSING — add IRB waiver statement |
| 7 | Eligibility criteria (participant + data level) | §2.1 | ◐ PARTIAL — add data-level criteria (image quality) |
| 8 | Participant identification basis | §2.1 (open repositories) | ✓ PRESENT |
| 9 | Setting and dates | §2.1 | ◐ PARTIAL — add acquisition dates |
| 10 | Participant series (consecutive/random/convenience) | §2.1 | ◐ PARTIAL — state consecutive vs selected |
| 11 | Data source (routinely collected / open-source) | §2.1 + Declarations | ✓ PRESENT |
| 12 | Dataset annotation (who, experience, how) | §2.1 | ◐ PARTIAL — add annotator experience for TN5000/TN3K labels |
| 13 | Data capture devices and software | §2.1 (device listed) + §2.4 (software versions) | ✓ PRESENT |
| 14 | Data acquisition and preprocessing | §2.4 | ✓ PRESENT |
| 15a | Index test detail | §2.3 Model architecture | ✓ PRESENT |
| 15b | Index test development (train/val/test) | §2.4 + §3.2/3.4 | ✓ PRESENT |
| 15c | Index test cutoffs (prespecified vs exploratory) | §2.5 (fixed 0.5 threshold) | ✓ PRESENT |
| 15d | End-user specification and expertise | §4 Discussion | ◐ PARTIAL — add explicit end-user |
| 16a | Reference standard detail | §2.5 | ✓ PRESENT |
| 16b | Reference standard rationale | §2.5 | ✓ PRESENT |
| 16c | Reference standard cutoffs | §2.5 | ✓ PRESENT |
| 17a | Blinding (index test readers) | Not applicable (automated model) | N/A — model has no human reader |
| 17b | Blinding (reference standard assessors) | Not reported | ◐ PARTIAL — state reference labels were independent |
| 18 | Statistical methods for accuracy | §2.5 | ✓ PRESENT |
| 19 | Indeterminate results handling | §2.1 (25 Thy-Wise images excluded) | ✓ PRESENT |
| 20 | Missing data handling | §2.2 (clinical zeros) | ✓ PRESENT |
| 21 | Variability analyses | §3.5 (multi-seed) | ✓ PRESENT |
| 22 | Sample size justification | Not reported | ✗ MISSING — add note (all available public data used) |
| 23 | Fairness assessment (methods) | Not reported | ✗ MISSING — add subgroup analysis as exploratory fairness |
| 24 | Flow diagram | Fig 1 | ✓ PRESENT |
| 25 | Baseline characteristics (train/val/test) | Table 1 + §3.1 | ✓ PRESENT |
| 26a | Severity distribution | Table 1 (malignant %) | ✓ PRESENT |
| 26b | Alternative diagnoses | Not applicable (binary) | N/A |
| 27 | Time interval (index vs reference) | §2.1 | ◐ PARTIAL — add label timing |
| 28 | Test set representativeness | §4 (domain shift) | ✓ PRESENT |
| 29 | External evaluation differences | §3.3/3.4 | ✓ PRESENT |
| 30 | Cross-tabulation (index × reference) | Table 2/3 + Fig 4 | ✓ PRESENT |
| 31 | Accuracy estimates with CIs | Table 2/3 + text | ✓ PRESENT |
| 32 | Adverse events | Not applicable | N/A — non-invasive ultrasound |
| 33 | Study limitations | §4 Limitations | ✓ PRESENT |
| 34 | Clinical applicability | §4 Discussion | ✓ PRESENT |
| 35 | Ethical considerations and fairness | Not reported | ✗ MISSING — add fairness/ethics discussion |
| 36 | Registration | Not registered | ◐ PARTIAL — add "not registered" statement |
| 37 | Protocol | Not reported | ✗ MISSING — add "no protocol prepared" |
| 38 | Funding | Not reported | ✗ MISSING — add funding statement |
| 39 | Commercial interests | Not reported | ✗ MISSING — add COI statement |
| 40a | Data and code availability | Declarations | ◐ PARTIAL — code URL to be inserted |
| 40b | Audit and evaluation of outputs | Not reported | ◐ PARTIAL — add outputs stored statement |

**Summary**: 25 PRESENT, 11 PARTIAL, 7 MISSING, 2 N/A. Items 6 (ethics), 22 (sample size), 38 (funding), 39 (commercial interests) were added to Declarations on 2026-08-10. Items 23/35 (fairness) remain to be addressed — subgroup analysis (§3.6) partially covers this; consider adding an explicit fairness statement. GitHub code URL still pending.
