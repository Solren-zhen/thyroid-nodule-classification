# TRIPOD+AI Checklist (2024)

**TriPod+AI — reporting guideline for clinical prediction models using regression or machine learning methods**

Manuscript title: Multimodal fusion of ultrasound imaging and ACR TI-RADS features for thyroid nodule malignancy classification: a multi-dataset validation study
Date: 2026-08-10
Status: Author self-assessment (fill page numbers after final formatting)

Reference: Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378

| # | Item | Reported in Section | Status |
|---|------|--------------------|--------|
| 1 | Title — identify prediction model, target population, outcome | Title | ✓ PRESENT |
| 2 | Abstract | Abstract | ✓ PRESENT |
| 3a | Background — context and rationale | §1 Introduction | ✓ PRESENT |
| 3b | Target population and intended purpose | §1 + §4 | ✓ PRESENT |
| 3c | Health inequalities | Not reported | ✗ MISSING — optional, add if relevant |
| 4 | Objectives (development vs validation) | §1 | ✓ PRESENT |
| 5a | Data sources (development vs evaluation) | §2.1 | ✓ PRESENT |
| 5b | Dates of data collection | §2.1 | ◐ PARTIAL — add cohort dates |
| 6a | Study setting, centres | §2.1 (4 datasets) | ✓ PRESENT |
| 6b | Eligibility criteria | §2.1 | ✓ PRESENT |
| 6c | Treatments received | Not applicable | N/A — diagnostic study |
| 7 | Data preprocessing | §2.4 | ✓ PRESENT |
| 8a | Outcome definition | §2.5 | ✓ PRESENT |
| 8b | Outcome assessor qualifications | §2.5 (expert/cytological/pathological) | ✓ PRESENT |
| 8c | Blinding of outcome | Not reported | ◐ PARTIAL |
| 9a | Predictor choice rationale | §2.2 | ✓ PRESENT |
| 9b | Predictor definitions | §2.2 | ✓ PRESENT |
| 9c | Predictor assessor qualifications | §2.2 (expert TI-RADS) | ✓ PRESENT |
| 10 | Sample size justification | Not reported | ✗ MISSING — add all-data-used note |
| 11 | Missing data handling | §2.2 (clinical zeros) + §2.1 (25 images) | ✓ PRESENT |
| 12a | Data partitioning | §2.1 + §2.4 | ✓ PRESENT |
| 12b | Predictor handling (scaling) | §2.3 (MLP input) | ✓ PRESENT |
| 12c | Model type, building steps, internal validation | §2.3/2.4 | ✓ PRESENT |
| 12d | Heterogeneity across clusters | §3.3/3.4 (domain shift) | ✓ PRESENT |
| 12e | Performance measures and plots | §2.5 (AUC/ECE/DCA) + Figs | ✓ PRESENT |
| 12f | Model updating | Not applicable | N/A |
| 12g | How predictions calculated | §2.3 | ✓ PRESENT |
| 13 | Class imbalance methods | §2.4 (pos_weight) | ✓ PRESENT |
| 14 | Fairness approaches | §3.6 (subgroup) | ◐ PARTIAL — add explicit fairness framing |
| 15 | Model output and thresholds | §2.5 (0.5 threshold) | ✓ PRESENT |
| 16 | Development vs evaluation differences | §3.3/3.4 | ✓ PRESENT |
| 17 | Ethical approval | Not reported | ✗ MISSING — add IRB waiver |
| 18a | Funding | Not reported | ✗ MISSING |
| 18b | Conflicts of interest | Not reported | ✗ MISSING |
| 18c | Protocol | Not reported | ✗ MISSING — add "no protocol prepared" |
| 18d | Registration | Not reported | ◐ PARTIAL — add "not registered" |
| 18e | Data sharing | Declarations | ✓ PRESENT |
| 18f | Code sharing | Declarations (URL pending) | ◐ PARTIAL — insert GitHub URL |
| 19 | Patient and public involvement | Not reported | ✗ MISSING — add "no involvement" |
| 20a | Participant flow | Fig 1 + §2.1 | ✓ PRESENT |
| 20b | Characteristics by dataset | Table 1 | ✓ PRESENT |
| 20c | Development vs evaluation predictor distribution | §3.3 (domain shift) | ✓ PRESENT |
| 21 | Participants and events per analysis | §3.2-3.5 (n values) | ✓ PRESENT |
| 22 | Model specification (formula/code) | §2.3 + Code availability | ◐ PARTIAL — code URL pending |
| 23a | Model performance with CIs, subgroups | Table 2/3/4 | ✓ PRESENT |
| 23b | Heterogeneity across clusters | §3.3/3.4 | ✓ PRESENT |
| 24 | Model updating results | Not applicable | N/A |
| 25 | Interpretation incl. fairness | §4 Discussion | ◐ PARTIAL |
| 26 | Limitations | §4 Limitations | ✓ PRESENT |
| 27a | Poor-quality input handling | Not reported | ◐ PARTIAL — optional |
| 27b | User interaction and expertise | §4 | ◐ PARTIAL |
| 27c | Future research | §4 Future work | ✓ PRESENT |

**Summary**: 31 PRESENT, 10 PARTIAL, 6 MISSING, 3 N/A. Items 17 (ethics), 18a (funding), 18b (COI), 18c (protocol), 18d (registration), 19 (patient involvement), 10 (sample size) were added to Declarations on 2026-08-10. GitHub code URL and explicit fairness statement still pending.
