# TRIPOD+AI Checklist (2024)

**TriPod+AI — reporting guideline for clinical prediction models using regression or machine learning methods**

Manuscript title: External generalization and domain shift in thyroid ultrasound AI: quantifying cross-dataset performance, multi-dataset training, and the incremental value of structured features
Date: 2026-08-12
Status: Author self-assessment (fill page numbers after final formatting)

Reference: Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378

| # | Item | Reported in Section | Status |
|---|------|--------------------|--------|
| 1 | Title — identify prediction model, target population, outcome | Title | ✓ PRESENT |
| 2 | Abstract | Abstract | ✓ PRESENT |
| 3a | Background — context and rationale | §1 Introduction | ✓ PRESENT |
| 3b | Target population and intended purpose | §1 + §4 | ✓ PRESENT |
| 3c | Health inequalities | Not reported | ◐ PARTIAL — optional; fairness considered in §3.6 |
| 4 | Objectives (development vs validation) | §1 | ✓ PRESENT |
| 5a | Data sources (development vs evaluation) | §2.1 | ✓ PRESENT |
| 5b | Dates of data collection | §2.1 (institutions & devices); dataset publication years in refs | ◐ PARTIAL — acquisition dates not reported in public data |
| 6a | Study setting, centres | §2.1 (4 datasets) | ✓ PRESENT |
| 6b | Eligibility criteria | §2.1 | ✓ PRESENT |
| 6c | Treatments received | Not applicable | N/A — diagnostic study |
| 7 | Data preprocessing | §2.4 | ✓ PRESENT |
| 8a | Outcome definition | §2.5 | ✓ PRESENT |
| 8b | Outcome assessor qualifications | §2.5 (expert/cytological/pathological) | ✓ PRESENT |
| 8c | Blinding of outcome | §2.1 (reference labels from published datasets, independent of model) | ✓ PRESENT |
| 9a | Predictor choice rationale | §2.2 | ✓ PRESENT |
| 9b | Predictor definitions | §2.2 | ✓ PRESENT |
| 9c | Predictor assessor qualifications | §2.2 (expert TI-RADS) | ✓ PRESENT |
| 10 | Sample size justification | Declarations (all available public data used) | ✓ PRESENT |
| 11 | Missing data handling | §2.2 (no missing clinical features) + §2.1 (25 corrupted images excluded) | ✓ PRESENT |
| 12a | Data partitioning | §2.1 + §2.4 | ✓ PRESENT |
| 12b | Predictor handling (scaling) | §2.3 (MLP input) | ✓ PRESENT |
| 12c | Model type, building steps, internal validation | §2.3/2.4 | ✓ PRESENT |
| 12d | Heterogeneity across clusters | §3.3/3.4 (domain shift) | ✓ PRESENT |
| 12e | Performance measures and plots | §2.5 (AUC/ECE/DCA) + Figs | ✓ PRESENT |
| 12f | Model updating | Not applicable | N/A |
| 12g | How predictions calculated | §2.3 | ✓ PRESENT |
| 13 | Class imbalance methods | §2.4 (pos_weight) | ✓ PRESENT |
| 14 | Fairness approaches | §3.6 (exploratory fairness: sex/age/size/TI-RADS) | ✓ PRESENT |
| 15 | Model output and thresholds | §2.5 (0.5 threshold) | ✓ PRESENT |
| 16 | Development vs evaluation differences | §3.3/3.4 | ✓ PRESENT |
| 17 | Ethical approval | Declarations (de-identified public data; no IRB required) | ✓ PRESENT |
| 18a | Funding | Declarations (no specific grant) | ✓ PRESENT |
| 18b | Conflicts of interest | Declarations (no competing interests) | ✓ PRESENT |
| 18c | Protocol | Declarations (no protocol prepared) | ✓ PRESENT |
| 18d | Registration | Declarations (not registered) | ✓ PRESENT |
| 18e | Data sharing | Declarations | ✓ PRESENT |
| 18f | Code sharing | Declarations | ● PRESENT — Declarations; public GitHub repository |
| 19 | Patient and public involvement | Declarations (no patients/public involved) | ✓ PRESENT |
| 20a | Participant flow | Fig 1 + §2.1 | ✓ PRESENT |
| 20b | Characteristics by dataset | Table 1 | ✓ PRESENT |
| 20c | Development vs evaluation predictor distribution | §3.3 (domain shift) | ✓ PRESENT |
| 21 | Participants and events per analysis | §3.2-3.5 (n values) | ✓ PRESENT |
| 22 | Model specification (formula/code) | §2.3 + Code availability | ● PRESENT — Declarations; public GitHub repository |
| 23a | Model performance with CIs, subgroups | Table 2/3/4 | ✓ PRESENT |
| 23b | Heterogeneity across clusters | §3.3/3.4 | ✓ PRESENT |
| 24 | Model updating results | Not applicable | N/A |
| 25 | Interpretation incl. fairness | §3.6 + §4 Discussion | ✓ PRESENT |
| 26 | Limitations | §4 Limitations | ✓ PRESENT |
| 27a | Poor-quality input handling | Not reported | ◐ PARTIAL — optional |
| 27b | User interaction and expertise | §4 | ◐ PARTIAL |
| 27c | Future research | §4 Future work | ✓ PRESENT |

**Summary**: 42 PRESENT, 4 PARTIAL, 0 MISSING, 3 N/A. Updated 2026-08-12: ethics/funding/COI/registration/protocol/patient involvement/sample size in Declarations; fairness approaches in §3.6 (explicit exploratory-fairness statement). Remaining PARTIAL: health inequalities (3c), acquisition dates (5b), code URL (18f, pending GitHub), model-specification URL (22), poor-quality input handling (27a, optional), user interaction/expertise (27b). Page numbers to be filled after final docx formatting.
