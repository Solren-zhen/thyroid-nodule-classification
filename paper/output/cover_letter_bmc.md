# Cover Letter — BMC Medical Imaging

> 作者、通讯作者与 GitHub URL 已填写；日期在投稿当日更新。

13 August 2026

Dear Editors,

We submit our manuscript entitled "Multimodal fusion of ultrasound imaging and ACR TI-RADS features for thyroid nodule malignancy classification: an external multi-dataset validation study" for consideration in *BMC Medical Imaging*. The journal's emphasis on image-analysis methodology and its audience of medical-imaging researchers make it the natural venue for a study whose central contribution is a rigorous, reproducible quantification of how well deep-learning thyroid nodule classifiers generalise across institutions and devices.

The unresolved clinical question is not whether artificial intelligence can read thyroid ultrasound — dozens of single-centre reports have answered that with high internal accuracy — but whether that accuracy survives real-world deployment, where models trained on one institution's images are applied to patients scanned elsewhere. Most published models are never externally validated. Using four fully public datasets (TN5000, TN3K, Thy-Wise, ThyroidXL), this study quantifies cross-dataset domain shift directly: an image-only model achieving an internal test AUC of 0.915 drops to 0.712 on the full external TN3K cohort (0.729 on the matched official test split) — a 0.20 gap that single-cohort evaluation completely misses. We then test two practical mitigation strategies — joint multi-dataset training and fusion of structured clinical features — and report transparently what helps and what does not.

All experiments used leakage-free, patient-grouped splits (MD5-verified across cohorts), bootstrap confidence intervals, and full reporting of calibration (expected calibration error) and decision curves, following STARD 2015 and TRIPOD+AI. Joint training recovered roughly half of the domain-shift gap on both the cohort added to training (external AUC 0.729 to 0.813 on the matched official test) and a fully unseen, pathology-confirmed cohort (Thy-Wise nodule-level AUC 0.608 to 0.710). On ThyroidXL, fusing image features with five structured clinical features yielded a small but statistically significant gain over image-only classification (nodule-level AUC 0.947 vs 0.939; paired bootstrap ΔAUC +0.007, 95% CI 0.002–0.013, p = 0.012; three-seed ensemble AUC 0.951), with the benefit concentrated in low-to-intermediate risk subgroups. The negative results are reported as candidly as the positive ones: the clinical-only model was markedly inferior and poorly calibrated, and the fusion gain was modest on this single-device cohort.

The findings matter to your readership for three reasons. First, they provide concrete, public-data evidence for a message the field increasingly recognises: single-dataset AUCs substantially overestimate real-world performance, and external validation is a prerequisite for deployment. Second, they establish a reproducible protocol and open benchmark — public data, released code, and completed STARD 2015 / TRIPOD+AI checklists — that other groups can adopt. Third, by reporting calibration, decision curves, subgroup analyses and error patterns alongside discrimination, the study demonstrates the reporting standards now expected of imaging-AI papers. Code is available at https://github.com/ojdanajakir848-a11y/thyroid.

We confirm that this manuscript is original, has not been published previously, and is not under consideration by any other journal. All authors have read and approved the manuscript and its submission. The authors declare no competing interests. Ethical approval was not required because the study used only de-identified images from publicly available, licensed datasets. We would be happy to suggest potential reviewers on request.

Yours sincerely,

Chaohui Zhen (甄朝晖)
Corresponding author: Chaohui Zhen, Wenzhou Medical University, chaohui0408.medical@outlook.com
