# Cover Letter — BMC Medical Imaging

13 August 2026

Dear Editors,

We submit our manuscript entitled "External generalization and domain shift in thyroid ultrasound AI: quantifying cross-dataset performance, multi-dataset training, and the incremental value of structured features" for consideration in *BMC Medical Imaging*. The journal's focus on image-analysis methodology and its readership of medical-imaging researchers make it a natural fit for a study that quantifies how well deep-learning thyroid nodule classifiers generalise across institutions and devices.

Most published thyroid ultrasound classifiers report high accuracy on a single internal test set, and few are evaluated on images acquired at other institutions or with other devices. Using four fully public datasets (TN5000, TN3K, Thy-Wise, ThyroidXL), this study measures the resulting gap directly. An image-only model that reached an internal test AUC of 0.915 fell to 0.712 on the full external TN3K cohort (0.729 on the official test split) and to 0.608 on the completely unseen Thy-Wise cohort. Joint multi-dataset training recovered roughly half of the observed AUC loss on the cohort added to training (0.729 to 0.813 on the matched official test) and produced a comparable gain on Thy-Wise (0.608 to 0.710). An equal-sample-size control indicated that the benefit was not explained by sample size alone.

On ThyroidXL we compared image-only, clinical-only, full five-feature fusion and feature-specific fusion configurations. The incremental value of structured information was mainly attributable to the expert ACR TI-RADS score: adding TI-RADS to the image model improved the nodule-level AUC from 0.939 to 0.960 (paired ΔAUC +0.021, 95% CI 0.008–0.035, p = 0.003), whereas age, sex and nodule dimensions added little. The clinical-only model was markedly inferior (AUC 0.814) and poorly calibrated. Throughout, we report calibration (expected calibration error, Brier score), decision curves, subgroup analyses and error patterns alongside discrimination, using leakage-free patient-grouped splits and bootstrap confidence intervals, following STARD 2015 and TRIPOD+AI.

We believe the study offers your readers two practical things: public-data evidence on how much single-dataset AUCs overestimate real-world performance, and a reproducible protocol (public data, released code, completed STARD 2015 and TRIPOD+AI checklists) for evaluating domain shift and the incremental value of structured features in thyroid ultrasound AI. Code is available at https://github.com/ojdanajakir848-a11y/thyroid-nodule-classification.

We confirm that this manuscript is original, has not been published previously, and is not under consideration by any other journal. All authors have read and approved the manuscript and its submission. The authors declare no competing interests. Ethical approval was not required because the study used only de-identified images from publicly available, licensed datasets. We would be happy to suggest potential reviewers on request.

Yours sincerely,

Chaohui Zhen
Corresponding author: Chaohui Zhen, Wenzhou Medical University, chaohui0408.medical@outlook.com
