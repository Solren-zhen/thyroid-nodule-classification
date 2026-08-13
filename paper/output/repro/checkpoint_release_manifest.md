# Final-protocol checkpoints (release manifest)

These are the **final-protocol** model weights behind every number in the
manuscript (leakage-free patient-grouped 70/15/15 split, `pos_weight = 1.0`,
EfficientNetV2-S encoder). They are git-ignored in the code repository and are
provided here so reviewers can reproduce the paper without retraining.

- Generated: 2026-08 (seed-42 anchors + seed-robustness runs 123/2024)
- Framework: PyTorch 2.5.1 (CUDA 12.1), Python 3.11
- Place these files back under `checkpoints/thyroid/` (same relative paths) and run
  the evaluation commands in `REPRODUCIBILITY.md`.
- Expected outputs: the committed JSON files under `paper/output/repro/`
  (Table 2/3, S1a-c, S2, S3, ensemble CI, paired test).

## Quick self-check after download

| Check | Expected (paper) |
|---|---|
| TN5000-only on TN5000 internal test | AUC 0.915 (0.893-0.936) |
| Joint model on TN3K official test | AUC 0.813 (0.779-0.846) |
| Fusion on ThyroidXL test (nodule-level mean) | AUC 0.947 (0.932-0.960) |

## SHA-256 checksums

### A. Track A checkpoints (trained on open-access TN5000 / TN3K)

| File | Role | Size (bytes) | SHA-256 |
|---|---|---|---|
| `single_tn3k_eval/best.pt` | TN5000-only, seed 42 | 84931890 | `DE9E59A77E25B19101DC68E4ED61B35260CFE9FE5E7120E56107829771B5F95C` |
| `_pre_seedretrain_backup/best_joint_seed42_fixed.pt` | Joint (TN5000 + TN3K train), seed 42 | 84931954 | `73D75D7061BA4515BAA00B3A1D5ECB63366DA5A35DD8496F4A04EF8079F6D938` |
| `tn5000_seed123/best.pt` | TN5000-only, seed 123 | 84931954 | `C8838B6E6C43131DF21467683ED16183A78BF292A8D329610BEEF51E55CCE509` |
| `tn5000_seed2024/best.pt` | TN5000-only, seed 2024 | 84931954 | `0C6917B5E65018B2724FA2DAB310B15DCD21783D35209E71791F3EA0C874BC0B` |
| `joint_seed123/best.pt` | Joint, seed 123 | 84931954 | `09F423A7D5561131AB4F3561F4E804917F6DE964DC9064B5732407A488B053F7` |
| `joint_seed2024/best.pt` | Joint, seed 2024 | 84931954 | `2391DC3B452E8C97090529AAB7BEFDEE54D36D2637F4266BE481B013A71A2771` |

### B. ThyroidXL-derived checkpoints (gated dataset)

| File | Role | Size (bytes) | SHA-256 |
|---|---|---|---|
| `fusion/best.pt` | Fusion (image + 5 clinical features), seed 42 | 85041038 | `BCB9491FC5CB2C623F50E97E4977A0B5F751C8F5DB62FF36AD99ADD15883E1B2` |
| `image/best.pt` | Image-only (ThyroidXL), seed 42 | 84931954 | `7A59794CF8812F3A8CF500B27BD4AAB1D684BF5CA2665B58E9EA5F482A4D0E9B` |
| `clinical/best.pt` | Clinical-only, seed 42 | 254128 | `7605220F0080E33871AA18EA2EF5EE55A87E586E60F058A7161112D26E3E91D9` |
| `fusion_seed123_pw1.0/best.pt` | Fusion, seed 123 | 85041038 | `A97D6A008D4451573392CECAD074AB7E8191F9F3D72DF9504AF9AB101BF00D11` |
| `fusion_seed2024_pw1.0/best.pt` | Fusion, seed 2024 | 85041038 | `B1166BE86080E982A24CEEE934E8EE9BF211610F61C19208D55BB299CDA92317` |

## License / redistribution notice

- Track A checkpoints are trained on TN5000 (open access) and TN3K (open access);
  redistribution follows the dataset terms and this project's MIT code license.
- **ThyroidXL is a gated research-use dataset.** Before publicly releasing the
  ThyroidXL-derived checkpoints (Section B), confirm that its licence permits
  redistribution of trained models. If not, release only Section A publicly and
  make Section B available on request / under the same research-use terms.
- No patient-level data are included; checkpoints contain learned weights only.

