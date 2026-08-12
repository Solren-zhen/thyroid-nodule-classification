# Archive

This directory holds legacy / superseded artifacts that are **excluded from the
public repository** (everything except this file is git-ignored). They are kept
locally for provenance and are intentionally not part of the submitted
code/data package.

## Contents

- `legacy_protocols/` — artifacts from the pre-final protocol (pw0.35 /
  85-15 leaky split) that were superseded by the final leakage-free
  protocol (`pos_weight = 1.0`, patient-grouped 70/15/15 split):
  - `manifest_85_15_leaky.bak` — old TN5000 85/15 manifest (pre-fix split)
  - `best_joint_20260809_leaky.bak.pt` — leaky-era joint checkpoint
  - `best_joint_20260809.pt` — pre-fix joint checkpoint (superseded by
    `best_joint_20260810_fixed.pt`, retained under `checkpoints/`)
  - `fusion_seed42_pw0.35/` — seed-42 fusion model trained with the old
    class-weighted loss (pos_weight = 0.35)
  - `logs/train_fusion_seed*_pw035.log` — old-protocol training logs
  - `stale_eval_image_dir/` — evaluation JSONs that were produced while
    `checkpoints/thyroid/image/best.pt` transiently held joint / TN5000
    models; superseded by the canonical evals under `paper/output/repro/`
  - `thyroid_dataset.py.bak` — source backup superseded by `data/thyroid_dataset.py`

The manuscript and `REPRODUCIBILITY.md` refer only to the final-protocol
artifacts. Do not use the files in `legacy_protocols/` for new analyses.
