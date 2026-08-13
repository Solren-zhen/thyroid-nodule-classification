# External generalization and domain shift in thyroid ultrasound AI

[Paper]() · [Datasets]() · License: [MIT](LICENSE) · [Reproducibility](REPRODUCIBILITY.md)

Code accompanying the manuscript:

> **External generalization and domain shift in thyroid ultrasound AI: quantifying cross-dataset performance, multi-dataset training, and the incremental value of structured features**

This repository reproduces the experiments of the manuscript: an image-only
EfficientNetV2-S classifier trained on TN5000, external validation on TN3K and
Thy-Wise, joint multi-dataset training to quantify cross-dataset domain shift,
and a three-arm ablation (image-only / clinical-only / fusion) on ThyroidXL
fusing an image encoder with five structured clinical features (ACR TI-RADS
total score, nodule width, nodule height, age, sex).

## Key results (paper summary)

| Setting | AUC (nodule-level) |
|---|---|
| TN5000 internal test (image-only, single dataset) | 0.915 |
| TN3K external (official test, single dataset) | 0.729 |
| TN3K external (official test, joint training) | 0.813 (Δ +0.085) |
| Thy-Wise external (single dataset) | 0.608 |
| Thy-Wise external (joint training) | 0.710 (Δ +0.102) |
| ThyroidXL test — fusion | 0.947 |
| ThyroidXL test — image-only | 0.939 |
| ThyroidXL test — clinical-only | 0.814 |

See the manuscript (`paper/output/doc/`) for full details, tables, and figures.
Every reported number maps to an exact command and result file in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (canonical evaluation outputs are committed
under `paper/output/repro/`).

## Repository layout

```
config.py                  # experiment configuration
download_*.py              # dataset downloaders (HuggingFace / HF-mirror)
prepare_*.py               # build per-dataset manifest.csv (image paths + labels + clinical columns)
train_thyroid.py           # train / fine-tune / ablation (fusion|image|clinical)
eval_thyroid.py            # evaluate on val/test: AUC(95%CI), ACC, sens/spec, F1, ECE, DCA, confusion matrix
eval_thywise.py            # Thy-Wise nodule-level external evaluation (mean aggregation)
ensemble_thyroidxl.py      # multi-seed fusion ensemble evaluation
run_*.sh                   # end-to-end reproduction scripts (portable; set PYTHON=/path/to/python if needed)
tools/                     # data QC / TI-RADS annotation sampling helpers
paper/                     # manuscript source (markdown) + figure/docx generation scripts
```

## Installation

```
pip install -r requirements.txt
```

Requires Python 3.11+, PyTorch ≥ 2.0 with CUDA (tested on PyTorch 2.5.1 / CUDA 12.1 / RTX 3060 6 GB).

## Data preparation

All four datasets are public. Place each dataset under `data/thyroid/` and build
its manifest, e.g.:

```bash
# TN5000 (HF mirror: Johnyquest7/...)
python download_tn5000.py --dest data/thyroid/tn5000
python prepare_tn5000.py --data_root data/thyroid

# TN3K (HuggingFace: haifan-gong/TN3K)
#   官方 test（论文匹配口径，n=614）评估用 --split test；全量 3,493（Table 2 域漂移行）用 --split all
python prepare_tn3k.py --data_root data/thyroid/tn3k

# Thy-Wise (figshare, DOI 10.6084/m9.figshare.20417895, CC BY 4.0)
#   → prepare via prepare_thyroid_data.py or a Thy-Wise-specific manifest

# ThyroidXL (HuggingFace: hunglc007/ThyroidXL, gated access)
#   export HF_TOKEN  # read from ~/.huggingface/token; never commit tokens
python download_hf_dataset.py --repo hunglc007/ThyroidXL --dest data/thyroid/thyroidxl \
    --subdirs train/images train/labels test/images test/labels
python prepare_thyroidxl.py --root data/thyroid/thyroidxl

# Multi-dataset training manifest (TN5000 train/val + TN3K trainval; internal test kept separate)
python prepare_multi.py
```

Cross-dataset byte-level (MD5) duplicate checks (TN5000?TN3K and
Thy-Wise?(TN5000+TN3K), zero duplicates) are provided by
tools/tn5000_tn3k_dedup_check.py and tools/thywise_dedup_check.py, with
reports committed under paper/output/repro/.

> Data files are excluded from this repository (see `.gitignore`). To reproduce,
> download the data and re-run the prepare steps; each manifest stores absolute
> image paths resolved relative to the repository root at prepare time.

## Training

Single-dataset (image-only) and joint-training experiments:

```bash
# Joint training (TN5000 + TN3K trainval) — internal test held out (leakage-free)
python train_thyroid.py --data_root data/thyroid_multi --ablation image \
    --epochs 30 --batch_size 64 --seed 42 --pos_weight 1.0

# ThyroidXL three-arm ablation
python train_thyroid.py --data_root data/thyroid/thyroidxl --ablation fusion \
    --epochs 30 --batch_size 32 --clinical_columns tirads,width_mm,height_mm,age,gender --seed 42
python train_thyroid.py --data_root data/thyroid/thyroidxl --ablation image \
    --epochs 30 --batch_size 32 --clinical_columns tirads,width_mm,height_mm,age,gender --seed 42
python train_thyroid.py --data_root data/thyroid/thyroidxl --ablation clinical \
    --epochs 30 --batch_size 32 --clinical_columns tirads,width_mm,height_mm,age,gender --seed 42
```

Reproduction scripts run the full pipelines and skip already-finished steps:

```bash
bash run_joint_retrain.sh          # joint multi-dataset training (H1-fixed, no leakage)
bash run_thyroidxl_ablation.sh     # ThyroidXL image / clinical ablations
bash run_fusion_multiseed.sh       # fusion seeds 123 + 2024 (pos_weight=1.0)
```

Set `PYTHON=/path/to/python` to use a specific interpreter; otherwise `python` on PATH is used.

## Evaluation

```bash
# Internal / external image-level evaluation
python eval_thyroid.py --weights checkpoints/thyroid/fusion/best.pt --data_root data/thyroid/thyroidxl --split test --clinical_columns tirads,width_mm,height_mm,age,gender

# Thy-Wise nodule-level external evaluation
python eval_thywise.py --weights checkpoints/thyroid/joint/best.pt --data_root data/thyroid/thywise

# Multi-seed fusion ensemble (seeds 42/123/2024)
python paper/scripts/ensemble_fusion_pw1.py   # canonical 3-seed ensemble (AUC/CI/AUPRC)
```

Metrics reported: AUC with bootstrap 95% CI (n = 2000), sensitivity/specificity/
accuracy at a fixed 0.5 threshold, expected calibration error (ECE, 10
equal-width bins) and Brier score, decision curve analysis, confusion matrix,
and F1.

## Reporting

The manuscript follows the STARD 2015 and TRIPOD+AI reporting guidelines;
self-assessment checklists are in `paper/output/checklists/`.

## Citation

If you use this code or the results, please cite the manuscript (to be linked
once published):

```bibtex
@article{thyroid_fusion,
  title  = {External generalization and domain shift in thyroid ultrasound AI:
            quantifying cross-dataset performance, multi-dataset training, and the
            incremental value of structured features},
  author = {Zhen, Chaohui},
  journal = {TBD},
  year   = {2026}
}
```

> ⚠️ The author name above is a placeholder — update `author` to the actual
> author list before release.

## Data provenance and licences

- **TN5000** — Zhang et al., Scientific Data 2025; open access.
- **TN3K** — Gong et al., ISBI 2021 / CBM 2022; open access.
- **Thy-Wise** — Jin et al., International Journal of Cancer 2022; CC BY 4.0.
- **ThyroidXL** — Duong et al., MICCAI 2025; gated access for research use.
