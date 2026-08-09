#!/usr/bin/env python3
"""
甲状腺结节良恶性分类 — 训练脚本（开源数据路线）
================================================
支持三种消融: fusion（图像+临床） / image（仅图像） / clinical（仅临床）。

用法:
    # mock 冒烟测试（无需数据、无需预训练权重）
    python train_thyroid.py --mode mock --epochs 2 --no_pretrained

    # 真实数据
    python train_thyroid.py --data_root ./data/thyroid --epochs 80

    # 消融实验
    python train_thyroid.py --data_root ./data/thyroid --ablation image
    python train_thyroid.py --data_root ./data/thyroid --ablation clinical
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent

from thyroid.config import get_thyroid_config
from thyroid.models.thyroid import ThyroidClassifier
from thyroid.data.thyroid_dataset import ThyroidDataset


# ── 工具 ─────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def bootstrap_auc(labels: np.ndarray, probs: np.ndarray,
                  n_boot: int = 2000, seed: int = 0):
    """AUC 的 bootstrap 95% 置信区间"""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(labels[idx], probs[idx]))
        except ValueError:
            continue
    if not aucs:
        return 0.0, 0.0, 0.0
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def ece_score(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """期望校准误差（Expected Calibration Error）"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(probs)) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    acc = float((preds == labels).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    auprc = float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    auc_mean, auc_lo, auc_hi = bootstrap_auc(labels, probs)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return {
        "acc": acc, "precision": float(p), "recall": float(r), "f1": float(f1),
        "auc": auc, "auc_ci": (auc_lo, auc_hi), "auprc": auprc,
        "ece": ece_score(labels, probs),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sen, "specificity": spe,
    }


# ── 训练循环 ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    pbar = tqdm(loader, desc="[train]", leave=False)
    for batch in pbar:
        labels = batch["label"].to(device).unsqueeze(1)
        clinical = batch["clinical"].to(device)

        def step(images=None):
            outputs = model(images=images, clinical=clinical)
            return criterion(outputs["logits"], labels)

        if use_amp:
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=True):
                if model.use_image:
                    images = batch["image"].to(device)
                    loss = step(images)
                else:
                    loss = step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if model.use_image:
                images = batch["image"].to(device)
                loss = step(images)
            else:
                loss = step()
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        clinical = batch["clinical"].to(device)
        if model.use_image:
            images = batch["image"].to(device)
            outputs = model(images=images, clinical=clinical)
        else:
            outputs = model(clinical=clinical)
        all_probs.append(outputs["probs"].cpu().numpy().flatten())
        all_labels.append(batch["label"].numpy().flatten())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    return compute_metrics(labels, probs)


def main():
    parser = argparse.ArgumentParser(description="甲状腺结节良恶性分类训练")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--mode", type=str, default="real", choices=["real", "mock"])
    parser.add_argument("--ablation", type=str, default="fusion",
                        choices=["fusion", "image", "clinical"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True,
                        help="加载 ImageNet 预训练权重（默认开启）")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = get_thyroid_config()
    if args.data_root: cfg.data_root = args.data_root
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.workers is not None: cfg.num_workers = args.workers
    cfg.encoder_pretrained = args.pretrained
    cfg.seed = args.seed

    set_seed(cfg.seed)
    device = get_device()
    print(f"设备: {device} | 消融: {args.ablation} | 预训练: {cfg.encoder_pretrained}")

    data_root = cfg.data_root
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)
    mock = args.mode == "mock"
    train_ds = ThyroidDataset(data_root, split="train", image_size=cfg.image_size,
                              num_clinical_features=cfg.clinical_feature_dim,
                              clinical_columns=cfg.clinical_columns,
                              mock=mock, split_by_group=cfg.split_by_group, seed=cfg.seed)
    val_ds = ThyroidDataset(data_root, split="val", image_size=cfg.image_size,
                            num_clinical_features=cfg.clinical_feature_dim,
                            clinical_columns=cfg.clinical_columns,
                            mock=mock, split_by_group=cfg.split_by_group, seed=cfg.seed)

    persistent = cfg.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers,
                            persistent_workers=persistent)

    model = ThyroidClassifier(
        encoder_name=cfg.encoder_name,
        pretrained=cfg.encoder_pretrained,
        vis_dim=cfg.vis_dim,
        clinical_feature_dim=cfg.clinical_feature_dim,
        clinical_hidden_dim=cfg.clinical_hidden_dim,
        clinical_output_dim=cfg.clinical_output_dim,
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
        use_image=args.ablation != "clinical",
        use_clinical=args.ablation != "image",
    ).to(device)

    # 分层学习率（主干更小）
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    optimizer = AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)

    if cfg.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu",
                                  enabled=cfg.mixed_precision)
    use_amp = cfg.mixed_precision and device.type == "cuda"

    save_dir = PROJECT_ROOT / "checkpoints" / "thyroid" / args.ablation
    save_dir.mkdir(parents=True, exist_ok=True)
    log_csv = save_dir / "metrics.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_auc", "auc_lo", "auc_hi",
                                "acc", "sensitivity", "specificity", "f1", "ece"])

    best_auc, best_epoch, patience = 0.0, 0, 0
    model_config = {
        "encoder_name": cfg.encoder_name,
        "pretrained": cfg.encoder_pretrained,
        "vis_dim": cfg.vis_dim,
        "clinical_feature_dim": cfg.clinical_feature_dim,
        "clinical_hidden_dim": cfg.clinical_hidden_dim,
        "clinical_output_dim": cfg.clinical_output_dim,
        "head_hidden": cfg.head_hidden,
        "head_dropout": cfg.head_dropout,
        "use_image": args.ablation != "clinical",
        "use_clinical": args.ablation != "image",
    }

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val = evaluate(model, val_loader, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val["auc"])
        else:
            scheduler.step()

        print(f"\nEpoch {epoch:3d} | loss {loss:.4f} | val AUC {val['auc']:.4f} "
              f"(95% CI {val['auc_ci'][0]:.4f}-{val['auc_ci'][1]:.4f}) | "
              f"ACC {val['acc']:.4f} | ECE {val['ece']:.4f}")

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{loss:.4f}", f"{val['auc']:.4f}",
                f"{val['auc_ci'][0]:.4f}", f"{val['auc_ci'][1]:.4f}",
                f"{val['acc']:.4f}", f"{val['sensitivity']:.4f}",
                f"{val['specificity']:.4f}", f"{val['f1']:.4f}", f"{val['ece']:.4f}",
            ])

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "model_config": model_config, "val_metrics": val}
        torch.save(ckpt, save_dir / "last.pt")
        if val["auc"] > best_auc:
            best_auc, best_epoch, patience = val["auc"], epoch, 0
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  [NEW BEST] AUC={best_auc:.4f} @ epoch {epoch}")
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch}, AUC {best_auc:.4f})")
                break

    print(f"\nDone: {args.ablation} best AUC = {best_auc:.4f} @ epoch {best_epoch}")
    print(f"Model: {save_dir / 'best.pt'} | Log: {log_csv}")


if __name__ == "__main__":
    main()