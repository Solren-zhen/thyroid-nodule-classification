#!/usr/bin/env python3
"""Figure 1: 研究流程图（数据→切分→模型→评估）"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"


def box(ax, x, y, w, h, text, fc="#e8f0fe", ec="#1f77b4", fs=10):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                 fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#333333"})


def main():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # 数据来源
    box(ax, 0.3, 6.0, 2.4, 1.1, "4 public datasets\nTN5000 / TN3K\nThy-Wise / ThyroidXL", fc="#f0f4e8", ec="#2ca02c")
    arrow(ax, 2.7, 6.55, 3.6, 6.55)

    # 切分
    box(ax, 3.6, 6.0, 2.4, 1.1, "Patient-level split\n(train/val/test, seed 42)")
    arrow(ax, 6.0, 6.55, 6.9, 6.55)

    # 训练
    box(ax, 6.9, 6.0, 2.5, 1.1, "Model training\nEfficientNetV2-S +\nclinical MLP (3 arms)")
    arrow(ax, 9.4, 6.55, 10.3, 6.55)

    # 评估
    box(ax, 10.3, 6.0, 1.6, 1.1, "Internal\ntest")

    # 外部验证（下行）
    arrow(ax, 11.1, 6.0, 11.1, 4.9)
    box(ax, 9.6, 3.9, 3.0, 1.0, "External validation\nTN3K + Thy-Wise", fc="#fdecea", ec="#d62728")
    arrow(ax, 9.6, 4.4, 8.7, 4.4)

    # 融合消融
    box(ax, 5.9, 3.9, 2.8, 1.0, "Fusion ablation\nThyroidXL (739 nodules)", fc="#fdecea", ec="#d62728")
    arrow(ax, 5.9, 4.4, 5.0, 4.4)
    box(ax, 3.2, 3.9, 1.8, 1.0, "Metrics\nAUC/AUPRC/ECE/DCA")

    # 报告
    arrow(ax, 4.1, 3.9, 4.1, 2.6)
    box(ax, 1.8, 1.6, 4.6, 1.3, "Reporting\nSTARD 2015 + TRIPOD+AI", fc="#fff3d6", ec="#e6a817")

    # 图例说明
    ax.text(0.3, 0.4, "Domain shift / joint training experiments: TN5000 + TN3K + Thy-Wise",
            fontsize=9, color="#333333")
    ax.text(0.3, 0.05, "Fusion ablation: ThyroidXL (image vs clinical vs fusion)",
            fontsize=9, color="#333333")

    fig.tight_layout()
    out = FIG_DIR / "fig1_flow.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
