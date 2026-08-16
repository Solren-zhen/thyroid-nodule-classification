"""
甲状腺融合分类器组件
====================
图像嵌入 + 临床特征 → 良恶性分类。

融合架构（ThyroidClassifier 使用）:
  ┌─────────────────┐   ┌──────────────────┐
  │ Image Emb (512) │   │ Clinical MLP (64) │
  └───────┬─────────┘   └────────┬─────────┘
          └──────────┬───────────┘
                     ↓
              concat (B, 576)
                     ↓
              MLP Head → logits
                     ↓
               Sigmoid → [0,1]

临床分支（ClinicalMLP）将 ACR TI-RADS 等结构化特征映射到嵌入空间；
分类头（MetastasisHead，沿用通用命名）输出良恶性 logits。
"""

import torch
from torch import nn


class ClinicalMLP(nn.Module):
    """
    临床特征编码器 — 将表格特征映射到嵌入空间。

    甲状腺场景使用 6 维临床特征（ACR TI-RADS）:
      - composition / echogenicity / shape / margin / echogenic_foci
      - size_mm（ThyroidXL 则为 tirads 总分 + 宽高 + 年龄 + 性别）
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6) → (B, 128)"""
        return self.net(x)


class MetastasisHead(nn.Module):
    """良恶性分类头 — 多层 MLP + Sigmoid"""

    def __init__(
        self,
        input_dim: int = 768,  # 图像嵌入 + 临床嵌入拼接后的维度
        hidden_dims: list | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) → logits: (B, 1)"""
        return self.net(x)
