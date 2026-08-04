"""
甲状腺结节良恶性分类模型
========================
超声图像深度特征 + TI-RADS 临床特征融合 → 良恶性概率。

支持三种消融配置:
  - fusion       : 图像 + 临床特征融合（完整模型）
  - image        : 仅图像
  - clinical     : 仅临床特征

复用 encoder.ModalityEncoder（EfficientNet）与 fusion.ClinicalMLP / MetastasisHead。
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .encoder import ModalityEncoder
from .fusion import ClinicalMLP, MetastasisHead


class ThyroidClassifier(nn.Module):
    """甲状腺结节良恶性分类器。

    Args:
        encoder_name: 视觉主干（efficientnetv2_s / efficientnetv2_m / efficientnet_b4）
        pretrained: 是否加载 ImageNet 预训练权重
        vis_dim: 图像特征维度
        clinical_feature_dim: 临床特征输入维度（默认 6 = ACR TI-RADS 5 项 + 大小）
        clinical_hidden_dim: 临床 MLP 隐藏维度
        clinical_output_dim: 临床嵌入维度
        head_hidden: 预测头隐藏层
        head_dropout: 预测头 dropout
        use_image: 是否使用图像分支
        use_clinical: 是否使用临床分支
    """

    def __init__(
        self,
        encoder_name: str = "efficientnetv2_s",
        pretrained: bool = True,
        vis_dim: int = 512,
        clinical_feature_dim: int = 6,
        clinical_hidden_dim: int = 64,
        clinical_output_dim: int = 64,
        head_hidden: List[int] = [256, 128],
        head_dropout: float = 0.3,
        use_image: bool = True,
        use_clinical: bool = True,
    ):
        super().__init__()
        self.use_image = use_image
        self.use_clinical = use_clinical
        self.vis_dim = vis_dim
        self.clinical_out_dim = clinical_output_dim

        if use_image:
            self.encoder = ModalityEncoder(
                backbone=encoder_name,
                pretrained=pretrained,
                in_channels=3,
                output_dim=vis_dim,
            )
        if use_clinical:
            self.clinical_encoder = ClinicalMLP(
                input_dim=clinical_feature_dim,
                hidden_dim=clinical_hidden_dim,
                output_dim=clinical_output_dim,
                dropout=0.2,
            )

        head_in = 0
        if use_image:
            head_in += vis_dim
        if use_clinical:
            head_in += clinical_output_dim
        if head_in == 0:
            raise ValueError("use_image 与 use_clinical 至少一个必须为 True")

        self.head = MetastasisHead(
            input_dim=head_in,
            hidden_dims=list(head_hidden),
            dropout=head_dropout,
        )

    def forward(
        self,
        images: Optional[torch.Tensor] = None,      # (B, 3, H, W)
        clinical: Optional[torch.Tensor] = None,     # (B, clinical_feature_dim)
    ) -> dict:
        feats = []
        if self.use_image:
            if images is None:
                raise ValueError("use_image=True 但未提供 images")
            feats.append(self.encoder(images))       # (B, vis_dim)
        if self.use_clinical:
            if clinical is None:
                raise ValueError("use_clinical=True 但未提供 clinical")
            feats.append(self.clinical_encoder(clinical))  # (B, clinical_out)

        x = torch.cat(feats, dim=-1)
        logits = self.head(x)
        return {"logits": logits, "probs": torch.sigmoid(logits)}