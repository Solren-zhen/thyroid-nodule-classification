"""
多模态视觉编码器
================
包含白光编码器、荧光编码器（共享或独立）。
支持 EfficientNetV2 / ViT 作为主干，输出固定维度的视觉嵌入。

关键设计:
  - 白光和荧光可以共享编码器（节省参数 + 特征对齐）或独立编码
  - 荧光不可用时，使用可学习的"空荧光嵌入"替代
  - 支持冻结主干进行两阶段训练
"""
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _get_efficientnet_v2(
    name: str, pretrained: bool, in_channels: int = 3
) -> nn.Module:
    """获取 EfficientNetV2 主干并调整输入通道"""
    model_fn = getattr(tv_models, name, None)
    if model_fn is None:
        raise ValueError(f"不支持的模型: {name}。可用: efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l")

    weights = "IMAGENET1K_V1" if pretrained else None
    model = model_fn(weights=weights)

    # 调整第一层输入通道
    if in_channels != 3:
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # 如果 in_channels == 1 且 pretrained，复制第一通道均值
        if in_channels == 1 and pretrained:
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.features[0][0] = new_conv

    # 移除分类头，保留特征提取器
    model.classifier = nn.Identity()
    return model


def _get_efficientnet_b4(pretrained: bool, in_channels: int = 3) -> nn.Module:
    """EfficientNet-B4（备选）"""
    weights = "IMAGENET1K_V1" if pretrained else None
    model = tv_models.efficientnet_b4(weights=weights)

    if in_channels != 3:
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if in_channels == 1 and pretrained:
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.features[0][0] = new_conv
    model.classifier = nn.Identity()
    return model


class ModalityEncoder(nn.Module):
    """
    单模态视觉编码器包装器。

    接受 (B, C, H, W) 图像批次，输出 (B, D) 特征向量。
    """

    def __init__(
        self,
        backbone: str = "efficientnetv2_s",
        pretrained: bool = True,
        in_channels: int = 3,
        output_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.output_dim = output_dim

        # 创建主干
        if backbone.startswith("efficientnetv2"):
            # efficientnetv2_s / efficientnetv2_m / efficientnetv2_l
            name = backbone.replace("efficientnetv2", "efficientnet_v2")
            self.backbone = _get_efficientnet_v2(name, pretrained, in_channels)
            base_dim = 1280  # EfficientNetV2-S 特征维度
        elif backbone.startswith("efficientnet_b"):
            self.backbone = _get_efficientnet_b4(pretrained, in_channels)
            base_dim = 1792
        else:
            raise ValueError(f"不支持的主干: {backbone}")

        # 投影层（将主干输出映射到统一维度）
        if output_dim != base_dim:
            self.projection = nn.Sequential(
                nn.Linear(base_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.projection = nn.Identity()

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, output_dim)"""
        features = self.backbone(x)  # (B, base_dim)
        return self.projection(features)

    def freeze_backbone(self):
        """冻结主干参数（仅训练投影层）"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻主干参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class DualModalityEncoder(nn.Module):
    """
    双模态编码器：白光 + 荧光。

    支持两种模式:
      - "shared":  白光和荧光共用同一个编码器（推荐，数据量小时更好）
      - "separate": 各自独立编码器

    荧光缺失时使用可学习的 null_embedding 占位。
    """

    def __init__(
        self,
        backbone: str = "efficientnetv2_s",
        pretrained: bool = True,
        vis_dim: int = 512,
        fluorescence_channels: int = 1,
        fluorescence_mode: Literal["shared", "separate"] = "shared",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.fluorescence_mode = fluorescence_mode

        # 白光编码器（RGB 3通道）
        self.white_encoder = ModalityEncoder(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=3,
            output_dim=vis_dim,
            freeze_backbone=freeze_backbone,
        )

        # 荧光编码器
        if fluorescence_mode == "separate":
            self.fluorescence_encoder = ModalityEncoder(
                backbone=backbone,
                pretrained=pretrained,
                in_channels=fluorescence_channels,
                output_dim=vis_dim,
                freeze_backbone=freeze_backbone,
            )
        else:
            # 共享模式：注册为同一对象
            self.fluorescence_encoder = self.white_encoder

        # 可学习的空荧光嵌入（当荧光数据不可用时使用）
        self.null_fluorescence_embed = nn.Parameter(
            torch.randn(1, vis_dim) * 0.02
        )

    def forward(
        self,
        white_rois: torch.Tensor,                      # (B, 3, H, W)
        fluorescence_rois: Optional[torch.Tensor] = None,  # (B, 1, H, W) or None
        fluorescence_available: Optional[torch.Tensor] = None,  # (B,) bool mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            white_rois: 白光 ROI 批次 (B, 3, H, W)
            fluorescence_rois: 荧光 ROI 批次 (B, 1, H, W) 或 None
            fluorescence_available: (B,) bool 张量标记哪些样本有荧光

        Returns:
            white_embeds:    (B, vis_dim)
            fluo_embeds:     (B, vis_dim) — 荧光缺失时填充 null_embedding
        """
        B = white_rois.shape[0]
        device = white_rois.device

        # 编码白光
        white_embeds = self.white_encoder(white_rois)  # (B, vis_dim)

        # 编码荧光（或使用空嵌入）
        if fluorescence_rois is not None:
            # 共享编码器且荧光为单通道 → 复制到3通道
            if self.fluorescence_mode == "shared" and fluorescence_rois.shape[1] == 1:
                fluorescence_rois = fluorescence_rois.repeat(1, 3, 1, 1)

            if fluorescence_available is None:
                # 假设所有样本都有荧光
                fluo_embeds = self.fluorescence_encoder(fluorescence_rois)
            else:
                # 混合批次：有荧光的编码，无荧光的填充 null
                fluo_embeds = torch.zeros(B, white_embeds.shape[-1], device=device)
                has_fluo = fluorescence_available.bool()

                if has_fluo.any():
                    valid_fluo = fluorescence_rois[has_fluo]
                    encoded_fluo = self.fluorescence_encoder(valid_fluo)
                    fluo_embeds[has_fluo] = encoded_fluo

                # 无荧光的样本使用可学习空嵌入
                if not has_fluo.all():
                    fluo_embeds[~has_fluo] = self.null_fluorescence_embed.expand(
                        (~has_fluo).sum(), -1
                    )
        else:
            # 完全没有荧光输入
            fluo_embeds = self.null_fluorescence_embed.expand(B, -1)

        return white_embeds, fluo_embeds

    def freeze_backbone(self):
        self.white_encoder.freeze_backbone()
        if self.fluorescence_mode == "separate":
            self.fluorescence_encoder.freeze_backbone()

    def unfreeze_backbone(self):
        self.white_encoder.unfreeze_backbone()
        if self.fluorescence_mode == "separate":
            self.fluorescence_encoder.unfreeze_backbone()
