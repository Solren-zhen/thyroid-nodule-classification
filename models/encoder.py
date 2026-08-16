"""
视觉编码器
==========
将超声图像映射为固定维度特征向量。
支持 EfficientNetV2（默认 efficientnetv2_s）/ EfficientNet-B4 主干，
移除分类头后接可选投影层，输出 (B, D) 视觉嵌入。

关键设计:
  - 支持冻结主干（两阶段训练 / 迁移学习）
  - 可调整输入通道数（灰度图复制为 3 通道）
"""

import torch
import torchvision.models as tv_models
from torch import nn


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
