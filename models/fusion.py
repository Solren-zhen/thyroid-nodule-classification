"""
多模态融合模型
==============
Cross-Attention 融合白光 + 荧光 + 临床特征 → 转移概率输出。

融合架构:
  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
  │ White Emb (512) │   │ Fluo Emb (512)    │   │ Clinical (128)  │
  └───────┬─────────┘   └────────┬─────────┘   └───────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ↓
                     Cross-Attention Fusion
                      (Q: white+clinical
                       K,V: fluo)
                                 ↓
                          Fusion Embed
                                 ↓
                       MLP Head → [0,1]

荧光修正机制:
  - 训练时: 荧光 → "teacher" 信号，通过一致性损失约束白光预测
  - 推理时: 荧光可选，缺失时 null_embedding 替代
  - 模态丢弃训练确保白光-only 推理也准确
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import DualModalityEncoder


class ClinicalMLP(nn.Module):
    """
    临床特征编码器 — 将表格特征映射到嵌入空间。

    临床特征 (15维) 包括:
      - 淋巴结短径 (mm)
      - 淋巴结长径 (mm)
      - 长短径比
      - 边缘光滑度 (0/1)
      - 内部回声 (0/1/2)
      - 位置/分区编码 (1-14)
      - CT HU 均值
      - 患者年龄
      - 肿瘤标志物水平 (CEA/CA19-9)
      - ...等
    """

    def __init__(
        self,
        input_dim: int = 15,
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
        """x: (B, 15) → (B, 128)"""
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention 多模态融合模块。

    设计思想:
      白光特征 + 临床特征 作为 Query，荧光特征作为 Key/Value。
      这样荧光"指导"白光特征的解读——模拟临床中荧光标记修正肉眼判断的过程。
      荧光缺失时，Self-Attention 在白光特征上自修正。
    """

    def __init__(self, dim: int = 512, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Multi-head Cross-Attention 层
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Feed-Forward 层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim),
                nn.Dropout(0.1),
            )
            for _ in range(num_layers)
        ])

        # Layer Norm
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:       (B, dim) — 白光 + 临床特征的融合
            key_value:   (B, dim) — 荧光特征（或 null_embedding）

        Returns:
            fused: (B, dim)
        """
        # 添加序列维度 (B, 1, dim)
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)

        for attn, ffn, attn_norm, ffn_norm in zip(
            self.cross_attn_layers, self.ffn_layers,
            self.attn_norms, self.ffn_norms,
        ):
            # Cross-Attention: Q attends to KV
            attn_out, _ = attn(q, kv, kv)  # (B, 1, dim)
            q = attn_norm(q + attn_out)

            # Feed-Forward
            ffn_out = ffn(q)
            q = ffn_norm(q + ffn_out)

        return q.squeeze(1)  # (B, dim)


class GatedFusion(nn.Module):
    """
    门控融合 — 简单但有效的备选方案。

    学习一个门控权重 α ∈ [0,1]，动态决定白光和荧光的融合比例:
      fused = α * white + (1-α) * fluo
    当荧光缺失时 α → 1，完全依赖白光。
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, white_embeds: torch.Tensor, fluo_embeds: torch.Tensor
    ) -> torch.Tensor:
        """动态门控融合"""
        concat = torch.cat([white_embeds, fluo_embeds], dim=-1)  # (B, dim*2)
        alpha = self.gate(concat)  # (B, 1)
        fused = alpha * white_embeds + (1 - alpha) * fluo_embeds
        return fused, alpha


class MetastasisHead(nn.Module):
    """转移概率预测头 — 多层 MLP + Sigmoid"""

    def __init__(
        self,
        input_dim: int = 768,  # vis_dim(512) + clinical_dim(128) + fluo_dim 或 仅 vis_dim
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()
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


class LymphNodeFusionModel(nn.Module):
    """
    完整的淋巴结转移概率评估模型。

    输入:
      - white_rois:          (B, 3, 256, 256) 白光 ROI
      - fluorescence_rois:   (B, 1, 256, 256) 荧光 ROI (可选)
      - clinical_features:   (B, 15)           临床特征 (可选)
      - fluorescence_avail:  (B,)              bool mask

    输出:
      - metastasis_prob:  (B, 1)  转移概率 [0, 1]
      - aux_outputs:      dict    辅助输出（用于损失计算和可解释性）
    """

    def __init__(self, config=None):
        super().__init__()

        # 从配置提取参数（兼容字典和 dataclass）
        if config is None:
            from ..config import ModelConfig
            config = ModelConfig()

        # 视觉编码器
        self.encoder = DualModalityEncoder(
            backbone=getattr(config, "encoder_name", "efficientnetv2_s"),
            pretrained=getattr(config, "encoder_pretrained", True),
            vis_dim=getattr(config, "vis_dim", 512),
            fluorescence_channels=getattr(config, "fluorescence_channels", 1),
            fluorescence_mode=getattr(config, "fluorescence_encoder", "shared"),
        )

        self.vis_dim = getattr(config, "vis_dim", 512)

        # 临床特征编码器
        clinical_dim = getattr(config, "clinical_feature_dim", 15)
        clinical_out = getattr(config, "clinical_output_dim", 128)
        self.clinical_encoder = ClinicalMLP(
            input_dim=clinical_dim,
            hidden_dim=getattr(config, "clinical_hidden_dim", 128),
            output_dim=clinical_out,
        )
        self.clinical_out_dim = clinical_out

        # 融合模块
        fusion_method = getattr(config, "fusion_method", "cross_attention")
        self.fusion_method = fusion_method

        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(
                dim=self.vis_dim,
                num_heads=getattr(config, "cross_attention_heads", 4),
                num_layers=getattr(config, "cross_attention_layers", 2),
            )
            # 投影临床特征到视觉维度，用于构建 Query
            self.clinical_proj = nn.Linear(clinical_out, self.vis_dim)
        elif fusion_method == "gated":
            self.fusion = GatedFusion(dim=self.vis_dim)
        else:
            # concat 模式：无特殊融合层，直接拼接
            self.fusion = None

        # 预测头
        if fusion_method == "concat":
            head_input_dim = self.vis_dim * 2 + clinical_out
        else:
            head_input_dim = self.vis_dim + clinical_out

        self.head = MetastasisHead(
            input_dim=head_input_dim,
            hidden_dims=getattr(config, "head_hidden", [512, 256, 128]),
            dropout=getattr(config, "head_dropout", 0.3),
        )

    def forward(
        self,
        white_rois: torch.Tensor,
        fluorescence_rois: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
        fluorescence_available: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            white_rois: (B, 3, H, W)
            fluorescence_rois: (B, 1, H, W) or None
            clinical_features: (B, 15) or None
            fluorescence_available: (B,) bool or None
            return_aux: 是否返回辅助输出（注意力权重、门控值等）

        Returns:
            dict with:
                - logits:   (B, 1) 原始 logits
                - probs:    (B, 1) sigmoid 概率
                - white_embeds:  (B, vis_dim)  白光嵌入
                - fluo_embeds:   (B, vis_dim)  荧光嵌入
                - alpha:    (B, 1) 门控权重（仅 gated 模式）
        """
        B = white_rois.shape[0]
        device = white_rois.device

        # 1. 视觉编码
        white_embeds, fluo_embeds = self.encoder(
            white_rois, fluorescence_rois, fluorescence_available
        )

        # 2. 临床特征编码
        if clinical_features is not None:
            clinical_embeds = self.clinical_encoder(clinical_features)  # (B, 128)
        else:
            clinical_embeds = torch.zeros(B, self.clinical_out_dim, device=device)

        # 3. 融合
        aux = {}
        if self.fusion_method == "cross_attention":
            # 白光 + 临床 → Query; 荧光 → Key/Value
            clinical_proj = self.clinical_proj(clinical_embeds)  # (B, 128) → (B, vis_dim)
            query = white_embeds + clinical_proj
            fused_vis = self.fusion(query, fluo_embeds)  # (B, vis_dim)
            fused_all = torch.cat([fused_vis, clinical_embeds], dim=-1)

        elif self.fusion_method == "gated":
            fused_vis, alpha = self.fusion(white_embeds, fluo_embeds)
            fused_all = torch.cat([fused_vis, clinical_embeds], dim=-1)
            aux["alpha"] = alpha

        else:  # concat
            fused_all = torch.cat([white_embeds, fluo_embeds, clinical_embeds], dim=-1)

        # 4. 预测
        logits = self.head(fused_all)  # (B, 1)
        probs = torch.sigmoid(logits)

        outputs = {
            "logits": logits,
            "probs": probs,
            "white_embeds": white_embeds,
            "fluo_embeds": fluo_embeds,
        }
        outputs.update(aux)
        return outputs

    def predict(
        self,
        white_rois: torch.Tensor,
        fluorescence_rois: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
        fluorescence_available: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """简化的预测接口 — 仅返回概率"""
        return self.forward(
            white_rois, fluorescence_rois, clinical_features, fluorescence_available
        )["probs"]
