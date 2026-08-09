"""
甲状腺结节良恶性分类 — 独立配置（开源数据路线）
"""
from dataclasses import dataclass, field


# 默认 ACR TI-RADS 临床特征列（TN5000/TN3K/Thy-Wise manifest 使用）
ACR_CLINICAL_COLUMNS = [
    "composition", "echogenicity", "shape", "margin",
    "echogenic_foci", "size_mm",
]

# ThyroidXL 临床特征列（TIRADS 总分 + 尺寸 + 人口学）
THYROIDXL_CLINICAL_COLUMNS = [
    "tirads", "width_mm", "height_mm", "age", "gender",
]


@dataclass
class ThyroidConfig:
    """甲状腺分类实验配置（开源数据路线）"""

    # 数据
    data_root: str = "./data/thyroid"
    image_size: int = 224
    clinical_feature_dim: int = 6   # ACR TI-RADS: composition, echogenicity, shape, margin, echogenic_foci, size_mm
    clinical_columns: list = field(default_factory=lambda: list(ACR_CLINICAL_COLUMNS))
    split_by_group: bool = True     # 按 patient_id 切分，防止同患者多图泄漏

    # 模型
    encoder_name: str = "efficientnetv2_s"
    encoder_pretrained: bool = True
    vis_dim: int = 512
    clinical_hidden_dim: int = 64
    clinical_output_dim: int = 64
    head_hidden: list = field(default_factory=lambda: [256, 128])
    head_dropout: float = 0.3

    # 训练
    epochs: int = 80
    batch_size: int = 16
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    scheduler: str = "cosine"       # cosine / plateau
    warmup_epochs: int = 3
    mixed_precision: bool = True
    num_workers: int = 0
    early_stopping_patience: int = 15
    seed: int = 42

    # 输出
    save_dir: str = "./checkpoints/thyroid"
    experiment_name: str = "thyroid_fusion"


def get_thyroid_config() -> ThyroidConfig:
    """获取甲状腺实验默认配置"""
    return ThyroidConfig()
