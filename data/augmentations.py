"""
医学图像数据增强 — 腹腔镜淋巴结专用
=====================================
针对腹腔镜手术场景定制的增强策略:
  - 保持组织颜色真实性（限制色彩抖动幅度）
  - 模拟不同气腹压力下的组织形变
  - 模拟术中光照变化、烟雾、血污等
  - 荧光通道的独立增强
"""
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    roi_size: Tuple[int, int] = (256, 256),
    use_fluorescence: bool = True,
) -> A.Compose:
    """
    获取训练数据增强流水线。

    增强策略针对腹腔镜手术场景:
      - 小幅度旋转/翻转（淋巴结方向不固定）
      - 弹性变换模拟气腹组织形变
      - 亮度/对比度模拟术中光照变化
      - 限制色彩抖动（组织颜色有诊断意义）
      - 独立处理荧光通道

    Args:
        roi_size: 输出 ROI 尺寸
        use_fluorescence: 是否包含荧光通道增强
    """
    transforms = [
        # ── 空间变换 ──
        A.RandomResizedCrop(
            size=(roi_size[1], roi_size[0]),   # albumentations v2: (H, W)
            scale=(0.7, 1.0),                  # 保守缩放——淋巴结大小有意义
            ratio=(0.8, 1.25),                  # 保守长宽比
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(
            limit=(-30, 30),                    # 旋转不超过30度
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        # 弹性变形（模拟气腹组织形变）
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.2,
        ),

        # ── 像素级增强 — 模拟术中条件 ──
        # 光照变化（内窥镜光源变化）
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6,
        ),
        # 高斯噪声（摄像系统噪声）
        A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
        # 高斯模糊（运动/聚焦模糊）
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # 雾化（模拟镜头起雾/烟雾）
        A.RandomFog(
            fog_coef_range=(0.0, 0.1),
            alpha_coef=0.05,
            p=0.1,
        ),
        # 模拟血污遮挡 (v2: 使用分数)
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.0, 0.15),
            hole_width_range=(0.0, 0.15),
            fill=0,
            p=0.15,
        ),

        # ── 颜色增强（保守）──
        A.ColorJitter(
            brightness=(1.0, 1.0),       # v2: 乘法因子
            contrast=(1.0, 1.0),
            saturation=(0.9, 1.1),       # ±10% 饱和度
            hue=(-0.02, 0.02),           # 几乎不改变色调
            p=0.3,
        ),
        # CLAHE（增强组织纹理对比度）
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # ── 归一化 ──
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transforms(
    roi_size: Tuple[int, int] = (256, 256),
) -> A.Compose:
    """获取验证/测试数据变换（仅归一化 + 中心裁剪）"""
    return A.Compose([
        A.CenterCrop(height=roi_size[1], width=roi_size[0]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_fluorescence_transforms() -> A.Compose:
    """
    荧光通道专用增强（灰度图像）。
    荧光图像通常为单通道灰度，需要独立于 RGB 的增强策略。
    """
    return A.Compose([
        # 荧光信号强度变化（ICG 剂量/代谢差异）
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.2,      # 荧光对比度变化更大
            p=0.5,
        ),
        # 荧光噪声
        A.GaussNoise(std_range=(0.01, 0.02), p=0.3),
        # 荧光阈值增强
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        # 模拟荧光不均匀分布
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(mean=[0.5], std=[0.25]),
        ToTensorV2(),
    ])


# ── 自定义增强函数 — 可在数据集 __getitem__ 中直接调用 ──

def simulate_smoke(image: np.ndarray, severity: float = 0.1) -> np.ndarray:
    """模拟术中电刀烟雾效果"""
    h, w = image.shape[:2]
    smoke = np.random.normal(180, 30, (h, w)).astype(np.float32)
    smoke = cv2.GaussianBlur(smoke, (51, 51), 30)
    smoke = smoke / 255.0
    alpha = 1 - severity * smoke[..., np.newaxis]
    return (image.astype(np.float32) * alpha + 200 * (1 - alpha)).astype(np.uint8)


def simulate_specular_reflection(
    image: np.ndarray, num_spots: int = 2
) -> np.ndarray:
    """模拟腹腔镜镜头反光光斑"""
    h, w = image.shape[:2]
    result = image.copy()
    for _ in range(num_spots):
        cx, cy = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
        radius = np.random.randint(15, 50)
        intensity = np.random.uniform(0.3, 0.6)

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        spot = np.exp(-dist**2 / (2 * (radius/3)**2))
        spot = spot[..., np.newaxis]

        if image.ndim == 3:
            result = (result.astype(np.float32) * (1 - intensity * spot)
                      + 255 * intensity * spot).clip(0, 255).astype(np.uint8)

    return result

# ── 甲状腺超声专用增强 ──────────────────────────────

def get_thyroid_transforms(image_size: int = 224) -> A.Compose:
    """
    甲状腺超声结节分类训练增强。

    超声特点: 灰阶为主、结节方向相对固定、存在散斑噪声。
    因此只做温和的几何变换 + 亮度/对比度 + 少量噪声，不做大幅裁切。
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=0,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5,
        ),
        A.GaussNoise(std_range=(0.005, 0.02), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_thyroid_val_transforms() -> A.Compose:
    """甲状腺验证/测试变换（仅归一化）"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])