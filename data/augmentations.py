"""
甲状腺超声结节数据增强
====================
针对甲状腺超声图像定制的增强策略（灰阶为主、结节方向相对固定、存在散斑噪声）:
  - 温和的几何变换（水平翻转 / 小角度旋转 / 轻微缩放平移）
  - 亮度 / 对比度微调
  - 少量高斯噪声（模拟超声散斑）
  - CLAHE 纹理对比度增强

训练用 get_thyroid_transforms()，验证/测试仅用归一化（get_thyroid_val_transforms()）。
"""
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


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
