"""
甲状腺结节超声数据集 — manifest 驱动 + 病例级切分
==================================================
期望数据目录结构:

  data_root/
  ├── images/                 # 超声结节图像（png/jpg，灰度或 RGB 均可）
  │   ├── case_001.png
  │   └── ...
  └── manifest.csv            # 必需：image_path, patient_id, label
                              # 可选临床列: composition, echogenicity, shape,
                              #            margin, echogenic_foci, size_mm
                              # （缺失的临床列自动补 0）

安全规则（沿用淋巴项目的教训）:
  - 真实模式（mock=False）下 manifest.csv 缺失直接报错，绝不静默 mock
  - 图片文件缺失抛异常
  - 默认按 patient_id 分组切分，防止同患者多图进入 train/val 造成泄漏
"""
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .augmentations import get_thyroid_transforms, get_thyroid_val_transforms

# 默认临床特征列（ACR TI-RADS；ThyroidXL 用 config.THYROIDXL_CLINICAL_COLUMNS）
CLINICAL_COLUMNS = [
    "composition", "echogenicity", "shape", "margin",
    "echogenic_foci", "size_mm",
]


class ThyroidDataset(Dataset):
    """
    甲状腺结节良恶性分类数据集。

    每个样本:
      - image:  (3, H, W) 超声图像（灰度自动复制为 3 通道）
      - clinical: (num_clinical_features,) TI-RADS 特征向量
      - label:  0=良性, 1=恶性
      - patient_id: 用于分组切分与复现
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 224,
        num_clinical_features: int = 6,
        clinical_columns: Optional[List[str]] = None,
        mock: bool = False,
        split_by_group: bool = True,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.num_clinical_features = num_clinical_features
        self.clinical_columns = clinical_columns or list(CLINICAL_COLUMNS)
        self.mock = mock
        self.split_by_group = split_by_group
        self.seed = seed

        manifest_path = self.data_root / "manifest.csv"
        if not manifest_path.exists():
            if mock:
                print("  ⚠ mock 模式：manifest.csv 未找到，使用模拟数据（仅开发测试）。")
                self.samples = self._create_mock_samples()
            else:
                raise FileNotFoundError(
                    f"manifest.csv 不存在: {manifest_path}\n"
                    "真实训练需要 manifest.csv（image_path, patient_id, label）。"
                    "如确需模拟测试，请显式 mock=True。"
                )
        else:
            self.samples = self._load_manifest(manifest_path)

        self._has_split_col = any((s.get("split") or "") for s in self.samples)
        self.samples = self._apply_split()

        if split == "train":
            self.transform = get_thyroid_transforms(image_size)
        else:
            self.transform = get_thyroid_val_transforms()

        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        print(f"  [{split}] {len(self.samples)} 样本 (恶性 {n_pos}, "
              f"阳性率 {n_pos / max(len(self.samples), 1) * 100:.1f}%)")

    # ── 加载 ─────────────────────────────────────────

    def _load_manifest(self, manifest_path: Path) -> List[Dict]:
        samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = (row.get("image_path") or "").strip()
                patient_id = (row.get("patient_id") or "").strip()
                label_raw = (row.get("label") or "").strip()
                if not image_path or not patient_id:
                    raise ValueError(f"manifest.csv 存在空 image_path/patient_id: {row}")

                img_path = Path(image_path)
                if not img_path.is_absolute():
                    img_path = self.data_root / img_path
                if not img_path.exists():
                    raise FileNotFoundError(f"图片不存在: {img_path}")

                try:
                    label = int(float(label_raw))
                except (TypeError, ValueError):
                    raise ValueError(f"manifest.csv 中 label 不是 0/1: {row}")

                clinical = np.zeros(self.num_clinical_features, dtype=np.float32)
                for i, col in enumerate(self.clinical_columns[:self.num_clinical_features]):
                    v = (row.get(col) or "").strip()
                    if v:
                        try:
                            clinical[i] = float(v)
                        except ValueError:
                            pass  # 空/非法值按 0 处理

                samples.append({
                    "patient_id": patient_id,
                    "image_path": str(img_path),
                    "label": label,
                    "clinical": clinical,
                    "split": (row.get("split") or "").strip().lower(),
                })
        if not samples:
            raise ValueError(f"manifest.csv 为空: {manifest_path}")
        return samples

    def _create_mock_samples(self) -> List[Dict]:
        """模拟数据：60 例患者 × 10 张图 = 600 张（35% 阳性率）"""
        rng = np.random.RandomState(self.seed)
        samples = []
        for p in range(60):
            patient_label = int(rng.random_sample() < 0.35)
            for i in range(10):
                # 10% 图像标签与患者标签不一致（模拟标注噪声）
                label = patient_label if rng.random_sample() > 0.1 else 1 - patient_label
                clinical = np.array([
                    rng.randint(1, 3),      # composition
                    rng.randint(1, 3),      # echogenicity
                    rng.randint(1, 3),      # shape
                    rng.randint(1, 4),      # margin
                    rng.randint(1, 5),      # echogenic foci
                    rng.uniform(5, 40),     # size_mm
                ], dtype=np.float32)
                samples.append({
                    "patient_id": f"mock_p{p:03d}",
                    "image_path": None,
                    "label": label,
                    "clinical": clinical,
                })
        return samples

    def _apply_split(self) -> List[Dict]:
        """按 7:1.5:1.5 切分（默认按 patient_id 分组；split="all" 返回全部样本）"""
        if self.split == "all":
            return list(self.samples)
        if self.split == "all":
            return list(self.samples)
        if self.split == "all":
            return list(self.samples)
        if getattr(self, "_has_split_col", False):
            keep = [s for s in self.samples if (s.get("split") or "") == self.split]
            if not keep:
                raise ValueError(f"manifest has no samples for split='{self.split}'")
            return keep
        n = len(self.samples)
        rng = np.random.RandomState(self.seed)

        if self.split_by_group:
            group_map: Dict[str, List[int]] = {}
            for i, s in enumerate(self.samples):
                group_map.setdefault(s["patient_id"], []).append(i)
            groups = list(group_map.values())
            rng.shuffle(groups)
            perm = [i for g in groups for i in g]
        else:
            perm = rng.permutation(n).tolist()

        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        if self.split == "train":
            keep = perm[:n_train]
        elif self.split == "val":
            keep = perm[n_train:n_train + n_val]
        else:
            keep = perm[n_train + n_val:]

        return [self.samples[i] for i in keep]

    # ── 访问 ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        if sample["image_path"]:
            # Unicode 安全读法：cv2.imread 对含中文的绝对路径会失败（Windows），
            # 用 np.fromfile + imdecode 绕过。
            img = cv2.imdecode(
                np.fromfile(sample["image_path"], dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if img is None:
                raise IOError(f"无法读取图片: {sample['image_path']}")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = self._generate_mock_image(sample["label"])

        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)

        if self.transform:
            transformed = self.transform(image=img)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(
                img.transpose(2, 0, 1)
            ).float() / 255.0

        return {
            "image": image_tensor,                          # (3, H, W)
            "clinical": torch.from_numpy(sample["clinical"]),  # (C,)
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "patient_id": sample["patient_id"],
        }

    def _generate_mock_image(self, label: int) -> np.ndarray:
        """向量化生成模拟超声结节图（仅 mock 模式）"""
        h = w = self.image_size
        rng = np.random.RandomState()

        # 散斑背景
        bg = rng.normal(90, 15, (h, w))
        bg = cv2.GaussianBlur(bg, (5, 5), 1.2)

        # 结节：暗区椭圆
        yy, xx = np.mgrid[:h, :w]
        cx, cy = w // 2, h // 2
        rx, ry = int(w * 0.22), int(h * 0.18)
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

        nodule = rng.normal(55, 12, (h, w))
        # 恶性：边缘不规则 + 更强回声点
        if label == 1:
            edge_noise = rng.random((h, w)) > 0.86
            mask = mask | (edge_noise & (
                ((xx - cx) / (rx * 1.5)) ** 2 + ((yy - cy) / (ry * 1.5)) ** 2 <= 1.0
            ))
            bright = rng.random((h, w)) > 0.97
            nodule[bright] += rng.normal(120, 20, (h, w))[bright]

        img = np.where(mask, nodule, bg)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)