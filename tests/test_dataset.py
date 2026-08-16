"""数据集测试：mock 模式、真实模式缺 manifest 报错、按组切分。"""
import inspect

import albumentations as A
import pytest
import torch
from torch.utils.data import Dataset

from data.thyroid_dataset import ThyroidDataset


def _requires_albumentations_2x():
    """训练变换使用 GaussNoise(std_range=...)（albumentations 2.x API）。
    环境装的是 1.x 时跳过 train-split 相关断言（CI 使用 2.x 会完整执行）。"""
    sig = inspect.signature(A.GaussNoise.__init__)
    return "std_range" in sig.parameters



def test_mock_dataset_split_counts(tmp_path):
    if not _requires_albumentations_2x():
        pytest.skip("requires albumentations>=2.0.0 (GaussNoise std_range)")
    ds = ThyroidDataset(data_root=str(tmp_path), split="train", mock=True, seed=42)
    assert isinstance(ds, Dataset)
    assert len(ds) == 420  # 600 * 0.7 (patient-group split, seed 42)


def test_mock_dataset_all_split(tmp_path):
    ds = ThyroidDataset(data_root=str(tmp_path), split="all", mock=True, seed=42)
    assert len(ds) == 600


def test_mock_getitem_returns_expected_keys(tmp_path):
    if not _requires_albumentations_2x():
        pytest.skip("requires albumentations>=2.0.0 (GaussNoise std_range)")
    ds = ThyroidDataset(data_root=str(tmp_path), split="train", mock=True, seed=42,
                        image_size=64)
    item = ds[0]
    assert set(item.keys()) == {"image", "clinical", "label", "patient_id"}
    assert item["image"].shape[0] == 3
    assert item["label"].dtype == torch.float32
    assert item["clinical"].shape == (6,)
    assert isinstance(item["patient_id"], str)


def test_real_mode_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        ThyroidDataset(data_root=str(tmp_path), split="train", mock=False)


def test_split_by_group_no_patient_leakage(tmp_path):
    if not _requires_albumentations_2x():
        pytest.skip("requires albumentations>=2.0.0 (GaussNoise std_range)")
    ds = ThyroidDataset(data_root=str(tmp_path), split="train", mock=True, seed=42)
    all_ids = {ds.samples[i]["patient_id"] for i in range(len(ds))}
    # mock 模式每个患者 10 张图；train 420 张 = 42 个完整患者（420/10）
    assert len(all_ids) == 42
