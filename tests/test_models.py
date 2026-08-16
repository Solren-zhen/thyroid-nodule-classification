"""模型结构冒烟测试：ThyroidClassifier 三种消融配置的前向形状。"""
import torch

from models.thyroid import ThyroidClassifier


def _make_model(use_image: bool, use_clinical: bool) -> ThyroidClassifier:
    return ThyroidClassifier(
        encoder_name="efficientnetv2_s",
        pretrained=False,  # 不下载权重，纯结构测试
        vis_dim=512,
        clinical_feature_dim=6,
        use_image=use_image,
        use_clinical=use_clinical,
    ).eval()


def test_fusion_forward_shape():
    model = _make_model(True, True)
    images = torch.randn(2, 3, 224, 224)
    clinical = torch.randn(2, 6)
    out = model(images=images, clinical=clinical)
    assert out["logits"].shape == (2, 1)
    assert out["probs"].shape == (2, 1)
    assert bool(((out["probs"] > 0) & (out["probs"] < 1)).all())


def test_image_only_forward_shape():
    model = _make_model(True, False)
    out = model(images=torch.randn(2, 3, 224, 224))
    assert out["logits"].shape == (2, 1)


def test_clinical_only_forward_shape():
    model = _make_model(False, True)
    out = model(clinical=torch.randn(2, 6))
    assert out["logits"].shape == (2, 1)


def test_image_model_rejects_missing_images():
    model = _make_model(True, False)
    try:
        model(clinical=torch.randn(2, 6))
    except ValueError:
        return
    raise AssertionError("use_image=True 但未提供 images 时应抛 ValueError")


def test_clinical_model_rejects_missing_clinical():
    model = _make_model(False, True)
    try:
        model(images=torch.randn(2, 3, 224, 224))
    except ValueError:
        return
    raise AssertionError("use_clinical=True 但未提供 clinical 时应抛 ValueError")
