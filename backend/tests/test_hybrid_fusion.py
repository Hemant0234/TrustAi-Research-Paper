import pytest
import numpy as np
from app.xai.real_xai import hybrid_explanation_fusion, RealXAIEngine

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

def test_hybrid_explanation_fusion_numpy_lists():
    # Grad-CAM: [0.0, 10.0]
    gradcam = np.random.uniform(0.0, 10.0, (32, 32)).astype(np.float32)
    # SHAP: [-2.0, 5.0]
    shap = np.random.uniform(-2.0, 5.0, (32, 32)).astype(np.float32)
    # Integrated Gradients: [0.0, 0.05]
    ig = [[0.02] * 32 for _ in range(32)]

    unified = hybrid_explanation_fusion(gradcam, shap, ig, weights=(0.5, 0.3, 0.2))

    if has_torch:
        assert isinstance(unified, torch.Tensor)
        assert tuple(unified.shape) == (32, 32)
        assert float(unified.min()) >= 0.0
        assert float(unified.max()) <= 1.0 + 1e-6
    else:
        assert isinstance(unified, np.ndarray)
        assert unified.shape == (32, 32)
        assert float(unified.min()) >= 0.0
        assert float(unified.max()) <= 1.0 + 1e-6

@pytest.mark.skipif(not has_torch, reason="PyTorch not installed in this environment")
def test_hybrid_explanation_fusion_with_tensors():
    gradcam = torch.rand((32, 32)) * 10.0
    shap = torch.rand((32, 32)) * 7.0 - 2.0
    ig = torch.rand((32, 32)) * 0.05

    unified = hybrid_explanation_fusion(gradcam, shap, ig, weights=(0.5, 0.3, 0.2))

    assert isinstance(unified, torch.Tensor)
    assert unified.shape == (32, 32)
    assert float(unified.min()) >= 0.0
    assert float(unified.max()) <= 1.0 + 1e-6
    assert float(unified.max()) > 0.5

@pytest.mark.skipif(not has_torch, reason="PyTorch not installed in this environment")
def test_hybrid_explanation_fusion_with_mixed_types_and_resolutions():
    gradcam = np.random.uniform(0, 1, (16, 16)).astype(np.float32)
    shap = [[0.2] * 32 for _ in range(32)]
    ig = torch.ones((32, 32))

    unified = RealXAIEngine.hybrid_explanation_fusion(gradcam, shap, ig)

    assert isinstance(unified, torch.Tensor)
    assert unified.shape == (32, 32)
    assert float(unified.min()) >= 0.0
    assert float(unified.max()) <= 1.0

