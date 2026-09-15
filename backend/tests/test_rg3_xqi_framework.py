import pytest
import numpy as np
from app.quality.real_xqi import RealXQIEngine, RealXQIEvaluation

def test_rg3_evaluate_localization_bbox():
    # Saliency map with concentrated activation in bottom-right quadrant
    saliency = [[0.1] * 32 for _ in range(32)]
    for r in range(16, 28):
        for c in range(16, 28):
            saliency[r][c] = 0.95

    # Target bbox covering bottom-right quadrant [ymin, xmin, ymax, xmax]
    bbox = [0.50, 0.50, 0.90, 0.90]
    loc_score = RealXQIEngine.evaluate_localization(saliency, bbox)

    assert loc_score is not None
    assert loc_score >= 80.0

    # Divergent bbox in top-left quadrant [0.05, 0.05, 0.35, 0.35]
    divergent_bbox = [0.05, 0.05, 0.35, 0.35]
    divergent_score = RealXQIEngine.evaluate_localization(saliency, divergent_bbox)
    assert divergent_score is not None
    assert divergent_score < loc_score

    # When ground truth is unavailable, return None
    assert RealXQIEngine.evaluate_localization(saliency, None) is None

def test_rg3_evaluate_robustness():
    base_sal = [[float(r + c) / 64.0 for c in range(32)] for r in range(32)]
    # Slightly perturbed saliency
    pert_sal = [[base_sal[r][c] + 0.02 * (r % 2) for c in range(32)] for r in range(32)]

    score = RealXQIEngine.evaluate_robustness(base_sal, pert_sal)
    assert 90.0 <= score <= 100.0

def test_rg3_evaluate_stability():
    base_sal = [[0.5] * 32 for _ in range(32)]
    run_1 = [[0.5] * 32 for _ in range(32)]
    run_2 = [[0.52] * 32 for _ in range(32)]

    score = RealXQIEngine.evaluate_stability(base_sal, [run_1, run_2])
    assert score >= 80.0

def test_rg3_evaluate_human_agreement():
    # 5.0 out of 5.0 on Likert scale should be 100.0%
    assert RealXQIEngine.evaluate_human_agreement(5.0) == 100.0
    # 1.0 out of 5.0 should be 0.0%
    assert RealXQIEngine.evaluate_human_agreement(1.0) == 0.0
    # Average of [4.0, 5.0] = 4.5 -> ((4.5 - 1)/4)*100 = 87.5%
    assert RealXQIEngine.evaluate_human_agreement([4.0, 5.0]) == 87.5
    # None when uncollected
    assert RealXQIEngine.evaluate_human_agreement(None) is None

def test_rg3_evaluate_consistency():
    map_a = [[float(r) / 32.0 for _ in range(32)] for r in range(32)]
    map_b = [[float(r) / 32.0 + 0.01 for _ in range(32)] for r in range(32)]

    score = RealXQIEngine.evaluate_consistency({"Grad-CAM++": map_a, "SHAP": map_b})
    assert score >= 90.0

def test_rg3_complete_xqi_dynamic_renormalization():
    # Test with ALL 7 dimensions provided
    full_eval = RealXQIEngine.evaluate_complete_xqi(
        faithfulness=88.0,
        robustness=85.0,
        stability=83.0,
        consistency=86.0,
        uncertainty_score=0.12,
        localization=92.0,
        human_agreement=87.5,
        entropy=0.20,
        mc_variance=0.012
    )

    assert isinstance(full_eval, RealXQIEvaluation)
    assert full_eval.overall_xqi >= 80.0
    assert full_eval.status == "HIGH QUALITY — RESEARCH THRESHOLD"
    assert full_eval.localization == 92.0
    assert full_eval.human_agreement == 87.5
    assert sum(full_eval.weights_used.values()) == pytest.approx(1.0, 0.01)

    # Test with MISSING localization and human_agreement (None)
    sparse_eval = RealXQIEngine.evaluate_complete_xqi(
        faithfulness=85.0,
        robustness=82.0,
        stability=80.0,
        consistency=84.0,
        uncertainty_score=0.15,
        localization=None,
        human_agreement=None,
        entropy=0.22,
        mc_variance=0.015
    )

    assert sparse_eval.localization is None
    assert sparse_eval.human_agreement is None
    assert "localization" not in sparse_eval.weights_used
    assert "human_agreement" not in sparse_eval.weights_used
    assert sum(sparse_eval.weights_used.values()) == pytest.approx(1.0, 0.01)
    assert sparse_eval.mathematical_formulation is not None
