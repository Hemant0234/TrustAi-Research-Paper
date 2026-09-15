import pytest
from app.reliability.reliability_engine import ExplanationReliabilityEngine
from app.schemas.cases import UncertaintyResult, XQIDimensions, FusionResult
from app.services.synthetic_data import SyntheticCaseLibrary

def test_rg4_uncertainty_penalty_calculation():
    # Low uncertainty
    pen_low = ExplanationReliabilityEngine.calculate_uncertainty_penalty(
        entropy=0.20,
        mc_variance=0.010,
        composite_score=0.12
    )
    assert pen_low["total_penalty"] < 5.0
    assert pen_low["uncertainty_index"] < 0.25

    # High uncertainty (elevated entropy + high MC variance)
    pen_high = ExplanationReliabilityEngine.calculate_uncertainty_penalty(
        entropy=0.65,
        mc_variance=0.085,
        composite_score=0.60
    )
    assert pen_high["total_penalty"] > 25.0
    assert pen_high["uncertainty_index"] > 0.65
    assert pen_high["mc_variance_penalty"] > 10.0
    assert pen_high["entropy_penalty"] > 10.0

def test_rg4_reliable_case_preserves_high_trust():
    case_tx2048 = SyntheticCaseLibrary.get_case_tx2048()
    assert case_tx2048.uncertainty.level == "low"
    assert case_tx2048.reliability.score >= 85.0
    assert case_tx2048.reliability.level == "RELIABLE"
    assert case_tx2048.reliability.should_trust_explanation is True
    # Penalty is minimal
    assert case_tx2048.reliability.uncertainty_penalty < 10.0

def test_rg4_killer_demo_case_tx2047_penalized_below_trust_threshold():
    case_tx2047 = SyntheticCaseLibrary.get_case_tx2047()
    # High nominal confidence (93.2%)
    assert case_tx2047.prediction.probability >= 0.90
    # But high uncertainty
    assert case_tx2047.uncertainty.level in ["high", "very_high"]
    assert case_tx2047.uncertainty.monte_carlo_variance > 0.05
    # Must be penalized into REVIEW REQUIRED
    assert case_tx2047.reliability.score < 60.0
    assert case_tx2047.reliability.level == "REVIEW REQUIRED"
    assert case_tx2047.reliability.should_trust_explanation is False
    assert any("MC variance" in c or "σ²_MC" in c for c in case_tx2047.reliability.evidence_concerns)

def test_rg4_high_xqi_with_high_uncertainty_is_penalized():
    """
    CORE RESEARCH HYPOTHESIS TEST:
    Even if an explanation exhibits high XQI (88.0) and high consensus (90%),
    elevated MC variance (0.075) and high predictive entropy (0.70) must mathematically
    penalize the reliability score below the trust threshold (<60.0).
    """
    xqi = XQIDimensions(
        overall=88.0,
        faithfulness=90.0,
        localization=92.0,
        robustness=85.0,
        stability=88.0,
        consistency=90.0,
        human_agreement=85.0,
        uncertainty_alignment=40.0,
        weights={"faithfulness": 0.25, "localization": 0.20, "robustness": 0.15, "stability": 0.15, "consistency": 0.10, "human_agreement": 0.10, "uncertainty_alignment": 0.05},
        status="HIGH QUALITY — RESEARCH THRESHOLD",
        mathematical_formulation="Weighted Composite"
    )

    fusion = FusionResult(
        fused_matrix=[[0.5] * 32 for _ in range(32)],
        agreement_matrix=[[0.9] * 32 for _ in range(32)],
        disagreement_matrix=[[0.1] * 32 for _ in range(32)],
        overall_agreement=0.90,
        fusion_confidence=0.55,
        weights_used={"Grad-CAM++": 0.5, "SHAP": 0.3, "Integrated Gradients": 0.2},
        pairwise_agreement={"Grad-CAM++ ↔ SHAP": 0.90}
    )

    high_uncertainty = UncertaintyResult(
        score=0.72,
        level="high",
        entropy=0.75,
        calibration_error=0.15,
        monte_carlo_variance=0.078,
        interpretation="High epistemic uncertainty and dispersed class probabilities.",
        alignment_with_confidence="LOW"
    )

    assessment = ExplanationReliabilityEngine.evaluate_reliability(
        confidence_prob=0.91,
        uncertainty=high_uncertainty,
        xqi=xqi,
        fusion=fusion,
        localization_score=92.0
    )

    # Must be penalized into UNRELIABLE despite high XQI!
    assert assessment.score < 60.0
    assert assessment.level == "REVIEW REQUIRED"
    assert assessment.should_trust_explanation is False
    assert assessment.uncertainty_penalty > 20.0
