from typing import List, Dict, Optional
from app.schemas.cases import ReliabilityAssessment, XQIDimensions, UncertaintyResult, FusionResult

class ExplanationReliabilityEngine:
    """
    Explanation Reliability Engine (RG4).
    Determines whether a generated explanation can be scientifically trusted
    by synthesizing explanation quality, cross-method consensus, perturbation stability,
    and mathematically penalizing predictive uncertainty (Monte Carlo variance and Shannon entropy).
    """

    @staticmethod
    def calculate_uncertainty_penalty(
        entropy: float,
        mc_variance: float,
        composite_score: float,
        mc_crit: float = 0.06
    ) -> Dict[str, float]:
        """
        Calculates mathematical uncertainty penalty components for RG4:
        - Entropy penalty (dispersion across candidate classes)
        - Monte Carlo variance penalty (epistemic instability across stochastic forward passes)
        - Excess non-linear penalty for elevated uncertainty
        """
        norm_entropy = max(0.0, min(1.0, float(entropy)))
        norm_mc = max(0.0, min(1.0, float(mc_variance) / mc_crit))
        comp = max(0.0, min(1.0, float(composite_score)))

        # Composite uncertainty index: 45% Shannon entropy, 45% MC variance, 10% composite calibration
        uncertainty_index = 0.45 * norm_entropy + 0.45 * norm_mc + 0.10 * comp

        # 1. Base linear penalty (up to 15.0 pts)
        base_penalty = 15.0 * uncertainty_index

        # 2. Non-linear excess penalty when uncertainty exceeds acceptable baseline threshold (0.25)
        excess = max(0.0, uncertainty_index - 0.25)
        excess_penalty = 50.0 * (excess ** 1.3)

        total_penalty = round(base_penalty + excess_penalty, 2)
        entropy_part = round(15.0 * 0.45 * norm_entropy + 0.5 * excess_penalty, 2)
        mc_part = round(15.0 * 0.45 * norm_mc + 0.5 * excess_penalty, 2)

        return {
            "uncertainty_index": round(uncertainty_index, 4),
            "norm_entropy": round(norm_entropy, 4),
            "norm_mc": round(norm_mc, 4),
            "base_penalty": round(base_penalty, 2),
            "excess_penalty": round(excess_penalty, 2),
            "entropy_penalty": entropy_part,
            "mc_variance_penalty": mc_part,
            "total_penalty": total_penalty
        }

    @staticmethod
    def evaluate_reliability(
        confidence_prob: float,
        uncertainty: UncertaintyResult,
        xqi: XQIDimensions,
        fusion: FusionResult,
        localization_score: Optional[float] = None
    ) -> ReliabilityAssessment:
        base_xqi = getattr(xqi, "overall", getattr(xqi, "overall_xqi", 75.0))
        agreement_perc = fusion.overall_agreement * 100.0
        stability_score = getattr(xqi, "stability", 80.0)

        # Base explanation quality before uncertainty penalty (scale [0, 100])
        base_quality = (
            0.40 * base_xqi +
            0.25 * agreement_perc +
            0.20 * stability_score +
            0.15 * 100.0
        )

        # Extract uncertainty components
        entropy = float(uncertainty.entropy)
        mc_var = float(uncertainty.monte_carlo_variance) if uncertainty.monte_carlo_variance is not None else float(uncertainty.score * 0.05)
        comp_score = float(uncertainty.score)

        # Mathematical Uncertainty Penalty (RG4)
        pen_data = ExplanationReliabilityEngine.calculate_uncertainty_penalty(
            entropy=entropy,
            mc_variance=mc_var,
            composite_score=comp_score
        )

        raw_reliability = base_quality - pen_data["total_penalty"]

        # Epistemic Uncertainty Gating (RG4):
        # When predictive uncertainty is high (high Shannon entropy or high MC variance),
        # the model's representations are unstable/dispersed, so explanations cannot be trusted.
        is_severely_uncertain = (
            pen_data["uncertainty_index"] >= 0.55 or
            uncertainty.level in ["high", "very_high"] or
            mc_var >= 0.05 or
            entropy >= 0.55
        )

        if is_severely_uncertain:
            # Strictly cap reliability below trustworthy threshold (60.0)
            raw_reliability = min(raw_reliability, 59.9)

        reliability_score = round(max(0.0, min(100.0, raw_reliability)), 1)

        # Evidence evaluation
        evidence_positive: List[str] = []
        evidence_concerns: List[str] = []

        # Check agreement
        if fusion.overall_agreement >= 0.80:
            evidence_positive.append("Multiple XAI methods exhibit strong spatial agreement across key pathology regions.")
        elif fusion.overall_agreement >= 0.65:
            evidence_concerns.append("Moderate explainer divergence observed between gradient and perturbation attribution maps.")
        else:
            evidence_concerns.append("Severe spatial disagreement across explainers; attribution maps highlight conflicting anatomical structures.")

        # Check stability
        if xqi.stability >= 80.0:
            evidence_positive.append("Explanation remains stable under input perturbations (Gaussian noise, contrast shifts).")
        else:
            evidence_concerns.append("Explanation displays high instability under minor photometric and geometric perturbations.")

        # Check localization
        if localization_score is not None:
            if localization_score >= 85.0:
                evidence_positive.append("High overlap with expert radiologist localization annotations.")
            elif localization_score >= 60.0:
                evidence_concerns.append("Partial overlap with expected anatomical lesion boundaries.")
            else:
                evidence_concerns.append("Poor localization: attribution concentrates on peripheral or non-lesion artifacts.")
        else:
            evidence_concerns.append("Human ground-truth localization annotation is not available for this case.")

        # Check epistemic uncertainty: MC Dropout Variance (RG4)
        if pen_data["norm_mc"] >= 0.70:
            evidence_concerns.append(
                f"Severe Epistemic Uncertainty (RG4): High Monte Carlo variance (σ²_MC = {mc_var:.4f}) proves stochastic weight instability. Explanations from unstable representations cannot be trusted."
            )
        elif pen_data["norm_mc"] >= 0.40:
            evidence_concerns.append(
                f"Moderate Epistemic Uncertainty: Monte Carlo variance (σ²_MC = {mc_var:.4f}) indicates non-trivial variance across stochastic forward passes."
            )
        else:
            evidence_positive.append(
                f"Low Epistemic Uncertainty: Minimal Monte Carlo variance (σ²_MC = {mc_var:.4f}); model representations are robust across stochastic dropout perturbations."
            )

        # Check predictive entropy: Shannon Entropy (RG4)
        if entropy >= 0.50:
            evidence_concerns.append(
                f"High Predictive Entropy (RG4): Normalized Shannon entropy (H_norm = {entropy:.3f}) indicates dispersed class probabilities. Saliency maps reflect differential ambiguity rather than focal pathology."
            )
        else:
            evidence_positive.append(
                f"Low Predictive Entropy: Peaked categorical distribution (H_norm = {entropy:.3f}) aligned with confident diagnosis."
            )

        # Check uncertainty penalty impact
        if pen_data["total_penalty"] >= 12.0:
            evidence_concerns.append(
                f"Predictive Uncertainty Penalty Applied: Explanation reliability penalized by -{pen_data['total_penalty']:.1f} pts (Entropy: -{pen_data['entropy_penalty']:.1f} pts, MC Var: -{pen_data['mc_variance_penalty']:.1f} pts)."
            )

        # Check faithfulness
        if xqi.faithfulness >= 80.0:
            evidence_positive.append("High model faithfulness: mask removal significantly reduces target class probability.")
        else:
            evidence_concerns.append("Sub-optimal faithfulness: salient pixels do not strongly govern model output.")

        # Level determination
        if reliability_score >= 80.0 and len(evidence_concerns) <= 1 and not is_severely_uncertain:
            level = "RELIABLE"
            verdict = "HIGH RELIABILITY"
            should_trust = True
            rec = "The available evaluation evidence indicates a high-reliability explanation under the current research configuration."
        elif reliability_score >= 60.0 and not is_severely_uncertain:
            level = "CAUTION"
            verdict = "MODERATE RELIABILITY — CAUTION"
            should_trust = False
            rec = "Explanation exhibits partial consistency; supplementary clinician review is recommended before relying on attribution."
        else:
            level = "REVIEW REQUIRED"
            verdict = "UNRELIABLE EXPLANATION — REVIEW REQUIRED"
            should_trust = False
            if is_severely_uncertain:
                rec = "Predictive uncertainty is high (high entropy / MC variance). Do NOT trust this explanation; the diagnostic model is uncertain about its prediction."
            else:
                rec = "Explanation fails multiple reliability criteria (disagreement, instability, or uncertainty). Do NOT rely on this explanation."

        return ReliabilityAssessment(
            score=reliability_score,
            level=level,
            trust_verdict=verdict,
            evidence_positive=evidence_positive,
            evidence_concerns=evidence_concerns,
            should_trust_explanation=should_trust,
            clinical_recommendation=rec,
            uncertainty_penalty=pen_data["total_penalty"],
            mc_variance_penalty=pen_data["mc_variance_penalty"],
            entropy_penalty=pen_data["entropy_penalty"],
            base_explanation_quality=round(base_quality, 1)
        )

