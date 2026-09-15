from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field

class PredictionResult(BaseModel):
    label: str
    probability: float
    probabilities: Dict[str, float] = Field(default_factory=dict)
    logits: Optional[Dict[str, float]] = None

class UncertaintyResult(BaseModel):
    score: float = Field(..., description="Normalized uncertainty score in [0, 1]")
    level: str = Field(..., description="low, moderate, high, very_high")
    entropy: float = Field(..., description="Normalized predictive entropy")
    calibration_error: float = Field(..., description="Estimated calibration error ECE")
    monte_carlo_variance: Optional[float] = None
    interpretation: str
    alignment_with_confidence: str = Field(..., description="HIGH, MODERATE, LOW alignment")

class SaliencyMapData(BaseModel):
    method: str
    matrix: List[List[float]] = Field(..., description="Normalized 2D heatmap matrix values [0, 1]")
    grid_size: List[int] = Field(default=[32, 32])
    faithfulness: float
    localization: Optional[float] = None
    stability: float
    robustness: float
    consistency: float
    human_agreement: Optional[float] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)

class FusionResult(BaseModel):
    fused_matrix: List[List[float]]
    agreement_matrix: List[List[float]]
    disagreement_matrix: List[List[float]]
    overall_agreement: float
    fusion_confidence: float
    weights_used: Dict[str, float]
    pairwise_agreement: Dict[str, float]
    fusion_strategy: str = "Uncertainty-Weighted Quality-Aware Baseline"

class XQIDimensions(BaseModel):
    overall: float
    faithfulness: float
    localization: Optional[float] = None
    robustness: float
    stability: float
    consistency: float
    human_agreement: Optional[float] = None
    uncertainty_alignment: float
    weights: Dict[str, float]
    status: str
    mathematical_formulation: str
    is_research_baseline: bool = True

class ReliabilityEvidence(BaseModel):
    factor: str
    status: str  # positive, warning, negative
    detail: str

class ReliabilityAssessment(BaseModel):
    score: float
    level: str  # RELIABLE, CAUTION, REVIEW REQUIRED
    trust_verdict: str
    evidence_positive: List[str]
    evidence_concerns: List[str]
    should_trust_explanation: bool
    clinical_recommendation: str
    uncertainty_penalty: Optional[float] = None
    mc_variance_penalty: Optional[float] = None
    entropy_penalty: Optional[float] = None
    base_explanation_quality: Optional[float] = None

class CaseSummary(BaseModel):
    case_id: str
    modality: str
    dataset: str
    model_name: str
    predicted_label: str
    confidence: float
    uncertainty_level: str
    uncertainty_score: float
    xqi_score: float
    reliability_score: float
    reliability_level: str
    overall_agreement: float
    is_demo: bool = True

class CaseAnalysisResponse(BaseModel):
    case_id: str
    modality: str
    dataset: str
    model_name: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    ground_truth_class: Optional[str] = None
    ground_truth_bbox: Optional[List[float]] = None
    prediction: PredictionResult
    uncertainty: UncertaintyResult
    explanations: Dict[str, SaliencyMapData]
    fusion: FusionResult
    xqi: Union[XQIDimensions, Any]
    reliability: ReliabilityAssessment
    is_demo: bool = True
    provenance: Dict[str, Any] = Field(default_factory=dict)
