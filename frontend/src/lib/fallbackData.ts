import { CaseAnalysis, CaseSummary } from '../types';

// Biomedical SVG representations for instant fallback rendering without backend
function createChestXRaySVG(caseId: string): string {
  const is2047 = caseId.includes('2047');
  const heartPath = is2047 ? "M 140 100 L 280 240 L 250 310 L 130 290 Z" : "M 160 100 L 235 230 L 210 290 L 150 270 Z";
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384" viewBox="0 0 384 384">
    <rect width="384" height="384" fill="#0f172a"/>
    <ellipse cx="115" cy="180" rx="65" ry="135" fill="#1e293b" stroke="#334155" stroke-width="2"/>
    <ellipse cx="269" cy="180" rx="65" ry="135" fill="#1e293b" stroke="#334155" stroke-width="2"/>
    <path d="${heartPath}" fill="#334155" stroke="#475569"/>
    <line x1="192" y1="20" x2="192" y2="360" stroke="#475569" stroke-width="6"/>
    <line x1="60" y1="60" x2="190" y2="85" stroke="#334155" stroke-width="3"/>
    <line x1="324" y1="60" x2="194" y2="85" stroke="#334155" stroke-width="3"/>
    <text x="14" y="24" fill="#94a3b8" font-family="monospace" font-size="11">TX-MED | ${caseId} | CHEST X-RAY</text>
    <text x="14" y="370" fill="#64748b" font-family="monospace" font-size="9">SYNTHETIC RESEARCH BENCHMARK</text>
  </svg>`;
  return `data:image/svg+xml;base64,${btoa(svg)}`;
}

function createGrid(centerR: number, centerC: number, radius: number): number[][] {
  const grid: number[][] = [];
  for (let r = 0; r < 32; r++) {
    const row: number[] = [];
    for (let c = 0; c < 32; c++) {
      const dist = Math.sqrt((r - centerR) ** 2 + (c - centerC) ** 2);
      const val = Math.max(0, 1 - dist / radius);
      row.push(Math.round(val * 100) / 100);
    }
    grid.push(row);
  }
  return grid;
}

export const FALLBACK_SUMMARIES: CaseSummary[] = [
  {
    case_id: 'TX-2047',
    modality: 'Chest X-Ray',
    dataset: 'CheXpert / CheXlocalize',
    model_name: 'DenseNet-121 (Radiology)',
    predicted_label: 'Cardiomegaly',
    confidence: 93.2,
    uncertainty_level: 'high',
    uncertainty_score: 0.58,
    xqi_score: 49.2,
    reliability_score: 46.1,
    reliability_level: 'REVIEW REQUIRED',
    overall_agreement: 0.44,
    is_demo: true,
  },
  {
    case_id: 'TX-2048',
    modality: 'Chest X-Ray',
    dataset: 'CheXpert / CheXlocalize',
    model_name: 'DenseNet-121 (Radiology)',
    predicted_label: 'Pneumonia',
    confidence: 91.4,
    uncertainty_level: 'low',
    uncertainty_score: 0.12,
    xqi_score: 87.5,
    reliability_score: 89.2,
    reliability_level: 'RELIABLE',
    overall_agreement: 0.88,
    is_demo: true,
  },
  {
    case_id: 'TX-2049',
    modality: 'Chest X-Ray',
    dataset: 'NIH ChestX-ray14',
    model_name: 'DenseNet-121',
    predicted_label: 'Pleural Effusion',
    confidence: 84.7,
    uncertainty_level: 'moderate',
    uncertainty_score: 0.32,
    xqi_score: 72.8,
    reliability_score: 71.4,
    reliability_level: 'CAUTION',
    overall_agreement: 0.74,
    is_demo: true,
  },
  {
    case_id: 'TX-2050',
    modality: 'Dermoscopy',
    dataset: 'ISIC 2019 / HAM10000',
    model_name: 'EfficientNet-B4 (Dermoscopy)',
    predicted_label: 'Melanocytic Nevus',
    confidence: 94.8,
    uncertainty_level: 'low',
    uncertainty_score: 0.09,
    xqi_score: 91.2,
    reliability_score: 93.4,
    reliability_level: 'RELIABLE',
    overall_agreement: 0.92,
    is_demo: true,
  },
];

export const FALLBACK_CASE_DATA: CaseAnalysis = {
  case_id: 'TX-2047',
  modality: 'Chest X-Ray',
  dataset: 'CheXpert / CheXlocalize',
  model_name: 'DenseNet-121 (Radiology Backbone)',
  image_base64: createChestXRaySVG('TX-2047'),
  ground_truth_class: 'Cardiomegaly',
  ground_truth_bbox: [0.35, 0.40, 0.75, 0.85],
  prediction: {
    label: 'Cardiomegaly',
    probability: 0.932,
    probabilities: {
      'Cardiomegaly': 0.932,
      'Atelectasis': 0.038,
      'Effusion': 0.021,
      'No Finding': 0.009,
    },
  },
  uncertainty: {
    score: 0.58,
    level: 'high',
    entropy: 0.62,
    calibration_error: 0.182,
    monte_carlo_variance: 0.088,
    interpretation: 'Elevated predictive uncertainty and high Monte Carlo variance despite high top-class softmax confidence.',
    alignment_with_confidence: 'LOW',
  },
  explanations: {
    'Grad-CAM++': {
      method: 'Grad-CAM++',
      matrix: createGrid(18, 14, 9),
      grid_size: [32, 32],
      faithfulness: 54.0,
      localization: 42.0,
      stability: 48.0,
      robustness: 45.0,
      consistency: 52.0,
      human_agreement: 48.0,
      provenance: { layer: 'features.denseblock4.denselayer16.conv2', hook_type: 'forward_and_backward' },
    },
    'SHAP': {
      method: 'SHAP',
      matrix: createGrid(12, 22, 8),
      grid_size: [32, 32],
      faithfulness: 48.0,
      localization: 38.0,
      stability: 41.0,
      robustness: 39.0,
      consistency: 46.0,
      human_agreement: 40.0,
      provenance: { n_samples: 250, algorithm: 'KernelSHAP' },
    },
    'Integrated Gradients': {
      method: 'Integrated Gradients',
      matrix: createGrid(16, 17, 7),
      grid_size: [32, 32],
      faithfulness: 56.0,
      localization: 51.0,
      stability: 44.0,
      robustness: 42.0,
      consistency: 50.0,
      human_agreement: 46.0,
      provenance: { steps: 50, baseline: 'black_image' },
    },
    'Attention Rollout': {
      method: 'Attention Rollout',
      matrix: createGrid(24, 20, 10),
      grid_size: [32, 32],
      faithfulness: 42.0,
      localization: 35.0,
      stability: 38.0,
      robustness: 36.0,
      consistency: 40.0,
      human_agreement: 38.0,
      provenance: { head_fusion: 'mean' },
    },
  },
  fusion: {
    fused_matrix: createGrid(17, 16, 8),
    agreement_matrix: createGrid(17, 16, 5),
    disagreement_matrix: createGrid(14, 21, 6),
    overall_agreement: 0.44,
    fusion_confidence: 0.46,
    weights_used: {
      'Grad-CAM++': 0.35,
      'SHAP': 0.20,
      'Integrated Gradients': 0.30,
      'Attention Rollout': 0.15,
    },
    pairwise_agreement: {
      'GradCAM++_SHAP': 0.38,
      'GradCAM++_IG': 0.54,
      'GradCAM++_Attn': 0.36,
      'SHAP_IG': 0.42,
    },
    fusion_strategy: 'uncertainty_weighted_consensus',
  },
  xqi: {
    overall: 49.2,
    faithfulness: 51.0,
    localization: 42.0,
    robustness: 41.0,
    stability: 44.0,
    consistency: 48.0,
    human_agreement: 44.0,
    uncertainty_alignment: 35.0,
    weights: {
      faithfulness: 0.25,
      localization: 0.20,
      robustness: 0.15,
      stability: 0.15,
      consistency: 0.15,
      uncertainty_alignment: 0.10,
    },
    status: 'POOR',
    mathematical_formulation: 'XQI = \\sum w_i \\cdot D_i - \\alpha \\cdot \\text{Pen}(U, \\text{Conf})',
  },
  reliability: {
    score: 46.1,
    level: 'REVIEW REQUIRED',
    trust_verdict: 'Do Not Rely Blindly: High Predictive Uncertainty Anomaly detected.',
    should_trust_explanation: false,
    clinical_recommendation: 'Secondary radiologist review required. Saliency consensus is low (44%) and uncertainty is elevated (0.58).',
    evidence_positive: [
      'Model prediction aligns with cardiac enlargement silhouette.',
      'Integrated Gradients shows moderate localization over ventricular border.',
    ],
    evidence_concerns: [
      'Disagreement between Grad-CAM++ and SHAP exceeds clinical tolerance (>40%).',
      'High Monte Carlo variance indicates epistemic model instability for this scan.',
    ],
  },
  is_demo: true,
  provenance: {
    source: 'synthetic_research_benchmark',
    simulated: true,
    dataset_source: 'Stanford CheXpert Reference Library',
    weights_version: 'v1.2-baseline',
  },
};
