# TrustXAI-Med
### Uncertainty-Aware Hybrid Explainable AI for Medical Image Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg)](https://www.typescriptlang.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-amber.svg)](#research-status)

<p align="center">
  <img src="docs/images/trustxai_dashboard.png" alt="TrustXAI-Med Primary Command Center" width="100%" />
</p>

> **Can we trust the explanation behind an AI diagnosis?**

**TrustXAI-Med** is a research-grade, full-stack platform for investigating the reliability of Explainable AI (XAI) in medical image diagnosis. Instead of stopping at:

```text
Medical Image → AI Prediction → Heatmap
```

TrustXAI-Med investigates the complete trustworthy-AI pipeline:

```text
Medical Image
      ↓
Diagnostic Model
      ↓
Prediction ───────────────┐
      ↓                   │
Multiple XAI Methods      │
      ↓                   │
Explanation Fusion        │
      ↓                   │
Explanation Quality       │
      ↓                   │
XQI + Uncertainty ────────┘
      ↓
Explanation Reliability
      ↓
Clinical Trust
      ↓
Decision Support
```

The core research question is:
> *Can predictive uncertainty and explanation-quality metrics be combined to determine whether a medical AI explanation should—or should not—be trusted?*

---

## Table of Contents
- [Overview](#overview)
- [Why TrustXAI-Med?](#why-trustxai-med)
- [Research Gaps](#research-gaps)
- [Core Architecture](#core-architecture)
- [Key Features](#key-features)
- [The Killer Demo](#the-killer-demo)
- [XAI Methods](#xai-methods)
- [Explanation Fusion](#explanation-fusion)
- [Explanation Quality Index (XQI)](#explanation-quality-index-xqi)
- [Explanation Reliability](#explanation-reliability)
- [Uncertainty Analysis](#uncertainty-analysis)
- [Robustness Analysis](#robustness-analysis)
- [Cross-Domain Validation](#cross-domain-validation)
- [Clinician Trust Study](#clinician-trust-study)
- [Experiments and Ablation](#experiments-and-ablation)
- [Research Reports](#research-reports)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Training Pipeline](#model-training-pipeline)
- [Demo Mode](#demo-mode)
- [Real Research Mode](#real-research-mode)
- [Datasets](#datasets)
- [Running Tests](#running-tests)
- [Research Reproducibility](#research-reproducibility)
- [Limitations](#limitations)
- [Research Status](#research-status)
- [Medical Safety Disclaimer](#medical-safety-disclaimer)
- [License](#license)

---

## Overview

Traditional medical XAI systems often focus on generating visual explanations such as saliency maps or heatmaps. While these visualizations can make model decisions appear interpretable, a critical question remains:

**Is the explanation itself reliable?**

A model can produce:
- High prediction confidence,
- High classification accuracy, and
- A visually convincing heatmap,

while the underlying explanation may still be **unstable, poorly localized, inconsistent across XAI methods, sensitive to small perturbations, or poorly aligned with model uncertainty**.

TrustXAI-Med addresses this problem by separating:
$$\text{Prediction Reliability} \neq \text{Explanation Reliability}$$

The platform evaluates not only what the model predicts, but also:
1. How uncertain the prediction is.
2. Why the model made the prediction.
3. Whether multiple explanation methods agree.
4. Whether explanations remain stable under perturbations.
5. Whether explanations align with available annotations.
6. Whether the explanation satisfies a quantitative quality framework.
7. Whether the available evidence supports trusting the explanation.

---

## Why TrustXAI-Med?

A conventional medical AI workflow may look like:
```text
Upload Image → Predict Disease → 91% Confidence → Show Heatmap → Done
```

TrustXAI-Med asks essential clinical and methodological questions:
- *91% confidence... But: Is the model calibrated?*
- *Is predictive uncertainty low?*
- *Do different XAI methods agree?*
- *Is the salient region meaningful?*
- *Is the explanation stable under noise?*
- *Does it align with available expert annotations?*
- *Is the explanation faithful to the model's inner representations?*
- *What is the overall explanation quality?*
- *Should the explanation actually be trusted?*

---

## Research Gaps

TrustXAI-Med is designed around five core research gaps (RG1 – RG5):

- **RG1 — Explanation-Level Fusion Gap:** Existing systems generate multiple explanations independently without a principled mechanism for synthesizing them. TrustXAI-Med introduces a **Quality-Aware + Agreement-Weighted + Uncertainty-Gated Explanation Fusion Engine** to combine complementary XAI evidence into a unified explanation.
- **RG2 — Explanation Reliability Gap:** High prediction confidence does not necessarily imply explanation reliability. TrustXAI-Med explicitly separates **Prediction Confidence & Uncertainty** from **Explanation Quality & Reliability**.
- **RG3 — Quantitative Explanation Quality Gap:** Visual inspection alone is insufficient. TrustXAI-Med introduces the **Explanation Quality Index (XQI)**, a 7-dimensional evaluation framework combining Faithfulness, Localization, Robustness, Stability, Consistency, Human Agreement, and Uncertainty Alignment.
- **RG4 — Uncertainty → Explanation Reliability Gap:** Investigates whether predictive entropy, calibration error (ECE), and dispersion can serve as quantitative indicators of when explanations should receive additional scrutiny.
- **RG5 — Cross-Domain and Clinical Validation Gap:** Validates explanation reliability across Chest X-ray, Dermoscopy, and Brain MRI with a 4-condition clinician trust-study protocol.

---

## Core Architecture

```text
               MEDICAL IMAGE (DICOM / PNG / JPEG)
                               │
                               ▼
                     ┌──────────────────┐
                     │ DIAGNOSTIC MODEL │
                     │  DenseNet / ViT  │
                     │      / Swin      │
                     └─────────┬────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌──────────────┐              ┌──────────────┐
        │  PREDICTION  │              │ UNCERTAINTY  │
        │ Probability  │              │   Entropy    │
        │    Logits    │              │     ECE      │
        └───────┬──────┘              └───────┬──────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    MULTIPLE XAI     │
                    ├─────────────────────┤
                    │ Grad-CAM++          │
                    │ SHAP                │
                    │ Integrated Gradients│
                    │ Attention Rollout   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ EXPLANATION FUSION  │
                    │  Quality-Aware      │
                    │  Agreement-Weighted │
                    │  Uncertainty-Gated  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ UNIFIED EXPLANATION │
                    │ Agreement Map       │
                    │ Disagreement Map    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │         XQI         │
                    │ 7-D Quality Score   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │     EXPLANATION     │
                    │     RELIABILITY     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   CLINICAL TRUST    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  DECISION SUPPORT   │
                    │  • Reliable         │
                    │  • Caution          │
                    │  • Review Required  │
                    └─────────────────────┘
```

---

## Key Features

### 1. Clinical Case Analysis Command Center
- **Medical Image Viewer:** Pan, zoom, opacity slider, thresholding, contour toggle, and side-by-side comparison.
- **Diagnostic Insights:** Class prediction probabilities, alternative differential findings, and predictive uncertainty metrics.
- **Explainability Suite:** Individual XAI overlays, unified fused explanation, 7-D XQI breakdown, and explanation reliability assessment.

### 2. Multi-Method XAI
- **Grad-CAM++:** Gradient-based localization of convolutional feature activations.
- **SHAP:** Game-theoretic superpixel attribution inspired by Shapley values.
- **Integrated Gradients:** Path-integral axiomatic attribution relative to a baseline reference.
- **Attention Rollout:** Recursive attention flow tracking across Vision Transformer (ViT / Swin) heads.

<p align="center">
  <img src="docs/images/explanation_lab.png" alt="Multi-Method Explanation Lab with Radar Metrics" width="100%" />
</p>

---

## Explanation Fusion

Rather than treating a single heatmap as infallible ground truth, the **Hybrid Explanation Fusion Engine** synthesizes multiple attribution maps:

```text
Grad-CAM++ ──────┐
SHAP ────────────┤
Integrated Grad. ├──→ [ FUSION ENGINE ] ──► [ UNIFIED EXPLANATION ]
Attention ───────┤      (Quality + Agreement    (Agreement Map /
Uncertainty ─────┤       + Uncertainty Gating)   Disagreement Map)
Quality Metrics ─┘
```

**Fusion Components:**
1. Robust percentile normalization ($P_{99.5}$)
2. Spatial alignment & resolution matching
3. Pairwise spatial correlation ($\rho_{a,b}$)
4. Uncertainty-aware dynamic weighting
5. Spatial agreement and variance/disagreement maps

<p align="center">
  <img src="docs/images/fusion_lab.png" alt="Explanation Fusion and Pairwise Agreement Matrix" width="100%" />
</p>

---

## Explanation Quality Index (XQI)

The **Explanation Quality Index (XQI)** is a multi-dimensional research formulation in $[0, 100]$:

| Dimension | Description |
| :--- | :--- |
| **Faithfulness** | How accurately the explanation reflects internal model decision logic |
| **Localization** | Spatial alignment (IoU / Pointing Game) with expert radiologist annotations |
| **Robustness** | Resistance to degradation under input noise and perturbations |
| **Stability** | Consistency of explanations across initialization seeds and inputs |
| **Consistency** | Consensus across multiple explanation techniques |
| **Human Agreement** | Concordance with clinician Likert evaluation in reader studies |
| **Uncertainty Alignment** | Coherence between predictive uncertainty and explanation quality |

$$XQI = \sum_{d \in D} \tilde{w}_d \cdot S_d$$

> [!NOTE]
> XQI is a proposed research framework and metric within this project and is not an established clinical or regulatory standard.

<p align="center">
  <img src="docs/images/reliability_xqi.png" alt="7-Dimension XQI and Explanation Reliability Gauge" width="100%" />
</p>

---

## Explanation Reliability

Prediction confidence and explanation reliability are decoupled. The platform classifies assessments into:
- `RELIABLE`: Strong positive research evidence across metrics.
- `CAUTION`: Moderate uncertainty or mixed evaluation evidence.
- `REVIEW REQUIRED`: Severe explainer discordance, high entropy, or poor stability.

---

## The Killer Demo

### Case TX-2047 — Cardiomegaly

<p align="center">
  <img src="docs/images/killer_demo_tx2047.png" alt="Killer Demo Case TX-2047 Review Required" width="100%" />
</p>

| Metric | Traditional XAI Dashboard | TrustXAI-Med |
| :--- | :--- | :--- |
| **AI Diagnosis** | Cardiomegaly | Cardiomegaly |
| **Prediction Confidence** | **93.2%** | **93.2%** |
| **Predictive Uncertainty** | *Not shown* | **High** |
| **Normalized Entropy** | *Not shown* | **0.62** |
| **Calibration (ECE)** | *Not shown* | **0.18** |
| **Multi-XAI Agreement** | *Not evaluated* | **51.5%** |
| **Spatial Disagreement** | *Not evaluated* | **48.5%** |
| **Explanation Quality (XQI)** | *Not measured* | **49.0 / 100** |
| **Explanation Reliability** | *Assumed Reliable* | **46.0 / 100** |
| **Recommended Action** | Accepted blindly | **REVIEW REQUIRED** |

```text
93.2% Prediction Confidence
      ↓
High Predictive Uncertainty
      ↓
Severe XAI Disagreement
      ↓
Low Explanation Quality (XQI 49)
      ↓
Low Explanation Reliability (46/100)
      ↓
REVIEW REQUIRED
```

---

## Robustness Analysis

The **Robustness Lab** simulates clinical and adversarial image perturbations:
- Gaussian noise ($\sigma = 0.1$)
- Brightness shifts ($+20\%$)
- Contrast gain ($-20\%$)
- Gaussian blur ($5\times 5$)
- JPEG compression ($50\%$)
- Rotation ($5^\circ$)
- Random crop ($80\%$)

Measures SSIM degradation, perturbation stability score, and attribution displacement.

<p align="center">
  <img src="docs/images/robustness_lab.png" alt="Robustness Lab Perturbation Stress Testing" width="100%" />
</p>

---

## Cross-Domain Validation

Evaluated across four primary clinical imaging benchmarks:
- **CheXpert** (Chest X-ray, 224,316 scans): Primary classification and uncertainty.
- **CheXlocalize** (Chest X-ray, 2,340 masks): Radiologist ground-truth localization.
- **ISIC 2024 / HAM10000** (Dermoscopy, 40,000+ images): Cross-domain oncology screening.
- **BraTS 2023** (Brain MRI, 1,251 3D cases): Cross-modality multi-parametric neuro-oncology.
- **VinDr-CXR & MIMIC-CXR-JPG**: External cohort and ICU domain shift validation.

---

## Clinician Trust Study

A 4-condition reader-study framework:
- **Condition A:** Prediction Only
- **Condition B:** Prediction + Grad-CAM
- **Condition C:** Prediction + Hybrid XAI
- **Condition D:** Prediction + Hybrid XAI + XQI + Uncertainty

Measures diagnostic accuracy, confidence, decision latency, trust rating, and reduction in automation bias.

---

## Technology Stack

- **Frontend:** React 18, TypeScript 5.6, Vite, Tailwind CSS, Lucide Icons, Canvas API.
- **Backend:** Python 3.11+, FastAPI, Pydantic v2, Uvicorn.
- **Machine Learning & XAI:** PyTorch, Torchvision, scikit-learn, NumPy, SciPy.
- **XAI Adapters:** Grad-CAM++, SHAP, Integrated Gradients, Attention Rollout.

---

## Project Structure

```text
TrustXAI-Med/
├── backend/
│   ├── app/
│   │   ├── clinical_study/   # Reader study protocol & benchmarks
│   │   ├── datasets/         # Dataset registries (CheXpert, ISIC, BraTS)
│   │   ├── experiments/      # Benchmark tracking & ablation matrix
│   │   ├── fusion/           # Normalization, agreement, & fusion engine
│   │   ├── models/           # Uncertainty estimator & model registry
│   │   ├── quality/          # 7-D XQI calculation & dynamic weights
│   │   ├── reliability/      # Explanation reliability assessment engine
│   │   ├── reports/          # Markdown, JSON, and CSV report generator
│   │   ├── robustness/       # Perturbation stability simulation
│   │   ├── schemas/          # Strict Pydantic v2 data models
│   │   ├── services/         # Synthetic case library & image renderer
│   │   └── xai/              # Grad-CAM++, SHAP, IG, Attention adapters
│   ├── tests/
│   │   ├── run_tests.py      # Standalone verification runner
│   │   └── test_algorithms.py # Pytest test suite
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── fusion/       # Fusion pipeline diagram & agreement matrix
│   │   │   ├── image-viewer/ # Canvas radiological viewer with controls
│   │   │   ├── layout/       # Sidebar navigation & Header
│   │   │   ├── prediction/   # AI diagnosis card & alternative findings
│   │   │   ├── reliability/  # Should I Trust This? evidence panel
│   │   │   ├── uncertainty/  # Predictive entropy & alignment card
│   │   │   ├── views/        # All 10 command center screens
│   │   │   ├── xai/          # Explainer cards with radar charts
│   │   │   └── xqi/          # 7-D XQI arc gauge & formula inspector
│   │   ├── lib/              # API client & demo fallback catalog
│   │   ├── types/            # TypeScript interfaces
│   │   ├── App.tsx           # Primary routing & view orchestration
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── architecture.md       # High-level component topology
│   ├── methodology.md        # Mathematical derivations & gaps
│   ├── xqi.md                # 7-D XQI metric formulation
│   ├── fusion.md             # Quality-aware agreement fusion math
│   ├── datasets.md           # Data access setup & credentials
│   ├── experiments.md        # Benchmark tables & ablation protocols
│   └── api.md                # REST API contract
├── package.json              # Root npm scripts
├── LICENSE                   # MIT License
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Start the Frontend Command Center
```powershell
npm run dev
```
*Runs the Vite development server at [http://localhost:5173](http://localhost:5173)*

### 2. Start the FastAPI Backend Server (in a separate terminal)
```powershell
cd backend
python -m uvicorn app.main:app --port 8008 --reload
```
*API docs available at [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs)*

---

## Model Training Pipeline

TrustXAI-Med provides a full diagnostic model training pipeline (`backend/app/training/train_classifier.py`) supporting DenseNet-121, ResNet-50, and EfficientNet-B4.

### 1. Install Training Dependencies (with CUDA GPU support):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas pillow tqdm
```

### 2. Training Execution Commands:

#### Train DenseNet-121 on CheXpert (Chest X-Ray):
```bash
python backend/app/training/train_classifier.py \
  --dataset chexpert \
  --data_dir ./data/chexpert \
  --arch densenet121 \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir ./checkpoints
```

#### Train EfficientNet-B4 on ISIC (Dermoscopy):
```bash
python backend/app/training/train_classifier.py \
  --dataset isic \
  --data_dir ./data/isic \
  --arch efficientnet_b4 \
  --epochs 20 \
  --batch_size 16 \
  --lr 5e-5 \
  --output_dir ./checkpoints
```

### 3. Integrated Trustworthy Training Features:
- **Monte Carlo (MC) Dropout:** Embeds `Dropout(p=0.3)` before the linear head to enable stochastic forward-pass sampling ($T=20$) for epistemic uncertainty $\sigma_{\text{MC}}^2$.
- **Temperature Scaling (Calibration):** Post-hoc optimization ($z / T$) on validation NLL to minimize Expected Calibration Error (ECE).
- **Multi-Label Loss:** Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) with positive-weight balancing for co-occurring pathologies.

### 4. Connect Trained Weights to the Platform:
Set the environment variable pointing to your `.pth` checkpoint:
```powershell
# Windows PowerShell
$env:MODEL_PATH="C:\path\to\checkpoints\densenet121_chexpert.pth"
$env:CHEXPERT_ROOT="C:\path\to\data\chexpert"
```
```bash
# Linux / macOS
export MODEL_PATH=/path/to/checkpoints/densenet121_chexpert.pth
export CHEXPERT_ROOT=/path/to/data/chexpert
```
For complete training details, see the [Model Training Guide](docs/training.md).

---

## Running Tests

Run the backend verification suite:
```powershell
python backend/tests/run_tests.py
```
Or via pytest:
```powershell
cd backend
pytest tests/ -v
```

---

## Research Reproducibility

Each experiment preserves full provenance configuration:
- `experiment_id`
- `dataset` and `dataset_version`
- `model_name` and `weights_version`
- `xai_ensemble_methods`
- `fusion_strategy`
- `uncertainty_score` and `entropy`
- `xqi_dimensional_weights`
- `software_version` and timestamp

---

## Limitations

1. **XQI is a Research Formulation:** The Explanation Quality Index is an experimental proposition and is not an established clinical standard.
2. **Context Dependency:** Saliency behavior varies across architectures, diseases, and acquisition protocols.
3. **Human Agreement Data:** Metric requires empirical clinician study validation.
4. **Demo Data Note:** Values displayed in Demo Mode (e.g. 92.41% accuracy, 87/100 XQI) are simulated to demonstrate the workflow without requiring multi-gigabyte hospital datasets or GPU clusters.

---

## Medical Safety Disclaimer

> [!IMPORTANT]
> **RESEARCH AND EVALUATION USE ONLY**
>
> TrustXAI-Med is an experimental research prototype intended for scientific evaluation, software development, and educational demonstration. It is **not intended for clinical diagnosis, treatment planning, patient management, or autonomous medical decision-making**.
>
> TrustXAI-Med has **not been certified or approved by the FDA, CE, or any other regulatory authority**. Predictions, explanations, confidence values, uncertainty estimates, XQI scores, and reliability assessments must not be interpreted as medical advice.

---

## Citation

If you use TrustXAI-Med in academic work, please cite:

```bibtex
@software{trustxai_med_2026,
  title  = {TrustXAI-Med: Uncertainty-Aware Hybrid XAI for Medical Image Diagnosis},
  author = {Hemanth Dhaka, Srinath Doggala and Contributors},
  year   = {2026},
  url    = {https://github.com/Hemant0234/TrustXAI-Research-Paper}
}
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
#   T r u s t A i - R e s e a r c h - P a p e r  
 