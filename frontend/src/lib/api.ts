import type { CaseAnalysis, CaseSummary, PerturbationResult } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || '';

// ---------------------------------------------------------------------------
// Typed API helpers
// ---------------------------------------------------------------------------
async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers as Record<string, string>) },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API object consumed by App.tsx & views
// ---------------------------------------------------------------------------
export interface DashboardSummary {
  accuracy: number;
  auc_roc?: number;
  macro_f1?: number;
  calibration: number;
  xqi: number;
  reliability: number;
  cases_analyzed: number;
  dataset: string;
  model_name: string;
  system_health: string;
  last_updated: string;
}

export const api = {
  // ── Health ───────────────────────────────────────────────────
  healthCheck: () => fetchJSON<{ status: string }>('/api/health'),

  // ── Settings & Reliability Penalty Coefficient ────────────────
  getSettings: () => fetchJSON<any>('/api/v1/settings'),
  saveSettings: (settings: any) =>
    fetchJSON<{ status: string; settings: any }>('/api/v1/settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    }),

  // ── Dashboard summary ────────────────────────────────────────
  getDashboardSummary: () => fetchJSON<DashboardSummary>('/api/dashboard/summary'),

  // ── Cases ───────────────────────────────────────────────────
  getCases: () => fetchJSON<CaseSummary[]>('/api/cases'),

  getCaseById: (caseId: string) =>
    fetchJSON<CaseAnalysis>(`/api/cases/${encodeURIComponent(caseId)}`),

  // ── Fusion ──────────────────────────────────────────────────
  recalculateFusion: (caseId: string, weights: Record<string, number>) =>
    fetchJSON<CaseAnalysis>('/api/fusion/custom', {
      method: 'POST',
      body: JSON.stringify({ case_id: caseId, weights }),
    }),

  // ── Live XQI & ERS Calculation (/api/v1/xqi) ────────────────
  recalculateXQI: (caseId: string, weights?: Record<string, number>, alphaPenalty?: number) =>
    fetchJSON<any>('/api/v1/xqi', {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        weights: weights || {},
        alpha_penalty: alphaPenalty,
      }),
    }),

  // ── Robustness / Perturbation ───────────────────────────────
  runPerturbation: (
    caseId: string,
    perturbationType: string,
    intensity: number,
    xaiMethod: string,
  ) =>
    fetchJSON<PerturbationResult>('/api/robustness/perturb', {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        perturbation_type: perturbationType,
        intensity,
        xai_method: xaiMethod,
      }),
    }),

  // ── Datasets ────────────────────────────────────────────────
  getDatasets: () => fetchJSON<any[]>('/api/datasets'),
  getRealDatasets: () => fetchJSON<any[]>('/api/datasets/real'),
  scanDataset: (payload: {
    root_path: string;
    dataset_name?: string;
    modality?: string;
    train_pct?: number;
    val_pct?: number;
    test_pct?: number;
    seed?: number;
    enforce_patient_split?: boolean;
  }) =>
    fetchJSON<any>('/api/datasets/scan', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // ── Models ──────────────────────────────────────────────────
  getModels: () => fetchJSON<any[]>('/api/models'),
  getRealModels: () => fetchJSON<any[]>('/api/models/real'),
  getModelsRegistry: () => fetchJSON<any[]>('/api/v1/models/registry'),

  // ── Training ────────────────────────────────────────────────
  startTraining: (config: any) =>
    fetchJSON<{ job_id: string }>('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  getTrainingStatus: (jobId: string) =>
    fetchJSON<any>(`/api/training/${encodeURIComponent(jobId)}/status`),

  // ── Live Inference (/api/v1/inference) ──────────────────────
  runV1Inference: async (payload: {
    image_base64?: string;
    image_path?: string;
    alpha_penalty?: number;
    modality?: string;
    dataset_type?: string;
  }) =>
    fetchJSON<any>('/api/v1/inference', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  uploadForInference: async (file: File, modality?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (modality) {
      form.append('modality', modality);
      form.append('dataset_type', modality);
    }
    const res = await fetch(`${BASE_URL}/api/v1/inference`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
    return res.json();
  },

  // Alias used across views
  uploadImageForInference: async (file: File, modality?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (modality) {
      form.append('modality', modality);
      form.append('dataset_type', modality);
    }
    const res = await fetch(`${BASE_URL}/api/v1/inference`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
    return res.json();
  },

  // ── Experiments ─────────────────────────────────────────────
  getExperiments: () => fetchJSON<any[]>('/api/experiments'),
  getAblationMatrix: () => fetchJSON<any[]>('/api/experiments/ablation'),

  // ── Clinical Study ──────────────────────────────────────────
  getStudyConditions: () => fetchJSON<any[]>('/api/clinical-study/conditions'),
  getStudyBenchmarks: () => fetchJSON<any[]>('/api/clinical-study/benchmarks'),
  getStudyResponses: () => fetchJSON<any[]>('/api/clinical-study/responses'),

  // ── Reports ─────────────────────────────────────────────────
  getMarkdownReport: (caseId: string) =>
    fetchJSON<any>(`/api/reports/${encodeURIComponent(caseId)}/markdown`),
  getJsonReport: (caseId: string) =>
    fetchJSON<any>(`/api/reports/${encodeURIComponent(caseId)}/json`),
};

// ---------------------------------------------------------------------------
// Named convenience exports used by individual views
// ---------------------------------------------------------------------------
export const fetchReportMarkdown = (caseId: string) => api.getMarkdownReport(caseId);
export const fetchReportJSON = (caseId: string) => api.getJsonReport(caseId);
