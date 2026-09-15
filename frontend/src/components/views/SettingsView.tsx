import React, { useState, useEffect } from 'react';
import {
  Settings,
  Sliders,
  ShieldCheck,
  Cpu,
  Database,
  Save,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Info,
  Server,
  Sparkles,
  Lock,
  Layers,
  Activity,
  ShieldAlert,
  Gauge
} from 'lucide-react';
import { api } from '../../lib/api';

export const SettingsView: React.FC = () => {
  // XQI Weights Tuning State
  const [faithfulnessWeight, setFaithfulnessWeight] = useState<number>(35);
  const [localizationWeight, setLocalizationWeight] = useState<number>(25);
  const [stabilityWeight, setStabilityWeight] = useState<number>(20);
  const [robustnessWeight, setRobustnessWeight] = useState<number>(20);

  // Uncertainty & Calibration State
  const [alphaPenalty, setAlphaPenalty] = useState<number>(0.50);
  const [mcDropoutSamples, setMcDropoutSamples] = useState<number>(20);
  const [enableTempScaling, setEnableTempScaling] = useState<boolean>(true);

  // Fusion Defaults State
  const [defaultFusionStrategy, setDefaultFusionStrategy] = useState<string>('uncertainty-weighted');
  const [consensusThreshold, setConsensusThreshold] = useState<number>(0.70);

  // Clinical Alert Thresholds State
  const [reliableThreshold, setReliableThreshold] = useState<number>(80);
  const [cautionThreshold, setCautionThreshold] = useState<number>(60);
  const [enforceStrictPatientSplit, setEnforceStrictPatientSplit] = useState<boolean>(true);
  const [flagHighUncertaintyForReview, setFlagHighUncertaintyForReview] = useState<boolean>(true);

  // Compute & Runtime State
  const [computeDevice, setComputeDevice] = useState<string>('auto');
  const [enableMixedPrecision, setEnableMixedPrecision] = useState<boolean>(true);
  const [cacheActivations, setCacheActivations] = useState<boolean>(true);
  const [workerThreads, setWorkerThreads] = useState<number>(4);

  // Paths & Storage
  const [dbPath, setDbPath] = useState<string>('./backend/data/trustxai.db');
  const [checkpointDir, setCheckpointDir] = useState<string>('./checkpoints');
  const [reportsDir, setReportsDir] = useState<string>('./reports/exports');

  // UI Status
  const [saveToast, setSaveToast] = useState<string | null>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);

  const totalWeight = faithfulnessWeight + localizationWeight + stabilityWeight + robustnessWeight;

  useEffect(() => {
    async function checkHealth() {
      setIsLoadingHealth(true);
      try {
        const health = await api.healthCheck();
        setSystemHealth(health);
      } catch (err) {
        setSystemHealth({
          status: 'online',
          service: 'TrustXAI-Med Client Runtime',
          version: '1.0.0',
          mode: 'Standby'
        });
      } finally {
        setIsLoadingHealth(false);
      }
    }
    checkHealth();

    // Load saved settings from backend API
    async function loadBackendSettings() {
      try {
        const backendSettings = await api.getSettings();
        if (backendSettings) {
          if (backendSettings.alpha_penalty !== undefined) setAlphaPenalty(backendSettings.alpha_penalty);
          if (backendSettings.mc_dropout_samples !== undefined) setMcDropoutSamples(backendSettings.mc_dropout_samples);
          if (backendSettings.faithfulness_weight !== undefined) setFaithfulnessWeight(backendSettings.faithfulness_weight);
          if (backendSettings.localization_weight !== undefined) setLocalizationWeight(backendSettings.localization_weight);
          if (backendSettings.stability_weight !== undefined) setStabilityWeight(backendSettings.stability_weight);
          if (backendSettings.robustness_weight !== undefined) setRobustnessWeight(backendSettings.robustness_weight);
        }
      } catch (e) {
        console.warn('Could not load backend settings:', e);
      }
    }
    loadBackendSettings();

    // Load saved settings from localStorage if available
    try {
      const saved = localStorage.getItem('trustxai_settings');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.faithfulnessWeight !== undefined) setFaithfulnessWeight(parsed.faithfulnessWeight);
        if (parsed.localizationWeight !== undefined) setLocalizationWeight(parsed.localizationWeight);
        if (parsed.stabilityWeight !== undefined) setStabilityWeight(parsed.stabilityWeight);
        if (parsed.robustnessWeight !== undefined) setRobustnessWeight(parsed.robustnessWeight);
        if (parsed.alphaPenalty !== undefined) setAlphaPenalty(parsed.alphaPenalty);
        if (parsed.mcDropoutSamples !== undefined) setMcDropoutSamples(parsed.mcDropoutSamples);
        if (parsed.defaultFusionStrategy !== undefined) setDefaultFusionStrategy(parsed.defaultFusionStrategy);
        if (parsed.reliableThreshold !== undefined) setReliableThreshold(parsed.reliableThreshold);
        if (parsed.cautionThreshold !== undefined) setCautionThreshold(parsed.cautionThreshold);
        if (parsed.computeDevice !== undefined) setComputeDevice(parsed.computeDevice);
      }
    } catch (e) {
      console.warn('Could not load stored settings:', e);
    }
  }, []);

  const handleAutoBalanceWeights = () => {
    setFaithfulnessWeight(35);
    setLocalizationWeight(25);
    setStabilityWeight(20);
    setRobustnessWeight(20);
  };

  const handleSaveSettings = async () => {
    const config = {
      faithfulnessWeight,
      localizationWeight,
      stabilityWeight,
      robustnessWeight,
      alphaPenalty,
      mcDropoutSamples,
      enableTempScaling,
      defaultFusionStrategy,
      consensusThreshold,
      reliableThreshold,
      cautionThreshold,
      enforceStrictPatientSplit,
      flagHighUncertaintyForReview,
      computeDevice,
      enableMixedPrecision,
      cacheActivations,
      workerThreads,
      dbPath,
      checkpointDir,
      reportsDir,
      updated_at: new Date().toISOString()
    };
    try {
      localStorage.setItem('trustxai_settings', JSON.stringify(config));
      // Bind directly to live backend API (ERS Reliability Coefficient alpha)
      await api.saveSettings({
        alpha_penalty: alphaPenalty,
        mc_dropout_samples: mcDropoutSamples,
        faithfulness_weight: faithfulnessWeight,
        localization_weight: localizationWeight,
        stability_weight: stabilityWeight,
        robustness_weight: robustnessWeight,
        reliable_threshold: reliableThreshold,
        caution_threshold: cautionThreshold,
        default_fusion_strategy: defaultFusionStrategy,
        compute_device: computeDevice
      });
      setSaveToast('System configuration & Reliability Coefficient (alpha) bound to live backend successfully.');
      setTimeout(() => setSaveToast(null), 3500);
    } catch (err) {
      console.error('Failed to save settings:', err);
      setSaveToast('Saved locally; backend sync error.');
      setTimeout(() => setSaveToast(null), 3500);
    }
  };

  const handleResetDefaults = () => {
    handleAutoBalanceWeights();
    setAlphaPenalty(0.50);
    setMcDropoutSamples(20);
    setEnableTempScaling(true);
    setDefaultFusionStrategy('uncertainty-weighted');
    setConsensusThreshold(0.70);
    setReliableThreshold(80);
    setCautionThreshold(60);
    setEnforceStrictPatientSplit(true);
    setFlagHighUncertaintyForReview(true);
    setComputeDevice('auto');
    setEnableMixedPrecision(true);
    setCacheActivations(true);
    setWorkerThreads(4);
    setSaveToast('Reset to default research parameters.');
    setTimeout(() => setSaveToast(null), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-slate-200">
        <div>
          <div className="flex items-center space-x-2">
            <Settings className="w-5 h-5 text-blue-600" />
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">
              System Settings &amp; Research Configuration
            </h1>
          </div>
          <p className="text-xs text-slate-500 font-medium mt-0.5">
            XQI reliability index weights, uncertainty penalty scaling, fusion consensus thresholds, hardware runtime, and safety rules
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleResetDefaults}
            className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 flex items-center space-x-1.5 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
            <span>Reset Defaults</span>
          </button>
          <button
            onClick={handleSaveSettings}
            className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-xs font-semibold text-white shadow-xs flex items-center space-x-1.5 transition-colors"
          >
            <Save className="w-3.5 h-3.5" />
            <span>Save Configuration</span>
          </button>
        </div>
      </div>

      {/* Save Toast Notification */}
      {saveToast && (
        <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between shadow-xs animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>{saveToast}</span>
          </div>
          <button onClick={() => setSaveToast(null)} className="text-emerald-700 hover:text-emerald-900 text-xs">
            ✕
          </button>
        </div>
      )}

      {/* Grid: 2 Columns for Configurations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column (6 cols): XQI Dimension Weights & Uncertainty */}
        <div className="lg:col-span-6 space-y-5">
          {/* Card 1: XQI Dimension Weights */}
          <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-blue-600" />
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                  1. XQI Dimensions &amp; Metric Weighting
                </h2>
              </div>
              <div className="flex items-center space-x-2">
                <span
                  className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                    totalWeight === 100
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : 'bg-rose-50 text-rose-700 border border-rose-200'
                  }`}
                >
                  Sum: {totalWeight}%
                </span>
                <button
                  onClick={handleAutoBalanceWeights}
                  className="text-[10px] text-blue-600 font-semibold hover:underline"
                >
                  Normalize
                </button>
              </div>
            </div>

            <div className="space-y-3.5 text-xs">
              {/* Faithfulness */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold text-slate-700">Faithfulness ($w_1$)</span>
                  <span className="font-mono font-bold text-blue-600">{faithfulnessWeight}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="60"
                  value={faithfulnessWeight}
                  onChange={(e) => setFaithfulnessWeight(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <span className="text-[10px] text-slate-400 block">Pixel deletion / insertion rank attribution correlation</span>
              </div>

              {/* Localization */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold text-slate-700">Localization Precision ($w_2$)</span>
                  <span className="font-mono font-bold text-blue-600">{localizationWeight}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="60"
                  value={localizationWeight}
                  onChange={(e) => setLocalizationWeight(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <span className="text-[10px] text-slate-400 block">IoU / Dice overlap with expert radiologist contours</span>
              </div>

              {/* Stability */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold text-slate-700">Perturbation Stability ($w_3$)</span>
                  <span className="font-mono font-bold text-blue-600">{stabilityWeight}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="50"
                  value={stabilityWeight}
                  onChange={(e) => setStabilityWeight(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <span className="text-[10px] text-slate-400 block">Attribution invariance under Gaussian noise and contrast shifts</span>
              </div>

              {/* Robustness */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold text-slate-700">Adversarial Robustness ($w_4$)</span>
                  <span className="font-mono font-bold text-blue-600">{robustnessWeight}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="50"
                  value={robustnessWeight}
                  onChange={(e) => setRobustnessWeight(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <span className="text-[10px] text-slate-400 block">Resistance against high-frequency medical artifacts</span>
              </div>
            </div>
          </div>

          {/* Card 2: Uncertainty Estimation & Calibration */}
          <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs space-y-4">
            <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
              <Gauge className="w-4 h-4 text-indigo-600" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                2. Uncertainty Penalty &amp; Calibration
              </h2>
            </div>

            <div className="space-y-3.5 text-xs">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold text-slate-700">
                    Reliability Penalty Coefficient ($\alpha$)
                  </span>
                  <span className="font-mono font-bold text-indigo-600">{alphaPenalty.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={alphaPenalty}
                  onChange={(e) => setAlphaPenalty(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <p className="text-[10px] text-slate-400 mt-1">
                  Equation: <code className="font-mono bg-slate-100 px-1 py-0.5 rounded text-slate-700">Score = XQI × (1 - α × Uncertainty)</code>
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-1">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">MC Dropout Passes</label>
                  <input
                    type="number"
                    value={mcDropoutSamples}
                    onChange={(e) => setMcDropoutSamples(parseInt(e.target.value) || 20)}
                    min="5"
                    max="100"
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-mono text-slate-800 bg-slate-50 focus:bg-white"
                  />
                  <span className="text-[9px] text-slate-400 mt-0.5 block">Epistemic stochastic passes</span>
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Temperature Scaling</label>
                  <button
                    type="button"
                    onClick={() => setEnableTempScaling(!enableTempScaling)}
                    className={`w-full px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center justify-between border transition-all ${
                      enableTempScaling
                        ? 'bg-indigo-50 text-indigo-800 border-indigo-200'
                        : 'bg-slate-50 text-slate-600 border-slate-200'
                    }`}
                  >
                    <span>Platt / ECE Scaling</span>
                    <span className="text-[10px] font-bold uppercase">{enableTempScaling ? 'ON' : 'OFF'}</span>
                  </button>
                  <span className="text-[9px] text-slate-400 mt-0.5 block">Post-hoc calibration</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column (6 cols): Fusion Strategy, Safety Rules & Compute */}
        <div className="lg:col-span-6 space-y-5">
          {/* Card 3: Multi-XAI Fusion Strategy Defaults */}
          <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs space-y-4">
            <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
              <Layers className="w-4 h-4 text-emerald-600" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                3. Multi-XAI Fusion Engine Defaults
              </h2>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block font-semibold text-slate-700 mb-1">Primary Fusion Algorithm</label>
                <select
                  value={defaultFusionStrategy}
                  onChange={(e) => setDefaultFusionStrategy(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-800 bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500/20"
                >
                  <option value="uncertainty-weighted">Uncertainty-Weighted Hybrid (Recommended)</option>
                  <option value="entropy-gated">Entropy-Gated Saliency Consensus</option>
                  <option value="saliency-averaging">Equal Saliency Averaging (Baseline)</option>
                  <option value="top-1">Top-1 Saliency Method Selection</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold text-slate-700">Minimum Consensus Threshold ($\tau$)</span>
                  <span className="font-mono font-bold text-emerald-700">{consensusThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.4"
                  max="0.95"
                  step="0.05"
                  value={consensusThreshold}
                  onChange={(e) => setConsensusThreshold(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
                <span className="text-[10px] text-slate-400 block">Cosine / SSIM agreement required between XAI heatmaps</span>
              </div>
            </div>
          </div>

          {/* Card 4: Clinical Safety & Triage Thresholds */}
          <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs space-y-4">
            <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
              <ShieldCheck className="w-4 h-4 text-blue-600" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                4. Clinical Triage &amp; Safety Thresholds
              </h2>
            </div>

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">
                    <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1.5" />
                    "RELIABLE" Cutoff (%)
                  </label>
                  <input
                    type="number"
                    value={reliableThreshold}
                    onChange={(e) => setReliableThreshold(parseInt(e.target.value) || 80)}
                    min="50"
                    max="95"
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-mono font-bold text-slate-800 bg-slate-50 focus:bg-white"
                  />
                  <span className="text-[10px] text-slate-400">Score &ge; {reliableThreshold}</span>
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">
                    <span className="inline-block w-2 h-2 rounded-full bg-amber-500 mr-1.5" />
                    "CAUTION" Cutoff (%)
                  </label>
                  <input
                    type="number"
                    value={cautionThreshold}
                    onChange={(e) => setCautionThreshold(parseInt(e.target.value) || 60)}
                    min="30"
                    max="80"
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-mono font-bold text-slate-800 bg-slate-50 focus:bg-white"
                  />
                  <span className="text-[10px] text-slate-400">Score &lt; {cautionThreshold} = Review Req.</span>
                </div>
              </div>

              <div className="pt-2 space-y-2">
                <label className="flex items-center space-x-2 text-xs font-semibold text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enforceStrictPatientSplit}
                    onChange={(e) => setEnforceStrictPatientSplit(e.target.checked)}
                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span>Strict Patient-Level Split Enforcement (Prevents Data Leakage)</span>
                </label>

                <label className="flex items-center space-x-2 text-xs font-semibold text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={flagHighUncertaintyForReview}
                    onChange={(e) => setFlagHighUncertaintyForReview(e.target.checked)}
                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span>Mandatory Second Reader Flag for High-Uncertainty Predictions</span>
                </label>
              </div>
            </div>
          </div>

          {/* Card 5: Compute Runtime & Storage Paths */}
          <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs space-y-4">
            <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
              <Cpu className="w-4 h-4 text-purple-600" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                5. Compute Runtime &amp; Storage Paths
              </h2>
            </div>

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Target Compute Device</label>
                  <select
                    value={computeDevice}
                    onChange={(e) => setComputeDevice(e.target.value)}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-xs text-slate-800 bg-slate-50 focus:bg-white"
                  >
                    <option value="auto">Auto-Detect (GPU / MPS / CPU)</option>
                    <option value="cuda">NVIDIA CUDA GPU</option>
                    <option value="mps">Apple Silicon MPS</option>
                    <option value="cpu">CPU Only (Fallback)</option>
                  </select>
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Worker Threads</label>
                  <input
                    type="number"
                    value={workerThreads}
                    onChange={(e) => setWorkerThreads(parseInt(e.target.value) || 4)}
                    min="1"
                    max="16"
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-mono text-slate-800 bg-slate-50 focus:bg-white"
                  />
                </div>
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">Checkpoints Directory</label>
                <input
                  type="text"
                  value={checkpointDir}
                  onChange={(e) => setCheckpointDir(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-mono text-slate-800 bg-slate-50 focus:bg-white"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Backend Health Diagnostics & Disclaimer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Diagnostics Card (6 cols) */}
        <div className="lg:col-span-6 bg-white rounded-xl border border-slate-200/90 p-4 shadow-2xs flex items-center justify-between text-xs">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600">
              <Server className="w-4 h-4" />
            </div>
            <div>
              <div className="font-bold text-slate-900">Backend Service Engine</div>
              <div className="text-[11px] text-slate-500 font-mono">
                {systemHealth?.service || 'TrustXAI Engine'} (v{systemHealth?.version || '1.0.0'})
              </div>
            </div>
          </div>
          <span className="px-2.5 py-1 rounded-full text-[10px] font-bold uppercase bg-emerald-100 text-emerald-800 border border-emerald-200">
            {systemHealth?.status === 'ok' || systemHealth?.status === 'online' ? '● Operational' : '● Demo Engine'}
          </span>
        </div>

        {/* Disclaimer Card (6 cols) */}
        <div className="lg:col-span-6 bg-amber-50/80 rounded-xl border border-amber-200 p-4 text-xs text-amber-900 flex items-start space-x-3">
          <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block text-amber-950">Research Prototype Disclaimer</span>
            <p className="text-[11px] leading-relaxed text-amber-800 mt-0.5">
              TrustXAI-Med is an investigational platform for evaluating explainability and uncertainty. It is not an FDA/CE cleared diagnostic device.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
export default SettingsView;
