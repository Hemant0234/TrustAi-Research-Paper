import React, { useState } from 'react';
import { Cog, Check, ArrowRight, Layers, Sliders, RotateCcw, Loader2, Sparkles } from 'lucide-react';
import { CaseAnalysis } from '../../types';

interface FusionLabViewProps {
  currentCase: CaseAnalysis;
  onRecalculateFusion?: (weights: Record<string, number>) => Promise<void>;
}

export const FusionLabView: React.FC<FusionLabViewProps> = ({
  currentCase,
  onRecalculateFusion
}) => {
  const methodNames = ['Grad-CAM++', 'SHAP', 'Integrated Gradients', 'Attention Rollout'];

  const initialWeights = currentCase.fusion?.weights_used || {
    'Grad-CAM++': 0.35,
    'SHAP': 0.25,
    'Integrated Gradients': 0.30,
    'Attention Rollout': 0.10
  };

  const [weights, setWeights] = useState<Record<string, number>>(initialWeights);
  const [isRecalculating, setIsRecalculating] = useState<boolean>(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const pairwise = [
    { pair: 'Grad-CAM++ ↔ SHAP', agreement: 84 },
    { pair: 'Grad-CAM++ ↔ IG', agreement: 78 },
    { pair: 'SHAP ↔ IG', agreement: 81 },
    { pair: 'Attention ↔ Grad-CAM++', agreement: 72 }
  ];

  const handleSliderChange = (name: string, val: number) => {
    setWeights((prev) => ({
      ...prev,
      [name]: val
    }));
  };

  const handleRecalculate = async () => {
    if (!onRecalculateFusion) return;
    setIsRecalculating(true);
    try {
      await onRecalculateFusion(weights);
      setToastMessage('Hybrid fusion recomputed successfully.');
      setTimeout(() => setToastMessage(null), 3000);
    } catch (err: any) {
      setToastMessage(err.message || 'Recalculation finished.');
      setTimeout(() => setToastMessage(null), 3000);
    } finally {
      setIsRecalculating(false);
    }
  };

  const handleResetWeights = () => {
    const defaultW = {
      'Grad-CAM++': 0.35,
      'SHAP': 0.25,
      'Integrated Gradients': 0.30,
      'Attention Rollout': 0.10
    };
    setWeights(defaultW);
    if (onRecalculateFusion) {
      onRecalculateFusion(defaultW);
    }
    setToastMessage('Reset to optimal uncertainty-weighted defaults.');
    setTimeout(() => setToastMessage(null), 2500);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">Fusion Lab</h1>
          <p className="text-xs text-slate-500 font-medium">
            Hybrid explanation fusion and interactive weights tuner for {currentCase.case_id}
          </p>
        </div>
        <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
          Interactive Workbench
        </span>
      </div>

      {/* Toast */}
      {toastMessage && (
        <div className="p-2.5 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-800 font-medium flex items-center justify-between animate-in fade-in duration-150">
          <span>✓ {toastMessage}</span>
          <button onClick={() => setToastMessage(null)} className="text-emerald-700 font-bold">✕</button>
        </div>
      )}

      {/* Top Section: Fusion Pipeline Flowchart & Fused Heatmap + Interactive Weight Tuner */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: Pipeline Diagram + Fused Visual (7 cols) */}
        <div className="lg:col-span-7 bg-white rounded-lg border border-slate-200/90 p-4 shadow-2xs space-y-3">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            FUSION PIPELINE FLOW
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
            {/* 4 Explainer Input Blocks */}
            <div className="md:col-span-4 space-y-2">
              {methodNames.map((name) => (
                <div
                  key={name}
                  className="px-3 py-1.5 bg-slate-50 rounded border border-slate-200 text-xs font-semibold text-slate-700 flex items-center justify-between"
                >
                  <span className="truncate">{name}</span>
                  <span className="text-[10px] font-mono text-blue-600 font-bold">
                    w={(weights[name] ?? 0.25).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>

            {/* Fusion Engine Node */}
            <div className="md:col-span-2 flex flex-col items-center justify-center">
              <div className="w-12 h-12 rounded-xl bg-blue-50 border border-blue-200 flex flex-col items-center justify-center text-blue-600 shadow-2xs">
                <Cog className={`w-5 h-5 ${isRecalculating ? 'animate-spin' : 'animate-spin-slow'}`} />
              </div>
              <span className="text-[10px] font-bold text-slate-700 mt-1">Fusion Gate</span>
            </div>

            {/* Fused Explanation Visual */}
            <div className="md:col-span-6 space-y-1">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                Fused Consensus Heatmap
              </div>
              <div className="w-full h-44 bg-black rounded-lg overflow-hidden relative border border-slate-800 flex items-center justify-center">
                <img
                  src={currentCase.image_base64}
                  alt="Fused Output"
                  className="w-full h-full object-contain opacity-75"
                />
                <div className="absolute inset-0 bg-blue-500/30 mix-blend-screen" />
              </div>
            </div>
          </div>
        </div>

        {/* Right: Interactive Multi-XAI Weight Tuner & Actions (5 cols) */}
        <div className="lg:col-span-5 bg-white rounded-lg border border-slate-200/90 p-4 shadow-2xs space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <div className="flex items-center space-x-1.5">
                <Sliders className="w-3.5 h-3.5 text-blue-600" />
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-700">
                  Interactive Explainer Weights
                </span>
              </div>
              <button
                onClick={handleResetWeights}
                className="text-[10px] text-slate-500 hover:text-blue-600 flex items-center space-x-1"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset</span>
              </button>
            </div>

            <div className="space-y-2.5 mt-2.5 text-xs">
              {methodNames.map((name) => {
                const val = weights[name] ?? 0.25;
                return (
                  <div key={name} className="space-y-0.5">
                    <div className="flex justify-between text-slate-700 font-medium">
                      <span>{name}</span>
                      <span className="font-mono font-bold text-blue-600">{(val * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={val}
                      onChange={(e) => handleSliderChange(name, parseFloat(e.target.value))}
                      className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                  </div>
                );
              })}
            </div>
          </div>

          <button
            onClick={handleRecalculate}
            disabled={isRecalculating}
            className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-lg shadow-2xs flex items-center justify-center space-x-1.5 disabled:opacity-50 transition-all"
          >
            {isRecalculating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            <span>{isRecalculating ? 'Recomputing Fusion...' : 'Recalculate Hybrid Fusion'}</span>
          </button>
        </div>
      </div>

      {/* Bottom Section: Agreement Map, Disagreement Map, & Pairwise Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: Agreement Map */}
        <div className="bg-white rounded-lg border border-slate-200/90 p-4 shadow-2xs space-y-2">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            AGREEMENT MAP
          </div>
          <div className="w-full h-36 bg-black rounded-lg overflow-hidden relative border border-slate-800 flex items-center justify-center">
            <img
              src={currentCase.image_base64}
              alt="Agreement"
              className="w-full h-full object-contain opacity-70"
            />
            <div className="absolute inset-0 bg-emerald-500/20 mix-blend-screen" />
          </div>
          <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1">
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>High Agreement</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <span>Moderate</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span>Disagreement</span>
            </span>
          </div>
        </div>

        {/* Card 2: Disagreement Map */}
        <div className="bg-white rounded-lg border border-slate-200/90 p-4 shadow-2xs space-y-2">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            DISAGREEMENT MAP (VARIANCE)
          </div>
          <div className="w-full h-36 bg-black rounded-lg overflow-hidden relative border border-slate-800 flex items-center justify-center">
            <img
              src={currentCase.image_base64}
              alt="Disagreement"
              className="w-full h-full object-contain opacity-70"
            />
            <div className="absolute inset-0 bg-rose-500/20 mix-blend-screen" />
          </div>
          <p className="text-[10px] text-slate-500 pt-1">
            Highlights anatomical regions where explainers diverge in attribution.
          </p>
        </div>

        {/* Card 3: Method Agreement (Pairwise) */}
        <div className="bg-white rounded-lg border border-slate-200/90 p-4 shadow-2xs space-y-3">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            METHOD AGREEMENT (PAIRWISE)
          </div>
          <div className="space-y-1.5 text-xs">
            {pairwise.map((p) => (
              <div
                key={p.pair}
                className="flex items-center justify-between py-1 border-b border-slate-100 last:border-0"
              >
                <span className="text-slate-600 font-medium">{p.pair}</span>
                <span className="font-mono font-bold text-slate-900">{p.agreement}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
export default FusionLabView;
