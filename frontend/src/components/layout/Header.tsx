import React from 'react';
import { Shield, Activity, Database, Cpu, AlertTriangle, Sparkles } from 'lucide-react';
import { CaseSummary } from '../../types';

interface HeaderProps {
  selectedCaseId?: string;
  onSelectCase?: (caseId: string) => void;
  cases?: CaseSummary[];
}

export const Header: React.FC<HeaderProps> = ({
  selectedCaseId = '',
  onSelectCase,
  cases = []
}) => {
  return (
    <header className="bg-white border-b border-slate-200/90 px-6 py-2.5 flex items-center justify-between select-none">
      {/* Left Title & Status */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2">
          <span className="font-extrabold text-sm text-slate-900 tracking-tight font-mono">
            TRUSTXAI-MED
          </span>
          <span className="text-[10px] px-2 py-0.5 font-bold rounded bg-blue-50 text-blue-700 border border-blue-200">
            v1.0.0
          </span>
          <span className="text-[10px] px-2 py-0.5 font-bold rounded bg-blue-50 text-blue-700 border border-blue-200 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-pulse" /> Live Inference Active
          </span>
        </div>
      </div>

      {/* Right Telemetry info */}
      <div className="flex items-center space-x-4 text-xs text-slate-600">
        <div className="hidden md:flex items-center space-x-3 text-[11px] text-slate-500 font-medium">
          <span>Model: <strong>DenseNet-121</strong></span>
          <span>•</span>
          <span>Benchmark: <strong>CheXpert / CheXlocalize</strong></span>
        </div>

        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-100 border border-slate-200 text-[10px] text-slate-600 font-medium">
          <AlertTriangle className="w-3 h-3 text-slate-500" />
          <span>Research Prototype • Not for Clinical Diagnosis</span>
        </div>
      </div>
    </header>
  );
};
