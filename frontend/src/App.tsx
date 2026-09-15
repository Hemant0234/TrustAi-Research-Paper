import React, { useState, useEffect } from 'react';
import { Sidebar, NavView } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { OverviewView } from './components/views/OverviewView';
import { AnalyzeCaseView } from './components/views/AnalyzeCaseView';
import { ExplanationLabView } from './components/views/ExplanationLabView';
import { FusionLabView } from './components/views/FusionLabView';
import { ReliabilityView } from './components/views/ReliabilityView';
import { RobustnessView } from './components/views/RobustnessView';
import { ValidationView } from './components/views/ValidationView';
import { ExperimentsView } from './components/views/ExperimentsView';
import { ClinicalStudyView } from './components/views/ClinicalStudyView';
import { ReportsView } from './components/views/ReportsView';
import { DatasetsView } from './components/views/DatasetsView';
import { ModelsView } from './components/views/ModelsView';
import { SettingsView } from './components/views/SettingsView';
import { api } from './lib/api';
import { CaseAnalysis, CaseSummary } from './types';
import { Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<NavView>('overview');
  const [selectedCaseId, setSelectedCaseId] = useState<string>('');
  const [caseData, setCaseData] = useState<CaseAnalysis | null>(null);
  const [caseSummaries, setCaseSummaries] = useState<CaseSummary[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Load cases catalog from live API
  useEffect(() => {
    async function loadCatalog() {
      try {
        const summaries = await api.getCases();
        setCaseSummaries(summaries);

        if (summaries.length > 0) {
          setSelectedCaseId((prev) => (prev && summaries.some((c) => c.case_id === prev) ? prev : summaries[0].case_id));
        }
      } catch (err) {
        console.warn('Live catalog load error, attempting retry:', err);
      }
    }
    loadCatalog();
  }, []);

  const uploadedCasesRef = React.useRef<Record<string, CaseAnalysis>>({});

  // Load active case details
  useEffect(() => {
    async function loadCaseDetails() {
      if (!selectedCaseId) return;

      if (selectedCaseId.startsWith('UPLOAD-')) {
        const cached = uploadedCasesRef.current[selectedCaseId];
        if (cached) {
          setCaseData(cached);
        }
        return;
      }

      setIsLoading(true);
      try {
        const data = await api.getCaseById(selectedCaseId);
        setCaseData(data);
      } catch (err) {
        console.error('Failed to load case data:', err);
      } finally {
        setIsLoading(false);
      }
    }
    loadCaseDetails();
  }, [selectedCaseId]);

  const handleSelectCase = (caseId: string) => {
    setSelectedCaseId(caseId);
    setActiveView('analyze');
  };

  const handleRecalculateFusion = async (weights: Record<string, number>) => {
    if (!caseData) return;
    try {
      const updated = await api.recalculateFusion(caseData.case_id, weights);
      setCaseData(updated);
    } catch (err) {
      console.error(err);
    }
  };

  const handleRecalculateXQI = async (weights: Record<string, number>) => {
    if (!caseData) return;
    try {
      const savedSettings = JSON.parse(localStorage.getItem('trustxai_settings') || '{}');
      const alpha = savedSettings.alphaPenalty !== undefined ? savedSettings.alphaPenalty : 0.50;
      const updated = await api.recalculateXQI(caseData.case_id, weights, alpha);
      if (updated.xqi && updated.reliability) {
        setCaseData((prev) => (prev ? { ...prev, xqi: updated.xqi, reliability: updated.reliability } : null));
      } else if (updated.case_id) {
        setCaseData(updated);
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#f8fafc] text-slate-800 font-sans antialiased">
      {/* Sidebar Navigation matching reference */}
      <Sidebar activeView={activeView} onSelectView={setActiveView} />

      {/* Main Content Workspace */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header Bar */}
        <Header
          selectedCaseId={selectedCaseId}
          onSelectCase={setSelectedCaseId}
          cases={caseSummaries}
        />

        {/* Scrollable View Area */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 custom-scrollbar">
          {isLoading && !caseData ? (
            <div className="h-full flex flex-col items-center justify-center space-y-3">
              <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
              <p className="text-xs text-slate-500 font-medium">Loading clinical case analysis...</p>
            </div>
          ) : (
            <>
              {activeView === 'overview' && (
                <OverviewView
                  cases={caseSummaries}
                  onSelectCase={handleSelectCase}
                  onNavigateToAnalyze={() => setActiveView('analyze')}
                />
              )}

              {activeView === 'analyze' && caseData && (
                <AnalyzeCaseView
                  currentCase={caseData}
                  allCaseSummaries={caseSummaries}
                  onSelectCase={setSelectedCaseId}
                  onRecalculateFusion={handleRecalculateFusion}
                  onRecalculateXQI={handleRecalculateXQI}
                  onNavigateToReports={() => setActiveView('reports')}
                  onCaseUploaded={(newCase) => {
                    uploadedCasesRef.current[newCase.case_id] = newCase;
                    setCaseData(newCase);
                    setCaseSummaries(prev => {
                      const exists = prev.find(c => c.case_id === newCase.case_id);
                      if (exists) return prev;
                      return [{
                        case_id: newCase.case_id,
                        modality: newCase.modality,
                        dataset: newCase.dataset,
                        model_name: newCase.model_name,
                        predicted_label: newCase.prediction.label,
                        confidence: newCase.prediction.probability,
                        uncertainty_level: newCase.uncertainty.level,
                        uncertainty_score: newCase.uncertainty.score,
                        xqi_score: newCase.xqi.overall,
                        reliability_score: newCase.reliability.score,
                        reliability_level: newCase.reliability.level,
                        overall_agreement: newCase.fusion.overall_agreement,
                        is_demo: newCase.is_demo
                      }, ...prev];
                    });
                    setSelectedCaseId(newCase.case_id);
                  }}
                />
              )}

              {activeView === 'explanation-lab' && caseData && (
                <ExplanationLabView
                  currentCase={caseData}
                  onRecalculateFusion={handleRecalculateFusion}
                />
              )}

              {activeView === 'fusion-lab' && caseData && (
                <FusionLabView
                  currentCase={caseData}
                  onRecalculateFusion={handleRecalculateFusion}
                />
              )}

              {activeView === 'reliability' && caseData && (
                <ReliabilityView
                  currentCase={caseData}
                  onRecalculateXQI={handleRecalculateXQI}
                />
              )}

              {activeView === 'robustness' && caseData && (
                <RobustnessView currentCase={caseData} />
              )}

              {activeView === 'validation' && <ValidationView />}

              {activeView === 'experiments' && <ExperimentsView />}

              {activeView === 'clinical-study' && <ClinicalStudyView />}

              {activeView === 'reports' && caseData && (
                <ReportsView currentCase={caseData} />
              )}

              {activeView === 'datasets' && <DatasetsView />}

              {activeView === 'models' && <ModelsView />}

              {activeView === 'settings' && <SettingsView />}
            </>
          )}
        </main>
      </div>
    </div>
  );
};
export default App;
