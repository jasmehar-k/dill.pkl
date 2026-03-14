import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { stages, PipelineStage, StageStatus } from "@/data/pipelineStages";
import PipelineNode from "./PipelineNode";
import PipelineConnector from "./PipelineConnector";
import StageDetailPanel from "./StageDetailPanel";
import PipelineHeader from "./PipelineHeader";
import PipelineLogs from "./PipelineLogs";
import DatasetUpload from "./DatasetUpload";
import PredictionTarget from "./PredictionTarget";
import DatasetSummary from "./DatasetSummary";
import ResultsPanel from "./ResultsPanel";
import ChatBot from "./ChatBot";
import NeuralBackground from "./NeuralBackground";

const STAGE_DELAY = 1800;

const Pipeline = () => {
  const [statuses, setStatuses] = useState<StageStatus[]>(stages.map(() => "waiting"));
  const [selectedStage, setSelectedStage] = useState<PipelineStage | null>(null);
  const [datasetLoaded, setDatasetLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const runPipeline = async () => {
      for (let i = 0; i < stages.length; i++) {
        if (cancelled) return;
        setStatuses((prev) => prev.map((s, j) => (j === i ? "running" : s)));
        await new Promise((r) => setTimeout(r, STAGE_DELAY));
        if (cancelled) return;
        setStatuses((prev) => prev.map((s, j) => (j === i ? "completed" : s)));
      }
    };
    const timer = setTimeout(runPipeline, 800);
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  const handleNodeClick = useCallback((stage: PipelineStage) => {
    setSelectedStage(stage);
  }, []);

  const completedCount = statuses.filter((s) => s === "completed").length;
  const progress = (completedCount / stages.length) * 100;
  const activeStageIndex = statuses.findIndex((s) => s === "running");
  const activeStageId = activeStageIndex >= 0 ? stages[activeStageIndex].id : null;
  const completedStageIds = stages.filter((_, i) => statuses[i] === "completed").map((s) => s.id);
  const isComplete = completedCount === stages.length;

  return (
    <>
      <NeuralBackground />
      <div className="relative z-10 w-full max-w-6xl mx-auto px-4 py-8 space-y-5">
        <PipelineHeader completedCount={completedCount} totalStages={stages.length} progress={progress} />

        <DatasetUpload onUpload={() => setDatasetLoaded(true)} />

        <PredictionTarget datasetLoaded={datasetLoaded} />

        <DatasetSummary datasetLoaded={datasetLoaded} />

        {/* Pipeline nodes */}
        <div className="flex items-start justify-center overflow-x-auto pb-4 pt-2">
          {stages.map((stage, i) => (
            <div key={stage.id} className="flex items-start">
              <PipelineNode
                stage={stage}
                status={statuses[i]}
                index={i}
                onClick={() => handleNodeClick(stage)}
              />
              {i < stages.length - 1 && (
                <PipelineConnector
                  completed={statuses[i] === "completed"}
                  running={statuses[i + 1] === "running"}
                />
              )}
            </div>
          ))}
        </div>

        {completedCount > 0 && (
          <motion.p
            className="text-center text-xs text-muted-foreground"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            Hover for details • Click completed stages to explore
          </motion.p>
        )}

        <PipelineLogs activeStageId={activeStageId} completedStageIds={completedStageIds} />

        <ResultsPanel isComplete={isComplete} />

        <StageDetailPanel stage={selectedStage} onClose={() => setSelectedStage(null)} />
        <ChatBot />
      </div>
    </>
  );
};

export default Pipeline;
