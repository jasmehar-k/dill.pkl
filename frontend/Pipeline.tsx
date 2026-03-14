import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { stages, type PipelineStage, type StageStatus } from "@/data/pipelineStages";
import {
  getDatasetColumns,
  getDatasetSummary,
  getMetrics,
  getPipelineLogs,
  getPipelineStatus,
  getStageResult,
  runPipelineStage,
  setTargetColumn,
  uploadDataset,
  type DatasetColumn,
  type DatasetSummary,
  type MetricsResponse,
  type PipelineConfig,
  type PipelineStatusResponse,
  type TaskType,
} from "@/lib/api";
import ChatBot from "./ChatBot";
import DatasetSummaryCard from "./DatasetSummary";
import DatasetUpload from "./DatasetUpload";
import NeuralBackground from "./NeuralBackground";
import PipelineConnector from "./PipelineConnector";
import PipelineHeader from "./PipelineHeader";
import PipelineLogs from "./PipelineLogs";
import PipelineNode from "./PipelineNode";
import PredictionTarget from "./PredictionTarget";
import ResultsPanel from "./ResultsPanel";
import StageDetailPanel from "./StageDetailPanel";

const STAGE_ORDER = stages.map((stage) => stage.id);
const DEFAULT_CONFIG = {
  test_size: 0.2,
  random_state: 42,
} satisfies Omit<PipelineConfig, "task_type">;

const createInitialStatuses = (): Record<string, StageStatus> =>
  Object.fromEntries(STAGE_ORDER.map((stageId) => [stageId, "waiting"])) as Record<string, StageStatus>;

const createEmptyLogs = (): Record<string, string[]> =>
  Object.fromEntries(STAGE_ORDER.map((stageId) => [stageId, []]));

const Pipeline = () => {
  const [selectedStage, setSelectedStage] = useState<PipelineStage | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [datasetColumns, setDatasetColumns] = useState<DatasetColumn[]>([]);
  const [pipelineStatus, setPipelineStatus] = useState<Record<string, StageStatus>>(createInitialStatuses);
  const [stageLogs, setStageLogs] = useState<Record<string, string[]>>(createEmptyLogs);
  const [stageResults, setStageResults] = useState<Record<string, Record<string, unknown>>>({});
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [selectedColumn, setSelectedColumnState] = useState<string | null>(null);
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [isUploading, setIsUploading] = useState(false);
  const [isSavingTarget, setIsSavingTarget] = useState(false);
  const [isRunningPipeline, setIsRunningPipeline] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const syncStageResults = useCallback(async (statuses: Record<string, StageStatus>) => {
    const completedStages = STAGE_ORDER.filter((stageId) => statuses[stageId] === "completed");
    if (completedStages.length === 0) {
      setStageResults({});
      return;
    }

    const responses = await Promise.all(
      completedStages.map(async (stageId) => {
        try {
          const response = await getStageResult(stageId);
          return [stageId, response.result || {}] as const;
        } catch {
          return [stageId, {}] as const;
        }
      }),
    );

    setStageResults(Object.fromEntries(responses));
  }, []);

  const refreshLogs = useCallback(async () => {
    try {
      const logs = await getPipelineLogs();
      setStageLogs({ ...createEmptyLogs(), ...logs.logs });
    } catch {
      setStageLogs(createEmptyLogs());
    }
  }, []);

  const refreshPipelineData = useCallback(async () => {
    let statusResponse: PipelineStatusResponse;

    try {
      statusResponse = await getPipelineStatus();
    } catch (err) {
      throw err instanceof Error ? err : new Error("Unable to reach the backend API.");
    }

    const normalizedStages = { ...createInitialStatuses(), ...statusResponse.stages };
    setPipelineStatus(normalizedStages);
    setSelectedColumnState(statusResponse.target_column);

    if (!statusResponse.dataset_loaded) {
      setDatasetSummary(null);
      setDatasetColumns([]);
      setStageLogs(createEmptyLogs());
      setStageResults({});
      setMetrics(null);
      return;
    }

    const [summary, columnResponse] = await Promise.all([getDatasetSummary(), getDatasetColumns()]);
    setDatasetSummary(summary);
    setDatasetColumns(columnResponse.columns);

    if (statusResponse.target_column) {
      setTaskType(inferTaskType(columnResponse.columns, statusResponse.target_column));
    }

    await refreshLogs();
    await syncStageResults(normalizedStages);

    if (normalizedStages.evaluation === "completed" || normalizedStages.results === "completed") {
      try {
        setMetrics(await getMetrics());
      } catch {
        setMetrics(null);
      }
    } else {
      setMetrics(null);
    }
  }, [refreshLogs, syncStageResults]);

  useEffect(() => {
    void refreshPipelineData().catch((err) => {
      const message = err instanceof Error ? err.message : "Unable to load pipeline state.";
      if (!message.includes("No dataset uploaded")) {
        setError(message);
      }
    });
  }, [refreshPipelineData]);

  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);

    try {
      await uploadDataset(file);
      setMetrics(null);
      setStageResults({});
      setStageLogs(createEmptyLogs());
      await refreshPipelineData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset upload failed.");
    } finally {
      setIsUploading(false);
    }
  }, [refreshPipelineData]);

  const handleTargetSelect = useCallback(async (column: string) => {
    setIsSavingTarget(true);
    setError(null);

    try {
      await setTargetColumn(column);
      setSelectedColumnState(column);
      setTaskType(inferTaskType(datasetColumns, column));
      await refreshPipelineData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to set target column.");
    } finally {
      setIsSavingTarget(false);
    }
  }, [datasetColumns, refreshPipelineData]);

  const handleAutoDetectTarget = useCallback(() => {
    const suggestion = suggestTargetColumn(datasetColumns);
    if (suggestion) {
      void handleTargetSelect(suggestion);
    }
  }, [datasetColumns, handleTargetSelect]);

  const handleRunPipeline = useCallback(async () => {
    if (!datasetSummary || !selectedColumn) {
      setError("Upload a dataset and select a target column before running the pipeline.");
      return;
    }

    const config: PipelineConfig = {
      task_type: taskType,
      ...DEFAULT_CONFIG,
    };

    setIsRunningPipeline(true);
    setError(null);
    setPipelineStatus(createInitialStatuses());
    setStageResults({});
    setMetrics(null);
    setStageLogs(createEmptyLogs());

    for (let index = 0; index < STAGE_ORDER.length; index += 1) {
      const stageId = STAGE_ORDER[index];

      setPipelineStatus((current) => ({
        ...current,
        [stageId]: "running",
      }));

      try {
        const response = await runPipelineStage(stageId, config);
        setPipelineStatus((current) => ({
          ...current,
          [stageId]: response.status,
        }));

        if (response.result) {
          setStageResults((current) => ({
            ...current,
            [stageId]: response.result,
          }));
        }

        await refreshLogs();

        if (stageId === "evaluation" || stageId === "results") {
          try {
            setMetrics(await getMetrics());
          } catch {
            setMetrics(null);
          }
        }
      } catch (err) {
        setPipelineStatus((current) => ({
          ...current,
          [stageId]: "failed",
        }));
        await refreshLogs();
        setError(err instanceof Error ? err.message : `The ${stageId} stage failed.`);
        break;
      }
    }

    await refreshPipelineData().catch(() => {
      // Keep the UI state we already have if the final refresh fails.
    });
    setIsRunningPipeline(false);
  }, [datasetSummary, refreshLogs, refreshPipelineData, selectedColumn, taskType]);

  const completedCount = STAGE_ORDER.filter((stageId) => pipelineStatus[stageId] === "completed").length;
  const progress = (completedCount / stages.length) * 100;
  const activeStageId = STAGE_ORDER.find((stageId) => pipelineStatus[stageId] === "running") || null;
  const completedStageIds = STAGE_ORDER.filter((stageId) => pipelineStatus[stageId] === "completed");
  const isComplete = completedCount === stages.length;
  const modelName =
    (metrics?.model_name as string | null | undefined) ||
    (stageResults.training?.model_name as string | undefined) ||
    null;
  const canRun = Boolean(datasetSummary && selectedColumn && !isUploading && !isSavingTarget);

  return (
    <>
      <NeuralBackground />
      <div className="relative z-10 mx-auto w-full max-w-6xl space-y-5 px-4 py-8">
        <PipelineHeader
          completedCount={completedCount}
          totalStages={stages.length}
          progress={progress}
          datasetName={datasetSummary?.filename || null}
          targetColumn={selectedColumn}
          modelName={modelName}
          metrics={metrics}
          taskType={taskType}
          canRun={canRun}
          isRunning={isRunningPipeline}
          onRun={handleRunPipeline}
          onTaskTypeChange={setTaskType}
        />

        {error && (
          <motion.div
            className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {error}
          </motion.div>
        )}

        <DatasetUpload
          fileName={datasetSummary?.filename || null}
          isUploading={isUploading}
          error={null}
          onUpload={handleUpload}
        />

        <PredictionTarget
          datasetLoaded={Boolean(datasetSummary)}
          columns={datasetColumns}
          selectedColumn={selectedColumn}
          isSaving={isSavingTarget}
          onSelect={handleTargetSelect}
          onAutoDetect={handleAutoDetectTarget}
        />

        <DatasetSummaryCard
          summary={datasetSummary}
          columns={datasetColumns}
          targetColumn={selectedColumn}
        />

        <div className="flex items-start justify-center overflow-x-auto pb-4 pt-2">
          {stages.map((stage, index) => (
            <div key={stage.id} className="flex items-start">
              <PipelineNode
                stage={stage}
                status={pipelineStatus[stage.id] || "waiting"}
                index={index}
                onClick={() => setSelectedStage(stage)}
              />
              {index < stages.length - 1 && (
                <PipelineConnector
                  completed={pipelineStatus[stage.id] === "completed"}
                  running={pipelineStatus[stages[index + 1].id] === "running"}
                />
              )}
            </div>
          ))}
        </div>

        {(completedCount > 0 || activeStageId) && (
          <motion.p
            className="text-center text-xs text-muted-foreground"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            Hover for stage context. Click completed or failed stages for details.
          </motion.p>
        )}

        <PipelineLogs
          activeStageId={activeStageId}
          completedStageIds={completedStageIds}
          stageLogs={stageLogs}
          stageResults={stageResults}
          metrics={metrics}
        />

        <ResultsPanel
          isComplete={isComplete}
          metrics={metrics}
          results={stageResults.results || null}
        />

        <StageDetailPanel
          stage={selectedStage}
          stageResult={selectedStage ? stageResults[selectedStage.id] || null : null}
          datasetSummary={datasetSummary}
          metrics={metrics}
          stageLogs={selectedStage ? stageLogs[selectedStage.id] || [] : []}
          onClose={() => setSelectedStage(null)}
        />

        <ChatBot
          datasetName={datasetSummary?.filename || null}
          targetColumn={selectedColumn}
          taskType={taskType}
          activeStageId={activeStageId}
          stageLogs={stageLogs}
          metrics={metrics}
        />
      </div>
    </>
  );
};

const suggestTargetColumn = (columns: DatasetColumn[]) => {
  if (columns.length === 0) return null;

  const preferredNames = ["target", "label", "class", "outcome", "y", "price", "sale_price"];
  const preferred = columns.find((column) =>
    preferredNames.some((name) => column.name.toLowerCase() === name || column.name.toLowerCase().includes(name)),
  );

  return preferred?.name || columns[columns.length - 1]?.name || null;
};

const inferTaskType = (columns: DatasetColumn[], selectedColumn: string): TaskType => {
  const column = columns.find((entry) => entry.name === selectedColumn);
  return column?.is_numeric ? "regression" : "classification";
};

export default Pipeline;
