import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import type { PipelineStage } from "@/data/pipelineStages";
import type { DatasetSummary, MetricsResponse } from "@/lib/api";
import { getDatasetPreview } from "@/lib/api";
import { StageVisualization } from "./StageVisualizations";

interface StageDetailPanelProps {
  stage: PipelineStage | null;
  stageResult: Record<string, unknown> | null;
  datasetSummary: DatasetSummary | null;
  metrics: MetricsResponse | null;
  stageLogs: string[];
  onClose: () => void;
}

const StageDetailPanel = ({
  stage,
  stageResult,
  datasetSummary,
  metrics,
  stageLogs,
  onClose,
}: StageDetailPanelProps) => {
  const [datasetPreview, setDatasetPreview] = useState<{ rows: Array<Record<string, unknown>>; columns: string[] } | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const highlights = stage ? buildHighlights(stage.id, stageResult, datasetSummary, metrics) : [];
  const isPreprocessing = stage?.id === "preprocessing";
  const aboutText = useMemo(() => {
    if (!stage) return "";
    if (isPreprocessing) {
      return "This step prepares your dataset so it can be used by machine learning models. Missing values are filled in, categories are converted into numbers, numeric values are scaled, and the data is split into training and testing sets to evaluate performance.";
    }
    return stage.details;
  }, [isPreprocessing, stage]);

  useEffect(() => {
    if (!isPreprocessing || !stage) return;
    setIsLoadingPreview(true);
    void getDatasetPreview(5)
      .then(setDatasetPreview)
      .catch(() => setDatasetPreview(null))
      .finally(() => setIsLoadingPreview(false));
  }, [isPreprocessing, stage]);

  return (
    <AnimatePresence>
      {stage && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-background/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.div
            className="glass-card fixed right-0 top-0 z-50 h-full w-full max-w-lg overflow-y-auto border-l border-border/50 scrollbar-thin"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
          >
            <div className="space-y-6 p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{stage.icon}</span>
                  <div>
                    <h2 className="text-xl font-semibold text-foreground">{stage.label}</h2>
                    <p className="text-sm text-muted-foreground">{stage.description}</p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="space-y-2">
                <h3 className="text-sm font-medium text-accent">About this stage</h3>
                <p className="text-sm leading-relaxed text-secondary-foreground">{aboutText}</p>
              </div>

              {highlights.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-sm font-medium text-accent">Stage highlights</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {highlights.map((highlight) => (
                      <div key={highlight.label} className="glass-card space-y-1 p-3">
                        <p className="text-[10px] text-muted-foreground">{highlight.label}</p>
                        <p className="font-mono text-sm text-foreground">{highlight.value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="space-y-2">
                <h3 className="text-sm font-medium text-accent">{isPreprocessing ? "Dataset overview" : "Visualization"}</h3>
                <div className="glass-card p-4">
                  <StageVisualization
                    stage={stage}
                    stageResult={stageResult}
                    datasetSummary={datasetSummary}
                    metrics={metrics}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <h3 className="text-sm font-medium text-accent">
                  {isPreprocessing ? "How we prepared your data" : "Stage logs"}
                </h3>
                <div className="glass-card space-y-2 p-4 text-sm leading-relaxed text-secondary-foreground">
                  {isPreprocessing ? (
                    <PreprocessExplanation stageResult={stageResult} datasetSummary={datasetSummary} />
                  ) : (
                    <div className="max-h-48 space-y-2 overflow-y-auto font-mono text-[11px] scrollbar-thin">
                      {stageLogs.length > 0 ? (
                        stageLogs.map((log, index) => (
                          <p key={`${stage.id}-${index}`} className="whitespace-pre-wrap leading-relaxed text-foreground/75">
                            {log}
                          </p>
                        ))
                      ) : (
                        <p className="text-muted-foreground">No logs recorded for this stage yet.</p>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {isPreprocessing && (
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-accent">Dataset preview</h3>
                  <div className="glass-card overflow-auto p-3">
                    {isLoadingPreview && <p className="text-muted-foreground text-sm">Loading sample rows...</p>}
                    {!isLoadingPreview && datasetPreview && datasetPreview.rows.length > 0 ? (
                      <table className="min-w-full text-left text-[11px]">
                        <thead className="text-muted-foreground">
                          <tr>
                            {datasetPreview.columns.map((col) => (
                              <th key={col} className="px-2 py-1 font-medium">
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {datasetPreview.rows.map((row, idx) => (
                            <tr key={idx} className="border-b border-border/50">
                              {datasetPreview.columns.map((col) => (
                                <td key={col} className="px-2 py-1">
                                  {String(row[col] ?? "")}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      !isLoadingPreview && <p className="text-muted-foreground text-sm">Preview not available.</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

const buildHighlights = (
  stageId: string,
  stageResult: Record<string, unknown> | null,
  datasetSummary: DatasetSummary | null,
  metrics: MetricsResponse | null,
) => {
  switch (stageId) {
    case "analysis":
      return [
        { label: "Rows", value: String(stageResult?.row_count ?? datasetSummary?.rows ?? "Pending") },
        { label: "Features", value: String(stageResult?.feature_count ?? "Pending") },
        {
          label: "Recommendations",
          value: String(((stageResult?.recommendations as string[] | undefined) || []).length || "Pending"),
        },
      ];
    case "preprocessing":
      return [
        { label: "Training data", value: `${stageResult?.train_size ?? "Pending"} rows` },
        { label: "Evaluation data", value: `${stageResult?.test_size ?? "Pending"} rows` },
        { label: "Model features", value: `${stageResult?.feature_count ?? "Pending"} columns used` },
      ];
    case "features":
      return [
        { label: "Selected", value: String(stageResult?.final_feature_count ?? "Pending") },
        {
          label: "Scored features",
          value: String(Object.keys((stageResult?.feature_scores as Record<string, number> | undefined) || {}).length || 0),
        },
        {
          label: "PCA",
          value: stageResult?.pca_result ? "Enabled" : "Not used",
        },
      ];
    case "training":
      return [
        { label: "Model", value: String(stageResult?.model_name ?? metrics?.model_name ?? "Pending") },
        { label: "Best CV", value: formatMetric(stageResult?.best_score as number | undefined, false) },
        { label: "Test score", value: formatMetric(stageResult?.test_score as number | undefined, false) },
      ];
    case "loss":
      return [
        { label: "Best epoch", value: String(stageResult?.best_epoch ?? "Pending") },
        {
          label: "Final train loss",
          value: formatLast(stageResult?.train_loss as number[] | undefined),
        },
        {
          label: "Final val loss",
          value: formatLast(stageResult?.val_loss as number[] | undefined),
        },
      ];
    case "evaluation":
      return [
        {
          label: metrics?.task_type === "regression" ? "R2" : "Accuracy",
          value:
            metrics?.task_type === "regression"
              ? formatMetric(metrics?.r2, false)
              : formatMetric(metrics?.accuracy, true),
        },
        {
          label: metrics?.task_type === "regression" ? "RMSE" : "F1",
          value:
            metrics?.task_type === "regression"
              ? formatMetric(metrics?.rmse, false)
              : formatMetric(metrics?.f1, true),
        },
        { label: "Decision", value: String(metrics?.deployment_decision ?? "Pending") },
      ];
    case "results":
      return [
        { label: "Pipeline ID", value: String(stageResult?.pipeline_id ?? "Pending") },
        { label: "Saved model", value: String(stageResult?.model_path ?? "Pending") },
        { label: "Status", value: stageResult?.deployment_success ? "Exported" : "Pending" },
      ];
    default:
      return [];
  }
};

const formatMetric = (value: number | null | undefined, asPercent = false) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "Pending";
  return asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
};

const formatLast = (values?: number[]) => {
  if (!values || values.length === 0) return "Pending";
  return values[values.length - 1].toFixed(3);
};

const PreprocessExplanation = ({
  stageResult,
  datasetSummary,
}: {
  stageResult: Record<string, unknown> | null;
  datasetSummary: DatasetSummary | null;
}) => {
  const summary = (stageResult?.explanation as string | undefined) || "We filled missing values, simplified categories, and prepared numeric columns so the model can learn reliably.";
  const train = stageResult?.train_size as number | undefined;
  const test = stageResult?.test_size as number | undefined;
  const total = datasetSummary?.rows;
  const integrityOK = typeof train === "number" && typeof test === "number" && typeof total === "number" && train + test === total;

  return (
    <div className="space-y-2">
      <p className="text-foreground">{summary}</p>
      <ul className="ml-4 list-disc space-y-1 text-sm text-secondary-foreground">
        <li>Any missing values were filled in automatically.</li>
        <li>Columns with categories were converted into numbers so the model can use them.</li>
        <li>Numeric features were standardized so large ranges don&apos;t dominate the model.</li>
      </ul>
      {(train || test) && (
        <p className="text-sm text-secondary-foreground">
          Finally, the data was split into {train ? train.toLocaleString() : "?"} rows for training and{" "}
          {test ? test.toLocaleString() : "?"} rows to test how well the model performs on new data.
        </p>
      )}
      {integrityOK && <p className="text-sm font-medium text-primary">✔ Dataset integrity preserved (0 rows dropped)</p>}
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
        {["Raw data", "Clean missing values", "Convert categories", "Prepare numeric columns", "Train/Test split"].map(
          (step, index, arr) => (
            <div key={step} className="flex items-center gap-1">
              <span className="rounded bg-secondary px-2 py-1">{step}</span>
              {index < arr.length - 1 && <span className="text-muted-foreground">→</span>}
            </div>
          ),
        )}
      </div>
    </div>
  );
};

export default StageDetailPanel;
