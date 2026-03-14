import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import type { PipelineStage } from "@/data/pipelineStages";
import type { DatasetSummary, MetricsResponse } from "@/lib/api";
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
  const highlights = stage ? buildHighlights(stage.id, stageResult, datasetSummary, metrics) : [];

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
                <p className="text-sm leading-relaxed text-secondary-foreground">{stage.details}</p>
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
                <h3 className="text-sm font-medium text-accent">Visualization</h3>
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
                <h3 className="text-sm font-medium text-accent">Stage logs</h3>
                <div className="glass-card max-h-48 space-y-2 overflow-y-auto p-4 font-mono text-[11px] scrollbar-thin">
                  {stageLogs.length > 0 ? (
                    stageLogs.map((log, index) => (
                      <p
                        key={`${stage.id}-${index}`}
                        className={`whitespace-pre-wrap leading-relaxed ${
                          log.includes(" summary:")
                            ? "text-accent"
                            : log.includes(" overall:")
                              ? "text-primary"
                              : "text-foreground/75"
                        }`}
                      >
                        {log}
                      </p>
                    ))
                  ) : (
                    <p className="text-muted-foreground">No logs recorded for this stage yet.</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <h3 className="text-sm font-medium text-accent">Code</h3>
                <pre className="glass-card overflow-x-auto whitespace-pre-wrap p-4 font-mono text-[12px] leading-relaxed text-foreground/80">
                  {stage.codeSnippet}
                </pre>
              </div>
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
        { label: "Train size", value: String(stageResult?.train_size ?? "Pending") },
        { label: "Test size", value: String(stageResult?.test_size ?? "Pending") },
        { label: "Features", value: String(stageResult?.feature_count ?? "Pending") },
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

export default StageDetailPanel;
