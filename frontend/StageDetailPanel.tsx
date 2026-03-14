import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import type { PipelineStage } from "@/data/pipelineStages";
import type { DataQualityProfile, DatasetColumn, DatasetPreviewResponse, DatasetSummary, MetricsResponse, QualityFlag, TaskType } from "@/lib/api";
import { getDatasetPreview } from "@/lib/api";
import FeatureEngineeringDashboard from "./FeatureEngineeringDashboard";
import EvaluationDashboard from "./EvaluationDashboard";
import { StageVisualization } from "./StageVisualizations";
import DatasetSummaryCard from "./DatasetSummary";

interface StageDetailPanelProps {
  stage: PipelineStage | null;
  stageResult: Record<string, unknown> | null;
  lossStageResult?: Record<string, unknown> | null;
  datasetSummary: DatasetSummary | null;
  datasetColumns?: DatasetColumn[];
  metrics: MetricsResponse | null;
  stageLogs: string[];
  lossStageLogs?: string[];
  taskType: TaskType;
  targetColumn: string | null;
  explanation?: Record<string, unknown> | null;
  onClose: () => void;
}

const StageDetailPanel = ({
  stage,
  stageResult,
  lossStageResult,
  datasetSummary,
  datasetColumns = [],
  metrics,
  stageLogs,
  lossStageLogs,
  taskType,
  targetColumn,
  explanation,
  onClose,
}: StageDetailPanelProps) => {
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreviewResponse | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);

  const isFeatureStage = stage?.id === "features";
  const isPreprocessing = stage?.id === "preprocessing";
  const isModelSelection = stage?.id === "model_selection";
  const isEvaluation = stage?.id === "evaluation";
  const isResults = stage?.id === "results";

  const highlights = stage ? buildHighlights(stage.id, stageResult, datasetSummary, metrics) : [];
  const panelWidthClass = getPanelWidthClass(stage?.id ?? "");

  const explanationSummary = explanation?.summary ? String(explanation.summary) : "";
  const pipelineSummary = explanation?.pipeline_summary ? String(explanation.pipeline_summary) : "";
  const explanationBullets = Array.isArray(explanation?.explanations)
    ? (explanation?.explanations as string[]).map((item) => String(item)).filter(Boolean)
    : [];

  const summaryLines = explanationSummary
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => line.toLowerCase() !== "model explanation summary:");
  const summaryBullets = summaryLines.filter((line) => /^\d+\./.test(line)).map((line) => line.replace(/^\d+\.\s*/, ""));
  const summaryText = summaryLines.filter((line) => !/^\d+\./.test(line) && !line.toLowerCase().startsWith("final decision:"));
  const summaryDecision = summaryLines.find((line) => line.toLowerCase().startsWith("final decision:"));

  const pipelineLines = pipelineSummary
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^-+\s*/, ""));

  const aboutText = useMemo(() => {
    if (!stage) return "";
    if (isPreprocessing) {
      return "This stage prepares your dataset using deterministic AutoML-style rules. It detects feature types, removes weak or suspicious columns, chooses how to handle missing values, encodes categories carefully, applies scaling or skew-aware transforms when needed, and creates a leakage-safe train/test split so the model can be evaluated fairly.";
    }
    return stage.details;
  }, [isPreprocessing, stage]);

  useEffect(() => {
    if (!isPreprocessing || !stage) {
      setDatasetPreview(null);
      setIsLoadingPreview(false);
      return;
    }

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
            className={`glass-card fixed right-0 top-0 z-50 h-full w-full overflow-y-auto border-l border-border/50 scrollbar-thin ${panelWidthClass}`}
            data-chat-context-label={isFeatureStage ? "Feature engineering panel" : `${stage.label} panel`}
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
          >
            <div className="space-y-6 p-6">
              <div className="flex justify-end">
                <button
                  onClick={onClose}
                  className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              {isFeatureStage ? (
                <FeatureEngineeringDashboard
                  stage={stage}
                  stageResult={stageResult}
                  metrics={metrics}
                  stageLogs={stageLogs}
                  taskType={taskType}
                  targetColumn={targetColumn}
                />
              ) : isEvaluation ? (
                <EvaluationDashboard
                  stage={stage}
                  stageResult={stageResult}
                  lossStageResult={lossStageResult ?? null}
                  metrics={metrics}
                  stageLogs={[...(lossStageLogs ?? []), ...stageLogs]}
                  taskType={taskType}
                  targetColumn={targetColumn}
                />
              ) : (
                <>
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-accent">About this stage</h3>
                    <p className="text-sm leading-relaxed text-secondary-foreground">{aboutText}</p>
                  </div>

                  {!isPreprocessing && <StageSummaryCard stage={stage} stageResult={stageResult} />}

                  {isResults && (explanationSummary || pipelineSummary || explanationBullets.length > 0) && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium text-accent">Pipeline recap</h3>
                      {(summaryText.length > 0 || summaryBullets.length > 0 || summaryDecision) && (
                        <div className="glass-card space-y-2 p-4 text-[11px] text-secondary-foreground">
                          {summaryText.map((line) => (
                            <p key={line} className="text-secondary-foreground">
                              {line}
                            </p>
                          ))}
                          {summaryBullets.length > 0 && (
                            <div className="space-y-1">
                              {summaryBullets.map((item, index) => (
                                <div key={`${item}-${index}`} className="flex items-start gap-2">
                                  <span className="text-accent">•</span>
                                  <span>{item}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          {summaryDecision && (
                            <p className="text-[11px] font-semibold uppercase tracking-wide text-primary">
                              {summaryDecision}
                            </p>
                          )}
                        </div>
                      )}

                      {explanationBullets.length > 0 && (
                        <div className="glass-card space-y-1 p-4 text-[11px] text-secondary-foreground">
                          {explanationBullets.map((item, index) => (
                            <div key={`${item}-${index}`} className="flex items-start gap-2">
                              <span className="text-accent">•</span>
                              <span>{item}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {pipelineLines.length > 0 && (
                        <div className="glass-card space-y-1 p-4 text-[11px] text-secondary-foreground">
                          {pipelineLines.map((line, index) => (
                            <div key={`${line}-${index}`} className="flex items-start gap-2">
                              <span className="text-accent">•</span>
                              <span>{line}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {highlights.length > 0 && !isModelSelection && (
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

                  {stage.id === "training" && buildTrainingComparisonPanel(stageResult)}

                  {stage.id === "analysis" && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-accent">Dataset overview</h3>
                      <DatasetSummaryCard
                        summary={datasetSummary}
                        columns={datasetColumns}
                        targetColumn={targetColumn}
                      />
                    </div>
                  )}

                  {stage.id === "analysis" && buildAnalysisQualityPanel(stageResult)}

                  {stage.id === "analysis" && (stageResult?.correlations as Record<string, unknown> | undefined) && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-accent">Correlation heatmap</h3>
                      <div className="glass-card p-4">
                        <StageVisualization
                          stage={stage}
                          stageResult={stageResult}
                          datasetSummary={datasetSummary}
                          metrics={metrics}
                        />
                      </div>
                    </div>
                  )}

                  {isModelSelection && buildModelSelectionPanel(stageResult)}

                  {isPreprocessing ? (
                    <>
                      {!isResults && (
                        <div className="space-y-2">
                          <h3 className="text-sm font-medium text-accent">Dataset overview</h3>
                          <div className="glass-card p-4">
                            <StageVisualization
                              stage={stage}
                              stageResult={stageResult}
                              datasetSummary={datasetSummary}
                              metrics={metrics}
                            />
                          </div>
                        </div>
                      )}

                      {!isResults && (
                        <div className="space-y-2">
                          <h3 className="text-sm font-medium text-accent">How we prepared your data</h3>
                          <div className="glass-card space-y-2 p-4 text-sm leading-relaxed text-secondary-foreground">
                            <PreprocessExplanation stageResult={stageResult} datasetSummary={datasetSummary} />
                          </div>
                        </div>
                      )}

                      {!isResults && (
                        <div className="space-y-2">
                          <h3 className="text-sm font-medium text-accent">Dataset preview</h3>
                          <DatasetPreviewCard datasetPreview={datasetPreview} isLoadingPreview={isLoadingPreview} />
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      {!isModelSelection && !isResults && (
                        <StageLogs stage={stage} stageLogs={stageLogs} stageResult={stageResult} metrics={metrics} />
                      )}
                    </>
                  )}
                </>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

const DatasetPreviewCard = ({
  datasetPreview,
  isLoadingPreview,
}: {
  datasetPreview: DatasetPreviewResponse | null;
  isLoadingPreview: boolean;
}) => (
  <div className="glass-card overflow-auto p-3">
    {isLoadingPreview && <p className="text-sm text-muted-foreground">Loading sample rows...</p>}
    {!isLoadingPreview && datasetPreview && datasetPreview.rows.length > 0 ? (
      <table className="min-w-full text-left text-[11px]">
        <thead className="text-muted-foreground">
          <tr>
            {datasetPreview.columns.map((column) => (
              <th key={column} className="px-2 py-1 font-medium">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {datasetPreview.rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="border-b border-border/50">
              {datasetPreview.columns.map((column) => (
                <td key={column} className="px-2 py-1">
                  {String(row[column] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    ) : (
      !isLoadingPreview && <p className="text-sm text-muted-foreground">Preview not available.</p>
    )}
  </div>
);

const StageLogs = ({ stage, stageLogs, stageResult, metrics }: { stage: PipelineStage; stageLogs: string[]; stageResult?: Record<string, unknown> | null; metrics?: MetricsResponse | null }) => {
  if (stage.id === "training") {
    const model = (stageResult?.model_name as string | undefined) || "the selected model";
    const trainRows = stageResult?.train_size as number | undefined;
    const testRows = stageResult?.test_size as number | undefined;
    const cv = metrics?.best_score ?? (stageResult?.best_score as number | undefined);
    const test = metrics?.r2 ?? (stageResult?.test_score as number | undefined);
    return (
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-accent">How the model was trained</h3>
        <div className="glass-card space-y-2 p-4 text-sm leading-relaxed text-secondary-foreground">
          <p>
            The system evaluated several models and trained {model} because it performed best on the dataset. It learned from{" "}
            {trainRows ? trainRows.toLocaleString() : "the"} training rows and kept {testRows ? testRows.toLocaleString() : "some"} rows aside for testing.
          </p>
          <ul className="ml-4 list-disc space-y-1">
            <li>Used cross-validation to check performance: {typeof cv === "number" ? cv.toFixed(3) : "n/a"} R².</li>
            <li>Held-out test performance: {typeof test === "number" ? test.toFixed(3) : "n/a"} R².</li>
            <li>Learned how features relate to the target to make price predictions.</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
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
  );
};

const StageSummaryCard = ({
  stage,
  stageResult,
}: {
  stage: PipelineStage;
  stageResult: Record<string, unknown> | null;
}) => {
  const agentSummary = stageResult?._agent_summary as
    | { step_summary?: string; decisions_made?: string[] }
    | undefined;

  const summary =
    agentSummary?.step_summary ||
    (stageResult?.explanation_details as { summary?: string } | undefined)?.summary ||
    (stageResult?.explanation as string | undefined) ||
    "";

  const decisions =
    (agentSummary?.decisions_made as string[] | undefined)?.filter(Boolean)?.slice(0, 3) || [];

  if (!summary && decisions.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-accent">What happened in this stage</h3>
      <div className="glass-card space-y-2 p-4 text-sm leading-relaxed text-secondary-foreground">
        {summary && <p className="text-foreground">{summary}</p>}
        {decisions.length > 0 && (
          <ul className="ml-4 list-disc space-y-1">
            {decisions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

const getPanelWidthClass = (stageId: string) => {
  if (stageId === "features") return "max-w-[min(96vw,1200px)]";
  if (stageId === "evaluation") return "max-w-[min(96vw,1240px)]";
  if (stageId === "model_selection" || stageId === "preprocessing" || stageId === "analysis") {
    return "max-w-[min(92vw,960px)]";
  }
  return "max-w-lg";
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
        { label: "Risk level", value: String(stageResult?.risk_level ?? "Pending") },
        {
          label: "Recommendations",
          value: String(((stageResult?.recommendations as string[] | undefined) || []).length || "Pending"),
        },
      ];
    case "preprocessing":
      return [
        { label: "Training data", value: `${stageResult?.train_size ?? "Pending"} rows` },
        {
          label: "After drops",
          value: `${stageResult?.feature_count_after_column_drops ?? stageResult?.feature_count ?? "Pending"} columns`,
        },
        {
          label: "Transformed",
          value: `${stageResult?.transformed_feature_count ?? stageResult?.feature_count ?? "Pending"} columns`,
        },
      ];
    case "features":
      return [
        { label: "Selected", value: String(stageResult?.final_feature_count ?? "Pending") },
        {
          label: "Scored features",
          value: String(
            Object.keys((stageResult?.feature_scores as Record<string, number> | undefined) || {}).length || 0,
          ),
        },
        {
          label: "PCA",
          value: stageResult?.pca_result ? "Enabled" : "Not used",
        },
      ];
    case "model_selection":
      return [
        {
          label: "Candidate set",
          value: String(
            ((stageResult?.top_candidates as Array<Record<string, unknown>> | undefined) || [])
              .map((item) => String(item.model_name ?? ""))
              .filter(Boolean)
              .join(", ") || "Pending",
          ),
        },
      ];
    case "training":
      return [
        { label: "Selected model", value: String(stageResult?.model_name ?? metrics?.model_name ?? "Pending") },
        { label: "Training time", value: formatDuration(stageResult?.training_time as number | undefined) },
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

const SEVERITY_STYLES: Record<string, string> = {
  high: "bg-red-500/15 text-red-400 border border-red-500/30",
  medium: "bg-yellow-500/15 text-yellow-400 border border-yellow-500/30",
  low: "bg-blue-500/15 text-blue-400 border border-blue-500/30",
};

const buildAnalysisQualityPanel = (stageResult: Record<string, unknown> | null) => {
  if (!stageResult) return null;

  const dq = (stageResult.data_quality as DataQualityProfile | undefined) || {};
  const qualityFlags = (stageResult.quality_flags as QualityFlag[] | undefined) || [];
  const analysisScore = String(stageResult.analysis_summary || "").trim();
  const recommendations = (stageResult.recommendations as string[] | undefined) || [];

  const kpis = [
    {
      label: "Missing rows",
      value: dq.missing_rows_pct != null ? `${dq.missing_rows_pct.toFixed(1)}%` : "—",
    },
    {
      label: "Duplicate rows",
      value: dq.duplicate_rows != null ? `${dq.duplicate_rows.toLocaleString()} (${(dq.duplicate_pct ?? 0).toFixed(1)}%)` : "—",
    },
    {
      label: "Outlier columns",
      value: dq.outlier_columns_count != null ? String(dq.outlier_columns_count) : "—",
    },
    {
      label: "High cardinality",
      value: dq.high_cardinality_count != null ? String(dq.high_cardinality_count) : "—",
    },
    {
      label: "Missing columns",
      value: dq.missing_columns_count != null ? String(dq.missing_columns_count) : "—",
    },
    {
      label: "Leakage signals",
      value: dq.leakage_risk_columns != null ? String(dq.leakage_risk_columns.length) : "—",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-accent">Data quality metrics</h3>
        <div className="grid grid-cols-3 gap-2">
          {kpis.map((kpi) => (
            <div key={kpi.label} className="glass-card space-y-1 p-3">
              <p className="text-[10px] text-muted-foreground">{kpi.label}</p>
              <p className="font-mono text-sm text-foreground">{kpi.value}</p>
            </div>
          ))}
        </div>
      </div>

      {qualityFlags.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-accent">Quality flags</h3>
          <div className="space-y-2">
            {qualityFlags.map((flag, index) => (
              <div
                key={`${flag.field}-${index}`}
                className={`flex items-start gap-3 rounded-lg px-3 py-2 text-xs ${SEVERITY_STYLES[flag.severity] ?? "bg-secondary/50 text-foreground"}`}
              >
                <span className="mt-px shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide opacity-80">
                  {flag.severity}
                </span>
                <span className="leading-relaxed">{flag.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {analysisScore && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-accent">LLM analysis summary</h3>
          <div className="glass-card p-4 text-sm text-secondary-foreground">{analysisScore}</div>
        </div>
      )}

      {recommendations.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-accent">Recommendations</h3>
          <div className="glass-card space-y-2 p-4">
            {recommendations.map((rec, index) => (
              <div key={`rec-${index}`} className="flex items-start gap-2 text-sm text-secondary-foreground">
                <span className="mt-0.5 shrink-0 text-accent">•</span>
                <span>{rec}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const MODEL_EXPLANATIONS: Record<string, string> = {
  RandomForest:
    "Random Forest is an ensemble of decision trees that averages many models to improve accuracy and reduce overfitting.",
  GradientBoosting:
    "Gradient Boosting builds trees sequentially to correct errors, often delivering strong tabular performance.",
  XGBoost:
    "XGBoost is a high-performance gradient boosting variant known for strong accuracy on structured data.",
  LogisticRegression:
    "Logistic Regression is a linear model that provides a fast, interpretable classification baseline.",
  Ridge:
    "Ridge Regression is a linear model with regularization to reduce overfitting on correlated features.",
  SVM:
    "Support Vector Machines separate classes with a margin and can capture non-linear boundaries with kernels.",
  SVR:
    "Support Vector Regression applies SVM principles to continuous targets with kernel-based flexibility.",
  LinearRegression:
    "Linear Regression is a simple baseline for continuous targets with easy interpretability.",
};

const MODEL_CAPABILITIES: Record<string, string[]> = {
  RandomForest: ["Handles non-linear relationships", "Robust to outliers", "Works with mixed data types"],
  GradientBoosting: ["Captures complex interactions", "Strong tabular performance", "Handles mixed feature types"],
  XGBoost: ["High predictive performance", "Handles non-linear patterns", "Scales to larger datasets"],
  LogisticRegression: ["Fast baseline classifier", "Interpretable coefficients", "Works with sparse features"],
  Ridge: ["Stable for collinear features", "Fast to train", "Good linear baseline"],
  SVM: ["Effective on medium-size datasets", "Handles non-linear boundaries", "Works with complex margins"],
  SVR: ["Captures non-linear regression", "Works with smaller datasets", "Kernel-based flexibility"],
  LinearRegression: ["Simple regression baseline", "Fast training", "Easy to interpret"],
};

const PARAM_HINTS: Record<string, string> = {
  n_estimators: "Number of trees in the ensemble.",
  max_depth: "Limits tree depth to reduce overfitting.",
  min_samples_split: "Minimum samples required to split a node.",
  min_samples_leaf: "Minimum samples required at a leaf node.",
  max_features: "Controls the number of features considered per split.",
  learning_rate: "Step size for gradient updates.",
  subsample: "Fraction of samples used for each boosting round.",
  colsample_bytree: "Fraction of features used per tree.",
  C: "Inverse regularization strength.",
  alpha: "Regularization strength for linear models.",
  kernel: "Kernel function used by the model.",
  gamma: "Kernel coefficient controlling decision boundary.",
  epsilon: "Margin of tolerance for regression.",
};

const formatNumber = (value: number) => value.toLocaleString();

const getTopCandidates = (stageResult: Record<string, unknown> | null) =>
  ((stageResult?.top_candidates as Array<Record<string, unknown>> | undefined) || []).slice(0, 3);

const getSelectedModel = (stageResult: Record<string, unknown> | null) =>
  String(getTopCandidates(stageResult)[0]?.model_name ?? "");

const getCandidates = (stageResult: Record<string, unknown> | null) =>
  getTopCandidates(stageResult).map((candidate) => String(candidate.model_name ?? "")).filter(Boolean);

const getSignals = (stageResult: Record<string, unknown> | null) =>
  (stageResult?.analysis_signals as Record<string, unknown> | undefined) || {};

const getParamEntries = (stageResult: Record<string, unknown> | null) => {
  const topCandidate = getTopCandidates(stageResult)[0];
  const params = topCandidate?.fixed_params as Record<string, unknown> | undefined;
  if (!params) return [];
  return Object.entries(params).sort(([a], [b]) => a.localeCompare(b));
};

const getModelComparisons = (stageResult: Record<string, unknown> | null) =>
  (stageResult?.model_comparisons as Array<Record<string, unknown>> | undefined) || [];

const formatComparisonParams = (params: Record<string, unknown> | undefined) => {
  if (!params) return "Default params";
  const entries = Object.entries(params).slice(0, 3);
  if (entries.length === 0) return "Default params";
  return entries.map(([key, value]) => `${key}=${String(value)}`).join(", ");
};

const formatParamValue = (value: unknown) => {
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "null";
  return String(value);
};

const formatFamilyLabel = (modelFamily: unknown) => {
  if (typeof modelFamily !== "string" || !modelFamily.trim()) return "other";
  return modelFamily.split("_").join(" ");
};

const describeDatasetSize = (nSamples: number) => {
  if (nSamples >= 50000) return "Large";
  if (nSamples >= 5000) return "Medium";
  return "Small";
};

const describeFeatureCount = (nFeatures: number) => {
  if (nFeatures >= 100) return "High";
  if (nFeatures >= 20) return "Moderate";
  return "Low";
};

const buildComplexity = (nSamples: number, modelName: string) => {
  const lower = modelName.toLowerCase();
  const heavyModels = ["xgboost", "gradient", "svm", "svr"];
  const base = heavyModels.some((key) => lower.includes(key)) ? "High" : "Medium";
  if (nSamples < 5000) return { complexity: "Low", time: "1–3 seconds", memory: "Low" };
  if (nSamples > 80000) return { complexity: base === "High" ? "High" : "Medium", time: "8–20 seconds", memory: "High" };
  return { complexity: base, time: "3–10 seconds", memory: base === "High" ? "High" : "Moderate" };
};

const buildNotChosenReason = (candidate: string, selected: string, nSamples: number) => {
  if (candidate === selected) return "Selected as the best fit for this dataset.";
  const lower = candidate.toLowerCase();
  if (lower.includes("linear") || lower.includes("logistic") || lower.includes("ridge")) {
    return "May underfit non-linear relationships in this dataset.";
  }
  if (lower.includes("xgboost")) {
    return nSamples > 50000 ? "Higher training cost at this scale." : "Complexity not needed for this dataset.";
  }
  if (lower.includes("svm") || lower.includes("svr")) {
    return nSamples > 10000 ? "Training cost grows quickly with dataset size." : "Less robust to noisy tabular features.";
  }
  if (lower.includes("gradient")) {
    return "Longer training time with similar expected accuracy.";
  }
  return "Selected model offers a better balance of accuracy and efficiency.";
};

const buildModelSelectionPanel = (stageResult: Record<string, unknown> | null) => {
  if (!stageResult) return null;

  const selectedModel = getSelectedModel(stageResult);
  const topCandidates = getTopCandidates(stageResult);
  const candidates = getCandidates(stageResult);
  const params = getParamEntries(stageResult);
  const signals = getSignals(stageResult);
  const nSamples = Number(stageResult.n_samples || 0);
  const nFeatures = Number(stageResult.n_features || 0);
  const taskType = String(stageResult.task_type || "");
  const classBalance = stageResult.class_balance as number | undefined;
  const datasetSizeLabel = nSamples ? `${describeDatasetSize(nSamples)} (${formatNumber(nSamples)} rows)` : "Pending";
  const featureCountLabel = nFeatures ? `${describeFeatureCount(nFeatures)} (${nFeatures})` : "Pending";
  const capabilities = MODEL_CAPABILITIES[selectedModel] || [];
  const explanation = MODEL_EXPLANATIONS[selectedModel] || "This model was selected as the best fit for the dataset.";
  const complexity = buildComplexity(nSamples, selectedModel);
  const llmSummary = String(stageResult.llm_summary ?? "").trim();
  const selectionReasoning = String(stageResult.selection_reasoning ?? "").trim();

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="glass-card space-y-2 p-4">
          <h3 className="text-sm font-medium text-accent">Candidate set</h3>
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-accent/10 px-2 py-1 font-mono text-[12px] text-accent">
              {topCandidates.map((candidate) => String(candidate.model_name ?? "")).filter(Boolean).join(", ") || "Pending"}
            </span>
          </div>
        </div>
        <div className="glass-card space-y-2 p-4">
          <h3 className="text-sm font-medium text-accent">About this candidate set</h3>
          <p className="text-[11px] text-secondary-foreground">{explanation}</p>
        </div>
      </div>

      {llmSummary && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-accent">Model selection summary</h3>
          <div className="glass-card p-4 text-sm text-secondary-foreground">{llmSummary}</div>
        </div>
      )}

      {selectionReasoning && (
        <div className="glass-card p-4 text-[11px] text-secondary-foreground">{selectionReasoning}</div>
      )}

      <div className="glass-card space-y-2 p-4">
        <h3 className="text-sm font-medium text-accent">Candidate models</h3>
        <div className="space-y-2">
          {candidates.length > 0 ? (
            topCandidates.map((candidate, index) => {
              const candidateName = String(candidate.model_name ?? "");
              const candidateFamily = String(candidate.model_family ?? "other").split("_").join(" ");
              const candidateReasoning = String(candidate.reasoning ?? "") || buildNotChosenReason(candidateName, selectedModel, nSamples);
              return (
              <div key={candidateName} className="flex items-center justify-between rounded-lg bg-secondary/50 px-3 py-2 text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">
                    #{index + 1}
                  </span>
                  <span className="font-mono text-[11px] text-foreground">{candidateName}</span>
                  <span className="text-[10px] text-muted-foreground">({candidateFamily})</span>
                </div>
                <span className="text-[10px] text-muted-foreground">
                  {candidateReasoning}
                </span>
              </div>
            );})
          ) : (
            <p className="text-xs text-muted-foreground">Candidate models will appear after selection runs.</p>
          )}
        </div>
      </div>

      <div className="glass-card space-y-2 p-4">
        <h3 className="text-sm font-medium text-accent">Model configuration</h3>
        <div className="space-y-2">
          {params.length > 0 ? (
            params.map(([key, value]) => (
              <div key={key} className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-mono text-[11px] text-foreground">{key}</p>
                  {PARAM_HINTS[key] && <p className="text-[10px] text-muted-foreground">{PARAM_HINTS[key]}</p>}
                </div>
                <span className="rounded-md bg-secondary px-2 py-1 font-mono text-[10px] text-foreground/80">
                  {String(value)}
                </span>
              </div>
            ))
          ) : (
            <p className="text-xs text-muted-foreground">Hyperparameters will appear after selection runs.</p>
          )}
        </div>
      </div>

      <div className="glass-card space-y-2 p-4">
        <h3 className="text-sm font-medium text-accent">Signals used</h3>
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="rounded-md bg-secondary/50 p-2">
            <p className="text-muted-foreground">Dataset size</p>
            <p className="font-mono text-foreground">{datasetSizeLabel}</p>
          </div>
          <div className="rounded-md bg-secondary/50 p-2">
            <p className="text-muted-foreground">Feature count</p>
            <p className="font-mono text-foreground">{featureCountLabel}</p>
          </div>
          <div className="rounded-md bg-secondary/50 p-2">
            <p className="text-muted-foreground">Task type</p>
            <p className="font-mono text-foreground">{taskType || "Pending"}</p>
          </div>
          <div className="rounded-md bg-secondary/50 p-2">
            <p className="text-muted-foreground">Class balance</p>
            <p className="font-mono text-foreground">
              {typeof classBalance === "number" ? `${(classBalance * 100).toFixed(1)}%` : "N/A"}
            </p>
          </div>
          <div className="rounded-md bg-secondary/50 p-2">
            <p className="text-muted-foreground">High correlations</p>
            <p className="font-mono text-foreground">{String(signals.high_correlation_count ?? "Pending")}</p>
          </div>
          <div className="rounded-md bg-secondary/50 p-2">
            <p className="text-muted-foreground">Outlier risk</p>
            <p className="font-mono text-foreground">
              {signals.has_outliers ? "Detected" : signals.has_outliers === false ? "Low" : "Pending"}
            </p>
          </div>
        </div>
      </div>

      <div className="glass-card space-y-2 p-4">
        <h3 className="text-sm font-medium text-accent">Expected model behavior</h3>
        <div className="space-y-1 text-[11px] text-secondary-foreground">
          {capabilities.length > 0 ? (
            capabilities.map((item) => (
              <div key={item} className="flex items-center gap-2">
                <span className="text-accent">✓</span>
                <span>{item}</span>
              </div>
            ))
          ) : (
            <p className="text-muted-foreground">No capability profile available for this model yet.</p>
          )}
        </div>
      </div>

      <div className="glass-card space-y-2 p-4">
        <h3 className="text-sm font-medium text-accent">Training estimate</h3>
        <div className="flex flex-wrap gap-3 text-[11px] text-secondary-foreground">
          <div>
            <p className="text-muted-foreground">Complexity</p>
            <p className="font-mono text-foreground">{complexity.complexity}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Expected time</p>
            <p className="font-mono text-foreground">{complexity.time}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Memory</p>
            <p className="font-mono text-foreground">{complexity.memory}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

const buildTrainingComparisonPanel = (stageResult: Record<string, unknown> | null) => {
  if (!stageResult) return null;

  const comparisons = getModelComparisons(stageResult)
    .slice()
    .sort((a, b) => {
      const scoreA = typeof a.cv_mean === "number" ? a.cv_mean : Number.NEGATIVE_INFINITY;
      const scoreB = typeof b.cv_mean === "number" ? b.cv_mean : Number.NEGATIVE_INFINITY;
      return scoreB - scoreA;
    });
  const trainingMode = String(stageResult.training_mode || "");
  const hasComparisons = comparisons.length > 0;
  const bestModelName = String(stageResult.model_name || comparisons[0]?.model_name || "Pending");
  const bestCv = typeof comparisons[0]?.cv_mean === "number" ? comparisons[0].cv_mean : null;
  const rawWorstCv = comparisons.length > 0 ? comparisons[comparisons.length - 1]?.cv_mean : null;
  const worstCv = typeof rawWorstCv === "number" ? rawWorstCv : bestCv;
  const scoreSpan = bestCv !== null && worstCv !== null ? Math.max(bestCv - worstCv, 1e-6) : 1;

  if (!hasComparisons && trainingMode !== "multi_model") {
    return null;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-accent">Model training experiments</h3>

      <div className="grid gap-2 md:grid-cols-3">
        <div className="glass-card space-y-1 p-3">
          <p className="text-[10px] text-muted-foreground">Compared models</p>
          <p className="font-mono text-sm text-foreground">{hasComparisons ? comparisons.length : "Pending"}</p>
        </div>
        <div className="glass-card space-y-1 p-3">
          <p className="text-[10px] text-muted-foreground">Best CV score</p>
          <p className="font-mono text-sm text-foreground">{bestCv !== null ? bestCv.toFixed(4) : "Pending"}</p>
        </div>
        <div className="glass-card space-y-1 p-3">
          <p className="text-[10px] text-muted-foreground">Selected model</p>
          <p className="truncate font-mono text-sm text-foreground">{bestModelName}</p>
        </div>
      </div>

      <div className="glass-card space-y-3 p-4 text-[11px] text-secondary-foreground">
        {hasComparisons ? (
          <div className="space-y-3">
            {comparisons.map((item, index) => {
              const modelName = String(item.model_name || "Model");
              const family = formatFamilyLabel(item.model_family);
              const cvMean = typeof item.cv_mean === "number" ? item.cv_mean : null;
              const cvStd = typeof item.cv_std === "number" ? item.cv_std : null;
              const cvTime = typeof item.cv_time === "number" ? item.cv_time : null;
              const foldScores = Array.isArray(item.cv_scores)
                ? item.cv_scores.filter((value): value is number => typeof value === "number")
                : [];
              const params = item.hyperparameters as Record<string, unknown> | undefined;
              const paramEntries = Object.entries(params || {})
                .sort(([left], [right]) => left.localeCompare(right))
                .slice(0, 8);
              const reasoning = typeof item.reasoning === "string" ? item.reasoning.trim() : "";
              const scoreWidth =
                cvMean !== null && bestCv !== null
                  ? Math.max(((cvMean - (worstCv ?? cvMean)) / scoreSpan) * 100, 8)
                  : 8;
              const isWinner = modelName === bestModelName;

              return (
                <div
                  key={`${modelName}-${index}`}
                  className={`rounded-lg border p-3 ${
                    isWinner ? "border-accent/60 bg-accent/10" : "border-border/60 bg-secondary/40"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="rounded-md bg-secondary px-2 py-1 text-[10px] text-muted-foreground">#{index + 1}</span>
                      <span className="truncate font-mono text-[11px] text-foreground">{modelName}</span>
                      <span className="rounded-md bg-secondary px-2 py-1 text-[10px] text-muted-foreground">{family}</span>
                      {isWinner && <span className="rounded-md bg-accent/20 px-2 py-1 text-[10px] text-accent">selected</span>}
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-[11px] text-foreground">
                        {cvMean !== null ? cvMean.toFixed(4) : "n/a"}
                        <span className="text-muted-foreground"> ± {cvStd !== null ? cvStd.toFixed(4) : "n/a"}</span>
                      </p>
                      <p className="text-[10px] text-muted-foreground">CV mean ± std</p>
                    </div>
                  </div>

                  <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${scoreWidth}%`,
                        background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))",
                      }}
                    />
                  </div>

                  <div className="mt-2 grid gap-2 md:grid-cols-2">
                    <div className="space-y-1">
                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Fold scores</p>
                      {foldScores.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {foldScores.map((score, scoreIndex) => (
                            <span key={`${modelName}-fold-${scoreIndex}`} className="rounded bg-secondary px-2 py-1 font-mono text-[10px] text-foreground/90">
                              F{scoreIndex + 1}: {score.toFixed(3)}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[10px] text-muted-foreground">Fold-level scores unavailable.</p>
                      )}
                    </div>

                    <div className="space-y-1">
                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Hyperparameters</p>
                      {paramEntries.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {paramEntries.map(([key, value]) => (
                            <span key={`${modelName}-${key}`} className="rounded bg-secondary px-2 py-1 font-mono text-[10px] text-foreground/90">
                              {key}={formatParamValue(value)}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[10px] text-muted-foreground">{formatComparisonParams(params)}</p>
                      )}
                    </div>
                  </div>

                  <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[10px] text-muted-foreground">
                    <span>Training time: {cvTime !== null ? `${cvTime.toFixed(2)}s` : "n/a"}</span>
                    {reasoning && <span className="truncate">{reasoning}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Multi-model comparison is enabled, but experiment results are not available yet.
          </p>
        )}
      </div>
    </div>
  );
};

const formatMetric = (value: number | null | undefined, asPercent = false) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "Pending";
  return asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
};

const formatDuration = (seconds: number | undefined) => {
  if (typeof seconds !== "number" || Number.isNaN(seconds)) return "Pending";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rem = Math.round(seconds % 60);
  return `${minutes}m ${rem}s`;
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
  const summary =
    (stageResult?.explanation as string | undefined) ||
    "We reviewed the dataset and applied deterministic preprocessing rules before modeling.";
  const missingSummary = (stageResult?.missing_summary as Record<string, unknown> | undefined) || {};
  const categoricalSummary = (stageResult?.categorical_summary as Record<string, unknown> | undefined) || {};
  const scalingSummary = (stageResult?.scaling_summary as Record<string, unknown> | undefined) || {};
  const transformSummary = (stageResult?.transform_summary as Record<string, unknown> | undefined) || {};
  const droppedColumns = Array.isArray(stageResult?.dropped_columns)
    ? (stageResult?.dropped_columns as Array<Record<string, unknown>>)
    : [];
  const train = stageResult?.train_size as number | undefined;
  const test = stageResult?.test_size as number | undefined;
  const total = datasetSummary?.rows;
  const droppedRows = Number(missingSummary?.dropped_rows_count ?? 0);
  const strategyUsed = String(missingSummary?.strategy_used ?? "none");
  const imputedNumeric = Array.isArray(missingSummary?.imputed_numeric_columns)
    ? (missingSummary.imputed_numeric_columns as string[])
    : [];
  const imputedCategorical = Array.isArray(missingSummary?.imputed_categorical_columns)
    ? (missingSummary.imputed_categorical_columns as string[])
    : [];
  const highCardinality = Array.isArray(categoricalSummary?.high_cardinality_columns)
    ? (categoricalSummary.high_cardinality_columns as string[])
    : [];
  const rareGrouped = Array.isArray(categoricalSummary?.rare_category_grouped_columns)
    ? (categoricalSummary.rare_category_grouped_columns as string[])
    : [];
  const logColumns = Array.isArray(transformSummary?.log_transformed_columns)
    ? (transformSummary.log_transformed_columns as string[])
    : [];
  const scaler = String(scalingSummary?.scaler ?? "None");
  const splitMatchesModelingRows =
    typeof train === "number" &&
    typeof test === "number" &&
    typeof total === "number" &&
    train + test <= total;

  const bullets = [
    droppedColumns.length > 0
      ? `Dropped ${droppedColumns.length} low-value column${droppedColumns.length === 1 ? "" : "s"}, including ${droppedColumns
          .slice(0, 3)
          .map((item) => String(item.column ?? ""))
          .filter(Boolean)
          .join(", ")}.`
      : "No columns needed to be removed as obvious IDs, constants, or leakage risks.",
    strategyUsed === "none"
      ? "No missing values were found after cleanup, so no imputation or row dropping was needed."
      : strategyUsed === "drop_rows"
        ? `Dropped ${droppedRows.toLocaleString()} row${droppedRows === 1 ? "" : "s"} with missing values because the affected share was small.`
        : `Missing values were handled with a ${strategyUsed} strategy using train-fitted rules.`,
    highCardinality.length > 0
      ? `Used safer encoding for high-cardinality columns like ${highCardinality.slice(0, 3).join(", ")} instead of naive one-hot expansion.`
      : "Categorical columns were encoded with conservative train-fitted rules.",
    scaler !== "None"
      ? `Scaled numeric columns with ${scaler}${logColumns.length > 0 ? ` after log-transforming ${logColumns.slice(0, 3).join(", ")}` : ""}.`
      : logColumns.length > 0
        ? `Applied log transforms to ${logColumns.slice(0, 3).join(", ")} without additional scaling.`
        : "No numeric scaling or skew transform was needed.",
  ].filter(Boolean);

  return (
    <div className="space-y-2">
      <p className="text-foreground">{summary}</p>
      <ul className="ml-4 list-disc space-y-1 text-sm text-secondary-foreground">
        {bullets.map((bullet) => (
          <li key={bullet}>{bullet}</li>
        ))}
      </ul>
      {(imputedNumeric.length > 0 || imputedCategorical.length > 0 || rareGrouped.length > 0) && (
        <p className="text-sm text-secondary-foreground">
          {imputedNumeric.length > 0 && `Numeric imputation: ${imputedNumeric.slice(0, 4).join(", ")}.`}{" "}
          {imputedCategorical.length > 0 && `Categorical imputation: ${imputedCategorical.slice(0, 4).join(", ")}.`}{" "}
          {rareGrouped.length > 0 && `Rare levels grouped in ${rareGrouped.slice(0, 4).join(", ")}.`}
        </p>
      )}
      {(train || test) && (
        <p className="text-sm text-secondary-foreground">
          Finally, the data was split into {train ? train.toLocaleString() : "?"} rows for training and{" "}
          {test ? test.toLocaleString() : "?"} rows to test how well the model performs on new data.
        </p>
      )}
      {splitMatchesModelingRows && (
        <p className="text-sm font-medium text-primary">
          {droppedRows > 0
            ? `${droppedRows.toLocaleString()} row${droppedRows === 1 ? "" : "s"} were removed before the final split.`
            : "No feature rows had to be removed for missingness before the final split."}
        </p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
        {["Raw data", "Drop weak columns", "Handle missingness", "Encode categories", "Scale and split"].map(
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
