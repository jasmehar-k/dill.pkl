import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import type { PipelineStage } from "@/data/pipelineStages";
import type {
  DatasetPreviewResponse,
  DatasetSummary,
  MetricsResponse,
  TaskType,
} from "@/lib/api";
import { getDatasetPreview } from "@/lib/api";
import FeatureEngineeringDashboard from "./FeatureEngineeringDashboard";
import { StageVisualization } from "./StageVisualizations";

interface StageDetailPanelProps {
  stage: PipelineStage | null;
  stageResult: Record<string, unknown> | null;
  datasetSummary: DatasetSummary | null;
  metrics: MetricsResponse | null;
  stageLogs: string[];
  taskType: TaskType;
  targetColumn: string | null;
  onClose: () => void;
}

const StageDetailPanel = ({
  stage,
  stageResult,
  datasetSummary,
  metrics,
  stageLogs,
  taskType,
  targetColumn,
  onClose,
}: StageDetailPanelProps) => {
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreviewResponse | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);

  const isFeatureStage = stage?.id === "features";
  const isPreprocessing = stage?.id === "preprocessing";
  const isModelSelection = stage?.id === "model_selection";
  const highlights = stage ? buildHighlights(stage.id, stageResult, datasetSummary, metrics) : [];
  const panelWidthClass = getPanelWidthClass(stage?.id ?? "");
  const aboutText = useMemo(() => {
    if (!stage) return "";
    if (isPreprocessing) {
      return "This step prepares your dataset so it can be used by machine learning models. Missing values are filled in, categories are converted into numbers, numeric values are scaled, and the data is split into training and testing sets to evaluate performance.";
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
              ) : (
                <>
                  <StageHeader stage={stage} />

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

                  {isModelSelection && buildModelSelectionPanel(stageResult)}

                  {isPreprocessing ? (
                    <>
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

                      <div className="space-y-2">
                        <h3 className="text-sm font-medium text-accent">How we prepared your data</h3>
                        <div className="glass-card space-y-2 p-4 text-sm leading-relaxed text-secondary-foreground">
                          <PreprocessExplanation stageResult={stageResult} datasetSummary={datasetSummary} />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <h3 className="text-sm font-medium text-accent">Dataset preview</h3>
                        <DatasetPreviewCard
                          datasetPreview={datasetPreview}
                          isLoadingPreview={isLoadingPreview}
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      {!isModelSelection && (
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
                      )}

                      {!isModelSelection && (
                        <>
                          <StageLogs stage={stage} stageLogs={stageLogs} />

                          <div className="space-y-2">
                            <h3 className="text-sm font-medium text-accent">Code</h3>
                            <pre className="glass-card overflow-x-auto whitespace-pre-wrap p-4 font-mono text-[12px] leading-relaxed text-foreground/80">
                              {stage.codeSnippet}
                            </pre>
                          </div>
                        </>
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

const StageHeader = ({ stage }: { stage: PipelineStage }) => (
  <div className="flex items-start gap-3">
    <span className="text-3xl">{stage.icon}</span>
    <div className="space-y-1">
      <h2 className="text-xl font-semibold text-foreground">{stage.label}</h2>
      <p className="text-sm text-muted-foreground">{stage.description}</p>
    </div>
  </div>
);

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

const StageLogs = ({
  stage,
  stageLogs,
}: {
  stage: PipelineStage;
  stageLogs: string[];
}) => (
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

const getPanelWidthClass = (stageId: string) => {
  if (stageId === "features") return "max-w-[min(96vw,1200px)]";
  if (stageId === "model_selection" || stageId === "preprocessing") {
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
      return [{ label: "Selected model", value: String(stageResult?.selected_model ?? "Pending") }];
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

const getSelectedModel = (stageResult: Record<string, unknown> | null) =>
  String(stageResult?.selected_model ?? "");

const getCandidates = (stageResult: Record<string, unknown> | null) =>
  (stageResult?.candidate_models as string[] | undefined) || [];

const getSignals = (stageResult: Record<string, unknown> | null) =>
  (stageResult?.analysis_signals as Record<string, unknown> | undefined) || {};

const getParamEntries = (stageResult: Record<string, unknown> | null) => {
  const params = stageResult?.hyperparameters as Record<string, unknown> | undefined;
  if (!params) return [];
  return Object.entries(params).sort(([a], [b]) => a.localeCompare(b));
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
  if (nSamples < 5000) return { complexity: "Low", time: "1-3 seconds", memory: "Low" };
  if (nSamples > 80000) {
    return { complexity: base === "High" ? "High" : "Medium", time: "8-20 seconds", memory: "High" };
  }
  return { complexity: base, time: "3-10 seconds", memory: base === "High" ? "High" : "Moderate" };
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

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="glass-card space-y-2 p-4">
          <h3 className="text-sm font-medium text-accent">Selected model</h3>
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-accent/10 px-2 py-1 font-mono text-[12px] text-accent">
              {selectedModel || "Pending"}
            </span>
          </div>
        </div>
        <div className="glass-card space-y-2 p-4">
          <h3 className="text-sm font-medium text-accent">About {selectedModel || "this model"}</h3>
          <p className="text-[11px] text-secondary-foreground">{explanation}</p>
        </div>
      </div>

      {llmSummary && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-accent">Model selection summary</h3>
          <div className="glass-card p-4 text-sm text-secondary-foreground">{llmSummary}</div>
        </div>
      )}

      <div className="glass-card space-y-2 p-4">
        <h3 className="text-sm font-medium text-accent">Candidate models</h3>
        <div className="space-y-2">
          {candidates.length > 0 ? (
            candidates.map((candidate) => (
              <div key={candidate} className="flex items-center justify-between rounded-lg bg-secondary/50 px-3 py-2 text-xs">
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] ${candidate === selectedModel ? "text-accent" : "text-muted-foreground"}`}>
                    {candidate === selectedModel ? "Selected" : "Option"}
                  </span>
                  <span className="font-mono text-[11px] text-foreground">{candidate}</span>
                </div>
                <span className="text-[10px] text-muted-foreground">
                  {buildNotChosenReason(candidate, selectedModel, nSamples)}
                </span>
              </div>
            ))
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
                <span className="text-accent">+</span>
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
  const summary =
    (stageResult?.explanation as string | undefined) ||
    "We filled missing values, simplified categories, and prepared numeric columns so the model can learn reliably.";
  const train = stageResult?.train_size as number | undefined;
  const test = stageResult?.test_size as number | undefined;
  const total = datasetSummary?.rows;
  const integrityOK =
    typeof train === "number" && typeof test === "number" && typeof total === "number" && train + test === total;

  return (
    <div className="space-y-2">
      <p className="text-foreground">{summary}</p>
      <ul className="ml-4 list-disc space-y-1 text-sm text-secondary-foreground">
        <li>Any missing values were filled in automatically.</li>
        <li>Columns with categories were converted into numbers so the model can use them.</li>
        <li>Numeric features were standardized so large ranges do not dominate the model.</li>
      </ul>
      {(train || test) && (
        <p className="text-sm text-secondary-foreground">
          Finally, the data was split into {train ? train.toLocaleString() : "?"} rows for training and{" "}
          {test ? test.toLocaleString() : "?"} rows to test how well the model performs on new data.
        </p>
      )}
      {integrityOK && <p className="text-sm font-medium text-primary">Dataset integrity preserved (0 rows dropped)</p>}
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
        {["Raw data", "Clean missing values", "Convert categories", "Prepare numeric columns", "Train/Test split"].map(
          (step, index, arr) => (
            <div key={step} className="flex items-center gap-1">
              <span className="rounded bg-secondary px-2 py-1">{step}</span>
              {index < arr.length - 1 && <span className="text-muted-foreground">-&gt;</span>}
            </div>
          ),
        )}
      </div>
    </div>
  );
};

export default StageDetailPanel;
