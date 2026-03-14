import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";

interface ModelArchitectureVizProps {
  modelName: string;
  taskType: "regression" | "classification";
  featureCount?: number;
  sampleCount?: number;
  targetColumn?: string;
  topFeature?: string;
}

const transition = { duration: 0.8, ease: [0.2, 0, 0, 1] as [number, number, number, number] };

const seededRandom = (seed: number) => {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
};

export const ModelArchitectureViz = ({ modelName, taskType, featureCount, sampleCount, targetColumn, topFeature }: ModelArchitectureVizProps) => {
  const name = modelName.toLowerCase();
  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null);

  const formatStats = (base: string) => {
    const parts: string[] = [];
    if (sampleCount) parts.push(`${sampleCount.toLocaleString()} rows`);
    if (featureCount) parts.push(`${featureCount} features`);
    if (targetColumn) parts.push(`target: ${targetColumn}`);
    return parts.length > 0 ? `${base} — ${parts.join(" · ")}` : base;
  };

  const showTooltip = (text: string, evt: React.MouseEvent<SVGElement, MouseEvent>) => {
    const rect = (evt.currentTarget.ownerSVGElement || evt.currentTarget).getBoundingClientRect();
    setTooltip({
      text,
      x: evt.clientX - rect.left + 8,
      y: evt.clientY - rect.top + 8,
    });
  };

  const hideTooltip = () => setTooltip(null);

  if (name.includes("forest") || name.includes("tree") || name.includes("boost") || name.includes("xgb")) {
    const isEnsemble = name.includes("forest") || name.includes("boost") || name.includes("xgb");
    return (
      <TreeViz
        title={modelName}
        isEnsemble={isEnsemble}
        summary={{ featureCount, sampleCount, targetColumn }}
        onHover={showTooltip}
        onLeave={hideTooltip}
        tooltip={tooltip}
        formatStats={formatStats}
        topFeature={topFeature}
        taskType={taskType}
      />
    );
  }

  if (name.includes("linear") || name.includes("logistic") || name.includes("ridge") || name.includes("lasso")) {
    return (
      <LinearViz
        isClassification={name.includes("logistic")}
        summary={{ featureCount, sampleCount, targetColumn }}
        onHover={showTooltip}
        onLeave={hideTooltip}
        tooltip={tooltip}
        formatStats={formatStats}
      />
    );
  }

  if (name.includes("svm") || name.includes("svr") || name.includes("support")) {
    return (
      <SVMViz
        isRegression={name.includes("svr") || taskType === "regression"}
        summary={{ featureCount, sampleCount, targetColumn }}
        onHover={showTooltip}
        onLeave={hideTooltip}
        tooltip={tooltip}
        formatStats={formatStats}
      />
    );
  }

  if (name.includes("knn") || name.includes("neighbor")) {
    return (
      <KNNViz
        summary={{ featureCount, sampleCount, targetColumn }}
        onHover={showTooltip}
        onLeave={hideTooltip}
        tooltip={tooltip}
        formatStats={formatStats}
      />
    );
  }

  if (name.includes("neural") || name.includes("mlp") || name.includes("deep")) {
    return (
      <NeuralNetViz
        summary={{ featureCount, sampleCount, targetColumn }}
        onHover={showTooltip}
        onLeave={hideTooltip}
        tooltip={tooltip}
        formatStats={formatStats}
      />
    );
  }

  return <GenericModelViz name={modelName} summary={{ featureCount, sampleCount, targetColumn }} tooltip={tooltip} />;
};

const VizContainer = ({
  label,
  sublabel,
  children,
  extra,
  summary,
  tooltip,
  formatStats,
}: {
  label: string;
  sublabel: string;
  children: React.ReactNode;
  extra?: React.ReactNode;
  summary?: { featureCount?: number; sampleCount?: number; targetColumn?: string };
  tooltip?: { text: string; x: number; y: number } | null;
  formatStats?: (base: string) => string;
}) => (
  <div className="relative flex h-64 w-full items-center justify-center overflow-hidden rounded-xl border border-border bg-card p-6">
    <div className="absolute left-4 top-4 flex flex-col gap-0.5">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">{label}</span>
      <span className="text-sm font-medium text-foreground">{sublabel}</span>
    </div>
    {summary && (summary.featureCount || summary.sampleCount || summary.targetColumn) && (
      <div className="absolute right-4 top-4 flex flex-wrap items-center gap-2 text-[10px] text-muted-foreground">
        {summary.sampleCount && <span className="rounded bg-secondary/60 px-2 py-1">Rows: {summary.sampleCount.toLocaleString()}</span>}
        {summary.featureCount && <span className="rounded bg-secondary/60 px-2 py-1">Features: {summary.featureCount}</span>}
        {summary.targetColumn && <span className="rounded bg-secondary/60 px-2 py-1">Target: {summary.targetColumn}</span>}
      </div>
    )}
    {children}
    {extra}
    {tooltip && (
      <div
        className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md bg-popover px-2 py-1 text-[11px] text-popover-foreground shadow-lg ring-1 ring-border/70"
        style={{ left: tooltip.x, top: tooltip.y }}
      >
        {tooltip.text}
      </div>
    )}
  </div>
);

/* ── Decision Tree / Ensemble ── */
const TreeViz = ({
  title,
  isEnsemble,
  summary,
  onHover,
  onLeave,
  tooltip,
  formatStats,
  topFeature,
  taskType,
}: {
  title: string;
  isEnsemble: boolean;
  summary?: { featureCount?: number; sampleCount?: number; targetColumn?: string };
  onHover: (text: string, evt: React.MouseEvent<SVGElement, MouseEvent>) => void;
  onLeave: () => void;
  tooltip: { text: string; x: number; y: number } | null;
  formatStats: (base: string) => string;
  topFeature?: string;
  taskType: "regression" | "classification";
}) => {
  const statsText =
    summary && (summary.sampleCount || summary.featureCount || summary.targetColumn)
      ? [
          summary.sampleCount ? `${summary.sampleCount.toLocaleString()} rows` : null,
          summary.featureCount ? `${summary.featureCount} features` : null,
          summary.targetColumn ? `target: ${summary.targetColumn}` : null,
        ]
          .filter(Boolean)
          .join(" · ")
      : "";
  const targetLabel =
    taskType === "classification"
      ? `class label${summary?.targetColumn ? ` (${summary.targetColumn})` : ""}`
      : summary?.targetColumn
        ? `${summary.targetColumn} value`
        : "target value";

  return (
  <VizContainer
    label="Architecture"
    sublabel={isEnsemble ? "Parallel Decision Paths" : "Decision Logic"}
    summary={summary}
    tooltip={tooltip}
    extra={
      isEnsemble ? (
        <div className="absolute bottom-4 right-4 flex items-center gap-1.5">
          <span className="mr-1 text-[10px] text-muted-foreground">ensemble</span>
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-1.5 w-1.5 rounded-full bg-node-active opacity-40" />
          ))}
        </div>
      ) : undefined
    }
  >
    <svg width="320" height="160" viewBox="0 0 320 160" fill="none" className="overflow-visible">
      <motion.path
        d="M160 20 L80 80 M160 20 L240 80"
        stroke="hsl(var(--border))"
        strokeWidth="1.5"
        fill="none"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={transition}
      />
      <motion.path
        d="M80 80 L40 140 M80 80 L120 140 M240 80 L200 140 M240 80 L280 140"
        stroke="hsl(var(--border))"
        strokeWidth="1.5"
        fill="none"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ ...transition, delay: 0.2 }}
      />
      {[
        { x: 160, y: 20, r: 7, type: "root" as const },
        { x: 80, y: 80, r: 5.5, type: "branch" as const },
        { x: 240, y: 80, r: 5.5, type: "branch" as const },
        { x: 40, y: 140, r: 4.5, type: "leaf" as const },
        { x: 120, y: 140, r: 4.5, type: "leaf" as const },
        { x: 200, y: 140, r: 4.5, type: "leaf" as const },
        { x: 280, y: 140, r: 4.5, type: "leaf" as const },
      ].map((node, i) => (
        <motion.circle
          key={i}
          cx={node.x}
          cy={node.y}
          r={node.r}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 20, delay: i * 0.06 }}
          fill={
            node.type === "root"
              ? "hsl(var(--node-active))"
              : node.type === "leaf"
                ? "hsl(var(--node-leaf))"
                : "hsl(var(--muted-foreground))"
          }
          opacity={node.type === "root" ? 1 : 0.75}
          className="cursor-pointer"
          onMouseEnter={(evt) =>
              onHover(
                formatStats(
                  node.type === "root"
                    ? `Root split on ${topFeature || "strongest feature"}: splits the dataset to better predict the ${targetLabel}. ${statsText}`
                    : node.type === "leaf"
                      ? `Leaf: stores the final ${targetLabel} learned from rows that followed this path. ${statsText}`
                      : `Branch split: routes rows using the ${topFeature || "chosen feature"} threshold to refine the prediction. ${statsText}`,
                ),
              evt,
            )
          }
          onMouseMove={(evt) =>
            onHover(
              formatStats(
                node.type === "root"
                  ? `Root split on ${topFeature || "strongest feature"}: splits the dataset to better predict the ${targetLabel}. ${statsText}`
                  : node.type === "leaf"
                    ? `Leaf: stores the final ${targetLabel} learned from rows that followed this path. ${statsText}`
                    : `Branch split: routes rows using the ${topFeature || "chosen feature"} threshold to refine the prediction. ${statsText}`,
              ),
              evt,
            )
          }
          onMouseLeave={onLeave}
        >
          <title>
            {node.type === "root"
              ? "Root split: chooses the strongest feature to divide all samples into two groups."
              : node.type === "leaf"
                ? "Leaf: stores the final decision/prediction for samples reaching this path."
                : "Branch split: applies a single feature threshold to route samples left/right."}
          </title>
        </motion.circle>
      ))}
        <motion.text
          x="118"
          y="48"
          fontSize="9"
          fill="hsl(var(--muted-foreground))"
          textAnchor="middle"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.5 }}
          transition={{ delay: 0.6 }}
          onMouseEnter={(evt) => onHover(formatStats("Left branch: rows with feature x₁ below threshold θ."), evt)}
          onMouseMove={(evt) => onHover(formatStats("Left branch: rows with feature x₁ below threshold θ."), evt)}
          onMouseLeave={onLeave}
          className="cursor-pointer"
        >
          x₁ {"<"} θ
        </motion.text>
        <motion.text
          x="202"
          y="48"
          fontSize="9"
          fill="hsl(var(--muted-foreground))"
          textAnchor="middle"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.5 }}
          transition={{ delay: 0.6 }}
          onMouseEnter={(evt) => onHover(formatStats("Right branch: rows with feature x₁ at or above threshold θ."), evt)}
          onMouseMove={(evt) => onHover(formatStats("Right branch: rows with feature x₁ at or above threshold θ."), evt)}
          onMouseLeave={onLeave}
          className="cursor-pointer"
        >
          x₁ ≥ θ
        </motion.text>
      </svg>
    </VizContainer>
  );
};

/* ── Linear / Logistic ── */
const LinearViz = ({
  isClassification,
  summary,
  onHover,
  onLeave,
  tooltip,
  formatStats,
}: {
  isClassification: boolean;
  summary?: { featureCount?: number; sampleCount?: number; targetColumn?: string };
  onHover: (text: string, evt: React.MouseEvent<SVGElement, MouseEvent>) => void;
  onLeave: () => void;
  tooltip: { text: string; x: number; y: number } | null;
  formatStats: (base: string) => string;
}) => {
  const points = useMemo(
    () =>
      Array.from({ length: 16 }).map((_, i) => ({
        x: 15 + i * 14 + (seededRandom(i * 3 + 1) - 0.5) * 12,
        y: 125 - i * 7 + (seededRandom(i * 3 + 2) - 0.5) * 24,
        cls: i > 8 ? 1 : 0,
      })),
    [],
  );

  return (
    <VizContainer label="Feature Mapping" sublabel={isClassification ? "Decision Boundary" : "Linear Fit"} summary={summary} tooltip={tooltip}>
      <svg width="260" height="150" viewBox="0 0 260 150" className="overflow-visible">
        <line x1="0" y1="140" x2="250" y2="140" stroke="hsl(var(--border))" strokeWidth="1" />
        <line x1="0" y1="0" x2="0" y2="140" stroke="hsl(var(--border))" strokeWidth="1" />

        {points.map((p, i) => (
          <motion.circle
            key={i}
            cx={p.x}
            cy={p.y}
            r="3"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 0.5, scale: 1 }}
            transition={{ delay: i * 0.03, type: "spring", stiffness: 200, damping: 15 }}
            fill={isClassification && p.cls === 1 ? "hsl(var(--node-active))" : "hsl(var(--accent))"}
            className="cursor-pointer"
            onMouseEnter={(evt) =>
              onHover(
                formatStats(
                  isClassification
                    ? "Sample in feature space; color shows its class label."
                    : "Sample (x, y) plotted against the model's linear fit line.",
                ),
                evt,
              )
            }
            onMouseMove={(evt) =>
              onHover(
                formatStats(
                  isClassification
                    ? "Sample in feature space; color shows its class label."
                    : "Sample (x, y) plotted against the model's linear fit line.",
                ),
                evt,
              )
            }
            onMouseLeave={onLeave}
          />
        ))}

        {isClassification ? (
          <motion.path
            d="M0,130 C60,128 100,80 130,70 S200,15 250,10"
            stroke="hsl(var(--node-active))"
            strokeWidth="2"
            fill="none"
            strokeDasharray="4 3"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={transition}
          />
        ) : (
          <motion.line
            x1="0"
            y1="135"
            x2="250"
            y2="25"
            stroke="hsl(var(--node-active))"
            strokeWidth="2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={transition}
          />
        )}

        <motion.text
          x="240"
          y="18"
          fontSize="9"
          fill="hsl(var(--muted-foreground))"
          textAnchor="end"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.5 }}
          transition={{ delay: 0.8 }}
        >
          {isClassification ? "P(y=1)" : "ŷ = wᵀx + b"}
        </motion.text>
      </svg>
    </VizContainer>
  );
};

/* ── SVM / SVR ── */
const SVMViz = ({
  isRegression,
  summary,
  onHover,
  onLeave,
  tooltip,
  formatStats,
}: {
  isRegression: boolean;
  summary?: { featureCount?: number; sampleCount?: number; targetColumn?: string };
  onHover: (text: string, evt: React.MouseEvent<SVGElement, MouseEvent>) => void;
  onLeave: () => void;
  tooltip: { text: string; x: number; y: number } | null;
  formatStats: (base: string) => string;
}) => {
  const classA = useMemo(
    () => Array.from({ length: 8 }).map((_, i) => ({ x: 30 + seededRandom(i * 7) * 60, y: 30 + seededRandom(i * 7 + 1) * 40 })),
    [],
  );
  const classB = useMemo(
    () => Array.from({ length: 8 }).map((_, i) => ({ x: 160 + seededRandom(i * 11) * 60, y: 80 + seededRandom(i * 11 + 1) * 40 })),
    [],
  );

  return (
    <VizContainer label="Decision Geometry" sublabel={isRegression ? "ε-Tube Regression" : "Maximum Margin"} summary={summary} tooltip={tooltip}>
      <svg width="280" height="150" viewBox="0 0 280 150" className="overflow-visible">
        <motion.path d="M10,140 L260,10" stroke="hsl(var(--node-active))" strokeWidth="2" fill="none" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={transition} />
        <motion.path d="M10,120 L260,-10" stroke="hsl(var(--border))" strokeWidth="1" strokeDasharray="4 3" fill="none" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ ...transition, delay: 0.15 }} />
        <motion.path d="M10,160 L260,30" stroke="hsl(var(--border))" strokeWidth="1" strokeDasharray="4 3" fill="none" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ ...transition, delay: 0.15 }} />

        {classA.map((p, i) => (
          <motion.circle
            key={`a-${i}`}
            cx={p.x}
            cy={p.y}
            r="3.5"
            fill="hsl(var(--node-active))"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 0.7, scale: 1 }}
            transition={{ delay: 0.3 + i * 0.04, type: "spring", stiffness: 200, damping: 15 }}
            className="cursor-pointer"
            onMouseEnter={(evt) => onHover(formatStats("Support vector (Class A): defines one side of the margin."), evt)}
            onMouseMove={(evt) => onHover(formatStats("Support vector (Class A): defines one side of the margin."), evt)}
            onMouseLeave={onLeave}
          />
        ))}
        {classB.map((p, i) => (
          <motion.circle
            key={`b-${i}`}
            cx={p.x}
            cy={p.y}
            r="3.5"
            fill="hsl(var(--accent))"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 0.6, scale: 1 }}
            transition={{ delay: 0.3 + i * 0.04, type: "spring", stiffness: 200, damping: 15 }}
            className="cursor-pointer"
            onMouseEnter={(evt) => onHover(formatStats("Support vector (Class B): defines the opposite side of the margin."), evt)}
            onMouseMove={(evt) => onHover(formatStats("Support vector (Class B): defines the opposite side of the margin."), evt)}
            onMouseLeave={onLeave}
          />
        ))}

        <motion.text x="240" y="48" fontSize="9" fill="hsl(var(--muted-foreground))" textAnchor="end" initial={{ opacity: 0 }} animate={{ opacity: 0.5 }} transition={{ delay: 0.9 }}>
          margin
        </motion.text>
      </svg>
    </VizContainer>
  );
};

/* ── KNN ── */
const KNNViz = ({
  summary,
  onHover,
  onLeave,
  tooltip,
  formatStats,
}: {
  summary?: { featureCount?: number; sampleCount?: number; targetColumn?: string };
  onHover: (text: string, evt: React.MouseEvent<SVGElement, MouseEvent>) => void;
  onLeave: () => void;
  tooltip: { text: string; x: number; y: number } | null;
  formatStats: (base: string) => string;
}) => {
  const points = useMemo(
    () =>
      Array.from({ length: 14 }).map((_, i) => ({
        x: 30 + seededRandom(i * 5) * 210,
        y: 15 + seededRandom(i * 5 + 1) * 115,
        cls: i % 3 === 0 ? 1 : 0,
      })),
    [],
  );
  const queryPt = { x: 140, y: 70 };

  return (
    <VizContainer label="Instance-Based" sublabel="K-Nearest Neighbors" summary={summary} tooltip={tooltip}>
      <svg width="280" height="150" viewBox="0 0 280 150" className="overflow-visible">
        <motion.circle cx={queryPt.x} cy={queryPt.y} r="50" fill="none" stroke="hsl(var(--node-active))" strokeWidth="1" strokeDasharray="3 3" initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 0.35 }} transition={{ ...transition, delay: 0.4 }} />

        {points.map((p, i) => (
          <motion.circle
            key={i}
            cx={p.x}
            cy={p.y}
            r="3.5"
            fill={p.cls === 1 ? "hsl(var(--node-active))" : "hsl(var(--accent))"}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 0.6, scale: 1 }}
            transition={{ delay: i * 0.04, type: "spring", stiffness: 200, damping: 15 }}
            className="cursor-pointer"
          onMouseEnter={(evt) => onHover(formatStats("Neighbor sample: one of the k closest points used to vote on the query."), evt)}
          onMouseMove={(evt) => onHover(formatStats("Neighbor sample: one of the k closest points used to vote on the query."), evt)}
          onMouseLeave={onLeave}
        />
        ))}

        <motion.circle
          cx={queryPt.x}
          cy={queryPt.y}
          r="5"
          fill="hsl(var(--node-active))"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 20, delay: 0.2 }}
          className="cursor-pointer"
          onMouseEnter={(evt) => onHover(formatStats("Query point: new sample to predict; neighbors inside the circle drive the vote."), evt)}
          onMouseMove={(evt) => onHover(formatStats("Query point: new sample to predict; neighbors inside the circle drive the vote."), evt)}
          onMouseLeave={onLeave}
        />
        <motion.text x={queryPt.x + 10} y={queryPt.y - 8} fontSize="9" fill="hsl(var(--muted-foreground))" initial={{ opacity: 0 }} animate={{ opacity: 0.6 }} transition={{ delay: 0.7 }}>
          query
        </motion.text>
      </svg>
    </VizContainer>
  );
};

/* ── Neural Network / MLP ── */
const NeuralNetViz = ({
  summary,
  onHover,
  onLeave,
  tooltip,
  formatStats,
}: {
  summary?: { featureCount?: number; sampleCount?: number; targetColumn?: string };
  onHover: (text: string, evt: React.MouseEvent<SVGElement, MouseEvent>) => void;
  onLeave: () => void;
  tooltip: { text: string; x: number; y: number } | null;
  formatStats: (base: string) => string;
}) => {
  const layers = [3, 5, 5, 2];
  const layerX = [40, 110, 180, 250];
  const nodePositions = layers.map((count, li) =>
    Array.from({ length: count }).map((_, ni) => ({
      x: layerX[li],
      y: 25 + ni * (110 / (count - 1 || 1)),
    })),
  );

  return (
    <VizContainer label="Architecture" sublabel="Feed-Forward Network" summary={summary} tooltip={tooltip}>
      <svg width="290" height="150" viewBox="0 0 290 150" className="overflow-visible">
        {nodePositions.slice(0, -1).map((layer, li) =>
          layer.map((from, fi) =>
            nodePositions[li + 1].map((to, ti) => (
              <motion.line
                key={`${li}-${fi}-${ti}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke="hsl(var(--border))"
                strokeWidth="0.8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.25 }}
                transition={{ delay: li * 0.15 + fi * 0.02 }}
              />
            )),
          ),
        )}
        {nodePositions.map((layer, li) =>
          layer.map((node, ni) => (
            <motion.circle
              key={`n-${li}-${ni}`}
              cx={node.x}
              cy={node.y}
              r={li === 0 || li === layers.length - 1 ? 5 : 4}
              fill={
                li === 0
                  ? "hsl(var(--muted-foreground))"
                  : li === layers.length - 1
                    ? "hsl(var(--node-active))"
                    : "hsl(var(--accent))"
              }
              opacity={li === 0 || li === layers.length - 1 ? 0.85 : 0.5}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 20, delay: li * 0.1 + ni * 0.03 }}
              className="cursor-pointer"
              onMouseEnter={(evt) =>
                onHover(
                  formatStats(
                    li === 0
                      ? "Input neuron: receives one normalized feature value."
                      : li === layers.length - 1
                        ? "Output neuron: combines previous activations into the final prediction."
                        : "Hidden neuron: learns feature interactions via weighted sums + activation.",
                  ),
                  evt,
                )
              }
              onMouseMove={(evt) =>
                onHover(
                  formatStats(
                    li === 0
                      ? "Input neuron: receives one normalized feature value."
                      : li === layers.length - 1
                        ? "Output neuron: combines previous activations into the final prediction."
                        : "Hidden neuron: learns feature interactions via weighted sums + activation.",
                  ),
                  evt,
                )
              }
              onMouseLeave={onLeave}
            />
          )),
        )}
        {["Input", "Hidden", "Hidden", "Output"].map((label, i) => (
          <motion.text
            key={label + i}
            x={layerX[i]}
            y="148"
            fontSize="8"
            fill="hsl(var(--muted-foreground))"
            textAnchor="middle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            transition={{ delay: 0.8 }}
          >
            {label}
          </motion.text>
        ))}
      </svg>
    </VizContainer>
  );
};

/* ── Fallback ── */
const GenericModelViz = ({
  name,
  summary,
  tooltip,
}: {
  name: string;
  summary?: { featureCount?: number; sampleCount?: number; targetColumn?: string };
  tooltip?: { text: string; x: number; y: number } | null;
}) => {
  return (
    <VizContainer label="Model flow" sublabel="High-level pipeline" summary={summary} tooltip={tooltip ?? null}>
      <div className="flex h-24 w-full items-center justify-center rounded-xl border border-dashed border-border bg-card/60">
        <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Structural View: {name}</span>
      </div>
    </VizContainer>
  );
};

export default ModelArchitectureViz;
