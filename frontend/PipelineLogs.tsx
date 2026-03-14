import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, BarChart3 } from "lucide-react";

interface LogEntry {
  stage: string;
  message: string;
}

const LOG_MESSAGES: Record<string, string[]> = {
  analysis: [
    "loading dataset housing_prices.csv",
    "detecting column types... 12 numerical, 3 categorical",
    "computing correlation matrix",
    "identified 2 outliers in 'price' column",
    "EDA complete ✓",
  ],
  preprocessing: [
    "handling missing values (2.1% in income, 5.3% in score)",
    "encoding categorical: city → one-hot",
    "applying StandardScaler to numerical features",
    "train/test split: 80/20",
    "preprocessing complete ✓",
  ],
  features: [
    "running SelectKBest (f_classif, k=10)",
    "top features: income (0.92), age (0.78), score (0.65)",
    "creating interaction feature: income × age",
    "feature engineering complete ✓",
  ],
  training: [
    "initializing RandomForestClassifier(n_estimators=100)",
    "fitting model on 1,600 samples...",
    "epoch 1/10 — loss: 0.654",
    "epoch 5/10 — loss: 0.223",
    "epoch 10/10 — loss: 0.061",
    "training complete ✓",
  ],
  loss: [
    "computing training loss: 0.061",
    "computing validation loss: 0.089",
    "no overfitting detected",
    "optimal stopping at epoch 10",
    "loss analysis complete ✓",
  ],
  evaluation: [
    "running predictions on test set (400 samples)",
    "accuracy: 0.9120",
    "precision: 0.947 | recall: 0.920",
    "F1 score: 0.933",
    "evaluation complete ✓",
  ],
  results: [
    "saving model → model.pkl 🥒",
    "model size: 2.4 MB",
    "generating classification report",
    "pipeline finished — all stages complete ✓",
  ],
};

interface PipelineLogsProps {
  activeStageId: string | null;
  completedStageIds: string[];
}

const PipelineLogs = ({ activeStageId, completedStageIds }: PipelineLogsProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState<"visual" | "debug">("debug");
  const logEntries: LogEntry[] = [];

  const stageOrder = ["analysis", "preprocessing", "features", "training", "loss", "evaluation", "results"];
  for (const stageId of stageOrder) {
    if (completedStageIds.includes(stageId) || stageId === activeStageId) {
      const msgs = LOG_MESSAGES[stageId] || [];
      const isActive = stageId === activeStageId;
      const displayMsgs = isActive ? msgs.slice(0, Math.min(2, msgs.length)) : msgs;
      displayMsgs.forEach((m) => logEntries.push({ stage: stageId, message: m }));
    }
  }

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logEntries.length]);

  const stageLabel = (id: string) => {
    const map: Record<string, string> = {
      analysis: "Analysis", preprocessing: "Preprocess", features: "Features",
      training: "Training", loss: "Loss", evaluation: "Eval", results: "Results",
    };
    return map[id] || id;
  };

  return (
    <div className="glass-card overflow-hidden">
      {/* Tab header */}
      <div className="flex items-center gap-0 border-b border-border/30">
        <button
          onClick={() => setActiveTab("visual")}
          className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors ${
            activeTab === "visual" ? "text-accent border-b-2 border-accent" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          Visual
        </button>
        <button
          onClick={() => setActiveTab("debug")}
          className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors ${
            activeTab === "debug" ? "text-accent border-b-2 border-accent" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Terminal className="w-3.5 h-3.5" />
          Debug
        </button>
        <div className="ml-auto flex gap-1 pr-3">
          <div className="w-2 h-2 rounded-full bg-destructive/60" />
          <div className="w-2 h-2 rounded-full bg-primary/40" />
          <div className="w-2 h-2 rounded-full bg-accent/60" />
        </div>
      </div>

      {activeTab === "visual" ? (
        <div className="p-4 space-y-4">
          {completedStageIds.length === 0 && !activeStageId ? (
            <p className="text-sm text-muted-foreground text-center py-6">Pipeline hasn't started yet...</p>
          ) : (
            <>
              {completedStageIds.includes("evaluation") && (
                <div className="grid grid-cols-4 gap-3">
                  {[
                    { label: "Accuracy", value: "91.2%" },
                    { label: "Precision", value: "94.7%" },
                    { label: "Recall", value: "92.0%" },
                    { label: "F1", value: "93.3%" },
                  ].map((m) => (
                    <div key={m.label} className="text-center p-2 rounded-lg bg-secondary/50">
                      <div className="text-sm font-bold gradient-text">{m.value}</div>
                      <div className="text-[10px] text-muted-foreground">{m.label}</div>
                    </div>
                  ))}
                </div>
              )}
              {completedStageIds.includes("features") && (
                <div className="space-y-1">
                  <p className="text-[11px] text-muted-foreground font-medium">Feature Importance</p>
                  {[
                    { name: "income", value: 0.92 },
                    { name: "age", value: 0.78 },
                    { name: "score", value: 0.65 },
                  ].map((f) => (
                    <div key={f.name} className="flex items-center gap-2">
                      <span className="w-12 text-[10px] text-muted-foreground font-mono text-right">{f.name}</span>
                      <div className="flex-1 h-3 bg-secondary rounded-sm overflow-hidden">
                        <div className="h-full rounded-sm" style={{ width: `${f.value * 100}%`, background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))" }} />
                      </div>
                      <span className="text-[10px] font-mono text-foreground/60 w-6">{(f.value * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
              {!completedStageIds.includes("features") && (
                <p className="text-sm text-muted-foreground text-center py-4">Waiting for more stages to complete...</p>
              )}
            </>
          )}
        </div>
      ) : (
        <div ref={scrollRef} className="p-3 h-44 overflow-y-auto scrollbar-thin font-mono text-[11px] space-y-0.5">
          <AnimatePresence>
            {logEntries.map((entry, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.02 }}
                className="flex gap-2"
              >
                <span className="text-accent shrink-0">[{stageLabel(entry.stage)}]</span>
                <span className="text-foreground/70">{entry.message}</span>
              </motion.div>
            ))}
          </AnimatePresence>
          {activeStageId && (
            <motion.div
              className="text-primary"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              █
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
};

export default PipelineLogs;
