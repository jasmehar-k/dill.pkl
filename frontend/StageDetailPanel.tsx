import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { PipelineStage } from "@/data/pipelineStages";
import { StageVisualization } from "./StageVisualizations";

interface StageDetailPanelProps {
  stage: PipelineStage | null;
  onClose: () => void;
}

const StageDetailPanel = ({ stage, onClose }: StageDetailPanelProps) => {
  return (
    <AnimatePresence>
      {stage && (
        <>
          <motion.div
            className="fixed inset-0 bg-background/60 backdrop-blur-sm z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.div
            className="fixed right-0 top-0 h-full w-full max-w-lg z-50 glass-card border-l border-border/50 overflow-y-auto scrollbar-thin"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
          >
            <div className="p-6 space-y-6">
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
                  className="p-2 rounded-lg hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-2">
                <h3 className="text-sm font-medium text-accent">About this stage</h3>
                <p className="text-sm text-secondary-foreground leading-relaxed">{stage.details}</p>
              </div>

              {/* Feature Engineering specific content */}
              {stage.id === "features" && (
                <div className="space-y-3">
                  <h3 className="text-sm font-medium text-accent">Feature Details</h3>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="glass-card p-3 space-y-1">
                      <p className="text-[10px] text-muted-foreground">Selected</p>
                      <p className="text-sm font-mono text-accent">10</p>
                    </div>
                    <div className="glass-card p-3 space-y-1">
                      <p className="text-[10px] text-muted-foreground">Dropped</p>
                      <p className="text-sm font-mono text-destructive">5</p>
                    </div>
                    <div className="glass-card p-3 space-y-1">
                      <p className="text-[10px] text-muted-foreground">Generated</p>
                      <p className="text-sm font-mono text-primary">3</p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button className="px-3 py-1.5 rounded-lg bg-primary/10 text-primary text-[11px] font-medium hover:bg-primary/20 transition-colors">
                      Correlation Heatmap
                    </button>
                    <button className="px-3 py-1.5 rounded-lg bg-accent/10 text-accent text-[11px] font-medium hover:bg-accent/20 transition-colors">
                      Feature Importance
                    </button>
                  </div>
                </div>
              )}

              <div className="space-y-2">
                <h3 className="text-sm font-medium text-accent">Visualization</h3>
                <div className="glass-card p-4">
                  <StageVisualization stage={stage} />
                </div>
              </div>

              <div className="space-y-2">
                <h3 className="text-sm font-medium text-accent">Code</h3>
                <pre className="glass-card p-4 text-[12px] font-mono text-foreground/80 overflow-x-auto leading-relaxed whitespace-pre-wrap">
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

export default StageDetailPanel;
