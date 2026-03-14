import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { Activity, Database, Brain, Target } from "lucide-react";

interface PipelineHeaderProps {
  completedCount: number;
  totalStages: number;
  progress: number;
}

const PipelineHeader = ({ completedCount, totalStages, progress }: PipelineHeaderProps) => {
  const isComplete = completedCount === totalStages;
  const isRunning = completedCount > 0 && !isComplete;

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🥒</span>
          <div>
            <h1 className="text-xl font-bold font-mono gradient-text">dill.pkl</h1>
            <div className="flex items-center gap-2 mt-0.5">
              <motion.div
                className={`w-2 h-2 rounded-full ${isComplete ? "bg-accent" : isRunning ? "bg-primary" : "bg-muted-foreground"}`}
                animate={isRunning ? { scale: [1, 1.3, 1], opacity: [1, 0.5, 1] } : {}}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-xs font-mono text-muted-foreground">
                {isComplete ? "Pipeline Complete" : isRunning ? "Running..." : "Idle"}
              </span>
            </div>
          </div>
        </div>
        <span className="text-sm font-mono text-primary">{completedCount}/{totalStages} stages</span>
      </div>

      <div className="space-y-1.5">
        <div className="h-2 bg-secondary rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full"
            style={{ background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))" }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
          <span>Progress</span>
          <span>{Math.round(progress)}%</span>
        </div>
      </div>

      {completedCount > 0 && (
        <motion.div
          className="flex flex-wrap gap-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <InfoChip icon={<Brain className="w-3 h-3" />} label="Model" value="RandomForestClassifier" />
          <InfoChip icon={<Database className="w-3 h-3" />} label="Dataset" value="housing_prices.csv" />
          {isComplete && (
            <>
              <InfoChip icon={<Target className="w-3 h-3" />} label="Accuracy" value="91.2%" highlight />
              <InfoChip icon={<Activity className="w-3 h-3" />} label="F1" value="93.3%" highlight />
            </>
          )}
        </motion.div>
      )}
    </div>
  );
};

const InfoChip = ({ icon, label, value, highlight }: { icon: ReactNode; label: string; value: string; highlight?: boolean }) => (
  <div className="flex items-center gap-1.5 text-[11px] font-mono">
    <span className="text-muted-foreground">{icon}</span>
    <span className="text-muted-foreground">{label}:</span>
    <span className={highlight ? "text-accent font-semibold" : "text-foreground"}>{value}</span>
  </div>
);

export default PipelineHeader;
