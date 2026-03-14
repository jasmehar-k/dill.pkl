import type { TaskType } from "@/lib/api";
import { ModelArchitectureViz } from "@/components/ModelArchitectureViz";

interface TrainingVisualizationProps {
  modelName?: string;
  featureCount?: number;
  sampleCount?: number;
  targetColumn?: string | null;
  topFeature?: string | null;
  taskType: TaskType;
}

const TrainingVisualization = ({
  modelName = "",
  featureCount,
  sampleCount,
  targetColumn,
  topFeature,
  taskType,
}: TrainingVisualizationProps) => {
  if (!modelName) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium text-accent">Model visualization</h3>
      <ModelArchitectureViz
        modelName={modelName || "Selected model"}
        taskType={taskType}
        featureCount={featureCount}
        sampleCount={sampleCount}
        targetColumn={targetColumn || undefined}
        topFeature={topFeature || undefined}
      />
    </div>
  );
};

export default TrainingVisualization;
