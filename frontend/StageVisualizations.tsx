import type { ReactNode } from "react";
import { PipelineStage } from "@/data/pipelineStages";

const Heatmap = () => {
  const data = [
    [1.0, 0.8, 0.3, -0.2, 0.5],
    [0.8, 1.0, 0.4, -0.1, 0.6],
    [0.3, 0.4, 1.0, 0.7, -0.3],
    [-0.2, -0.1, 0.7, 1.0, 0.2],
    [0.5, 0.6, -0.3, 0.2, 1.0],
  ];
  const labels = ["age", "income", "score", "hours", "rating"];

  return (
    <div className="space-y-1">
      <div className="flex gap-1">
        <div className="w-12" />
        {labels.map((l) => (
          <div key={l} className="w-12 text-[10px] text-muted-foreground text-center truncate">{l}</div>
        ))}
      </div>
      {data.map((row, i) => (
        <div key={i} className="flex gap-1 items-center">
          <div className="w-12 text-[10px] text-muted-foreground text-right pr-1 truncate">{labels[i]}</div>
          {row.map((val, j) => {
            const hue = val > 0 ? 265 : 145;
            const opacity = Math.abs(val);
            return (
              <div
                key={j}
                className="w-12 h-10 rounded-sm flex items-center justify-center text-[10px] font-mono text-foreground/80"
                style={{ backgroundColor: `hsl(${hue} 80% 60% / ${opacity * 0.6})` }}
              >
                {val.toFixed(1)}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

const LossCurve = () => {
  const trainLoss = [0.9, 0.65, 0.45, 0.32, 0.22, 0.16, 0.12, 0.09, 0.07, 0.06];
  const valLoss = [0.92, 0.7, 0.52, 0.4, 0.33, 0.29, 0.27, 0.26, 0.26, 0.27];
  const h = 120;
  const w = 280;

  const toPath = (data: number[]) =>
    data.map((v, i) => `${(i / (data.length - 1)) * w},${h - v * h}`).join(" ");

  return (
    <svg viewBox={`-10 -10 ${w + 20} ${h + 30}`} className="w-full max-w-xs">
      <polyline points={toPath(trainLoss)} fill="none" stroke="hsl(265 80% 60%)" strokeWidth="2" />
      <polyline points={toPath(valLoss)} fill="none" stroke="hsl(145 70% 50%)" strokeWidth="2" strokeDasharray="4" />
      <text x={w / 2} y={h + 20} textAnchor="middle" className="fill-muted-foreground text-[10px]">Epoch</text>
      <text x={w - 10} y={h - trainLoss[trainLoss.length - 1] * h - 5} className="fill-primary text-[9px]">train</text>
      <text x={w - 10} y={h - valLoss[valLoss.length - 1] * h - 5} className="fill-accent text-[9px]">val</text>
    </svg>
  );
};

const BarChart = () => {
  const features = [
    { name: "income", value: 0.92 },
    { name: "age", value: 0.78 },
    { name: "score", value: 0.65 },
    { name: "hours", value: 0.45 },
    { name: "rating", value: 0.3 },
  ];
  return (
    <div className="space-y-2">
      {features.map((f) => (
        <div key={f.name} className="flex items-center gap-2">
          <span className="w-14 text-[11px] text-muted-foreground font-mono text-right">{f.name}</span>
          <div className="flex-1 h-5 bg-secondary rounded-sm overflow-hidden">
            <div
              className="h-full rounded-sm"
              style={{ width: `${f.value * 100}%`, background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))" }}
            />
          </div>
          <span className="text-[11px] font-mono text-foreground/70 w-8">{(f.value * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
};

const DataTable = () => {
  const cols = ["feature", "missing", "type", "action"];
  const rows = [
    ["age", "0%", "int", "scale"],
    ["income", "2.1%", "float", "scale"],
    ["city", "0%", "str", "encode"],
    ["score", "5.3%", "float", "impute"],
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] font-mono">
        <thead>
          <tr className="border-b border-border/50">
            {cols.map((c) => (
              <th key={c} className="text-left py-1 px-2 text-muted-foreground font-medium">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-border/20">
              {row.map((cell, j) => (
                <td key={j} className="py-1 px-2 text-foreground/80">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const ConfusionMatrix = () => {
  const matrix = [[142, 8], [12, 138]];
  const labels = ["Pos", "Neg"];
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex gap-1">
        <div className="w-10" />
        {labels.map((l) => (
          <div key={l} className="w-16 text-center text-[10px] text-muted-foreground">{l}</div>
        ))}
      </div>
      {matrix.map((row, i) => (
        <div key={i} className="flex gap-1 items-center">
          <div className="w-10 text-[10px] text-muted-foreground text-right">{labels[i]}</div>
          {row.map((val, j) => (
            <div
              key={j}
              className={`w-16 h-14 rounded-md flex items-center justify-center font-mono text-sm font-semibold
                ${i === j ? "bg-primary/20 text-primary" : "bg-destructive/10 text-destructive"}`}
            >
              {val}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

const MetricsCard = () => {
  const metrics = [
    { label: "Accuracy", value: "93.4%" },
    { label: "Precision", value: "94.7%" },
    { label: "Recall", value: "92.0%" },
    { label: "F1 Score", value: "93.3%" },
  ];
  return (
    <div className="grid grid-cols-2 gap-3">
      {metrics.map((m) => (
        <div key={m.label} className="glass-card p-3 text-center">
          <div className="text-lg font-bold gradient-text">{m.value}</div>
          <div className="text-[10px] text-muted-foreground mt-1">{m.label}</div>
        </div>
      ))}
    </div>
  );
};

export const StageVisualization = ({ stage }: { stage: PipelineStage }) => {
  const vizMap: Record<string, ReactNode> = {
    heatmap: <Heatmap />,
    lossCurve: <LossCurve />,
    barChart: <BarChart />,
    table: <DataTable />,
    confusionMatrix: <ConfusionMatrix />,
    metrics: <MetricsCard />,
  };
  return <>{vizMap[stage.vizType] || null}</>;
};
