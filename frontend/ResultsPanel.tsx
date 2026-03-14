import { motion } from "framer-motion";
import { BarChart3, Download, FileCode, FileText } from "lucide-react";
import { getDownloadUrl, type MetricsResponse } from "@/lib/api";

interface ResultsPanelProps {
  isComplete: boolean;
  metrics: MetricsResponse | null;
  results: Record<string, unknown> | null;
}

const ResultsPanel = ({ isComplete, metrics, results }: ResultsPanelProps) => {
  if (!isComplete) return null;

  const downloadMetrics = () => {
    if (!metrics) return;

    const blob = new Blob([JSON.stringify(metrics, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "metrics.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const downloadDeploymentSummary = () => {
    if (!results) return;

    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "deployment-summary.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const downloads = [
    {
      icon: <FileCode className="h-4 w-4" />,
      label: "Download model (.pkl)",
      sublabel: results?.model_path ? "Backend artifact" : "Unavailable",
      onClick: () => window.open(getDownloadUrl("model"), "_blank", "noopener,noreferrer"),
      disabled: !results?.model_path,
    },
    {
      icon: <FileText className="h-4 w-4" />,
      label: "Download training logs",
      sublabel: "Pipeline JSON log",
      onClick: () => window.open(getDownloadUrl("logs"), "_blank", "noopener,noreferrer"),
      disabled: false,
    },
    {
      icon: <BarChart3 className="h-4 w-4" />,
      label: "Download metrics",
      sublabel: metrics?.performance_summary || "Metrics JSON",
      onClick: downloadMetrics,
      disabled: !metrics,
    },
    {
      icon: <Download className="h-4 w-4" />,
      label: "Download deployment summary",
      sublabel: results?.pipeline_id ? `Pipeline ${String(results.pipeline_id)}` : "Summary JSON",
      onClick: downloadDeploymentSummary,
      disabled: !results,
    },
  ];

  return (
    <motion.div
      className="glass-card space-y-3 p-4"
      data-chat-context-label="Results panel"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
        <span className="text-accent">Success</span> Pipeline complete. Export results.
      </h3>
      <div className="grid gap-2 md:grid-cols-2">
        {downloads.map((download) => (
          <button
            key={download.label}
            onClick={download.onClick}
            disabled={download.disabled}
            className="group flex items-center gap-3 rounded-lg bg-secondary/50 p-3 text-left transition-colors hover:bg-surface-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span className="text-muted-foreground transition-colors group-hover:text-accent">{download.icon}</span>
            <div>
              <p className="text-xs font-medium text-foreground">{download.label}</p>
              <p className="text-[10px] text-muted-foreground">{download.sublabel}</p>
            </div>
          </button>
        ))}
      </div>
    </motion.div>
  );
};

export default ResultsPanel;
