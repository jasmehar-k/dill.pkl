import { motion } from "framer-motion";
import { BarChart3, FileCode, FileText, Package } from "lucide-react";
import { getDownloadUrl, type MetricsResponse } from "@/lib/api";

interface ResultsPanelProps {
  isComplete: boolean;
  metrics: MetricsResponse | null;
  results: Record<string, unknown> | null;
  explanation: Record<string, unknown> | null;
}

const ResultsPanel = ({ isComplete, metrics, results, explanation }: ResultsPanelProps) => {
  if (!isComplete) return null;

  const packageReady = Boolean(results?.package_ready);

  const downloadMetrics = () => {
    if (!metrics) return;
    const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "metrics.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const secondaryDownloads = [
    {
      icon: <FileCode className="h-4 w-4" />,
      label: "Download model (.pkl)",
      sublabel: results?.model_path ? "Raw model artifact" : "Unavailable",
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
  ];

  return (
    <motion.div
      className="glass-card space-y-4 p-4"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
        <span className="text-accent">Success</span> Pipeline complete. Export results.
      </h3>

      {/* Primary CTA – deployment package */}
      <button
        onClick={() => window.open(getDownloadUrl("deployment-package"), "_blank", "noopener,noreferrer")}
        disabled={!packageReady}
        className="group flex w-full items-center gap-3 rounded-xl border border-accent/30 bg-accent/10 p-4 text-left transition-all hover:border-accent/60 hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent/20 text-accent transition-colors group-hover:bg-accent/30">
          <Package className="h-5 w-5" />
        </span>
        <div className="min-w-0">
          <p className="text-sm font-semibold text-foreground">Download deployment package</p>
          <p className="text-[11px] text-muted-foreground">
            {packageReady
              ? "app.py · schema.json · model.pkl · Dockerfile · docker-compose.yml · README.md"
              : "Package not ready — run the full pipeline first"}
          </p>
        </div>
      </button>

      {/* Secondary downloads */}
      <div className="grid gap-2 md:grid-cols-3">
        {secondaryDownloads.map((download) => (
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
