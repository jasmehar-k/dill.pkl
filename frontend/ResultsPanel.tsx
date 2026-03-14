import { motion } from "framer-motion";
import { Download, FileText, FileCode, BarChart3 } from "lucide-react";

interface ResultsPanelProps {
  isComplete: boolean;
}

const ResultsPanel = ({ isComplete }: ResultsPanelProps) => {
  if (!isComplete) return null;

  const downloads = [
    { icon: <FileCode className="w-4 h-4" />, label: "Download Model (.pkl)", sublabel: "2.4 MB" },
    { icon: <Download className="w-4 h-4" />, label: "Download Pipeline", sublabel: "Config + Model" },
    { icon: <BarChart3 className="w-4 h-4" />, label: "Download Metrics Report", sublabel: "PDF" },
    { icon: <FileText className="w-4 h-4" />, label: "Download Training Logs", sublabel: "JSON" },
  ];

  return (
    <motion.div
      className="glass-card p-4 space-y-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
        <span className="text-accent">✓</span> Pipeline Complete — Export Results
      </h3>
      <div className="grid grid-cols-2 gap-2">
        {downloads.map((d) => (
          <button
            key={d.label}
            className="flex items-center gap-3 p-3 rounded-lg bg-secondary/50 hover:bg-surface-hover transition-colors text-left group"
          >
            <span className="text-muted-foreground group-hover:text-accent transition-colors">{d.icon}</span>
            <div>
              <p className="text-xs font-medium text-foreground">{d.label}</p>
              <p className="text-[10px] text-muted-foreground">{d.sublabel}</p>
            </div>
          </button>
        ))}
      </div>
    </motion.div>
  );
};

export default ResultsPanel;
