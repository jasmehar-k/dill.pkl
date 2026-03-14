import { motion } from "framer-motion";
import { Rows3, Columns3, AlertTriangle, Hash, Type, Eye } from "lucide-react";

interface DatasetSummaryProps {
  datasetLoaded: boolean;
}

const DatasetSummary = ({ datasetLoaded }: DatasetSummaryProps) => {
  if (!datasetLoaded) return null;

  const stats = [
    { icon: <Rows3 className="w-4 h-4" />, label: "Rows", value: "2,000" },
    { icon: <Columns3 className="w-4 h-4" />, label: "Columns", value: "15" },
    { icon: <AlertTriangle className="w-4 h-4" />, label: "Missing Values", value: "3.2%" },
    { icon: <Hash className="w-4 h-4" />, label: "Numeric", value: "12" },
    { icon: <Type className="w-4 h-4" />, label: "Categorical", value: "3" },
  ];

  return (
    <motion.div
      className="glass-card p-4 space-y-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Dataset Summary</h3>
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-secondary text-[11px] font-medium text-foreground hover:bg-surface-hover transition-colors">
          <Eye className="w-3 h-3" />
          Preview Data
        </button>
      </div>
      <div className="grid grid-cols-5 gap-3">
        {stats.map((s) => (
          <div key={s.label} className="flex flex-col items-center gap-1.5 p-3 rounded-lg bg-secondary/50">
            <span className="text-muted-foreground">{s.icon}</span>
            <span className="text-lg font-bold font-mono text-foreground">{s.value}</span>
            <span className="text-[10px] text-muted-foreground">{s.label}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

export default DatasetSummary;
