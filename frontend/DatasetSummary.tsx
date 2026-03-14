import { motion } from "framer-motion";
import { AlertTriangle, Columns3, Database, Hash, Rows3, Type } from "lucide-react";
import type { DatasetColumn, DatasetSummary as DatasetSummaryData } from "@/lib/api";

interface DatasetSummaryProps {
  summary: DatasetSummaryData | null;
  columns: DatasetColumn[];
  targetColumn: string | null;
}

const DatasetSummary = ({ summary, columns, targetColumn }: DatasetSummaryProps) => {
  if (!summary) return null;

  const numericCount = columns.filter((column) => column.is_numeric).length;
  const categoricalCount = columns.length - numericCount;
  const averageMissing =
    columns.length > 0
      ? columns.reduce((total, column) => total + column.missing_pct, 0) / columns.length
      : 0;

  const stats = [
    { icon: <Rows3 className="h-4 w-4" />, label: "Rows", value: summary.rows.toLocaleString() },
    { icon: <Columns3 className="h-4 w-4" />, label: "Columns", value: summary.columns.toString() },
    { icon: <AlertTriangle className="h-4 w-4" />, label: "Avg Missing", value: `${(averageMissing * 100).toFixed(1)}%` },
    { icon: <Hash className="h-4 w-4" />, label: "Numeric", value: numericCount.toString() },
    { icon: <Type className="h-4 w-4" />, label: "Categorical", value: categoricalCount.toString() },
  ];

  return (
    <motion.div
      className="glass-card relative z-0 space-y-4 p-4"
      data-chat-context-label="Dataset summary"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Dataset Summary</h3>
          <p className="mt-0.5 flex items-center gap-2 text-[11px] text-muted-foreground">
            <Database className="h-3.5 w-3.5" />
            {summary.filename}
          </p>
        </div>
        {targetColumn && (
          <span className="self-start rounded-lg bg-accent/10 px-3 py-1.5 text-[11px] font-medium text-accent">
            Target column: <span className="font-mono">{targetColumn}</span>
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        {stats.map((stat) => (
          <div key={stat.label} className="flex flex-col items-center gap-1.5 rounded-lg bg-secondary/50 p-3">
            <span className="text-muted-foreground">{stat.icon}</span>
            <span className="font-mono text-lg font-bold text-foreground">{stat.value}</span>
            <span className="text-[10px] text-muted-foreground">{stat.label}</span>
          </div>
        ))}
      </div>

      <div className="space-y-2">
        <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Columns in play</p>
        <div className="flex flex-wrap gap-2">
          {columns.slice(0, 12).map((column) => (
            <span
              key={column.name}
              className={`rounded-md px-2 py-1 text-[11px] font-mono ${
                column.name === targetColumn
                  ? "bg-accent/10 text-accent"
                  : "bg-secondary text-secondary-foreground"
              }`}
            >
              {column.name}
            </span>
          ))}
          {columns.length > 12 && (
            <span className="rounded-md bg-secondary px-2 py-1 text-[11px] text-muted-foreground">
              +{columns.length - 12} more
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default DatasetSummary;
