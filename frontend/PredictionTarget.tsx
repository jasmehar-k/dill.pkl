import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Sparkles, ChevronDown } from "lucide-react";

const COLUMNS = ["price", "age", "income", "city", "score", "bedrooms", "sqft", "rating", "hours", "zip_code", "year_built", "lot_size"];

interface PredictionTargetProps {
  datasetLoaded: boolean;
}

const PredictionTarget = ({ datasetLoaded }: PredictionTargetProps) => {
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [manualInput, setManualInput] = useState("");

  if (!datasetLoaded) return null;

  const filtered = COLUMNS.filter((c) => c.toLowerCase().includes(searchQuery.toLowerCase()));

  const handleSelect = (col: string) => {
    setSelectedColumn(col);
    setIsDropdownOpen(false);
    setSearchQuery("");
    setManualInput("");
  };

  const handleAutoDetect = () => {
    setSelectedColumn("price");
  };

  return (
    <motion.div
      className="glass-card p-4 space-y-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">What are you trying to predict?</h3>
          <p className="text-[11px] text-muted-foreground mt-0.5">Select or type the column the model should predict.</p>
        </div>
        <button
          onClick={handleAutoDetect}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary/10 text-primary text-[11px] font-medium hover:bg-primary/20 transition-colors"
        >
          <Sparkles className="w-3 h-3" />
          Auto-detect target
        </button>
      </div>

      <div className="flex gap-2">
        {/* Searchable dropdown */}
        <div className="relative flex-1">
          <button
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-secondary text-sm text-foreground hover:bg-surface-hover transition-colors"
          >
            <span className={selectedColumn ? "text-foreground" : "text-muted-foreground"}>
              {selectedColumn || "Select column..."}
            </span>
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          </button>

          {isDropdownOpen && (
            <motion.div
              className="absolute z-20 top-full mt-1 w-full glass-card border border-border/50 rounded-lg overflow-hidden"
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="p-2 border-b border-border/30">
                <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-secondary">
                  <Search className="w-3.5 h-3.5 text-muted-foreground" />
                  <input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search columns..."
                    className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                    autoFocus
                  />
                </div>
              </div>
              <div className="max-h-40 overflow-y-auto scrollbar-thin p-1">
                {filtered.map((col) => (
                  <button
                    key={col}
                    onClick={() => handleSelect(col)}
                    className="w-full text-left px-3 py-1.5 text-sm font-mono text-foreground rounded-md hover:bg-secondary transition-colors"
                  >
                    {col}
                  </button>
                ))}
                {filtered.length === 0 && (
                  <p className="text-xs text-muted-foreground px-3 py-2">No columns found</p>
                )}
              </div>
            </motion.div>
          )}
        </div>

        {/* Manual input */}
        <input
          value={manualInput}
          onChange={(e) => setManualInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && manualInput.trim()) {
              handleSelect(manualInput.trim());
            }
          }}
          placeholder="Or type column name..."
          className="flex-1 px-3 py-2 rounded-lg bg-secondary text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
        />
      </div>

      {selectedColumn && (
        <motion.div
          className="flex items-center gap-2 text-[11px] font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <span className="text-muted-foreground">Target:</span>
          <span className="px-2 py-0.5 rounded-md bg-accent/10 text-accent font-semibold">{selectedColumn}</span>
        </motion.div>
      )}
    </motion.div>
  );
};

export default PredictionTarget;
