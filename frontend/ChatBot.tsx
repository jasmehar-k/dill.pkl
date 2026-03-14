import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Bot, MessageCircle, Send, User, X } from "lucide-react";
import type { MetricsResponse, TaskType } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatBotProps {
  datasetName: string | null;
  targetColumn: string | null;
  taskType: TaskType;
  activeStageId: string | null;
  stageLogs: Record<string, string[]>;
  metrics: MetricsResponse | null;
}

const ChatBot = ({ datasetName, targetColumn, taskType, activeStageId, stageLogs, metrics }: ChatBotProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "I can explain what the pipeline is doing as your dataset moves through each stage.",
    },
  ]);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const contextSummary = useMemo(() => {
    const parts = [];
    if (datasetName) parts.push(`Dataset: ${datasetName}`);
    if (targetColumn) parts.push(`Target: ${targetColumn}`);
    parts.push(`Task: ${taskType}`);
    if (activeStageId) parts.push(`Active stage: ${activeStageId}`);
    if (metrics?.performance_summary) parts.push(metrics.performance_summary);
    return parts.join(" | ");
  }, [activeStageId, datasetName, metrics?.performance_summary, targetColumn, taskType]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    setMessages((current) => {
      const next = [...current];
      next[0] = {
        role: "assistant",
        content: contextSummary || "I can explain what the pipeline is doing as your dataset moves through each stage.",
      };
      return next;
    });
  }, [contextSummary]);

  const handleSend = () => {
    if (!input.trim()) return;

    const userMsg: Message = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    setTimeout(() => {
      const response = findResponse(userMsg.content, {
        datasetName,
        targetColumn,
        taskType,
        activeStageId,
        stageLogs,
        metrics,
      });
      setMessages((prev) => [...prev, { role: "assistant", content: response }]);
    }, 350);
  };

  return (
    <>
      <motion.button
        className="glow-purple fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-primary transition-transform hover:scale-110"
        onClick={() => setIsOpen((current) => !current)}
        whileTap={{ scale: 0.9 }}
      >
        {isOpen ? <X className="h-6 w-6 text-primary-foreground" /> : <MessageCircle className="h-6 w-6 text-primary-foreground" />}
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="glass-card fixed bottom-24 right-6 z-50 flex h-[28rem] w-80 flex-col overflow-hidden rounded-xl border border-border/50 sm:w-96"
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            transition={{ type: "spring", damping: 25 }}
          >
            <div className="flex items-center gap-2 border-b border-border/30 px-4 py-3">
              <Bot className="h-4 w-4 text-accent" />
              <span className="text-sm font-semibold text-foreground">dill.pkl Assistant</span>
              <span className="ml-auto text-[10px] font-mono text-accent">online</span>
            </div>

            <div ref={scrollRef} className="scrollbar-thin flex-1 space-y-3 overflow-y-auto p-3">
              {messages.map((message, index) => (
                <motion.div
                  key={`${message.role}-${index}`}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-2 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/20">
                      <Bot className="h-3 w-3 text-accent" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] whitespace-pre-line rounded-xl px-3 py-2 text-[12px] leading-relaxed ${
                      message.role === "user" ? "bg-primary/20 text-foreground" : "bg-secondary text-secondary-foreground"
                    }`}
                  >
                    {message.content}
                  </div>
                  {message.role === "user" && (
                    <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/20">
                      <User className="h-3 w-3 text-primary" />
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            <div className="border-t border-border/30 p-3">
              <div className="flex gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                  placeholder="Ask about this dataset or stage..."
                  className="flex-1 rounded-lg bg-secondary px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                />
                <button
                  onClick={handleSend}
                  className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary transition-colors hover:bg-primary/80"
                >
                  <Send className="h-4 w-4 text-primary-foreground" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

const findResponse = (
  input: string,
  context: {
    datasetName: string | null;
    targetColumn: string | null;
    taskType: TaskType;
    activeStageId: string | null;
    stageLogs: Record<string, string[]>;
    metrics: MetricsResponse | null;
  },
) => {
  const lower = input.toLowerCase().trim();

  if (lower.includes("dataset")) {
    return context.datasetName
      ? `The current dataset is ${context.datasetName}.`
      : "No dataset is loaded yet.";
  }

  if (lower.includes("target")) {
    return context.targetColumn
      ? `The current target column is ${context.targetColumn}.`
      : "A target column has not been selected yet.";
  }

  if (lower.includes("task")) {
    return `The pipeline is configured for a ${context.taskType} task.`;
  }

  if (lower.includes("stage") || lower.includes("status")) {
    if (!context.activeStageId) {
      return "The pipeline is idle right now. Upload a dataset and run the pipeline to see active stage updates.";
    }
    const stageLogs = context.stageLogs[context.activeStageId] || [];
    const latestLog = stageLogs[stageLogs.length - 1];
    return latestLog
      ? `The active stage is ${context.activeStageId}. Latest log: ${latestLog}`
      : `The active stage is ${context.activeStageId}.`;
  }

  if (lower.includes("metric") || lower.includes("accuracy") || lower.includes("f1") || lower.includes("r2")) {
    if (!context.metrics) {
      return "Metrics are not available yet. They appear after evaluation finishes.";
    }
    return context.metrics.performance_summary || "Metrics are available in the results panel.";
  }

  if (lower.includes("log")) {
    const activeLogs = context.activeStageId ? context.stageLogs[context.activeStageId] || [] : [];
    if (activeLogs.length === 0) {
      return "No logs are available for the active stage yet.";
    }
    return `Recent ${context.activeStageId} logs:\n${activeLogs.slice(-3).join("\n")}`;
  }

  return [
    context.datasetName ? `Dataset: ${context.datasetName}` : "No dataset loaded yet.",
    context.targetColumn ? `Target: ${context.targetColumn}` : "Select a target column to continue.",
    context.activeStageId ? `Active stage: ${context.activeStageId}` : "The pipeline is waiting to start.",
  ].join("\n");
};

export default ChatBot;
