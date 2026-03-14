import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, X, Send, Bot, User } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const QUICK_RESPONSES: Record<string, string> = {
  "what does preprocessing do": "Preprocessing prepares raw data for modeling. It handles missing values (imputation), encodes categorical variables (one-hot, label encoding), scales numerical features (StandardScaler, MinMaxScaler), and splits data into train/test sets.",
  "why is my model overfitting": "Overfitting happens when your model learns noise instead of patterns. Common fixes:\n• Increase training data\n• Use regularization (L1/L2)\n• Reduce model complexity\n• Apply dropout\n• Use cross-validation",
  "what does this loss curve mean": "The loss curve shows how your model's error decreases during training. A good curve:\n• Training loss decreases steadily\n• Validation loss follows but stays slightly higher\n\n⚠️ If validation loss starts increasing, that's overfitting.",
  "what is a confusion matrix": "A confusion matrix shows classifier performance:\n• True Positives (TP): Correctly predicted positive\n• True Negatives (TN): Correctly predicted negative\n• False Positives (FP): Type I error\n• False Negatives (FN): Type II error",
  "what is feature engineering": "Feature engineering creates new meaningful features from raw data:\n• Combining features (age × income)\n• Extracting date parts\n• Binning continuous variables\n• Text vectorization (TF-IDF)",
};

const findResponse = (input: string): string => {
  const lower = input.toLowerCase().trim();
  for (const [key, val] of Object.entries(QUICK_RESPONSES)) {
    if (lower.includes(key) || key.includes(lower)) return val;
  }
  if (lower.includes("preprocess")) return QUICK_RESPONSES["what does preprocessing do"];
  if (lower.includes("overfit")) return QUICK_RESPONSES["why is my model overfitting"];
  if (lower.includes("loss")) return QUICK_RESPONSES["what does this loss curve mean"];
  if (lower.includes("confusion")) return QUICK_RESPONSES["what is a confusion matrix"];
  if (lower.includes("feature")) return QUICK_RESPONSES["what is feature engineering"];
  return "I can help explain ML pipeline concepts! Try asking about:\n• Preprocessing\n• Overfitting\n• Loss curves\n• Confusion matrices\n• Feature engineering";
};

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hi! 🥒 I'm your ML pipeline assistant. Ask me anything about the pipeline stages!" },
  ]);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    const userMsg: Message = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setTimeout(() => {
      const response = findResponse(userMsg.content);
      setMessages((prev) => [...prev, { role: "assistant", content: response }]);
    }, 500);
  };

  return (
    <>
      <motion.button
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-primary flex items-center justify-center glow-purple hover:scale-110 transition-transform"
        onClick={() => setIsOpen(!isOpen)}
        whileTap={{ scale: 0.9 }}
      >
        {isOpen ? <X className="w-6 h-6 text-primary-foreground" /> : <MessageCircle className="w-6 h-6 text-primary-foreground" />}
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed bottom-24 right-6 z-50 w-80 sm:w-96 h-[28rem] glass-card border border-border/50 rounded-xl overflow-hidden flex flex-col"
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            transition={{ type: "spring", damping: 25 }}
          >
            <div className="px-4 py-3 border-b border-border/30 flex items-center gap-2">
              <Bot className="w-4 h-4 text-accent" />
              <span className="text-sm font-semibold text-foreground">dill.pkl Assistant</span>
              <span className="text-[10px] font-mono text-accent ml-auto">🥒 online</span>
            </div>

            <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-3">
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {msg.role === "assistant" && (
                    <div className="w-6 h-6 rounded-full bg-accent/20 flex items-center justify-center shrink-0 mt-0.5">
                      <Bot className="w-3 h-3 text-accent" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-xl px-3 py-2 text-[12px] leading-relaxed whitespace-pre-line ${
                      msg.role === "user" ? "bg-primary/20 text-foreground" : "bg-secondary text-secondary-foreground"
                    }`}
                  >
                    {msg.content}
                  </div>
                  {msg.role === "user" && (
                    <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center shrink-0 mt-0.5">
                      <User className="w-3 h-3 text-primary" />
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            <div className="p-3 border-t border-border/30">
              <div className="flex gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                  placeholder="Ask about ML pipeline..."
                  className="flex-1 bg-secondary rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                />
                <button
                  onClick={handleSend}
                  className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center hover:bg-primary/80 transition-colors"
                >
                  <Send className="w-4 h-4 text-primary-foreground" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default ChatBot;
