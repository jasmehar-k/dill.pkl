import type { FC } from "react";

interface GraphNode {
  id: string;
  label: string;
  x: number;
  y: number;
  accent?: boolean;
  dim?: boolean;
}

interface GraphEdge {
  from: string;
  to: string;
  dashed?: boolean;
}

interface ModelGraphProps {
  modelName?: string;
}

const DESCRIPTION_MAP: Record<string, string> = {
  forest: "Ensemble of shallow decision trees that vote on the final answer.",
  tree: "Single decision tree splitting data by the most informative features.",
  boost: "Sequential trees where each learner fixes the previous one's mistakes.",
  linear: "Weighted linear combination of features with a decision threshold.",
  svm: "Maximum-margin separator with support vectors defining the boundary.",
  knn: "Prediction from the nearest neighbours in feature space.",
  generic: "High-level view of how this model transforms features into predictions.",
};

const classifyModel = (raw: string): keyof typeof DESCRIPTION_MAP => {
  const name = raw.toLowerCase();
  if (name.includes("forest")) return "forest";
  if (name.includes("boost")) return "boost";
  if (name.includes("tree")) return "tree";
  if (name.includes("svm") || name.includes("svc") || name.includes("svr")) return "svm";
  if (name.includes("linear") || name.includes("logistic") || name.includes("ridge") || name.includes("lasso")) return "linear";
  if (name.includes("knn") || name.includes("neighbor")) return "knn";
  return "generic";
};

const buildGraph = (type: keyof typeof DESCRIPTION_MAP): { nodes: GraphNode[]; edges: GraphEdge[]; badge: string } => {
  switch (type) {
    case "forest":
      return {
        badge: "Ensemble",
        nodes: [
          { id: "t1", label: "Tree 1", x: 90, y: 50 },
          { id: "t2", label: "Tree 2", x: 180, y: 70 },
          { id: "t3", label: "Tree 3", x: 270, y: 50 },
          { id: "vote", label: "Majority vote", x: 180, y: 150, accent: true },
          { id: "predict", label: "Prediction", x: 180, y: 200, accent: true },
        ],
        edges: [
          { from: "t1", to: "vote" },
          { from: "t2", to: "vote" },
          { from: "t3", to: "vote" },
          { from: "vote", to: "predict" },
        ],
      };
    case "tree":
      return {
        badge: "Decision path",
        nodes: [
          { id: "root", label: "Root split", x: 180, y: 36, accent: true },
          { id: "n1", label: "Feature A ≤ 4.2", x: 110, y: 100 },
          { id: "n2", label: "Feature B > 1.3", x: 250, y: 100 },
          { id: "l1", label: "Leaf: Class 0", x: 70, y: 170 },
          { id: "l2", label: "Leaf: Class 1", x: 150, y: 170 },
          { id: "l3", label: "Leaf: Class 0", x: 230, y: 170 },
          { id: "l4", label: "Leaf: Class 1", x: 300, y: 170 },
        ],
        edges: [
          { from: "root", to: "n1" },
          { from: "root", to: "n2" },
          { from: "n1", to: "l1" },
          { from: "n1", to: "l2" },
          { from: "n2", to: "l3" },
          { from: "n2", to: "l4" },
        ],
      };
    case "boost":
      return {
        badge: "Boosted stumps",
        nodes: [
          { id: "s1", label: "Tree 1", x: 90, y: 70 },
          { id: "s2", label: "Tree 2", x: 180, y: 110 },
          { id: "s3", label: "Tree 3", x: 270, y: 70 },
          { id: "comb", label: "Weighted sum", x: 180, y: 160, accent: true },
          { id: "pred", label: "Final prediction", x: 180, y: 200, accent: true },
        ],
        edges: [
          { from: "s1", to: "comb" },
          { from: "s2", to: "comb" },
          { from: "s3", to: "comb" },
          { from: "comb", to: "pred" },
          { from: "s1", to: "s2", dashed: true },
          { from: "s2", to: "s3", dashed: true },
        ],
      };
    case "svm":
      return {
        badge: "Margin based",
        nodes: [
          { id: "sv1", label: "Support vector", x: 90, y: 120 },
          { id: "sv2", label: "Support vector", x: 140, y: 80 },
          { id: "sv3", label: "Support vector", x: 230, y: 140 },
          { id: "plane", label: "Decision boundary", x: 180, y: 60, accent: true },
          { id: "pred", label: "Class decision", x: 180, y: 190, accent: true },
        ],
        edges: [
          { from: "sv1", to: "plane" },
          { from: "sv2", to: "plane" },
          { from: "sv3", to: "plane" },
          { from: "plane", to: "pred" },
        ],
      };
    case "linear":
      return {
        badge: "Linear boundary",
        nodes: [
          { id: "f1", label: "Feature 1", x: 80, y: 70 },
          { id: "f2", label: "Feature 2", x: 80, y: 130 },
          { id: "f3", label: "Feature 3", x: 80, y: 190 },
          { id: "combine", label: "Weights & bias", x: 190, y: 120, accent: true },
          { id: "decision", label: "Prediction", x: 290, y: 120, accent: true },
        ],
        edges: [
          { from: "f1", to: "combine" },
          { from: "f2", to: "combine" },
          { from: "f3", to: "combine" },
          { from: "combine", to: "decision" },
        ],
      };
    case "knn":
      return {
        badge: "Instance based",
        nodes: [
          { id: "p1", label: "Neighbor 1", x: 110, y: 70 },
          { id: "p2", label: "Neighbor 2", x: 70, y: 150 },
          { id: "p3", label: "Neighbor 3", x: 150, y: 170 },
          { id: "query", label: "Query point", x: 230, y: 110, accent: true },
          { id: "vote", label: "Majority vote", x: 300, y: 150, accent: true },
        ],
        edges: [
          { from: "p1", to: "query" },
          { from: "p2", to: "query" },
          { from: "p3", to: "query" },
          { from: "query", to: "vote" },
        ],
      };
    default:
      return {
        badge: "Model flow",
        nodes: [
          { id: "input", label: "Features", x: 80, y: 120 },
          { id: "hidden", label: "Model logic", x: 190, y: 120, accent: true },
          { id: "output", label: "Prediction", x: 300, y: 120, accent: true },
        ],
        edges: [
          { from: "input", to: "hidden" },
          { from: "hidden", to: "output" },
        ],
      };
  }
};

const NodeCircle = ({ node }: { node: GraphNode }) => (
  <g>
    <circle
      cx={node.x}
      cy={node.y}
      r={22}
      fill={node.accent ? "var(--accent)" : "var(--secondary)"}
      fillOpacity={node.accent ? 0.18 : 0.9}
      stroke={node.accent ? "var(--accent)" : "var(--border)"}
      strokeWidth={node.accent ? 1.6 : 1.2}
      opacity={node.dim ? 0.65 : 1}
    />
    <text
      x={node.x}
      y={node.y}
      textAnchor="middle"
      dominantBaseline="middle"
      fill="var(--foreground)"
      style={{ fontSize: "10px", fontFamily: "var(--font-mono, 'JetBrains Mono', 'SFMono-Regular', monospace)" }}
    >
      {node.label}
    </text>
  </g>
);

const EdgeLine = ({ edge, nodeMap }: { edge: GraphEdge; nodeMap: Record<string, GraphNode> }) => {
  const from = nodeMap[edge.from];
  const to = nodeMap[edge.to];
  if (!from || !to) return null;
  return (
    <line
      x1={from.x}
      y1={from.y}
      x2={to.x}
      y2={to.y}
      stroke={edge.dashed ? "var(--muted-foreground)" : "var(--border)"}
      strokeWidth={edge.dashed ? 1 : 1.6}
      strokeDasharray={edge.dashed ? "4 4" : undefined}
      markerEnd="url(#arrowhead)"
      opacity={0.9}
    />
  );
};

const ModelGraph: FC<ModelGraphProps> = ({ modelName = "Selected model" }) => {
  const modelType = classifyModel(modelName);
  const { nodes, edges, badge } = buildGraph(modelType);
  const nodeMap = nodes.reduce<Record<string, GraphNode>>((acc, node) => {
    acc[node.id] = node;
    return acc;
  }, {});

  return (
    <div className="glass-card h-full border border-border/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Model sketch</p>
          <p className="text-sm font-semibold text-foreground">{modelName || "Selected model"}</p>
          <p className="text-[11px] text-secondary-foreground">{DESCRIPTION_MAP[modelType]}</p>
        </div>
        <span className="rounded-md bg-accent/10 px-2 py-1 text-[11px] font-medium text-accent">{badge}</span>
      </div>

      <div className="mt-4 h-56 w-full rounded-md bg-secondary/40 p-2">
        <svg viewBox="0 0 360 220" className="h-full w-full">
          <defs>
            <marker
              id="arrowhead"
              markerWidth="8"
              markerHeight="8"
              refX="4"
              refY="4"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L8,4 L0,8 z" fill="var(--border)" />
            </marker>
          </defs>
          {edges.map((edge) => (
            <EdgeLine key={`${edge.from}-${edge.to}`} edge={edge} nodeMap={nodeMap} />
          ))}
          {nodes.map((node) => (
            <NodeCircle key={node.id} node={node} />
          ))}
        </svg>
      </div>
    </div>
  );
};

export default ModelGraph;
