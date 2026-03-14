export type PipelineStageStatus = "waiting" | "running" | "completed" | "failed";
export type TaskType = "classification" | "regression";

export interface DatasetUploadResponse {
  dataset_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
}

export interface DatasetSummary {
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  column_types: Record<string, string>;
  missing_values: Record<string, number>;
  numeric_summary?: Record<string, Record<string, number>> | null;
}

export interface DatasetColumn {
  name: string;
  dtype: string;
  is_numeric: boolean;
  missing_pct: number;
}

export interface DatasetPreviewResponse {
  rows: Array<Record<string, unknown>>;
  columns: string[];
}

export interface ChatMessagePayload {
  role: "user" | "assistant";
  content: string;
}

export interface ChatSelectionContext {
  text: string;
  source_label?: string | null;
  surrounding_text?: string | null;
}

export interface ChatResponse {
  answer: string;
  llm_used: boolean;
}

export interface PipelineStatusResponse {
  pipeline_id: string | null;
  stages: Record<string, PipelineStageStatus>;
  target_column: string | null;
  dataset_loaded: boolean;
}

export interface PipelineLogsResponse {
  logs: Record<string, string[]>;
}

export interface PipelineConfig {
  task_type: TaskType;
  test_size: number;
  random_state: number;
}

export interface StageResultResponse {
  stage_id: string;
  status: PipelineStageStatus;
  result: Record<string, unknown> | null;
}

export interface MetricsResponse {
  task_type: TaskType;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  r2?: number | null;
  mae?: number | null;
  mse?: number | null;
  rmse?: number | null;
  best_score: number;
  model_name?: string | null;
  deployment_decision?: string | null;
  performance_summary?: string | null;
  confusion_matrix: number[][];
}

export type ExplanationResponse = Record<string, unknown>;

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://127.0.0.1:8000";

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.message || detail;
    } catch {
      // Ignore JSON parse errors and keep the fallback message.
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiRequest<DatasetUploadResponse>("/api/dataset/upload", {
    method: "POST",
    body: formData,
  });
}

export function getDatasetSummary(): Promise<DatasetSummary> {
  return apiRequest<DatasetSummary>("/api/dataset/summary");
}

export function getDatasetColumns(): Promise<{ columns: DatasetColumn[] }> {
  return apiRequest<{ columns: DatasetColumn[] }>("/api/dataset/columns");
}

export function getDatasetPreview(rows = 5): Promise<DatasetPreviewResponse> {
  return apiRequest<DatasetPreviewResponse>(`/api/dataset/preview?n=${rows}`);
}

export function setTargetColumn(targetColumn: string): Promise<{ target_column: string }> {
  return apiRequest<{ target_column: string }>("/api/dataset/target", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ target_column: targetColumn }),
  });
}

export function getPipelineStatus(): Promise<PipelineStatusResponse> {
  return apiRequest<PipelineStatusResponse>("/api/pipeline/status");
}

export function getPipelineLogs(): Promise<PipelineLogsResponse> {
  return apiRequest<PipelineLogsResponse>("/api/pipeline/logs");
}

export function runPipelineStage(stageId: string, config: PipelineConfig): Promise<StageResultResponse> {
  return apiRequest<StageResultResponse>(`/api/pipeline/stage/${stageId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  });
}

export function getStageResult(stageId: string): Promise<StageResultResponse> {
  return apiRequest<StageResultResponse>(`/api/stages/${stageId}/results`);
}

export function getMetrics(): Promise<MetricsResponse> {
  return apiRequest<MetricsResponse>("/api/results/metrics");
}

export function getExplanation(): Promise<ExplanationResponse> {
  return apiRequest<ExplanationResponse>("/api/results/explanation");
}

export function queryChat(
  question: string,
  history: ChatMessagePayload[],
  selectionContext?: ChatSelectionContext | null,
): Promise<ChatResponse> {
  return apiRequest<ChatResponse>("/api/chat/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      history,
      selection_context: selectionContext ?? null,
    }),
  });
}

export function getDownloadUrl(kind: "model" | "logs"): string {
  return `${API_BASE_URL}/api/results/download/${kind}`;
}
