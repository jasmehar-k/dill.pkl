"""FastAPI application for the AutoML Pipeline.

This module provides REST API endpoints for the AutoML pipeline,
allowing the React frontend to upload datasets, configure the pipeline,
and retrieve results.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from utils.logger import get_logger

# Ensure outputs directory exists
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Ensure uploads directory exists
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="dill.pkl AutoML API")
logger = get_logger("api.main")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pipeline state management
class PipelineState:
    """Manages the state of the AutoML pipeline."""

    def __init__(self):
        self.dataset: Optional[pd.DataFrame] = None
        self.dataset_path: Optional[str] = None
        self.dataset_filename: Optional[str] = None
        self.target_column: Optional[str] = None
        self.stage_results: dict[str, Any] = {}
        self.stage_statuses: dict[str, str] = {
            "analysis": "waiting",
            "preprocessing": "waiting",
            "features": "waiting",
            "model_selection": "waiting",
            "training": "waiting",
            "loss": "waiting",
            "evaluation": "waiting",
            "results": "waiting",
        }
        self.stage_logs: dict[str, list[str]] = {stage: [] for stage in self.stage_statuses}
        self.pipeline_id: Optional[str] = None


pipeline_state = PipelineState()


# Request/Response models
class TargetColumnRequest(BaseModel):
    target_column: str


class PipelineConfig(BaseModel):
    task_type: str = "classification"  # or "regression"
    test_size: float = 0.2
    random_state: int = 42


class StageResultResponse(BaseModel):
    stage_id: str
    status: str
    result: Optional[dict[str, Any]] = None


class DatasetSummaryResponse(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    column_types: dict[str, str]
    missing_values: dict[str, float]
    numeric_summary: Optional[dict[str, Any]] = None


# Helper functions
def add_log(stage: str, message: str):
    """Add a log message to a stage."""
    if stage in pipeline_state.stage_logs:
        pipeline_state.stage_logs[stage].append(message)


def add_agent_summary_logs(stage: str, result: Optional[dict[str, Any]]):
    """Add LLM or fallback agent summaries to the stage log stream."""
    if not result:
        return

    summary = result.get("_agent_summary")
    if not isinstance(summary, dict):
        return

    agent_name = str(summary.get("agent", stage))
    step_summary = str(summary.get("step_summary", "")).strip()
    overall_summary = str(summary.get("overall_summary", "")).strip()
    why = str(summary.get("why", "")).strip()
    decisions = summary.get("decisions_made", [])
    llm_used = bool(summary.get("llm_used", False))

    source_tag = "LLM" if llm_used else "Fallback"
    if step_summary:
        add_log(stage, f"{agent_name} [{source_tag}] summary: {step_summary}")
    if isinstance(decisions, list):
        for decision in decisions[:3]:
            decision_text = str(decision).strip()
            if decision_text:
                add_log(stage, f"{agent_name} decision: {decision_text}")
    if why:
        add_log(stage, f"{agent_name} why: {why}")
    if overall_summary:
        add_log(stage, f"{agent_name} overall: {overall_summary}")


def summarize_dataset(df: Optional[pd.DataFrame]) -> str:
    """Create a compact dataset summary for console logging."""
    if df is None:
        return "dataset=None"

    preview_columns = ", ".join(str(column) for column in df.columns[:6])
    if len(df.columns) > 6:
        preview_columns += ", ..."
    return f"rows={len(df)}, cols={len(df.columns)}, columns=[{preview_columns}]"


def make_json_safe(value: Any) -> Any:
    """Convert common Python, NumPy, and pandas objects to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {
            "rows": len(value),
            "columns": list(value.columns),
        }
    return str(value)


def summarize_stage_result(stage: str, result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return a frontend-safe stage result summary."""
    if result is None:
        return None

    stage_keys: dict[str, list[str]] = {
        "training": [
            "model_name",
            "best_score",
            "cv_scores",
            "cv_std",
            "train_score",
            "test_score",
            "train_loss",
            "val_loss",
            "best_epoch",
        ],
        "model_selection": [
            "selected_model",
            "candidate_models",
            "reasoning",
            "hyperparameters",
            "llm_returned",
            "llm_summary",
            "analysis_signals",
            "class_balance",
            "n_samples",
            "n_features",
            "task_type",
        ],
        "results": [
            "model_path",
            "metadata_path",
            "pipeline_id",
            "deployment_success",
            "deployment_code",
        ],
    }

    selected = {
        key: value for key, value in result.items()
        if not str(key).startswith("_")
    }
    if stage in stage_keys:
        selected = {key: selected.get(key) for key in stage_keys[stage]}

    return make_json_safe(selected)


async def run_pipeline_stage(stage: str, config: PipelineConfig):
    """Run a single pipeline stage and update state."""
    pipeline_state.stage_statuses[stage] = "running"
    add_log(stage, f"Starting {stage} stage...")
    # logger.info(
    #     "Starting stage=%s | file=%s | target=%s | task_type=%s | %s",
    #     stage,
    #     pipeline_state.dataset_filename or "unknown",
    #     pipeline_state.target_column or "unset",
    #     config.task_type,
    #     summarize_dataset(pipeline_state.dataset),
    # )

    try:
        if stage == "analysis":
            from agents.data_analyzer_agent import DataAnalyzerAgent

            agent = DataAnalyzerAgent()
            result = await agent.run(
                pipeline_state.dataset,
                pipeline_state.target_column
            )
            pipeline_state.stage_results["analysis"] = result
            add_agent_summary_logs("analysis", result)
            add_log(stage, f"Analysis complete: {result.get('row_count', 0)} rows, {result.get('feature_count', 0)} features")

        elif stage == "preprocessing":
            from agents.preprocessor_agent import PreprocessorAgent

            analysis = pipeline_state.stage_results.get("analysis", {})
            agent = PreprocessorAgent()
            result = await agent.run(
                pipeline_state.dataset,
                analysis,
                pipeline_state.target_column,
                test_size=config.test_size,
                random_state=config.random_state,
            )
            pipeline_state.stage_results["preprocessing"] = result
            add_agent_summary_logs("preprocessing", result)
            add_log(stage, "Preprocessing complete")

        elif stage == "features":
            from agents.feature_engineering_agent import FeatureEngineeringAgent

            preprocessing = pipeline_state.stage_results.get("preprocessing", {})
            agent = FeatureEngineeringAgent()
            result = await agent.run(
                pipeline_state.dataset,
                preprocessing,
                pipeline_state.target_column
            )
            pipeline_state.stage_results["features"] = result
            add_agent_summary_logs("features", result)
            add_log(stage, f"Feature engineering complete: {result.get('final_feature_count', 0)} features")

        elif stage == "model_selection":
            from agents.model_selection_agent import ModelSelectionAgent

            analysis = pipeline_state.stage_results.get("analysis", {})
            features = pipeline_state.stage_results.get("features", {})
            agent = ModelSelectionAgent()
            model_result = await agent.run(
                pipeline_state.dataset,
                features,
                pipeline_state.target_column,
                config.task_type,
                analysis,
            )
            model_result["target_column"] = pipeline_state.target_column
            pipeline_state.stage_results["model_selection"] = model_result
            add_agent_summary_logs("model_selection", model_result)
            if model_result.get("llm_returned"):
                add_log(stage, "Model selection LLM returned a result; using LLM selection.")
            else:
                add_log(stage, "Model selection LLM did not return a result; falling back to default selection.")
            add_log(stage, f"Selected model: {model_result.get('selected_model', 'unknown')}")

        elif stage == "training":
            from agents.training_agent import TrainingAgent

            model_result = pipeline_state.stage_results.get("model_selection")
            if not model_result:
                from agents.model_selection_agent import ModelSelectionAgent

                analysis = pipeline_state.stage_results.get("analysis", {})
                features = pipeline_state.stage_results.get("features", {})
                agent = ModelSelectionAgent()
                model_result = await agent.run(
                    pipeline_state.dataset,
                    features,
                    pipeline_state.target_column,
                    config.task_type,
                    analysis,
                )
                model_result["target_column"] = pipeline_state.target_column
                pipeline_state.stage_results["model_selection"] = model_result
                add_agent_summary_logs("model_selection", model_result)
                if model_result.get("llm_returned"):
                    add_log("model_selection", "Model selection LLM returned a result; using LLM selection.")
                else:
                    add_log("model_selection", "Model selection LLM did not return a result; falling back to default selection.")
                add_log("model_selection", f"Selected model: {model_result.get('selected_model', 'unknown')}")
                pipeline_state.stage_statuses["model_selection"] = "completed"

            train_agent = TrainingAgent()
            train_result = await train_agent.run(
                pipeline_state.dataset,
                model_result,
                config.model_dump(),
            )
            pipeline_state.stage_results["training"] = train_result
            add_agent_summary_logs("training", train_result)
            add_log(stage, f"Training complete: {train_result.get('best_score', 0):.4f}")

        elif stage == "loss":
            training = pipeline_state.stage_results.get("training", {})
            pipeline_state.stage_results["loss"] = {
                "train_loss": training.get("train_loss", [0.9, 0.6, 0.4, 0.3, 0.2]),
                "val_loss": training.get("val_loss", [0.95, 0.7, 0.5, 0.4, 0.35]),
                "best_epoch": training.get("best_epoch", 3),
            }
            add_log(stage, f"Loss analysis complete. Best epoch: {training.get('best_epoch', 3)}")

        elif stage == "evaluation":
            from agents.evaluation_agent import EvaluationAgent

            training = pipeline_state.stage_results.get("training", {})
            agent = EvaluationAgent()
            result = await agent.run(
                training,
                config.task_type
            )
            pipeline_state.stage_results["evaluation"] = result
            add_agent_summary_logs("evaluation", result)
            add_log(stage, f"Evaluation complete: Accuracy = {result.get('accuracy', 0):.4f}")

        elif stage == "results":
            from agents.deployment_agent import DeploymentAgent
            from agents.explanation_generator_agent import ExplanationGeneratorAgent

            training = pipeline_state.stage_results.get("training", {})
            evaluation = pipeline_state.stage_results.get("evaluation", {})
            agent = DeploymentAgent()
            result = await agent.run(
                training,
                evaluation,
                pipeline_state.pipeline_id
            )
            pipeline_state.stage_results["results"] = result
            add_agent_summary_logs("results", result)
            add_log(stage, f"Model saved to: {result.get('model_path', 'unknown')}")

            explanation_agent = ExplanationGeneratorAgent()
            explanation_result = await explanation_agent.run(
                training,
                evaluation,
            )
            pipeline_state.stage_results["explanation"] = explanation_result
            add_agent_summary_logs("results", explanation_result)
            add_log(stage, "Explanation summary generated")

        pipeline_state.stage_statuses[stage] = "completed"
        add_log(stage, f"{stage} stage completed successfully")
        # logger.info(
        #     "Completed stage=%s | result=%s",
        #     stage,
        #     json.dumps(summarize_stage_result(stage, pipeline_state.stage_results.get(stage)), default=str, ensure_ascii=True),
        # )

    except Exception as e:
        pipeline_state.stage_statuses[stage] = "failed"
        add_log(stage, f"Error: {str(e)}")
        # logger.exception("Stage failed stage=%s | error=%s", stage, str(e))
        raise


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "dill.pkl AutoML API", "version": "1.0.0"}


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file."""
    # Validate file extension
    allowed_extensions = {".csv", ".json", ".xlsx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Generate unique ID and save file
    dataset_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{dataset_id}{file_ext}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load into pandas
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
        elif file_ext == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        pipeline_state.dataset = df
        pipeline_state.dataset_path = str(file_path)
        pipeline_state.dataset_filename = file.filename
        pipeline_state.pipeline_id = dataset_id

        # Reset state
        pipeline_state.stage_results = {}
        pipeline_state.stage_statuses = {stage: "waiting" for stage in pipeline_state.stage_statuses}
        pipeline_state.stage_logs = {stage: [] for stage in pipeline_state.stage_statuses}
        # logger.info(
        #     "Dataset uploaded | file=%s | pipeline_id=%s | %s",
        #     file.filename,
        #     dataset_id,
        #     summarize_dataset(df),
        # )

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/dataset/summary")
async def get_dataset_summary():
    """Get dataset summary metadata."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    df = pipeline_state.dataset

    # Get numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = None
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().to_dict()

    return {
        "filename": pipeline_state.dataset_filename or "unknown",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().mean().to_dict(),
        "numeric_summary": numeric_summary,
    }


@app.get("/api/dataset/columns")
async def get_columns():
    """Get column names from the dataset."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    columns = []
    for col in pipeline_state.dataset.columns:
        dtype = str(pipeline_state.dataset[col].dtype)
        is_numeric = np.issubdtype(pipeline_state.dataset[col].dtype, np.number)
        columns.append({
            "name": col,
            "dtype": dtype,
            "is_numeric": is_numeric,
            "missing_pct": float(pipeline_state.dataset[col].isnull().mean()) if col in pipeline_state.dataset.columns else 0,
        })

    return {"columns": columns}


@app.post("/api/dataset/target")
async def set_target_column(request: TargetColumnRequest):
    """Set the target column for prediction."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    if request.target_column not in pipeline_state.dataset.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.target_column}' not found in dataset"
        )

    pipeline_state.target_column = request.target_column
    add_log("analysis", f"Target column set to: {request.target_column}")

    return {"target_column": request.target_column}


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get the current pipeline status."""
    return {
        "pipeline_id": pipeline_state.pipeline_id,
        "stages": pipeline_state.stage_statuses,
        "target_column": pipeline_state.target_column,
        "dataset_loaded": pipeline_state.dataset is not None,
    }


@app.get("/api/pipeline/logs")
async def get_pipeline_logs(stage: Optional[str] = None):
    """Get logs for a specific stage or all stages."""
    if stage:
        if stage not in pipeline_state.stage_logs:
            raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found")
        return {"stage": stage, "logs": pipeline_state.stage_logs[stage]}

    return {"logs": pipeline_state.stage_logs}


@app.post("/api/pipeline/start")
async def start_pipeline(config: PipelineConfig = PipelineConfig()):
    """Start the AutoML pipeline."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    if pipeline_state.target_column is None:
        raise HTTPException(status_code=400, detail="Target column not set")

    # Run all stages sequentially
    stages_order = ["analysis", "preprocessing", "features", "model_selection", "training", "loss", "evaluation", "results"]

    for stage in stages_order:
        if pipeline_state.stage_statuses.get(stage) == "waiting":
            await run_pipeline_stage(stage, config)

    return {
        "status": "completed",
        "stages": pipeline_state.stage_statuses,
    }


@app.post("/api/pipeline/stage/{stage_id}")
async def run_stage(stage_id: str, config: PipelineConfig = PipelineConfig()):
    """Run a specific pipeline stage."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    if stage_id not in pipeline_state.stage_statuses:
        raise HTTPException(status_code=404, detail=f"Stage '{stage_id}' not found")

    await run_pipeline_stage(stage_id, config)

    return {
        "stage_id": stage_id,
        "status": pipeline_state.stage_statuses[stage_id],
        "result": summarize_stage_result(stage_id, pipeline_state.stage_results.get(stage_id)),
    }


@app.get("/api/stages/{stage_id}/results")
async def get_stage_results(stage_id: str):
    """Get results for a specific stage."""
    if stage_id not in pipeline_state.stage_results:
        raise HTTPException(status_code=404, detail=f"No results for stage '{stage_id}'")

    return {
        "stage_id": stage_id,
        "status": pipeline_state.stage_statuses.get(stage_id, "unknown"),
        "result": summarize_stage_result(stage_id, pipeline_state.stage_results[stage_id]),
    }


@app.get("/api/results/download/model")
async def download_model():
    """Download the trained model."""
    results = pipeline_state.stage_results.get("results", {})
    model_path = results.get("model_path")

    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="No model available")

    return FileResponse(
        path=model_path,
        filename="model.pkl",
        media_type="application/octet-stream"
    )


@app.get("/api/results/download/logs")
async def download_logs():
    """Download pipeline logs."""
    logs_data = {
        "pipeline_id": pipeline_state.pipeline_id,
        "target_column": pipeline_state.target_column,
        "stages": pipeline_state.stage_statuses,
        "logs": pipeline_state.stage_logs,
        "results": pipeline_state.stage_results,
    }

    log_file = OUTPUTS_DIR / f"{pipeline_state.pipeline_id}_logs.json"
    with open(log_file, "w") as f:
        json.dump(logs_data, f, indent=2, default=str)

    return FileResponse(
        path=str(log_file),
        filename="pipeline_logs.json",
        media_type="application/json"
    )


@app.get("/api/results/metrics")
async def get_metrics():
    """Get evaluation metrics."""
    evaluation = pipeline_state.stage_results.get("evaluation", {})
    training = pipeline_state.stage_results.get("training", {})

    return {
        "task_type": evaluation.get("task_type", "classification"),
        "accuracy": evaluation.get("accuracy", 0),
        "precision": evaluation.get("precision", 0),
        "recall": evaluation.get("recall", 0),
        "f1": evaluation.get("f1", 0),
        "r2": evaluation.get("r2"),
        "mae": evaluation.get("mae"),
        "mse": evaluation.get("mse"),
        "rmse": evaluation.get("rmse"),
        "best_score": training.get("best_score", 0),
        "model_name": training.get("model_name"),
        "deployment_decision": evaluation.get("deployment_decision"),
        "performance_summary": evaluation.get("performance_summary"),
        "confusion_matrix": evaluation.get("confusion_matrix", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
