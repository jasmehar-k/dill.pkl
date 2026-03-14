"""Deployment Agent for AutoML Pipeline.

This agent handles model deployment including:
- Saving model to disk
- Generating deployment code
- Creating model artifacts
"""

import logging
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

# Ensure outputs directory exists
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


class DeploymentAgent(BaseAgent):
    """Agent for deploying trained models.

    This agent handles:
    - Saving model to disk
    - Generating deployment code
    - Creating model artifacts
    """

    def __init__(self) -> None:
        """Initialize the DeploymentAgent."""
        super().__init__("Deployment")

    async def execute(
        self,
        training_result: dict[str, Any],
        evaluation_result: dict[str, Any],
        pipeline_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Deploy the trained model.

        Args:
            training_result: Training results from TrainingAgent.
            evaluation_result: Evaluation results from EvaluationAgent.
            pipeline_id: Optional pipeline identifier.

        Returns:
            Dictionary containing deployment results including:
            - model_path
            - deployment_code
            - artifacts
        """
        try:
            # logger.info("Deploying model")

            try:
                import joblib
            except ModuleNotFoundError as exc:
                raise AgentExecutionError(
                    "joblib is required to export trained models. Install backend dependencies with 'pip install -r requirements.txt'.",
                    agent_name=self.name,
                ) from exc

            model = training_result.get("model")
            model_name = training_result.get("model_name", "model")
            task_type = evaluation_result.get("task_type", "classification")

            if model is None:
                raise AgentExecutionError(
                    "No model available for deployment",
                    agent_name=self.name,
                )

            # Generate pipeline ID if not provided
            pipeline_id = pipeline_id or "pipeline"

            # Save model
            model_path = OUTPUTS_DIR / f"{pipeline_id}_model.pkl"
            joblib.dump(model, model_path)

            # Save metadata
            metadata = {
                "pipeline_id": pipeline_id,
                "model_name": model_name,
                "task_type": task_type,
                "accuracy": evaluation_result.get("accuracy"),
                "f1": evaluation_result.get("f1"),
                "r2": evaluation_result.get("r2"),
                "deployment_decision": evaluation_result.get("deployment_decision"),
            }

            metadata_path = OUTPUTS_DIR / f"{pipeline_id}_metadata.json"
            import json
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Generate deployment code
            deployment_code = self._generate_deployment_code(
                model_name=model_name,
                task_type=task_type,
            )

            result = {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "deployment_code": deployment_code,
                "pipeline_id": pipeline_id,
                "deployment_success": True,
            }

            # logger.info(f"Model deployed to {model_path}")
            return result

        except Exception as e:
            # logger.exception(f"Error in deployment: {e}")
            raise AgentExecutionError(
                f"Deployment failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _generate_deployment_code(
        self,
        model_name: str,
        task_type: str,
    ) -> str:
        """Generate deployment code for the model."""
        code = f'''"""Deployment code for {model_name} model.

This code can be used to make predictions with the trained model.
"""

import joblib
import pandas as pd

# Load the model
model = joblib.load("outputs/{{pipeline_id}}_model.pkl")

def predict(data: pd.DataFrame):
    """Make predictions on new data.

    Args:
        data: DataFrame with the same features as training data.

    Returns:
        Predictions as a list or DataFrame.
    """
    # Ensure data has the same columns as training
    predictions = model.predict(data)
    return predictions

# Example usage:
# import pandas as pd
# new_data = pd.read_csv("new_data.csv")
# predictions = predict(new_data)
# print(predictions)
'''
        return code
