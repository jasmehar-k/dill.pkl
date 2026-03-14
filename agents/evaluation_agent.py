"""Evaluation Agent for AutoML Pipeline.

This agent evaluates trained models including:
- Computing classification metrics
- Generating confusion matrix
- Making deployment decisions
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    """Agent for evaluating trained models.

    This agent handles:
    - Computing classification metrics (accuracy, precision, recall, F1)
    - Generating confusion matrix
    - Making deployment recommendations
    """

    def __init__(self) -> None:
        """Initialize the EvaluationAgent."""
        super().__init__("Evaluation")

    async def execute(
        self,
        training_result: dict[str, Any],
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """Evaluate the trained model.

        Args:
            training_result: Training results from TrainingAgent.
            task_type: Type of task - "classification" or "regression".

        Returns:
            Dictionary containing evaluation results including:
            - accuracy, precision, recall, f1
            - confusion_matrix
            - classification_report
            - deployment_decision
            - performance_summary
        """
        try:
            logger.info("Evaluating model performance")

            model = training_result.get("model")
            X_test = training_result.get("X_test")
            y_test = training_result.get("y_test")

            if model is None or X_test is None or y_test is None:
                raise AgentExecutionError(
                    "Missing training results for evaluation",
                    agent_name=self.name,
                )

            # Make predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            if task_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                confusion_matrix_list = cm.tolist()

                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                result = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "confusion_matrix": confusion_matrix_list,
                    "classification_report": report,
                    "task_type": "classification",
                }
            else:
                # For regression, use R2 and error metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                result = {
                    "r2": float(r2),
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "task_type": "regression",
                }

            # Make deployment decision
            deployment_decision = self._make_deployment_decision(result, training_result)
            result["deployment_decision"] = deployment_decision

            # Performance summary
            performance_summary = self._generate_performance_summary(result)
            result["performance_summary"] = performance_summary

            logger.info(f"Evaluation complete: {deployment_decision}")
            return result

        except Exception as e:
            logger.exception(f"Error in evaluation: {e}")
            raise AgentExecutionError(
                f"Evaluation failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _make_deployment_decision(
        self,
        evaluation_result: dict[str, Any],
        training_result: dict[str, Any],
    ) -> str:
        """Make deployment decision based on metrics."""
        task_type = evaluation_result.get("task_type", "classification")

        if task_type == "classification":
            accuracy = evaluation_result.get("accuracy", 0)
            f1 = evaluation_result.get("f1", 0)
            train_score = training_result.get("train_score", 0)
            test_score = training_result.get("test_score", 0)

            # Check for overfitting
            overfitting_gap = train_score - test_score

            if accuracy >= 0.9 and f1 >= 0.9 and overfitting_gap < 0.1:
                return "deploy"
            elif accuracy >= 0.7 and f1 >= 0.7:
                if overfitting_gap > 0.15:
                    return "iterate"  # Possible overfitting
                return "deploy"
            elif accuracy >= 0.5:
                return "iterate"  # Performance could be improved
            else:
                return "reject"  # Poor performance
        else:
            r2 = evaluation_result.get("r2", 0)

            if r2 >= 0.8:
                return "deploy"
            elif r2 >= 0.5:
                return "iterate"
            else:
                return "reject"

    def _generate_performance_summary(self, evaluation_result: dict[str, Any]) -> str:
        """Generate human-readable performance summary."""
        task_type = evaluation_result.get("task_type", "classification")

        if task_type == "classification":
            accuracy = evaluation_result.get("accuracy", 0)
            f1 = evaluation_result.get("f1", 0)
            decision = evaluation_result.get("deployment_decision", "unknown")

            summary = f"Model achieved {accuracy:.1%} accuracy and {f1:.1%} F1 score. "
            summary += f"Deployment decision: {decision}."
        else:
            r2 = evaluation_result.get("r2", 0)
            rmse = evaluation_result.get("rmse", 0)
            decision = evaluation_result.get("deployment_decision", "unknown")

            summary = f"Model achieved R² = {r2:.3f} and RMSE = {rmse:.3f}. "
            summary += f"Deployment decision: {decision}."

        return summary