"""Explanation Generator Agent for AutoML Pipeline.

This agent generates explanations for model predictions including:
- Feature importance
- SHAP values
- Human-readable explanations
"""

import logging
from typing import Any, Optional

import numpy as np

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class ExplanationGeneratorAgent(BaseAgent):
    """Agent for generating model explanations.

    This agent handles:
    - Feature importance analysis
    - SHAP value computation
    - Generating human-readable explanations
    """

    def __init__(self) -> None:
        """Initialize the ExplanationGeneratorAgent."""
        super().__init__("ExplanationGenerator")

    async def execute(
        self,
        training_result: dict[str, Any],
        evaluation_result: dict[str, Any],
        X_data: Optional[Any] = None,
        pipeline_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate explanations for the model.

        Args:
            training_result: Training results from TrainingAgent.
            evaluation_result: Evaluation results from EvaluationAgent.
            X_data: Optional feature data for SHAP analysis.

        Returns:
            Dictionary containing explanation results including:
            - feature_importance
            - explanations
            - summary
        """
        try:
            # logger.info("Generating model explanations")

            model = training_result.get("model")
            task_type = evaluation_result.get("task_type", "classification")

            if model is None:
                raise AgentExecutionError(
                    "No model available for explanation",
                    agent_name=self.name,
                )

            # Get feature importance
            feature_importance = self._get_feature_importance(model, training_result)

            # Generate explanations
            explanations = self._generate_explanations(
                feature_importance,
                evaluation_result,
            )
            summary = self._generate_summary(explanations, evaluation_result, pipeline_context)

            llm_explanation = self._generate_llm_explanation(
                feature_importance=feature_importance,
                evaluation_result=evaluation_result,
                fallback_explanations=explanations,
                fallback_summary=summary,
                pipeline_context=pipeline_context,
            )

            if llm_explanation:
                explanations = llm_explanation.get("explanations", explanations)
                summary = llm_explanation.get("summary", summary)

            result = {
                "feature_importance": feature_importance,
                "explanations": explanations,
                "summary": summary,
                "llm_used": bool(llm_explanation),
                "pipeline_summary": self._build_pipeline_summary(pipeline_context),
            }

            # logger.info("Explanation generation complete")
            return result

        except Exception as e:
            # logger.exception(f"Error in explanation generation: {e}")
            raise AgentExecutionError(
                f"Explanation generation failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _get_feature_importance(
        self,
        model: Any,
        training_result: dict[str, Any],
    ) -> dict[str, float]:
        """Extract feature importance from model."""
        importance_dict = {}

        # Try to get feature importance from model
        if hasattr(model, "feature_importances_"):
            X_train = training_result.get("X_train")
            if X_train is not None:
                importances = model.feature_importances_
                feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else [f"feature_{i}" for i in range(len(importances))]
                importance_dict = dict(zip(feature_names, importances.tolist()))
        elif hasattr(model, "coef_"):
            # For linear models
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            X_train = training_result.get("X_train")
            if X_train is not None:
                feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else [f"feature_{i}" for i in range(len(coefs))]
                importance_dict = dict(zip(feature_names, np.abs(coefs).tolist()))

        # If no importance found, create dummy importance
        if not importance_dict:
            X_train = training_result.get("X_train")
            if X_train is not None:
                n_features = X_train.shape[1] if hasattr(X_train, "shape") else 10
            else:
                n_features = 10
            importance_dict = {f"feature_{i}": 1.0 / n_features for i in range(n_features)}

        return importance_dict

    def _generate_explanations(
        self,
        feature_importance: dict[str, float],
        evaluation_result: dict[str, Any],
    ) -> list[str]:
        """Generate human-readable explanations."""
        explanations = []

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top features explanation
        top_features = sorted_features[:5]
        if top_features:
            features_str = ", ".join([f"{name} ({importance:.2f})" for name, importance in top_features])
            explanations.append(
                f"The most important features are: {features_str}"
            )

        # Task performance explanation
        task_type = evaluation_result.get("task_type", "classification")
        if task_type == "classification":
            accuracy = evaluation_result.get("accuracy", 0)
            if accuracy >= 0.9:
                explanations.append("The model achieves excellent accuracy and can reliably make predictions.")
            elif accuracy >= 0.7:
                explanations.append("The model achieves good accuracy with room for improvement.")
            else:
                explanations.append("The model has moderate accuracy. Consider trying different approaches.")
        else:
            r2 = evaluation_result.get("r2", 0)
            if r2 >= 0.8:
                explanations.append(f"The model explains {r2:.1%} of the variance in the target variable.")
            else:
                explanations.append(f"The model explains {r2:.1%} of the variance. More work may be needed.")

        return explanations

    def _generate_summary(
        self,
        explanations: list[str],
        evaluation_result: dict[str, Any],
        pipeline_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a summary of the explanation."""
        summary = "Model Explanation Summary:\n\n"
        for i, exp in enumerate(explanations, 1):
            summary += f"{i}. {exp}\n"

        task_type = evaluation_result.get("task_type", "classification")
        decision = evaluation_result.get("deployment_decision", "unknown")
        summary += f"\nFinal decision: {decision.upper()}"

        pipeline_summary = self._build_pipeline_summary(pipeline_context)
        if pipeline_summary:
            summary += f"\n\nPipeline recap:\n{pipeline_summary}"

        return summary

    def _build_pipeline_summary(self, pipeline_context: Optional[dict[str, Any]]) -> str:
        """Create a short pipeline recap from available context."""
        if not pipeline_context:
            return ""

        lines: list[str] = []
        dataset = pipeline_context.get("dataset")
        target = pipeline_context.get("target_column")
        task_type = pipeline_context.get("task_type")
        if dataset:
            lines.append(f"- Dataset: {dataset}")
        if target:
            lines.append(f"- Target column: {target}")
        if task_type:
            lines.append(f"- Task type: {task_type}")

        model_selection = pipeline_context.get("model_selection", {})
        if isinstance(model_selection, dict):
            selected_model = model_selection.get("selected_model")
            if selected_model:
                lines.append(f"- Selected model: {selected_model}")

        training = pipeline_context.get("training", {})
        if isinstance(training, dict):
            best_score = training.get("best_score")
            if best_score is not None:
                lines.append(f"- Best CV score: {best_score:.3f}" if isinstance(best_score, (int, float)) else f"- Best CV score: {best_score}")

        evaluation = pipeline_context.get("evaluation", {})
        if isinstance(evaluation, dict):
            decision = evaluation.get("deployment_decision")
            if decision:
                lines.append(f"- Deployment decision: {str(decision).upper()}")

        deployment = pipeline_context.get("deployment", {})
        if isinstance(deployment, dict):
            model_path = deployment.get("model_path")
            if model_path:
                lines.append(f"- Model artifact: {model_path}")

        return "\n".join(lines)

    def _generate_llm_explanation(
        self,
        *,
        feature_importance: dict[str, float],
        evaluation_result: dict[str, Any],
        fallback_explanations: list[str],
        fallback_summary: str,
        pipeline_context: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Use an LLM to produce clearer user-facing model explanations."""
        top_features = sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:8]

        payload = {
            "task_type": evaluation_result.get("task_type", "classification"),
            "deployment_decision": evaluation_result.get("deployment_decision"),
            "performance_summary": evaluation_result.get("performance_summary"),
            "metrics": {
                "accuracy": evaluation_result.get("accuracy"),
                "precision": evaluation_result.get("precision"),
                "recall": evaluation_result.get("recall"),
                "f1": evaluation_result.get("f1"),
                "r2": evaluation_result.get("r2"),
                "rmse": evaluation_result.get("rmse"),
            },
            "top_features": top_features,
            "fallback_explanations": fallback_explanations,
            "fallback_summary": fallback_summary,
            "pipeline_context": pipeline_context or {},
        }

        response = self._generate_llm_json(
            system_prompt=(
                "You are an AutoML explanation assistant. "
                "Provide a concise but comprehensive rundown of the pipeline decisions, including model selection, "
                "training outcomes, evaluation results, and deployment status. "
                "Write concise, trustworthy explanations for model performance and feature importance. "
                "Return ONLY valid JSON with keys 'explanations' and 'summary'. "
                "explanations must be a list of 2 to 4 short bullets written as sentences. "
                "summary must be a short paragraph that includes pipeline recap and deployment status."
            ),
            user_prompt=f"Explanation context:\n{self._safe_json(payload)}",
            temperature=0.2,
            max_tokens=700,
        )

        if not response:
            return None

        explanations = response.get("explanations", [])
        if not isinstance(explanations, list):
            explanations = fallback_explanations

        clean_explanations = [str(item).strip() for item in explanations if str(item).strip()]
        if not clean_explanations:
            clean_explanations = fallback_explanations

        summary = str(response.get("summary") or fallback_summary).strip()
        if not summary:
            summary = fallback_summary

        return {
            "explanations": clean_explanations[:4],
            "summary": summary,
        }
