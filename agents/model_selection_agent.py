"""Model Selection Agent for AutoML Pipeline.

This agent recommends machine learning models based on:
- Data characteristics
- Task type (classification/regression)
- Dataset size
- Feature types
"""

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class ModelSelectionAgent(BaseAgent):
    """Agent for selecting appropriate ML models based on data characteristics.

    This agent recommends models based on:
    - Dataset size
    - Feature types
    - Task type (classification/regression)
    - Class imbalance
    """

    def __init__(self) -> None:
        """Initialize the ModelSelectionAgent."""
        super().__init__("ModelSelection")

    async def execute(
        self,
        df: pd.DataFrame,
        features: dict[str, Any],
        target_column: str,
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """Select appropriate models for the dataset.

        Args:
            df: The input DataFrame.
            features: Feature engineering results from FeatureEngineeringAgent.
            target_column: The target column name.
            task_type: Type of task - "classification" or "regression".

        Returns:
            Dictionary containing model selection results including:
            - selected_model
            - candidate_models
            - model_recommendations
            - hyperparameters
        """
        try:
            logger.info(f"Selecting models for {task_type} task")

            # Analyze data characteristics
            n_samples = len(df)
            n_features = features.get("final_feature_count", df.shape[1] - 1)

            # Check class distribution
            y = df[target_column]
            if task_type == "classification":
                class_counts = y.value_counts()
                min_class_ratio = class_counts.min() / class_counts.max() if len(class_counts) > 0 else 1.0
            else:
                min_class_ratio = 1.0

            # Select candidate models based on characteristics
            candidate_models = self._select_candidates(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                class_balance=min_class_ratio,
            )

            # Select best model
            selected_model = candidate_models[0] if candidate_models else "RandomForest"

            # Generate hyperparameters for selected model
            hyperparameters = self._get_default_hyperparameters(
                selected_model,
                n_samples,
                task_type,
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                class_balance=min_class_ratio,
                selected_model=selected_model,
            )

            llm_selection = self._generate_llm_selection(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                class_balance=min_class_ratio,
                candidate_models=candidate_models,
                default_model=selected_model,
                default_hyperparameters=hyperparameters,
                default_reasoning=reasoning,
                target_column=target_column,
                selected_features=features.get("selected_features", []),
            )

            if llm_selection:
                selected_model = llm_selection.get("selected_model", selected_model)
                hyperparameters = llm_selection.get("hyperparameters", hyperparameters)
                reasoning = llm_selection.get("reasoning", reasoning)

            result = {
                "selected_model": selected_model,
                "candidate_models": candidate_models,
                "reasoning": reasoning,
                "hyperparameters": hyperparameters,
                "task_type": task_type,
                "n_samples": n_samples,
                "n_features": n_features,
                "selection_source": "llm" if llm_selection else "heuristic",
            }

            logger.info(f"Model selection complete: {selected_model}")
            return result

        except Exception as e:
            logger.exception(f"Error in model selection: {e}")
            raise AgentExecutionError(
                f"Model selection failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _select_candidates(
        self,
        n_samples: int,
        n_features: int,
        task_type: str,
        class_balance: float,
    ) -> list[str]:
        """Select candidate models based on data characteristics."""
        candidates = []

        if task_type == "classification":
            if n_samples < 1000:
                candidates.extend(["LogisticRegression", "SVM", "RandomForest"])
            elif n_samples < 10000:
                candidates.extend(["RandomForest", "GradientBoosting", "LogisticRegression"])
            else:
                candidates.extend(["RandomForest", "GradientBoosting", "XGBoost"])

            # Adjust for class imbalance
            if class_balance < 0.1:
                candidates.append("XGBoost")
        else:
            if n_samples < 1000:
                candidates.extend(["Ridge", "SVR", "RandomForest"])
            elif n_samples < 10000:
                candidates.extend(["RandomForest", "GradientBoosting", "Ridge"])
            else:
                candidates.extend(["RandomForest", "GradientBoosting", "XGBoost"])

        return list(dict.fromkeys(candidates))

    def _get_default_hyperparameters(
        self,
        model_name: str,
        n_samples: int,
        task_type: str,
    ) -> dict[str, Any]:
        """Get default hyperparameters for a model."""
        base_params = {"random_state": 42}

        if model_name == "RandomForest":
            n_estimators = min(100, max(10, n_samples // 100))
            return {
                **base_params,
                "n_estimators": n_estimators,
                "max_depth": 10,
                "min_samples_split": 5,
            }
        elif model_name == "GradientBoosting":
            return {
                **base_params,
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
            }
        elif model_name == "XGBoost":
            return {
                **base_params,
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
        elif model_name == "LogisticRegression":
            return {
                **base_params,
                "max_iter": 1000,
                "C": 1.0,
            }
        elif model_name == "Ridge":
            return {"alpha": 1.0, **base_params}
        elif model_name == "SVR":
            return {"kernel": "rbf", "C": 1.0}
        else:
            return base_params

    def _generate_reasoning(
        self,
        n_samples: int,
        n_features: int,
        task_type: str,
        class_balance: float,
        selected_model: str,
    ) -> str:
        """Generate reasoning for model selection."""
        reasoning_parts = []

        reasoning_parts.append(f"Dataset has {n_samples} samples and {n_features} features.")

        if n_samples < 1000:
            reasoning_parts.append("Small dataset - simpler models are less likely to overfit.")
        elif n_samples > 10000:
            reasoning_parts.append("Large dataset - can leverage more complex models.")

        if class_balance < 0.1 and task_type == "classification":
            reasoning_parts.append("Class imbalance detected - ensemble methods recommended.")

        reasoning_parts.append(f"Selected {selected_model} as the primary model.")

        return " ".join(reasoning_parts)

    def _generate_llm_selection(
        self,
        *,
        n_samples: int,
        n_features: int,
        task_type: str,
        class_balance: float,
        candidate_models: list[str],
        default_model: str,
        default_hyperparameters: dict[str, Any],
        default_reasoning: str,
        target_column: str,
        selected_features: list[str],
    ) -> dict[str, Any] | None:
        """Use an LLM to refine model choice within deterministic guardrails."""
        payload = {
            "task_type": task_type,
            "target_column": target_column,
            "n_samples": n_samples,
            "n_features": n_features,
            "class_balance": class_balance,
            "candidate_models": candidate_models,
            "selected_features_preview": selected_features[:15],
            "default_model": default_model,
            "default_hyperparameters": default_hyperparameters,
            "default_reasoning": default_reasoning,
        }

        response = self._generate_llm_json(
            system_prompt=(
                "You are an AutoML model-selection assistant. "
                "Choose the best model ONLY from the provided candidate_models list and return JSON with keys "
                "'selected_model', 'hyperparameters', and 'reasoning'. "
                "Use only constructor-safe hyperparameters for the chosen model. "
                "Be conservative and practical."
            ),
            user_prompt=f"Model selection context:\n{self._safe_json(payload)}",
            temperature=0.1,
            max_tokens=500,
        )

        if not response:
            return None

        proposed_model = str(response.get("selected_model") or default_model)
        if proposed_model not in candidate_models:
            proposed_model = default_model

        merged_hyperparameters = self._get_default_hyperparameters(
            proposed_model,
            n_samples,
            task_type,
        )
        raw_hyperparameters = response.get("hyperparameters", {})
        if isinstance(raw_hyperparameters, dict):
            merged_hyperparameters.update(
                self._sanitize_hyperparameters(
                    proposed_model,
                    raw_hyperparameters,
                    task_type,
                )
            )

        reasoning = str(response.get("reasoning") or default_reasoning)
        return {
            "selected_model": proposed_model,
            "hyperparameters": merged_hyperparameters,
            "reasoning": reasoning,
        }

    def _sanitize_hyperparameters(
        self,
        model_name: str,
        hyperparameters: dict[str, Any],
        task_type: str,
    ) -> dict[str, Any]:
        """Keep only safe, supported hyperparameters for the selected model."""
        allowed_keys: dict[str, set[str]] = {
            "RandomForest": {"random_state", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight"},
            "GradientBoosting": {"random_state", "n_estimators", "max_depth", "learning_rate", "subsample"},
            "XGBoost": {"random_state", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "use_label_encoder", "eval_metric"},
            "LogisticRegression": {"random_state", "max_iter", "C", "class_weight", "solver", "penalty"},
            "Ridge": {"alpha", "random_state"},
            "SVM": {"C", "kernel", "gamma", "class_weight", "probability"},
            "SVR": {"C", "kernel", "gamma", "epsilon"},
        }

        key = model_name
        if model_name == "SVM" and task_type != "classification":
            key = "SVR"

        filtered = {}
        for param_name, value in hyperparameters.items():
            if param_name not in allowed_keys.get(key, set()):
                continue
            filtered[param_name] = self._coerce_hyperparameter_value(value)
        return filtered

    def _coerce_hyperparameter_value(self, value: Any) -> Any:
        """Convert JSON-like hyperparameter values to Python scalars."""
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, str):
            lowered = value.lower().strip()
            if lowered in {"true", "false"}:
                return lowered == "true"
            try:
                if "." in lowered:
                    return float(lowered)
                return int(lowered)
            except ValueError:
                return value
        if isinstance(value, list):
            return value
        return str(value)
