"""Preprocessor Agent for AutoML Pipeline.

This agent handles data preprocessing including:
- Missing value imputation
- Categorical encoding
- Feature scaling
- Train/test splitting
"""

import logging
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class PreprocessorAgent(BaseAgent):
    """Agent for preprocessing data before model training."""

    def __init__(self) -> None:
        super().__init__("Preprocessor")

    async def execute(
        self,
        df: pd.DataFrame,
        analysis: dict[str, Any],
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:

        try:
            logger.info(f"Starting preprocessing of dataset with {len(df)} rows.\n")

            df = df.copy()

            if target_column not in df.columns:
                raise AgentExecutionError(
                    f"Target column '{target_column}' not found in dataset",
                    agent_name=self.name,
                )

            y = df[target_column]
            X = df.drop(columns=[target_column])

            numeric_columns = analysis.get("numeric_columns", [])
            categorical_columns = analysis.get("categorical_columns", [])

            numeric_columns = [c for c in numeric_columns if c in X.columns]
            categorical_columns = [c for c in categorical_columns if c in X.columns]

            # Capture missing values before fixing
            missing_summary = {
                col: int(X[col].isnull().sum())
                for col in X.columns
                if X[col].isnull().sum() > 0
            }

            X = self._handle_missing_values(X, numeric_columns, categorical_columns)

            high_cardinality_threshold = 20
            categorical_columns = self._handle_high_cardinality(
                X, categorical_columns, threshold=high_cardinality_threshold
            )

            categorical_cardinality = {
                col: int(X[col].nunique()) for col in categorical_columns
            }

            X, encoding_mapping = self._encode_categorical(X, categorical_columns)

            X = self._scale_features(X, numeric_columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
                if y.dtype == "object" or y.dtype.name == "category"
                else None,
            )

            feature_actions = {
                col: "median imputation + standard scaling" for col in numeric_columns
            }

            feature_actions.update(
                {
                    col: f"mode imputation + one-hot encoding (top {high_cardinality_threshold} levels)"
                    for col in categorical_columns
                }
            )

            explanation_details, llm_used = self._generate_llm_explanation(
                dataset_rows=len(df),
                feature_columns=list(X.columns),
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                high_cardinality_threshold=high_cardinality_threshold,
                split_train=len(X_train),
                split_test=len(X_test),
                test_size_ratio=test_size,
                encoding_mapping=encoding_mapping,
                missing_summary=missing_summary,
                categorical_cardinality=categorical_cardinality,
                feature_actions=feature_actions,
            )

            # -------------------------------------------------
            # Beginner-friendly logs in paragraph style
            # -------------------------------------------------
            summary = explanation_details.get("summary", "")

            paragraph = (
                "We just finished preparing your dataset for machine learning. "
                + summary
                + f" This means we used {len(X_train):,} rows to train the model, "
                f"so it can learn patterns, and kept {len(X_test):,} rows separate "
                f"to check how well the model performs on new, unseen data."
            )

            logger.info(paragraph + "\n")
            logger.info("Preprocessing stage completed successfully.\n")

            preprocessing_config = {
                "test_size": test_size,
                "random_state": random_state,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "imputation_strategy": "median",
                "scaling_strategy": "standard",
                "encoding_strategy": "onehot",
            }

            return {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "missing_handled": True,
                "encoding_mapping": encoding_mapping,
                "preprocessing_config": preprocessing_config,
                "explanation": summary,
                "explanation_details": explanation_details,
                "llm_used": llm_used,
            }

        except Exception as e:
            logger.exception(f"Error preprocessing data: {e}")
            raise AgentExecutionError(
                f"Preprocessing failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> pd.DataFrame:

        df = df.copy()

        for col in numeric_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in categorical_columns:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

        return df

    def _build_agent_summary(  # type: ignore[override]
        self,
        result: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dataset_summary: Optional[str],
    ) -> dict[str, Any]:
        """Surface the beginner-friendly explanation in agent summaries/logs."""
        details = result.get("explanation_details", {}) if isinstance(result, dict) else {}
        summary = str(details.get("summary") or result.get("explanation") or "Preprocessing completed.").strip()
        decisions = details.get("decisions") if isinstance(details, dict) else []
        decisions_list = [str(d).strip() for d in decisions] if isinstance(decisions, list) else []
        why = str(details.get("why") or "These steps clean the data and make it ready for training.").strip()

        return {
            "agent": self.name,
            "step_summary": summary,
            "decisions_made": decisions_list[:3] if decisions_list else [],
            "why": why,
            "overall_summary": summary,
            "llm_used": bool(result.get("llm_used", False)),
        }

    def _handle_high_cardinality(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
        threshold: int = 20,
    ) -> list[str]:

        filtered = []

        for col in categorical_columns:
            if col in df.columns:
                unique_count = df[col].nunique()

                if unique_count > threshold:
                    top_categories = df[col].value_counts().nlargest(threshold).index
                    df[col] = df[col].apply(
                        lambda x: x if x in top_categories else "Other"
                    )

                filtered.append(col)

        return filtered

    def _encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
    ) -> tuple[pd.DataFrame, dict]:

        encoding_mapping = {}

        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                encoding_mapping[col] = list(df[col].unique())

        return df, encoding_mapping

    def _scale_features(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
    ) -> pd.DataFrame:

        # Placeholder: scaling logic could be added later
        return df

    def _generate_llm_explanation(
        self,
        *,
        dataset_rows: int,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        high_cardinality_threshold: int,
        split_train: int,
        split_test: int,
        test_size_ratio: float,
        encoding_mapping: dict[str, list[Any]],
        missing_summary: dict[str, int],
        categorical_cardinality: dict[str, int],
        feature_actions: dict[str, str],
    ) -> tuple[dict[str, Any], bool]:
        """Generate a beginner-friendly paragraph plus key decisions/why."""
        fallback = {
            "summary": (
                f"We cleaned missing values, tamed categorical columns, and set numeric columns up for scaling. "
                f"The dataset has {dataset_rows} rows and {len(feature_columns)} features. "
                f"We split it into {split_train} training rows and {split_test} testing rows."
            ),
            "decisions": [
                f"Imputed numeric columns with medians and categorical columns with the most frequent value ({len(missing_summary)} columns had gaps)",
                f"Capped categorical levels to the top {high_cardinality_threshold} values and prepared for one-hot encoding",
                "Kept a standard train/test split so the model can be evaluated on unseen data",
            ],
            "why": "These steps give the model clean, consistent numbers to learn from and keep evaluation fair.",
        }

        context = {
            "rows": dataset_rows,
            "feature_count": len(feature_columns),
            "example_columns": feature_columns[:6],
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_summary": missing_summary,
            "categorical_cardinality": categorical_cardinality,
            "imputation_strategy": "median (numeric), most frequent / Unknown (categorical)",
            "encoding_strategy": "one-hot (downstream)",
            "scaling_strategy": "standard (downstream)",
            "high_cardinality_threshold": high_cardinality_threshold,
            "train_rows": split_train,
            "test_rows": split_test,
            "test_size_ratio": test_size_ratio,
            "feature_actions": feature_actions,
        }

        response = self._generate_llm_json(
            system_prompt=(
                "You are an AutoML assistant explaining preprocessing to a beginner user.\n"
                "Explain what the preprocessing stage did in simple conversational language.\n"
                "Write it like a short paragraph a non-technical user could understand.\n\n"
                "Rules:\n"
                "- Explain what happened to the dataset\n"
                "- Mention missing value handling\n"
                "- Mention categorical handling\n"
                "- Mention scaling and train/test split\n"
                "- Avoid technical jargon when possible\n"
                "- Keep it under 120 words\n\n"
                "Return JSON only:\n"
                "{\n"
                '  \"summary\": \"paragraph explanation\",\n'
                '  \"decisions\": [\"simple decision\", \"simple decision\", \"simple decision\"],\n'
                '  \"why\": \"simple explanation of why these steps help models learn\"\n'
                "}"
            ),
            user_prompt=f"Preprocessor context:\\n{self._safe_json(context)}",
            temperature=0.2,
            max_tokens=500,
        )

        if not response:
            return fallback, False

        summary = str(response.get("summary") or "").strip()
        decisions = response.get("decisions")
        decisions_clean = [str(item).strip() for item in decisions] if isinstance(decisions, list) else []
        why = str(response.get("why") or "").strip()

        if not summary:
            return fallback, False

        return {
            "summary": summary,
            "decisions": decisions_clean[:3] if decisions_clean else fallback["decisions"],
            "why": why or fallback["why"],
        }, True
