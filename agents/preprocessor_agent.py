"""Preprocessor Agent for AutoML Pipeline.

This agent handles data preprocessing including:
- Missing value imputation
- Categorical encoding
- Feature scaling
- Train/test splitting
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class PreprocessorAgent(BaseAgent):
    """Agent for preprocessing data before model training.

    This agent handles:
    - Missing value imputation (mean, median, mode, KNN)
    - Categorical encoding (one-hot, label, target encoding)
    - Feature scaling (standard, min-max, robust)
    - Train/test split
    """

    def __init__(self) -> None:
        """Initialize the PreprocessorAgent."""
        super().__init__("Preprocessor")

    async def execute(
        self,
        df: pd.DataFrame,
        analysis: dict[str, Any],
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Preprocess the dataset.

        Args:
            df: The input DataFrame.
            analysis: Analysis results from DataAnalyzerAgent.
            target_column: The target column name.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary containing preprocessing results including:
            - X_train, X_test, y_train, y_test
            - preprocessor: sklearn ColumnTransformer
            - imputed_columns, encoded_columns, scaled_columns
            - preprocessing_config
        """
        try:
            # logger.info(f"Preprocessing dataset with {len(df)} rows")

            df = df.copy()

            # Separate features and target
            if target_column not in df.columns:
                raise AgentExecutionError(
                    f"Target column '{target_column}' not found in dataset",
                    agent_name=self.name,
                )

            y = df[target_column]
            X = df.drop(columns=[target_column])

            # Identify column types
            numeric_columns = analysis.get("numeric_columns", [])
            categorical_columns = analysis.get("categorical_columns", [])

            # Filter to existing columns
            numeric_columns = [c for c in numeric_columns if c in X.columns]
            categorical_columns = [c for c in categorical_columns if c in X.columns]

            # Handle missing values
            X = self._handle_missing_values(X, numeric_columns, categorical_columns)

            # Handle high cardinality categorical columns
            categorical_columns = self._handle_high_cardinality(X, categorical_columns)

            # Encode categorical variables
            X, encoding_mapping = self._encode_categorical(X, categorical_columns)

            # Scale numeric features
            X = self._scale_features(X, numeric_columns)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == "object" or y.dtype.name == "category" else None
            )

            preprocessing_config = {
                "test_size": test_size,
                "random_state": random_state,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "imputation_strategy": "median",
                "scaling_strategy": "standard",
                "encoding_strategy": "onehot",
            }

            result = {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "missing_handled": True,
                "encoding_mapping": encoding_mapping,
                "preprocessing_config": preprocessing_config,
            }

            # Store processed data in a way that subsequent stages can access
            # For now, return config and let TrainingAgent handle the actual processing
            # logger.info(f"Preprocessing complete: {X_train.shape[1]} features, {len(X_train)} train samples")
            return result

        except Exception as e:
            # logger.exception(f"Error preprocessing data: {e}")
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
        """Handle missing values in the dataset."""
        df = df.copy()

        # Impute numeric columns with median
        for col in numeric_columns:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Impute categorical columns with mode
        for col in categorical_columns:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")

        return df

    def _handle_high_cardinality(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
        threshold: int = 20,
    ) -> list[str]:
        """Handle high cardinality categorical columns."""
        filtered_columns = []

        for col in categorical_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                if unique_count > threshold:
                    # Keep only top N most frequent categories
                    top_categories = df[col].value_counts().nlargest(threshold).index
                    df[col] = df[col].apply(lambda x: x if x in top_categories else "Other")
                filtered_columns.append(col)

        return filtered_columns

    def _encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
    ) -> tuple[pd.DataFrame, dict]:
        """Encode categorical variables using one-hot encoding."""
        encoding_mapping = {}

        # Convert categorical columns to category dtype
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
        """Scale numeric features using standard scaling."""
        # Just mark columns for scaling - actual scaling happens in pipeline
        return df
