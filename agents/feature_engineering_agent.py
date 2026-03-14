"""Feature Engineering Agent for AutoML Pipeline.

This agent handles feature engineering including:
- Feature selection
- Polynomial feature generation
- PCA for dimensionality reduction
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseAgent):
    """Agent for feature engineering and selection.

    This agent handles:
    - Feature selection based on importance
    - Polynomial feature generation
    - PCA for dimensionality reduction
    """

    def __init__(self) -> None:
        """Initialize the FeatureEngineeringAgent."""
        super().__init__("FeatureEngineering")

    async def execute(
        self,
        df: pd.DataFrame,
        preprocessing: dict[str, Any],
        target_column: str,
        n_features_to_select: int = 10,
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
    ) -> dict[str, Any]:
        """Engineer features for the dataset.

        Args:
            df: The input DataFrame.
            preprocessing: Preprocessing results from PreprocessorAgent.
            target_column: The target column name.
            n_features_to_select: Number of features to select (0 = all).
            use_pca: Whether to apply PCA for dimensionality reduction.
            n_pca_components: Number of PCA components (None = auto).

        Returns:
            Dictionary containing feature engineering results including:
            - selected_features
            - feature_scores
            - final_feature_count
            - pca_components (if PCA used)
            - feature_engineering_config
        """
        try:
            # logger.info(f"Engineering features for dataset with {len(df.columns)} columns")

            df = df.copy()

            # Get column info
            numeric_columns = preprocessing.get("numeric_columns", [])
            categorical_columns = preprocessing.get("categorical_columns", [])

            # Separate features and target
            if target_column in df.columns:
                y = df[target_column]
                X = df.drop(columns=[target_column])
            else:
                y = None
                X = df

            # Encode target if needed for scoring
            if y is not None and y.dtype == "object":
                y_encoded = pd.Categorical(y).codes
            else:
                y_encoded = y

            # Get all feature columns
            all_features = list(X.columns)

            # Feature selection
            selected_features = all_features
            feature_scores = {}

            if n_features_to_select > 0 and n_features_to_select < len(all_features):
                X_numeric = X.select_dtypes(include=[np.number])

                if len(X_numeric.columns) > 0 and y_encoded is not None:
                    # Use SelectKBest for feature selection
                    is_classification = y_encoded.dtype == int or (
                        isinstance(y_encoded, pd.Series) and y_encoded.dtype.name == "int32"
                    )

                    score_func = f_classif if is_classification else f_regression

                    selector = SelectKBest(score_func=score_func, k=min(n_features_to_select, len(X_numeric.columns)))
                    selector.fit(X_numeric, y_encoded)

                    # Get feature scores
                    feature_scores = dict(zip(X_numeric.columns, selector.scores_.tolist()))

                    # Get selected features
                    selected_indices = selector.get_support(indices=True)
                    selected_features = X_numeric.columns[selected_indices].tolist()

                    # Add any categorical features
                    selected_features = list(set(selected_features + categorical_columns[:3]))

            # PCA if requested
            pca_result = {}
            if use_pca and len(selected_features) > 2:
                X_selected = X[selected_features].select_dtypes(include=[np.number])

                if len(X_selected.columns) > 0:
                    n_components = n_pca_components or min(10, len(X_selected.columns))
                    pca = PCA(n_components=n_components)
                    pca.fit(X_selected)

                    pca_result = {
                        "n_components": n_components,
                        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                    }

                    selected_features = [f"PC{i+1}" for i in range(n_components)]

            feature_engineering_config = {
                "n_features_to_select": n_features_to_select,
                "use_pca": use_pca,
                "n_pca_components": n_pca_components,
                "original_feature_count": len(all_features),
            }

            result = {
                "selected_features": selected_features,
                "feature_scores": feature_scores,
                "final_feature_count": len(selected_features),
                "pca_result": pca_result if pca_result else None,
                "feature_engineering_config": feature_engineering_config,
                "numeric_features": [f for f in selected_features if f in numeric_columns],
                "categorical_features": [f for f in selected_features if f in categorical_columns],
            }

            # logger.info(f"Feature engineering complete: {len(selected_features)} features selected")
            return result

        except Exception as e:
            # logger.exception(f"Error in feature engineering: {e}")
            raise AgentExecutionError(
                f"Feature engineering failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e
