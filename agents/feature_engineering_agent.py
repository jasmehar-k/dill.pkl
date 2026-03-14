"""Feature Engineering Agent for AutoML Pipeline.

This agent performs deterministic feature engineering before model training.
"""

import logging
import re
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseAgent):
    """Agent for deterministic feature engineering and feature selection."""

    INDEX_LIKE_NAMES = {"index", "row_id"}
    CORRELATION_THRESHOLD = 0.95
    IMPORTANCE_THRESHOLD = 0.01
    MAX_INTERACTION_NUMERIC_COLUMNS = 5
    RANDOM_STATE = 42
    IMPORTANCE_TREES = 50

    def __init__(self) -> None:
        """Initialize the FeatureEngineeringAgent."""
        super().__init__("FeatureEngineering")

    async def execute(
        self,
        df: pd.DataFrame,
        preprocessing: dict[str, Any],
        target_column: str,
        n_features_to_select: int = 0,
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
    ) -> dict[str, Any]:
        """Engineer features for the dataset."""
        try:
            logger.info("Engineering features for dataset with %s columns", len(df.columns))

            df = df.copy()
            preprocessing_numeric = preprocessing.get("numeric_columns", [])
            preprocessing_categorical = preprocessing.get("categorical_columns", [])

            if target_column in df.columns:
                y = df[target_column].copy()
                X = df.drop(columns=[target_column]).copy()
            else:
                y = None
                X = df.copy()

            dropped_columns: list[str] = []
            generated_features: list[str] = []
            applied_transformations: list[dict[str, Any]] = []
            pca_result: dict[str, Any] = {}

            X = self._drop_target_column_if_present(X, target_column)

            index_like_columns = [column for column in X.columns if self._is_index_like(column)]
            if index_like_columns:
                X = X.drop(columns=index_like_columns)
                dropped_columns.extend(index_like_columns)
                applied_transformations.append({
                    "type": "drop_index_like_columns",
                    "columns": sorted(index_like_columns),
                })

            numeric_columns, categorical_columns = self._get_feature_types(X)

            numeric_imputations: dict[str, float] = {}
            for column in numeric_columns:
                if X[column].isna().any():
                    fill_value = float(X[column].median()) if X[column].notna().any() else 0.0
                    X[column] = X[column].fillna(fill_value)
                    numeric_imputations[column] = fill_value
            if numeric_imputations:
                applied_transformations.append({
                    "type": "numeric_imputation",
                    "strategy": "median",
                    "fill_values": numeric_imputations,
                })

            categorical_imputations: dict[str, str] = {}
            for column in categorical_columns:
                if X[column].isna().any():
                    mode = X[column].mode(dropna=True)
                    fill_value = str(mode.iloc[0]) if not mode.empty else "missing"
                    X[column] = X[column].fillna(fill_value)
                    categorical_imputations[column] = fill_value
            if categorical_imputations:
                applied_transformations.append({
                    "type": "categorical_imputation",
                    "strategy": "most_frequent",
                    "fill_values": categorical_imputations,
                })

            log_transforms: list[dict[str, Any]] = []
            for column in numeric_columns:
                skewness = X[column].skew()
                if pd.notna(skewness) and float(skewness) > 1.0:
                    minimum = float(X[column].min())
                    shift = abs(minimum) + 1.0 if minimum <= -1.0 else 0.0
                    X[column] = np.log1p(X[column] + shift)
                    log_transforms.append({
                        "column": column,
                        "skewness": float(skewness),
                        "shift": shift,
                    })
            if log_transforms:
                applied_transformations.append({
                    "type": "log1p_transform",
                    "columns": log_transforms,
                })

            numeric_columns, categorical_columns = self._get_feature_types(X)

            interaction_source_columns = self._select_interaction_columns(X, numeric_columns)
            if interaction_source_columns:
                new_features = self._generate_interactions(X, interaction_source_columns)
                if new_features:
                    generated_features.extend(new_features)
                    applied_transformations.append({
                        "type": "generated_interactions",
                        "source_columns": interaction_source_columns,
                        "columns": new_features,
                    })

            correlated_columns = self._find_high_correlation_columns(X)
            if correlated_columns:
                X = X.drop(columns=correlated_columns)
                dropped_columns.extend(correlated_columns)
                applied_transformations.append({
                    "type": "correlation_filter",
                    "threshold": self.CORRELATION_THRESHOLD,
                    "columns": correlated_columns,
                })

            if generated_features:
                generated_features = [column for column in generated_features if column in X.columns]

            numeric_columns, categorical_columns = self._get_feature_types(X)

            if y is not None and y.isna().any():
                valid_mask = y.notna()
                X = X.loc[valid_mask].copy()
                y = y.loc[valid_mask].copy()
                applied_transformations.append({
                    "type": "drop_missing_target_rows",
                    "rows_removed": int((~valid_mask).sum()),
                })

            feature_scores = self._compute_feature_importance_scores(X, y)
            feature_scores.pop(target_column, None)

            selected_features = self._select_features_from_scores(
                X=X,
                feature_scores=feature_scores,
                n_features_to_select=n_features_to_select,
                target_column=target_column,
            )

            protected_base_features = self._collect_protected_base_features(selected_features, X.columns)
            low_importance_columns = [
                column for column in X.columns
                if column not in selected_features and column not in protected_base_features
            ]
            if low_importance_columns:
                dropped_columns.extend(low_importance_columns)
                applied_transformations.append({
                    "type": "feature_importance_filter",
                    "threshold": self.IMPORTANCE_THRESHOLD,
                    "columns": sorted(low_importance_columns),
                })

            selected_features = sorted(
                set(selected_features).union(protected_base_features),
                key=lambda column: list(X.columns).index(column),
            )

            if n_features_to_select > 0 and len(selected_features) > n_features_to_select:
                capped_features = self._apply_selection_cap(
                    selected_features=selected_features,
                    feature_scores=feature_scores,
                    feature_order=list(X.columns),
                    n_features_to_select=n_features_to_select,
                )
                capped_out_columns = [
                    column for column in selected_features
                    if column not in capped_features
                ]
                if capped_out_columns:
                    dropped_columns.extend(capped_out_columns)
                    applied_transformations.append({
                        "type": "selection_cap",
                        "max_features": n_features_to_select,
                        "columns": sorted(capped_out_columns),
                    })
                selected_features = capped_features

            if use_pca and selected_features:
                X, selected_features, generated_features, pca_result = self._apply_pca(
                    X=X,
                    selected_features=selected_features,
                    generated_features=generated_features,
                    n_pca_components=n_pca_components,
                )
                if pca_result:
                    applied_transformations.append({
                        "type": "pca",
                        "n_components": pca_result["n_components"],
                        "components": pca_result["components"],
                    })

            selected_features = [
                column for column in selected_features
                if column in X.columns and column != target_column
            ]
            feature_scores = {
                column: float(score)
                for column, score in self._rank_features(feature_scores)
                if column in X.columns and column != target_column
            }
            generated_features = [
                column for column in generated_features
                if column in X.columns and column != target_column
            ]

            selected_numeric, selected_categorical = self._split_selected_features(
                X,
                selected_features,
                preprocessing_numeric,
                preprocessing_categorical,
            )

            result = {
                "dropped_columns": sorted(dict.fromkeys(dropped_columns)),
                "generated_features": generated_features,
                "selected_features": selected_features,
                "feature_scores": feature_scores,
                "applied_transformations": applied_transformations,
                "final_feature_count": len(selected_features),
                "pca_result": pca_result if pca_result else None,
                "feature_engineering_config": {
                    "n_features_to_select": n_features_to_select,
                    "use_pca": use_pca,
                    "n_pca_components": n_pca_components,
                    "original_feature_count": len(df.columns) - (1 if target_column in df.columns else 0),
                    "post_engineering_feature_count": len(X.columns),
                    "importance_threshold": self.IMPORTANCE_THRESHOLD,
                    "max_interaction_numeric_columns": self.MAX_INTERACTION_NUMERIC_COLUMNS,
                },
                "numeric_features": selected_numeric,
                "categorical_features": selected_categorical,
                "_engineered_df": X,
            }

            logger.info(
                "Feature engineering complete: %s selected, %s generated, %s dropped",
                len(selected_features),
                len(generated_features),
                len(result["dropped_columns"]),
            )
            return result

        except Exception as e:
            logger.exception("Error in feature engineering: %s", e)
            raise AgentExecutionError(
                f"Feature engineering failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _drop_target_column_if_present(self, X: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Defensively remove the target column from the feature matrix."""
        if target_column in X.columns:
            return X.drop(columns=[target_column])
        return X

    def _is_index_like(self, column_name: str) -> bool:
        """Return True when the column looks like an index or identifier."""
        normalized = column_name.strip().lower()
        return (
            normalized.startswith("unnamed")
            or normalized in self.INDEX_LIKE_NAMES
            or normalized.endswith("_id")
        )

    def _get_feature_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Split features into numeric and categorical groups."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [column for column in X.columns if column not in numeric_columns]
        return numeric_columns, categorical_columns

    def _select_interaction_columns(self, X: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
        """Choose the top numeric columns by variance for interaction generation."""
        if not numeric_columns:
            return []

        variances = X[numeric_columns].var(ddof=0).fillna(0.0)
        ranked = sorted(
            numeric_columns,
            key=lambda column: (-float(variances.get(column, 0.0)), column),
        )
        ranked = [column for column in ranked if float(variances.get(column, 0.0)) > 0.0]
        return ranked[: self.MAX_INTERACTION_NUMERIC_COLUMNS]

    def _generate_interactions(self, X: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
        """Generate deterministic pairwise interactions among selected numeric columns."""
        generated: list[str] = []
        epsilon = 1e-6

        for left, right in combinations(numeric_columns, 2):
            product_name = f"{left}__mul__{right}"
            ratio_name = f"{left}__div__{right}"

            X[product_name] = X[left] * X[right]
            denominator = X[right].where(X[right].abs() > epsilon, np.nan)
            X[ratio_name] = (
                X[left] / denominator
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            generated.extend([product_name, ratio_name])

        spatial_columns = [column for column in ("x", "y", "z") if column in X.columns]
        if len(spatial_columns) == 3:
            volume_name = "x__mul__y__mul__z"
            if volume_name not in X.columns:
                X[volume_name] = X["x"] * X["y"] * X["z"]
                generated.append(volume_name)

        return generated

    def _find_high_correlation_columns(self, X: pd.DataFrame) -> list[str]:
        """Find redundant columns while preferring simpler/original features."""
        numeric_frame = X.select_dtypes(include=[np.number])
        if numeric_frame.empty:
            return []

        correlation_matrix = numeric_frame.corr().abs()
        variances = numeric_frame.var(ddof=0).fillna(0.0).to_dict()
        columns_to_drop: set[str] = set()
        numeric_columns = list(numeric_frame.columns)

        for index, left in enumerate(numeric_columns):
            if left in columns_to_drop:
                continue
            for right in numeric_columns[index + 1:]:
                if right in columns_to_drop:
                    continue
                correlation = correlation_matrix.loc[left, right]
                if pd.isna(correlation) or float(correlation) <= self.CORRELATION_THRESHOLD:
                    continue
                columns_to_drop.add(self._choose_correlated_drop(left, right, variances))

        return sorted(columns_to_drop)

    def _choose_correlated_drop(
        self,
        left: str,
        right: str,
        variances: dict[str, float],
    ) -> str:
        """Pick which correlated feature to drop, preferring simpler/original features."""
        left_is_derived = self._is_derived_feature(left)
        right_is_derived = self._is_derived_feature(right)

        if left_is_derived != right_is_derived:
            return left if left_is_derived else right

        left_variance = float(variances.get(left, 0.0))
        right_variance = float(variances.get(right, 0.0))
        if left_variance != right_variance:
            return left if left_variance < right_variance else right

        left_complexity = len(self._extract_base_features(left))
        right_complexity = len(self._extract_base_features(right))
        if left_complexity != right_complexity:
            return left if left_complexity > right_complexity else right

        return max(left, right)

    def _compute_feature_importance_scores(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
    ) -> dict[str, float]:
        """Train a lightweight RandomForest model and return feature importances."""
        if X.empty:
            return {}

        if y is None:
            return {column: 1.0 for column in X.columns}

        encoded_X = self._encode_features_for_model(X)
        encoded_y, is_classification = self._encode_target(y)

        if is_classification:
            model = RandomForestClassifier(
                n_estimators=self.IMPORTANCE_TREES,
                max_depth=8,
                random_state=self.RANDOM_STATE,
                n_jobs=1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.IMPORTANCE_TREES,
                max_depth=8,
                random_state=self.RANDOM_STATE,
                n_jobs=1,
            )

        model.fit(encoded_X, encoded_y)
        return {
            column: float(score)
            for column, score in zip(encoded_X.columns, model.feature_importances_)
        }

    def _select_features_from_scores(
        self,
        *,
        X: pd.DataFrame,
        feature_scores: dict[str, float],
        n_features_to_select: int,
        target_column: str,
    ) -> list[str]:
        """Select important features and retain base columns for selected interactions."""
        initial_selected = [
            column for column in X.columns
            if column != target_column and feature_scores.get(column, 0.0) >= self.IMPORTANCE_THRESHOLD
        ]

        if not initial_selected and len(X.columns) > 0:
            ranked_features = self._rank_features(feature_scores)
            fallback_count = n_features_to_select if n_features_to_select > 0 else 1
            initial_selected = [column for column, _ in ranked_features[:fallback_count]]

        protected_base_features = self._collect_protected_base_features(initial_selected, X.columns)
        selected = [
            column for column in X.columns
            if column in set(initial_selected).union(protected_base_features)
        ]
        return selected

    def _collect_protected_base_features(
        self,
        selected_features: list[str],
        available_columns: Any,
    ) -> set[str]:
        """Keep source features when a derived interaction is selected."""
        available = set(available_columns)
        protected: set[str] = set()
        for feature in selected_features:
            if not self._is_derived_feature(feature):
                continue
            protected.update(
                base_feature for base_feature in self._extract_base_features(feature)
                if base_feature in available
            )
        return protected

    def _apply_selection_cap(
        self,
        *,
        selected_features: list[str],
        feature_scores: dict[str, float],
        feature_order: list[str],
        n_features_to_select: int,
    ) -> list[str]:
        """Cap selected features while preserving bases for retained interactions."""
        order_index = {column: index for index, column in enumerate(feature_order)}
        ranked = sorted(
            selected_features,
            key=lambda column: (-feature_scores.get(column, 0.0), order_index.get(column, len(feature_order)), column),
        )
        capped = ranked[:n_features_to_select]
        protected = self._collect_protected_base_features(capped, feature_order)
        kept = [
            column for column in feature_order
            if column in set(capped).union(protected)
        ]
        return kept

    def _encode_features_for_model(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features deterministically for model-based scoring."""
        encoded = X.copy()
        for column in encoded.columns:
            if pd.api.types.is_numeric_dtype(encoded[column]):
                continue
            values = encoded[column].astype(str)
            categories = sorted(values.unique().tolist())
            encoded[column] = pd.Categorical(values, categories=categories).codes
        return encoded

    def _encode_target(self, y: pd.Series) -> tuple[pd.Series, bool]:
        """Encode target values and infer task type."""
        if self._is_classification_target(y):
            values = y.astype(str)
            categories = sorted(values.unique().tolist())
            encoded_y = pd.Series(
                pd.Categorical(values, categories=categories).codes,
                index=y.index,
            )
            return encoded_y, True
        return y.astype(float), False

    def _is_classification_target(self, y: pd.Series) -> bool:
        """Infer whether the target is classification-like."""
        if (
            pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_object_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
        ):
            return True
        if pd.api.types.is_integer_dtype(y):
            unique_count = int(y.nunique(dropna=True))
            return unique_count <= min(20, max(2, len(y) // 5))
        return False

    def _rank_features(self, feature_scores: dict[str, float]) -> list[tuple[str, float]]:
        """Return features sorted by descending importance, then by name."""
        return sorted(
            feature_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

    def _is_derived_feature(self, feature_name: str) -> bool:
        """Return True if the feature is a generated interaction or PCA output."""
        return "__mul__" in feature_name or "__div__" in feature_name or feature_name.startswith("PC")

    def _extract_base_features(self, feature_name: str) -> list[str]:
        """Extract source feature names from a derived feature."""
        if feature_name.startswith("PC"):
            return []
        if not self._is_derived_feature(feature_name):
            return [feature_name]
        return [part for part in re.split(r"__(?:mul|div)__", feature_name) if part]

    def _apply_pca(
        self,
        *,
        X: pd.DataFrame,
        selected_features: list[str],
        generated_features: list[str],
        n_pca_components: Optional[int],
    ) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
        """Apply PCA to selected numeric features while keeping categorical selections."""
        selected_numeric = [
            column for column in selected_features
            if column in X.columns and pd.api.types.is_numeric_dtype(X[column])
        ]
        selected_categorical = [
            column for column in selected_features
            if column in X.columns and column not in selected_numeric
        ]
        if len(selected_numeric) < 2:
            return X, selected_features, generated_features, {}

        n_components = n_pca_components or min(10, len(selected_numeric))
        n_components = min(n_components, len(selected_numeric))
        pca = PCA(n_components=n_components, svd_solver="full")
        components = pca.fit_transform(X[selected_numeric])
        component_names = [f"PC{i + 1}" for i in range(n_components)]

        pca_frame = pd.DataFrame(components, index=X.index, columns=component_names)
        X = X.drop(columns=selected_numeric)
        X = pd.concat([X, pca_frame], axis=1)

        generated_features = [column for column in generated_features if column not in selected_numeric]
        generated_features.extend(component_names)
        selected_features = selected_categorical + component_names
        pca_result = {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": component_names,
        }
        return X, selected_features, generated_features, pca_result

    def _split_selected_features(
        self,
        X: pd.DataFrame,
        selected_features: list[str],
        preprocessing_numeric: list[str],
        preprocessing_categorical: list[str],
    ) -> tuple[list[str], list[str]]:
        """Return selected feature groups after engineering."""
        numeric_selected = [
            column for column in selected_features
            if column in X.columns and pd.api.types.is_numeric_dtype(X[column])
        ]
        categorical_selected = [
            column for column in selected_features
            if column in X.columns and column not in numeric_selected
        ]

        for column in preprocessing_numeric:
            if column in selected_features and column not in numeric_selected:
                numeric_selected.append(column)
        for column in preprocessing_categorical:
            if column in selected_features and column not in categorical_selected:
                categorical_selected.append(column)

        return numeric_selected, categorical_selected
