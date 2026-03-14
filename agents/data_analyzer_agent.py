"""Data Analyzer Agent for AutoML Pipeline.

This agent analyzes the uploaded dataset to provide insights about
data quality, feature types, correlations, and recommends preprocessing strategies.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class DataAnalyzerAgent(BaseAgent):
    """Agent for analyzing datasets and providing preprocessing recommendations.

    This agent analyzes:
    - Row and column counts
    - Column data types (numeric, categorical, text)
    - Missing value patterns
    - Class distribution (for classification tasks)
    - Feature correlations
    - Outliers in numeric columns
    """

    def __init__(self) -> None:
        """Initialize the DataAnalyzerAgent."""
        super().__init__("DataAnalyzer")

    async def execute(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> dict[str, Any]:
        """Analyze the dataset and return comprehensive analysis results.

        Args:
            df: The input DataFrame to analyze.
            target_column: Optional target column name for classification/regression.

        Returns:
            Dictionary containing analysis results including:
            - row_count, column_count, feature_count
            - column_types: dict mapping column names to types
            - missing_values: dict with missing value percentages
            - numeric_columns, categorical_columns
            - correlations: correlation matrix for numeric features
            - class_distribution (if target provided)
            - recommendations: list of preprocessing recommendations
        """
        try:
            logger.info(f"Analyzing dataset with {len(df)} rows and {len(df.columns)} columns")

            # Basic stats
            row_count = len(df)
            column_count = len(df.columns)
            feature_count = column_count - (1 if target_column else 0)

            # Classify columns by type
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Handle target column classification
            if target_column:
                if target_column in numeric_columns:
                    numeric_columns.remove(target_column)
                if target_column in categorical_columns:
                    categorical_columns.remove(target_column)

            # Missing value analysis
            missing_values = df.isnull().mean().to_dict()
            high_missing_cols = [col for col, pct in missing_values.items() if pct > 0.3]

            # Correlation analysis for numeric columns
            correlations = None
            if len(numeric_columns) > 1:
                corr_matrix = df[numeric_columns].corr()
                correlations = corr_matrix.to_dict()

                # Find highly correlated pairs
                high_corr_pairs = []
                for i, col1 in enumerate(numeric_columns):
                    for j, col2 in enumerate(numeric_columns):
                        if i < j and abs(corr_matrix.loc[col1, col2]) > 0.8:
                            high_corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
            else:
                high_corr_pairs = []

            # Class distribution for target column
            class_distribution = None
            if target_column:
                target_series = df[target_column]
                if target_series.dtype in ["object", "category"] or not np.issubdtype(
                    target_series.dtype, np.number
                ):
                    # Classification task
                    class_distribution = target_series.value_counts().to_dict()
                else:
                    # Regression task
                    class_distribution = {
                        "min": float(target_series.min()),
                        "max": float(target_series.max()),
                        "mean": float(target_series.mean()),
                        "std": float(target_series.std()),
                        "median": float(target_series.median()),
                    }

            # Outlier analysis for numeric columns
            outliers = {}
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outlier_count > 0:
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(df) * 100),
                    }

            # Generate recommendations
            recommendations = self._generate_recommendations(
                missing_values=missing_values,
                high_missing_cols=high_missing_cols,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                high_corr_pairs=high_corr_pairs,
                outliers=outliers,
                class_distribution=class_distribution,
                target_column=target_column,
            )

            llm_insights = self._generate_llm_analysis_insights(
                row_count=row_count,
                column_count=column_count,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                high_missing_cols=high_missing_cols,
                high_corr_pairs=high_corr_pairs,
                outliers=outliers,
                class_distribution=class_distribution,
                target_column=target_column,
                deterministic_recommendations=recommendations,
            )

            if llm_insights.get("recommendations"):
                recommendations = self._merge_recommendations(
                    recommendations,
                    llm_insights["recommendations"],
                )

            result = {
                "row_count": row_count,
                "column_count": column_count,
                "feature_count": feature_count,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "missing_values": missing_values,
                "high_missing_columns": high_missing_cols,
                "correlations": correlations,
                "high_correlation_pairs": high_corr_pairs,
                "outliers": outliers,
                "class_distribution": class_distribution,
                "target_column": target_column,
                "recommendations": recommendations,
                "analysis_summary": llm_insights.get("analysis_summary"),
                "risk_level": llm_insights.get("risk_level"),
                "llm_used": llm_insights.get("llm_used", False),
            }

            logger.info(f"Analysis complete: {feature_count} features, {len(high_missing_cols)} cols with high missing")
            return result

        except Exception as e:
            logger.exception(f"Error analyzing dataset: {e}")
            raise AgentExecutionError(
                f"Data analysis failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _generate_recommendations(
        self,
        missing_values: dict[str, float],
        high_missing_cols: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict,
        class_distribution: Optional[dict],
        target_column: Optional[str],
    ) -> list[str]:
        """Generate preprocessing recommendations based on analysis."""
        recommendations = []

        # Missing value recommendations
        if high_missing_cols:
            recommendations.append(
                f"High missing values detected in {len(high_missing_cols)} columns. "
                "Consider dropping or imputing."
            )

        # Correlation recommendations
        if high_corr_pairs:
            recommendations.append(
                f"High correlation detected between {len(high_corr_pairs)} feature pairs. "
                "Consider removing redundant features."
            )

        # Outlier recommendations
        if outliers:
            recommendations.append(
                f"Outliers detected in {len(outliers)} columns. Consider winsorization or removal."
            )

        # Class imbalance
        if class_distribution and isinstance(class_distribution, dict):
            if all(isinstance(v, int) for v in class_distribution.values()):
                total = sum(class_distribution.values())
                max_pct = max(class_distribution.values()) / total
                min_pct = min(class_distribution.values()) / total
                if max_pct / min_pct > 10 if min_pct > 0 else False:
                    recommendations.append(
                        "Significant class imbalance detected. Consider SMOTE or class weights."
                    )

        # Default recommendations
        if not recommendations:
            recommendations.append("Data quality looks good. Proceed with standard preprocessing.")

        return recommendations

    def _generate_llm_analysis_insights(
        self,
        *,
        row_count: int,
        column_count: int,
        numeric_columns: list[str],
        categorical_columns: list[str],
        high_missing_cols: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict[str, Any],
        class_distribution: Optional[dict[str, Any]],
        target_column: Optional[str],
        deterministic_recommendations: list[str],
    ) -> dict[str, Any]:
        """Use an LLM to synthesize a concise analysis summary and recommendations."""
        payload = {
            "row_count": row_count,
            "column_count": column_count,
            "target_column": target_column,
            "numeric_columns": numeric_columns[:15],
            "categorical_columns": categorical_columns[:15],
            "high_missing_columns": high_missing_cols[:15],
            "high_correlation_pairs": [list(pair) for pair in high_corr_pairs[:10]],
            "outliers": outliers,
            "class_distribution": class_distribution,
            "deterministic_recommendations": deterministic_recommendations,
        }

        response = self._generate_llm_json(
            system_prompt=(
                "You are an AutoML data analysis assistant. "
                "Given structured dataset diagnostics, produce a concise risk summary and "
                "up to three practical preprocessing recommendations. "
                "Return ONLY valid JSON with keys "
                "'analysis_summary', 'risk_level', and 'recommendations'. "
                "risk_level must be one of: low, medium, high. "
                "Recommendations must be short, concrete, and implementation-oriented."
            ),
            user_prompt=f"Dataset diagnostics:\n{self._safe_json(payload)}",
            temperature=0.1,
            max_tokens=500,
        )

        if not response:
            return {
                "analysis_summary": self._build_default_summary(
                    row_count,
                    column_count,
                    high_missing_cols,
                    high_corr_pairs,
                    outliers,
                ),
                "risk_level": self._default_risk_level(high_missing_cols, high_corr_pairs, outliers),
                "recommendations": [],
                "llm_used": False,
            }

        recommendations = response.get("recommendations", [])
        if not isinstance(recommendations, list):
            recommendations = []

        return {
            "analysis_summary": str(response.get("analysis_summary") or ""),
            "risk_level": str(response.get("risk_level") or self._default_risk_level(high_missing_cols, high_corr_pairs, outliers)).lower(),
            "recommendations": [str(item) for item in recommendations[:3] if str(item).strip()],
            "llm_used": True,
        }

    def _merge_recommendations(self, base: list[str], extra: list[str]) -> list[str]:
        """Merge deterministic and LLM recommendations while preserving order."""
        merged: list[str] = []
        for item in [*base, *extra]:
            normalized = item.strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    def _default_risk_level(
        self,
        high_missing_cols: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict[str, Any],
    ) -> str:
        """Generate a fallback risk level without an LLM."""
        issue_count = int(bool(high_missing_cols)) + int(bool(high_corr_pairs)) + int(bool(outliers))
        if issue_count >= 3:
            return "high"
        if issue_count >= 1:
            return "medium"
        return "low"

    def _build_default_summary(
        self,
        row_count: int,
        column_count: int,
        high_missing_cols: list[str],
        high_corr_pairs: list[tuple],
        outliers: dict[str, Any],
    ) -> str:
        """Generate a concise deterministic fallback summary."""
        parts = [f"Dataset contains {row_count} rows and {column_count} columns."]
        if high_missing_cols:
            parts.append(f"{len(high_missing_cols)} columns have high missingness.")
        if high_corr_pairs:
            parts.append(f"{len(high_corr_pairs)} strongly correlated feature pairs were detected.")
        if outliers:
            parts.append(f"Outliers were found in {len(outliers)} numeric columns.")
        if len(parts) == 1:
            parts.append("Overall data quality looks stable for standard preprocessing.")
        return " ".join(parts)
