"""Tests for the agent modules.

This module contains tests for the base agent and specific agent
implementations for the AutoML pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.base_agent import BaseAgent
from agents.data_analyzer_agent import DataAnalyzerAgent
from agents.preprocessor_agent import PreprocessorAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.deployment_agent import DeploymentAgent
from core.exceptions import AgentExecutionError


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_base_agent_initialization(self) -> None:
        """Test that BaseAgent can be initialized."""

        class TestAgent(BaseAgent):
            async def execute(self, *args, **kwargs):
                return "test"

        agent = TestAgent(name="test_agent")
        assert agent.name == "test_agent"

    @pytest.mark.asyncio
    async def test_base_agent_run_wraps_execute(self) -> None:
        """Test that run() wraps execute() with error handling."""

        class TestAgent(BaseAgent):
            async def execute(self, *args, **kwargs):
                return f"processed: {args[0]}"

        agent = TestAgent(name="test")
        result = await agent.run("input")
        assert result == "processed: input"

    @pytest.mark.asyncio
    async def test_base_agent_run_handles_execution_error(self) -> None:
        """Test that run() handles execution errors."""

        class TestAgent(BaseAgent):
            async def execute(self, *args, **kwargs):
                raise ValueError("Test error")

        agent = TestAgent(name="test")
        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.run("input")

        assert "Test error" in str(exc_info.value)


class TestDataAnalyzerAgent:
    """Tests for the DataAnalyzerAgent class."""

    def test_data_analyzer_initialization(self) -> None:
        """Test DataAnalyzerAgent initialization."""
        agent = DataAnalyzerAgent()
        assert agent.name == "DataAnalyzer"

    @pytest.mark.asyncio
    async def test_data_analyzer_analyzes_dataframe(self) -> None:
        """Test that agent analyzes a DataFrame correctly."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "category": ["A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        })

        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        assert result["row_count"] == 5
        assert result["column_count"] == 4
        assert result["feature_count"] == 3
        assert "numeric_columns" in result
        assert "categorical_columns" in result

    @pytest.mark.asyncio
    async def test_data_analyzer_handles_missing_values(self) -> None:
        """Test that agent handles missing values."""
        df = pd.DataFrame({
            "feature1": [1, 2, np.nan, 4, 5],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 1, 0, 1, 0],
        })

        agent = DataAnalyzerAgent()
        result = await agent.execute(df, "target")

        assert "missing_values" in result
        assert result["missing_values"]["feature1"] == 0.2


class TestPreprocessorAgent:
    """Tests for the PreprocessorAgent class."""

    def test_preprocessor_initialization(self) -> None:
        """Test PreprocessorAgent initialization."""
        agent = PreprocessorAgent()
        assert agent.name == "Preprocessor"

    @pytest.mark.asyncio
    async def test_preprocessor_handles_categorical_columns(self) -> None:
        """Test that preprocessor handles categorical columns."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "category": ["A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        })

        analysis = {
            "numeric_columns": ["feature1"],
            "categorical_columns": ["category"],
        }

        agent = PreprocessorAgent()
        result = await agent.execute(df, analysis, "target")

        assert result["train_size"] > 0
        assert result["test_size"] > 0


class TestFeatureEngineeringAgent:
    """Tests for the FeatureEngineeringAgent class."""

    def test_feature_engineering_initialization(self) -> None:
        """Test FeatureEngineeringAgent initialization."""
        agent = FeatureEngineeringAgent()
        assert agent.name == "FeatureEngineering"

    @pytest.mark.asyncio
    async def test_feature_engineering_selects_features(self) -> None:
        """Test that agent selects features."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 10,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
            "feature3": [2, 3, 4, 5, 6] * 10,
            "target": [0, 1, 0, 1, 0] * 10,
        })

        preprocessing = {
            "numeric_columns": ["feature1", "feature2", "feature3"],
            "categorical_columns": [],
        }

        agent = FeatureEngineeringAgent()
        result = await agent.execute(df, preprocessing, "target", n_features_to_select=2)

        assert result["final_feature_count"] <= 2


class TestModelSelectionAgent:
    """Tests for the ModelSelectionAgent class."""

    def test_model_selection_initialization(self) -> None:
        """Test ModelSelectionAgent initialization."""
        agent = ModelSelectionAgent()
        assert agent.name == "ModelSelection"

    @pytest.mark.asyncio
    async def test_model_selection_recommends_model(self) -> None:
        """Test that agent recommends a model."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 10,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
            "target": [0, 1, 0, 1, 0] * 10,
        })

        features = {
            "final_feature_count": 2,
            "numeric_features": ["feature1", "feature2"],
        }

        agent = ModelSelectionAgent()
        result = await agent.execute(df, features, "target", "classification")

        assert "selected_model" in result
        assert "candidate_models" in result
        assert "hyperparameters" in result


class TestTrainingAgent:
    """Tests for the TrainingAgent class."""

    def test_training_initialization(self) -> None:
        """Test TrainingAgent initialization."""
        agent = TrainingAgent()
        assert agent.name == "Training"

    @pytest.mark.asyncio
    async def test_training_trains_model(self) -> None:
        """Test that agent trains a model."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })

        model_selection = {
            "selected_model": "RandomForest",
            "hyperparameters": {"n_estimators": 10, "random_state": 42},
            "task_type": "classification",
            "target_column": "target",
        }

        agent = TrainingAgent()
        result = await agent.execute(df, model_selection, {"test_size": 0.2, "random_state": 42})

        assert "model" in result
        assert "best_score" in result


class TestEvaluationAgent:
    """Tests for the EvaluationAgent class."""

    def test_evaluation_initialization(self) -> None:
        """Test EvaluationAgent initialization."""
        agent = EvaluationAgent()
        assert agent.name == "Evaluation"

    @pytest.mark.asyncio
    async def test_evaluation_computes_metrics(self) -> None:
        """Test that agent computes metrics."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })

        X = df[["feature1", "feature2"]]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        training_result = {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_score": model.score(X_train, y_train),
        }

        agent = EvaluationAgent()
        result = await agent.execute(training_result, "classification")

        assert "accuracy" in result
        assert "f1" in result
        assert "deployment_decision" in result


class TestDeploymentAgent:
    """Tests for the DeploymentAgent class."""

    def test_deployment_initialization(self) -> None:
        """Test DeploymentAgent initialization."""
        agent = DeploymentAgent()
        assert agent.name == "Deployment"

    @pytest.mark.asyncio
    async def test_deployment_saves_model(self) -> None:
        """Test that agent saves the model."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        training_result = {
            "model": model,
            "model_name": "RandomForest",
        }

        evaluation_result = {
            "task_type": "classification",
            "accuracy": 0.9,
            "f1": 0.85,
            "deployment_decision": "deploy",
        }

        agent = DeploymentAgent()
        result = await agent.execute(training_result, evaluation_result, "test_pipeline")

        assert "model_path" in result
        assert result["deployment_success"] is True