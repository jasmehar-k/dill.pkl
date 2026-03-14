"""Tests for the agent modules.

This module contains tests for the base agent and specific agent
implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.base_agent import BaseAgent
from agents.content_writer import ContentWriterAgent
from agents.editor_agent import EditorAgent
from agents.seo_optimizer_agent import SeoOptimizerAgent
from core.exceptions import AgentExecutionError


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_base_agent_initialization(self) -> None:
        """Test that BaseAgent can be initialized."""

        class TestAgent(BaseAgent):
            async def execute(self, text: str) -> str:
                return text

        agent = TestAgent(name="test_agent")
        assert agent.name == "test_agent"

    @pytest.mark.asyncio
    async def test_base_agent_run_wraps_execute(self) -> None:
        """Test that run() wraps execute() with error handling."""

        class TestAgent(BaseAgent):
            async def execute(self, text: str) -> str:
                return f"processed: {text}"

        agent = TestAgent(name="test")
        result = await agent.run("input")
        assert result == "processed: input"

    @pytest.mark.asyncio
    async def test_base_agent_run_handles_execution_error(self) -> None:
        """Test that run() handles execution errors."""

        class TestAgent(BaseAgent):
            async def execute(self, text: str) -> str:
                raise ValueError("Test error")

        agent = TestAgent(name="test")
        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.run("input")

        assert "Test error" in str(exc_info.value)


class TestContentWriterAgent:
    """Tests for the ContentWriterAgent class."""

    def test_content_writer_initialization(self) -> None:
        """Test ContentWriterAgent initialization."""
        agent = ContentWriterAgent()
        assert agent.name == "content_writer"

    @pytest.mark.asyncio
    async def test_content_writer_fallback(self) -> None:
        """Test fallback content generation."""
        with patch("agents.content_writer.get_openrouter_api_key", side_effect=ValueError("No API key")):
            agent = ContentWriterAgent()
            result = await agent.execute("Test Topic")

            assert "Test Topic" in result
            assert "first draft" in result


class TestEditorAgent:
    """Tests for the EditorAgent class."""

    def test_editor_initialization(self) -> None:
        """Test EditorAgent initialization."""
        agent = EditorAgent()
        assert agent.name == "editor"

    @pytest.mark.asyncio
    async def test_editor_fallback(self) -> None:
        """Test fallback editing."""
        with patch("agents.editor_agent.get_openrouter_api_key", side_effect=ValueError("No API key")):
            agent = EditorAgent()
            content = "This is a first draft about testing."
            result = await agent.execute(content)

            assert "polished draft" in result


class TestSeoOptimizerAgent:
    """Tests for the SeoOptimizerAgent class."""

    def test_seo_optimizer_initialization(self) -> None:
        """Test SeoOptimizerAgent initialization."""
        agent = SeoOptimizerAgent()
        assert agent.name == "seo_optimizer"

    @pytest.mark.asyncio
    async def test_seo_optimizer_passes_through_existing_keywords(self) -> None:
        """Test that existing SEO keywords are preserved."""
        agent = SeoOptimizerAgent()
        content = "# Test\n\n## SEO Keywords\n- existing keyword"
        result = await agent.execute(content)

        assert "## SEO Keywords" in result

    @pytest.mark.asyncio
    async def test_seo_optimizer_fallback_adds_keywords(self) -> None:
        """Test fallback SEO keyword addition."""
        with patch("agents.seo_optimizer_agent.get_openrouter_api_key", side_effect=ValueError("No API key")):
            agent = SeoOptimizerAgent()
            content = "# Test Content"
            result = await agent.execute(content)

            assert "## SEO Keywords" in result