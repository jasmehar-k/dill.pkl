"""Tests for the Orchestrator module.

This module contains tests for the Orchestrator class which
coordinates the multi-agent blog generation pipeline.
"""

import pytest

from core.message import Message
from core.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_orchestrator_pipeline_generates_output() -> None:
    """Test that the orchestrator generates output with all expected components."""
    orchestrator = Orchestrator()
    result = await orchestrator.run("Test Topic")

    assert "Test Topic" in result
    assert "SEO Keywords" in result
    assert len(orchestrator.memory.all()) == 4


@pytest.mark.asyncio
async def test_orchestrator_memory_stores_all_messages() -> None:
    """Test that memory stores all pipeline messages."""
    orchestrator = Orchestrator()
    await orchestrator.run("Memory Test")

    messages = orchestrator.memory.all()
    assert len(messages) == 4

    # Verify message roles
    roles = [msg.role for msg in messages]
    assert "user" in roles
    assert "content_writer" in roles
    assert "editor" in roles
    assert "seo_optimizer" in roles


@pytest.mark.asyncio
async def test_orchestrator_creates_agents() -> None:
    """Test that orchestrator initializes all agents."""
    orchestrator = Orchestrator()
    await orchestrator.run("Agent Test")

    assert orchestrator.writer is not None
    assert orchestrator.editor is not None
    assert orchestrator.seo is not None


def test_orchestrator_sync_run() -> None:
    """Test synchronous run method."""
    orchestrator = Orchestrator()
    result = orchestrator.run_sync("Sync Test")

    assert "Sync Test" in result
    assert "SEO Keywords" in result


def test_orchestrator_initialization() -> None:
    """Test orchestrator can be initialized without running."""
    orchestrator = Orchestrator()

    assert orchestrator.memory is not None
    assert orchestrator.writer is None  # Lazy initialization
    assert orchestrator.editor is None
    assert orchestrator.seo is None