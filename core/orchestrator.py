"""Orchestrator for the Blog Generator pipeline.

This module provides the Orchestrator class that coordinates the
multi-agent pipeline for blog content generation.
"""

import logging
from typing import Optional

from agents.content_writer import ContentWriterAgent
from agents.editor_agent import EditorAgent
from agents.seo_optimizer_agent import SeoOptimizerAgent
from core.exceptions import PipelineError
from core.memory_manager import MemoryManager
from core.message import Message


class Orchestrator:
    """Orchestrates the multi-agent blog generation pipeline.

    The orchestrator coordinates three agents in sequence:
    1. ContentWriterAgent - generates initial draft
    2. EditorAgent - polishes and improves the content
    3. SeoOptimizerAgent - adds SEO optimizations

    The orchestrator maintains memory of all interactions and handles
    error management across the pipeline.

    Attributes:
        memory: The memory manager for storing conversation history.
    """

    def __init__(self) -> None:
        """Initialize the Orchestrator with agent instances."""
        self._logger = logging.getLogger(__name__)
        self.memory = MemoryManager()
        self.writer: Optional[ContentWriterAgent] = None
        self.editor: Optional[EditorAgent] = None
        self.seo: Optional[SeoOptimizerAgent] = None

    def _initialize_agents(self) -> None:
        """Initialize all agent instances.

        Agents are initialized lazily to avoid creating them
        if the pipeline is not going to be run.
        """
        if self.writer is None:
            self.writer = ContentWriterAgent()
        if self.editor is None:
            self.editor = EditorAgent()
        if self.seo is None:
            self.seo = SeoOptimizerAgent()

    async def run(self, topic: str) -> str:
        """Run the complete blog generation pipeline.

        Executes the three-agent pipeline in sequence to generate
        a complete, SEO-optimized blog post.

        Args:
            topic: The blog topic to generate content for.

        Returns:
            The final optimized blog content as a markdown string.

        Raises:
            PipelineError: If any stage of the pipeline fails.
        """
        self._logger.info(f"Starting blog generation pipeline for topic: {topic}")
        self._initialize_agents()

        try:
            # Stage 1: Content Writing
            self._logger.info("Stage 1: Generating initial content")
            self.memory.add(Message(role="user", content=topic))
            draft = await self.writer.run(topic)
            self.memory.add(Message(role="content_writer", content=draft))
            self._logger.info(f"Stage 1 complete, draft length: {len(draft)} characters")

            # Stage 2: Editing
            self._logger.info("Stage 2: Editing content")
            edited = await self.editor.run(draft)
            self.memory.add(Message(role="editor", content=edited))
            self._logger.info(f"Stage 2 complete, edited length: {len(edited)} characters")

            # Stage 3: SEO Optimization
            self._logger.info("Stage 3: Optimizing for SEO")
            optimized = await self.seo.run(edited)
            self.memory.add(Message(role="seo_optimizer", content=optimized))
            self._logger.info(f"Stage 3 complete, final length: {len(optimized)} characters")

            self._logger.info("Pipeline completed successfully")
            return optimized

        except Exception as e:
            self._logger.exception(f"Pipeline failed at stage: {e}")
            raise PipelineError(
                f"Pipeline failed: {str(e)}",
                details={"topic": topic, "error": str(e)},
            ) from e

    def run_sync(self, topic: str) -> str:
        """Synchronous wrapper for running the pipeline.

        This method provides a synchronous interface to the async
        pipeline for use in non-async contexts like CLI.

        Args:
            topic: The blog topic to generate content for.

        Returns:
            The final optimized blog content as a markdown string.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(topic))
                    return future.result()
            return loop.run_until_complete(self.run(topic))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.run(topic))