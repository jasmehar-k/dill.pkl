"""Base agent class for the Blog Generator application.

This module defines the abstract base class that all agents inherit from,
providing common functionality for async execution, logging, and error handling.
"""

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING

from core.exceptions import AgentExecutionError

if TYPE_CHECKING:
    from langchain.schema import BaseMessage


class BaseAgent(ABC):
    """Abstract base class for all agents in the auto machine learning pipeline.

    This class provides common functionality including async execution,
    logging, and structured error handling. All agents must inherit from
    this class and implement the `execute` method.

    Attributes:
        name: The name of the agent used for logging and identification.
    """

    def __init__(self, name: str) -> None:
        """Initialize the agent.

        Args:
            name: The name of the agent.
        """
        self.name = name
        self._logger = logging.getLogger(name)

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """Execute the agent's primary task.

        This method must be implemented by subclasses to define the
        agent's specific behavior.

        Args:
            input_text: The input text to process.

        Returns:
            The processed output text.

        Raises:
            AgentExecutionError: If the agent fails to execute.
        """
        raise NotImplementedError("Subclasses must implement the execute method")

    async def run(self, input_text: str) -> str:
        """Run the agent with error handling and logging.

        This method wraps the `execute` method with error handling
        and logging to provide consistent behavior across all agents.

        Args:
            input_text: The input text to process.

        Returns:
            The processed output text.

        Raises:
            AgentExecutionError: If the agent fails to execute.
        """
        self._logger.info(f"Starting {self.name} execution")
        try:
            result = await self.execute(input_text)
            self._logger.info(f"Completed {self.name} execution")
            return result
        except AgentExecutionError:
            raise
        except Exception as e:
            self._logger.exception(f"Error in {self.name}: {e}")
            raise AgentExecutionError(
                f"Agent {self.name} failed: {str(e)}",
                agent_name=self.name,
                details={"original_error": str(e)},
            ) from e

    def _build_prompt(self, template: str, **kwargs: str) -> str:
        """Build a prompt from a template with variable substitution.

        Args:
            template: The prompt template string.
            **kwargs: Variables to substitute in the template.

        Returns:
            The formatted prompt string.
        """
        return template.format(**kwargs)