"""Configuration management for the AutoML Pipeline application.

This module provides centralized configuration using Pydantic Settings
for environment variable validation and type safety.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="dill.pkl AutoML", description="Application name")
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    log_level: str = Field(default="INFO", description="Logging level")

    # OpenRouter API settings (optional for local processing)
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key for LLM calls")
    model_name: str = Field(default="arcee-ai/trinity-large-preview:free", description="LLM model to use")
    model_temperature: float = Field(default=0.7, description="LLM temperature (0.0-1.0)")
    model_max_tokens: int = Field(default=2000, description="Maximum tokens for LLM response")

    # Production settings
    request_timeout: int = Field(default=60, description="HTTP request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")

    # Pipeline settings
    default_test_size: float = Field(default=0.2, description="Default test set proportion")
    default_random_state: int = Field(default=42, description="Default random seed")
    default_cv_folds: int = Field(default=5, description="Default cross-validation folds")


settings = Settings()


def get_openrouter_api_key() -> str:
    """Retrieve the OpenRouter API key.

    Returns:
        The API key string.

    Raises:
        ValueError: If API key is not configured.
    """
    api_key = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env file."
        )
    return api_key