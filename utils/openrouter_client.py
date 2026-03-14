"""Minimal OpenRouter client for optional agentic reasoning."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import get_openrouter_api_key, settings
from utils.logger import get_logger


class OpenRouterClient:
    """Small OpenRouter client with graceful fallback behavior."""

    def __init__(self, logger_name: str = "OpenRouter") -> None:
        self._logger = get_logger(f"{logger_name}.OpenRouter")

    def is_enabled(self) -> bool:
        """Return whether OpenRouter is configured."""
        return bool(settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"))

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a plain-text response from OpenRouter."""
        api_key = get_openrouter_api_key()
        payload = {
            "model": settings.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.model_temperature if temperature is None else temperature,
            "max_tokens": settings.model_max_tokens if max_tokens is None else max_tokens,
        }

        request = Request(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": settings.app_name,
            },
            method="POST",
        )
        request.add_unredirected_header("Authorization", f"Bearer {api_key}")

        try:
            with urlopen(request, timeout=settings.request_timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc

        payload = json.loads(body)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in text_parts if part).strip()

        if isinstance(content, str):
            return content.strip()

        raise RuntimeError("OpenRouter returned an unsupported content payload")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate a JSON object from OpenRouter."""
        response_text = self.generate_text(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_json_object(response_text)

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from a model response."""
        stripped = text.strip()

        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"OpenRouter did not return JSON: {stripped}")

        candidate = stripped[start : end + 1]
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise RuntimeError("OpenRouter JSON response was not an object")
        return payload
