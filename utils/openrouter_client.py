"""Minimal OpenRouter client for optional agentic reasoning."""

from __future__ import annotations

import json
import os
import re
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
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                    continue
                if isinstance(part.get("content"), str):
                    text_parts.append(part["content"])
                    continue
                if isinstance(part.get("value"), str):
                    text_parts.append(part["value"])
                    continue
                if isinstance(part.get("type"), str) and part.get("type") == "text" and isinstance(part.get("data"), str):
                    text_parts.append(part["data"])
            assembled = "\n".join(part for part in text_parts if part).strip()
            if assembled:
                return assembled
            self._logger.warning("OpenRouter returned an empty content parts payload: %s", json.dumps(payload, ensure_ascii=True))
            raise RuntimeError("OpenRouter returned an unsupported content payload")

        if isinstance(content, str):
            return content.strip()

        if content is None and isinstance(message.get("reasoning"), str):
            return message["reasoning"].strip()

        self._logger.warning("OpenRouter returned an unsupported content payload: %s", json.dumps(payload, ensure_ascii=True))
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
        try:
            return self._extract_json_object(response_text)
        except Exception as exc:
            self._logger.warning(
                "OpenRouter JSON parse failed. Raw response (truncated): %s",
                self._truncate_for_log(response_text),
            )
            raise RuntimeError(str(exc)) from exc

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from a model response."""
        stripped = self._normalize_json_text(text)
        candidates: list[str] = []

        candidates.append(stripped)
        fence_stripped = self._strip_code_fences(stripped)
        if fence_stripped != stripped:
            candidates.append(fence_stripped)

        balanced_object = self._find_balanced_object(stripped)
        if balanced_object:
            candidates.append(balanced_object)

        if fence_stripped != stripped:
            fenced_balanced = self._find_balanced_object(fence_stripped)
            if fenced_balanced:
                candidates.append(fenced_balanced)

        parse_errors: list[str] = []
        for candidate in self._dedupe_candidates(candidates):
            payload = self._try_parse_json_object(candidate, parse_errors)
            if payload is not None:
                return payload

            repaired = self._repair_common_json_issues(candidate)
            if repaired != candidate:
                payload = self._try_parse_json_object(repaired, parse_errors)
                if payload is not None:
                    return payload

        if parse_errors:
            raise RuntimeError(f"OpenRouter returned malformed JSON: {parse_errors[-1]}")

        raise RuntimeError(f"OpenRouter did not return JSON: {stripped}")

    def _normalize_json_text(self, text: str) -> str:
        """Normalize common model response wrappers before JSON extraction."""
        normalized = text.lstrip("\ufeff").strip()

        # Handle responses like: "json\n{...}" or "JSON: {...}"
        lowered = normalized.lower()
        if lowered.startswith("json"):
            remainder = normalized[4:]
            normalized = remainder.lstrip(" :\n\r\t")

        # Handle opening fence even if closing fence is missing.
        if normalized.startswith("```"):
            lines = normalized.splitlines()
            if lines:
                normalized = "\n".join(lines[1:]).strip()

        return normalized

    def _try_parse_json_object(
        self,
        text: str,
        parse_errors: list[str],
    ) -> Optional[dict[str, Any]]:
        """Attempt to parse text as a JSON object."""
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            parse_errors.append("JSON payload was not an object")
            return None
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            return None

    def _strip_code_fences(self, text: str) -> str:
        """Strip markdown code fences if present."""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) < 3:
            return stripped

        if lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()

        # If closing fence is missing, still remove the opening fence line.
        if lines[0].startswith("```"):
            return "\n".join(lines[1:]).strip()

        return stripped

    def _find_balanced_object(self, text: str) -> Optional[str]:
        """Find the first balanced JSON object in text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        return None

    def _repair_common_json_issues(self, text: str) -> str:
        """Repair common model JSON formatting issues deterministically."""
        repaired = text
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r"\n\s*```.*$", "", repaired, flags=re.MULTILINE)
        return repaired.strip()

    def _dedupe_candidates(self, candidates: list[str]) -> list[str]:
        """Return candidates preserving order and removing duplicates."""
        seen: set[str] = set()
        unique: list[str] = []
        for item in candidates:
            if not item or item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _truncate_for_log(self, text: str, max_chars: int = 4000) -> str:
        """Trim long LLM payloads for readable logs."""
        normalized = text.replace("\n", "\\n")
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars] + "... [truncated]"
