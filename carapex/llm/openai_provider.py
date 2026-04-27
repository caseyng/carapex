"""
OpenAIProvider — LLMProvider implementation for OpenAI-compatible APIs.

Registered as name="openai". Works with any OpenAI-compatible endpoint
(OpenAI, Azure OpenAI, llama.cpp server, etc.) by setting `url`.

Config fields:
  type:  "openai"           (required)
  url:   "https://api.openai.com"  (required)
  model: "gpt-4o"          (required)

The caller's api_key is forwarded as the Authorization Bearer token.
When api_key is empty, the request proceeds — the server decides whether
to accept it. Returns None on any failure (network, auth, bad response).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from carapex.core.registry import register_llm
from carapex.core.types import CompletionResult, UsageResult
from carapex.llm.base import LLMProvider

log = logging.getLogger(__name__)


@register_llm
class OpenAIProvider(LLMProvider):
    """LLMProvider for OpenAI-compatible Chat Completions APIs."""

    name = "openai"

    def __init__(
        self,
        url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # LLMProvider contract
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        api_key: str,
    ) -> CompletionResult | None:
        return self._call(messages, api_key=api_key, temperature=None)

    def complete_with_temperature(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        api_key: str,
    ) -> CompletionResult | None:
        return self._call(messages, api_key=api_key, temperature=temperature)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 — close must never raise
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(
        self,
        messages: list[dict[str, str]],
        api_key: str,
        temperature: float | None,
    ) -> CompletionResult | None:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                time.sleep(self._retry_delay * (2 ** (attempt - 1)))
            try:
                resp = self._client.post(
                    f"{self._url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                return self._parse(resp.json())
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                log.debug("LLM call transient error (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, e)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    log.debug("LLM call retryable status (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, e)
                else:
                    log.debug("LLM call failed (non-retryable %d): %s", e.response.status_code, e)
                    return None
            except Exception as e:  # noqa: BLE001 — never raise per contract
                log.debug("LLM call failed: %s", e)
                return None

        log.debug("LLM call failed after %d attempts", self._max_retries + 1)
        return None

    @staticmethod
    def _parse(data: dict[str, Any]) -> CompletionResult | None:
        """Parse an OpenAI Chat Completions response. Returns None on any issue."""
        try:
            choice = data["choices"][0]
            finish_reason: str = choice.get("finish_reason", "")
            content: str | None = choice["message"].get("content")
        except (KeyError, IndexError, TypeError):
            return None

        # Per spec §2: null content (content_filter, function_call) → None.
        # Empty string is not a valid OpenAI state → also None.
        if not content:
            return None

        usage_raw = data.get("usage") or {}
        usage = UsageResult(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
            completion_tokens=int(usage_raw.get("completion_tokens", 0)),
            total_tokens=int(usage_raw.get("total_tokens", 0)),
        )
        return CompletionResult(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
        )

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "OpenAIProvider":
        url = raw.get("url")
        model = raw.get("model")
        if not url:
            raise ValueError("OpenAIProvider config missing 'url'")
        if not model:
            raise ValueError("OpenAIProvider config missing 'model'")
        return cls(
            url=url,
            model=model,
            timeout=float(raw.get("timeout", 60.0)),
            max_retries=int(raw.get("max_retries", 3)),
            retry_delay=float(raw.get("retry_delay", 1.0)),
        )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "type": "openai",
            "url": "https://api.openai.com",
            "model": "gpt-4o",
            "timeout": 60.0,
        }

    def __repr__(self) -> str:
        return f"OpenAIProvider(url={self._url!r}, model={self._model!r})"
