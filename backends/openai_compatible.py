"""
backends/openai_compatible.py
------------------------------
LLM backend for any OpenAI-compatible HTTP endpoint.
Compatible with llama-server, Ollama, LM Studio, OpenAI API.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from .base import LLMBackend, BackendConfig

logger = logging.getLogger(__name__)


@dataclass
class OpenAICompatibleConfig(BackendConfig):
    """
    Config for OpenAI-compatible backends.
    Extends shared base with model_name field.
    """
    model_name: Optional[str] = None


class OpenAICompatibleBackend(LLMBackend):
    """Talks to any /v1/chat/completions endpoint."""

    name = "openai_compatible"

    @classmethod
    def from_config(cls, raw: dict) -> "OpenAICompatibleBackend":
        import dataclasses
        known    = {f.name for f in dataclasses.fields(OpenAICompatibleConfig)}
        # "type" is a routing key consumed by provide_backend() — not a config field.
        filtered = {k: v for k, v in raw.items() if k in known and v is not None}
        return cls(OpenAICompatibleConfig(**filtered))

    @classmethod
    def default_config(cls) -> dict:
        base = super().default_config()
        base["model_name"] = None
        return base

    def __init__(self, cfg: OpenAICompatibleConfig):
        try:
            import requests as _requests
            self._requests = _requests
        except ImportError:
            raise ImportError(
                "requests library required.\nRun: pip install requests"
            )

        self._base_url   = cfg.base_url.rstrip("/")
        self._api_key    = cfg.api_key
        self._model      = cfg.model_name
        self._temp       = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._timeout    = cfg.request_timeout_s
        self._endpoint   = f"{self._base_url}/v1/chat/completions"
        self._headers    = {"Content-Type": "application/json"}
        self._last_usage = {}

        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"

    def chat(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   Optional[float] = None,
    ) -> Optional[str]:
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens" : self._max_tokens,
            "stream"     : False,
        }
        if self._model:
            body["model"] = self._model

        resolved_temp = temperature if temperature is not None else self._temp
        if resolved_temp is not None:
            body["temperature"] = resolved_temp

        try:
            resp = self._requests.post(
                self._endpoint,
                headers = self._headers,
                json    = body,
                timeout = self._timeout,
            )
            resp.raise_for_status()
            data             = resp.json()
            content          = data["choices"][0]["message"]["content"].strip()
            self._last_usage = data.get("usage", {})
            return content

        except self._requests.exceptions.ConnectionError:
            logger.warning("Cannot reach LLM at %s", self._base_url)
            return None
        except self._requests.exceptions.Timeout:
            logger.warning("LLM request timed out after %ds", self._timeout)
            return None
        except self._requests.exceptions.HTTPError as e:
            logger.error("LLM HTTP error: %s", e)
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Unexpected response format: %s", e)
            return None

    def health_check(self) -> bool:
        try:
            resp = self._requests.get(
                f"{self._base_url}/health", timeout=5
            )
            if resp.status_code == 200:
                try:
                    status = resp.json().get("status", "ok")
                except (json.JSONDecodeError, ValueError):
                    # Server returned 200 with a non-JSON body (plain "OK",
                    # empty body, HTML). Not llama-server format — treat as
                    # healthy. The backend is reachable; format mismatch is
                    # not a reason to fail startup.
                    return True
                if status == "loading model":
                    logger.warning("llama-server still loading model")
                    return False
                return True
            return False
        except (
            self._requests.exceptions.ConnectionError,
            self._requests.exceptions.Timeout,
        ):
            return False

    def last_usage(self) -> dict:
        return self._last_usage

    def __repr__(self) -> str:
        return f"OpenAICompatibleBackend(base_url={self._base_url!r})"
