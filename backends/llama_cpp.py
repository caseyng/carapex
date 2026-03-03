"""
backends/llama_cpp.py
---------------------
Direct llama-cpp-python backend. Loads model in-process.

Requires: pip install llama-cpp-python
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional

from .base import LLMBackend, BackendConfig

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppConfig(BackendConfig):
    """
    Config for llama_cpp in-process backend.
    Extends shared base with llama.cpp-specific fields.
    """
    model_path:   Optional[str] = None
    n_threads:    int           = 4
    n_gpu_layers: int           = 0


class LlamaCppBackend(LLMBackend):
    """Direct llama-cpp-python in-process backend."""

    name = "llama_cpp"

    @classmethod
    def from_config(cls, raw: dict) -> "LlamaCppBackend":
        known    = {f.name for f in dataclasses.fields(LlamaCppConfig)}
        # "type" is a routing key consumed by provide_backend() — not a config field.
        filtered = {k: v for k, v in raw.items() if k in known and v is not None}
        return cls(LlamaCppConfig(**filtered))

    @classmethod
    def default_config(cls) -> dict:
        base = super().default_config()
        base["model_path"]   = None
        base["n_threads"]    = 4
        base["n_gpu_layers"] = 0
        return base

    def __init__(self, cfg: LlamaCppConfig):
        if not cfg.model_path:
            raise ValueError(
                "llama_cpp backend requires model_path in config"
            )
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python required.\n"
                "Run: pip install llama-cpp-python"
            )

        self._cfg        = cfg
        self._last_usage = {}
        self._llama_cls  = Llama  # retained for type reference
        self._model      = Llama(
            model_path   = cfg.model_path,
            n_threads    = cfg.n_threads,
            n_gpu_layers = cfg.n_gpu_layers,
            verbose      = False,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   Optional[float] = None,
    ) -> Optional[str]:
        try:
            resolved_temp = temperature if temperature is not None else self._cfg.temperature
            kwargs = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": self._cfg.max_tokens,
            }
            if resolved_temp is not None:
                kwargs["temperature"] = resolved_temp

            result           = self._model.create_chat_completion(**kwargs)
            self._last_usage = result.get("usage", {})
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            logger.error("Unexpected response structure from llama_cpp: %s", e)
            return None
        except RuntimeError as e:
            logger.error("llama_cpp inference failed: %s", e)
            return None

    def last_usage(self) -> dict:
        return self._last_usage

    def close(self) -> None:
        """
        Release the llama_cpp model and underlying C++ context.

        llama-cpp-python does not expose an explicit free() or close() on
        the Llama class. Resource release relies on __del__, which calls the
        C destructor. We lower the reference count via del rather than calling
        __del__ directly — direct __del__ invocation runs the finaliser code
        but does not decrement the reference count, so the C context may not
        actually be released if other references exist.

        Under CPython, del + None triggers __del__ via reference counting.
        Under PyPy, it schedules finalisation — the best available option
        given the library's interface.
        """
        if self._model is not None:
            model = self._model
            self._model = None
            try:
                del model
            except Exception:
                # Finaliser raised — swallow and continue. The reference has
                # already been nulled; we cannot do more without a proper
                # close() method in the library.
                pass

    def health_check(self) -> bool:
        # Model loads at construction time — if __init__ succeeded, the model
        # is ready. No network round-trip needed; readiness is local state.
        return self._model is not None

    def __repr__(self) -> str:
        return f"LlamaCppBackend(model={self._cfg.model_path!r})"
