"""
LLMProvider — the abstraction for all LLM implementations.

Every LLM in the pipeline (main, input guard, output guard, translator)
implements this contract.

To add a new LLM backend:
  1. Create a file in carapex/llm/.
  2. Subclass LLMProvider and set a unique `name` class attribute.
  3. Decorate with @register_llm.
  4. Implement complete() and close(). Override complete_with_temperature()
     if the backend supports per-call temperature control (RECOMMENDED for
     any LLM used as a guard or translator).

No existing files need to be edited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from carapex.core.types import CompletionResult


class LLMProvider(ABC):
    """Abstract base for all LLM implementations.

    Contract:
      - complete() MUST return None on any infrastructure failure — never raise.
        Raising breaks the fail-closed contract (§16).
      - complete() MUST return None when content is null or empty (§2 CompletionResult).
      - close() MUST be idempotent and never raise.
      - name MUST be a unique class attribute (string).
    """

    name: str  # set as a class attribute on each concrete implementation

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        api_key: str,
    ) -> CompletionResult | None:
        """Send messages to the LLM and return a CompletionResult, or None on failure.

        api_key is always a string (never None, may be empty). The implementation
        decides whether to use the supplied key or its own configured credentials.
        Never raises on infrastructure failures — return None instead.
        """

    def complete_with_temperature(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        api_key: str,
    ) -> CompletionResult | None:
        """Variant called by the pipeline when a specific temperature must be applied.

        The base implementation ignores temperature and delegates to complete().
        Override this method if the backend supports per-call temperature control.
        A conformant override MUST apply the temperature argument and pass api_key through.
        """
        return self.complete(messages, api_key)

    @abstractmethod
    def close(self) -> None:
        """Release any resources held. Idempotent. Never raises."""

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "LLMProvider":
        """Construct an instance from a raw config dict.

        Override in subclasses to handle implementation-specific fields.
        The base implementation constructs using only base fields.
        """
        return cls()  # type: ignore[call-arg]

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Return the default configuration dict for this implementation.

        Used by CarapexConfig.write_default() to generate a complete config file.
        Override to add implementation-specific fields.
        """
        return {"type": cls.name}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
