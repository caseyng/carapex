"""
Translator — translates non-English working text to English before input guard evaluation.

A TextTransformingChecker — replaces the working text on successful translation.
Reads translation_needed from the preceding ScriptChecker result via set_prior_result().

The original prompt delivered to the main LLM is NEVER the translated text —
translation is an evaluation tool only.

Failure conditions that produce translation_failed (§5):
  - Translator LLM returns None (unavailable)
  - Translator LLM returns text byte-for-byte identical to input (echo response)
  - Translator LLM returns empty string when translation was needed
"""

from __future__ import annotations

import logging

from carapex.core.types import SafetyResult, ScriptResult
from carapex.llm.base import LLMProvider
from carapex.safety.base import TextTransformingChecker

log = logging.getLogger(__name__)

_TRANSLATION_SYSTEM_PROMPT = (
    "Translate the following text to English. "
    "Return only the translated text — no explanation, no preamble. "
    "If the text is already in English, return it unchanged."
)


class Translator(TextTransformingChecker):
    """Translates non-English input to English before input guard evaluation."""

    name = "translator"

    def __init__(self, llm: LLMProvider, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature
        self._prior_result: ScriptResult | None = None
        self._output_text: str | None = None

    # ------------------------------------------------------------------
    # TextTransformingChecker protocol
    # ------------------------------------------------------------------

    def set_prior_result(self, result: SafetyResult) -> None:
        if result is None:
            raise ValueError("Translator.set_prior_result() received None")
        self._prior_result = result  # type: ignore[assignment]

    def get_output_text(self) -> str:
        if self._output_text is None:
            raise RuntimeError(
                "get_output_text() called before a successful inspect()"
            )
        return self._output_text

    # ------------------------------------------------------------------
    # SafetyChecker contract
    # ------------------------------------------------------------------

    def inspect(self, text: str) -> SafetyResult:
        if text is None:
            raise ValueError("Translator.inspect() received None")
        if self._prior_result is None:
            raise ValueError(
                "Translator.inspect() called without set_prior_result() — "
                "ScriptChecker must precede Translator in the pipeline."
            )

        prior: ScriptResult = self._prior_result  # type: ignore[assignment]
        self._output_text = None  # reset

        if not prior.translation_needed:
            # No translation needed — pass text through unchanged
            self._output_text = text
            return SafetyResult(safe=True)

        # Attempt translation
        messages = [
            {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        result = self._llm.complete_with_temperature(
            messages, self._temperature, api_key=""
        )

        if result is None:
            log.debug("Translator: LLM returned None — translation_failed")
            return SafetyResult(safe=False, failure_mode="translation_failed")

        translated = result.content

        # Echo response check (§5)
        if translated == text:
            log.debug("Translator: echo response detected — translation_failed")
            return SafetyResult(safe=False, failure_mode="translation_failed")

        # Empty translation check (§5)
        if not translated.strip():
            log.debug("Translator: empty translation — translation_failed")
            return SafetyResult(safe=False, failure_mode="translation_failed")

        self._output_text = translated
        return SafetyResult(safe=True)

    def close(self) -> None:
        # Do NOT close self._llm — the composition root owns it
        pass

    def __repr__(self) -> str:
        return f"Translator(llm={self._llm!r}, temperature={self._temperature!r})"
