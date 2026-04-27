"""
InputGuardChecker — semantic safety evaluation of the working text via an LLM.
OutputGuardChecker — semantic safety evaluation of the LLM response via an LLM.

Both generate 64-character hex delimiters from a cryptographically random source
per call (256 bits of entropy). Delimiters are never logged.

Guard response format (JSON only):
  {"safe": true}
  {"safe": false, "reason": "one sentence explanation"}

Only the boolean literal true is accepted as safe (§19). String "true",
integer 1, null, absent — all produce safe=False.
"""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path

from carapex.core.types import SafetyResult
from carapex.llm.base import LLMProvider
from carapex.safety.base import SafetyChecker

log = logging.getLogger(__name__)

# Path to built-in prompts alongside this module
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(path: str | None, default_filename: str) -> str:
    """Load a system prompt from a file path or the built-in default."""
    if path is not None:
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Guard system prompt not found: {path!r}")
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Guard system prompt file is empty: {path!r}")
        return text

    builtin = _PROMPTS_DIR / default_filename
    return builtin.read_text(encoding="utf-8").strip()


def _generate_delimiter() -> str:
    """Generate a 64-character hex delimiter (256 bits of entropy)."""
    return secrets.token_hex(32)  # 32 bytes → 64 hex chars


def _parse_guard_response(content: str) -> SafetyResult:
    """Parse the guard's JSON response into a SafetyResult.

    Only boolean literal true is accepted as safe.
    Any parse failure, missing key, or non-bool value → safe=False.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return SafetyResult(safe=False, failure_mode="guard_evaluation_corrupt")

    if not isinstance(parsed, dict):
        return SafetyResult(safe=False, failure_mode="guard_evaluation_corrupt")

    safe_value = parsed.get("safe")

    # Strict boolean literal check (§19)
    if safe_value is True:
        return SafetyResult(safe=True)

    if "safe" not in parsed:
        return SafetyResult(safe=False, failure_mode="guard_evaluation_corrupt")

    # safe key present but value is not boolean True or boolean False —
    # type coercion attempt ("true" string, 1 integer, null, etc.) → corrupt
    if safe_value is not False:
        return SafetyResult(safe=False, failure_mode="guard_evaluation_corrupt")

    # safe=False (boolean literal) — extract reason if present
    reason: str | None = None
    if isinstance(parsed.get("reason"), str):
        reason = parsed["reason"]

    return SafetyResult(safe=False, failure_mode="safety_violation", reason=reason)


class InputGuardChecker(SafetyChecker):
    """Semantic input safety evaluator. Stateful — holds a reference to the guard LLM."""

    name = "input_guard"

    def __init__(
        self,
        llm: LLMProvider,
        temperature: float = 0.1,
        system_prompt_path: str | None = None,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._system_prompt = _load_prompt(system_prompt_path, "input_guard.txt")

    def inspect(self, text: str) -> SafetyResult:
        if text is None:
            raise ValueError("InputGuardChecker.inspect() received None")
        return self._call_guard(text, api_key="")

    def inspect_with_key(self, text: str, api_key: str) -> SafetyResult:
        """Variant used by the orchestrator to pass the caller's api_key."""
        if text is None:
            raise ValueError("InputGuardChecker.inspect_with_key() received None")
        return self._call_guard(text, api_key=api_key)

    def _call_guard(self, text: str, api_key: str) -> SafetyResult:
        delimiter = _generate_delimiter()
        user_content = (
            f"Evaluate the text between the delimiters.\n"
            f"START_DELIMITER_{delimiter}\n"
            f"{text}\n"
            f"END_DELIMITER_{delimiter}"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = self._llm.complete_with_temperature(
            messages, self._temperature, api_key=api_key
        )
        if result is None:
            return SafetyResult(safe=False, failure_mode="guard_unavailable")
        return _parse_guard_response(result.content)

    def close(self) -> None:
        # Do NOT close self._llm — the composition root owns it
        pass

    def __repr__(self) -> str:
        return f"InputGuardChecker(llm={self._llm!r}, temperature={self._temperature!r})"


class OutputGuardChecker(SafetyChecker):
    """Semantic output safety evaluator. Stateful — holds a reference to the output guard LLM."""

    name = "output_guard"

    def __init__(
        self,
        llm: LLMProvider,
        temperature: float = 0.1,
        system_prompt_path: str | None = None,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._system_prompt = _load_prompt(system_prompt_path, "output_guard.txt")

    def inspect(self, text: str) -> SafetyResult:
        if text is None:
            raise ValueError("OutputGuardChecker.inspect() received None")
        return self._call_guard(text, api_key="")

    def inspect_with_key(self, text: str, api_key: str) -> SafetyResult:
        """Variant used by the orchestrator to pass the caller's api_key."""
        if text is None:
            raise ValueError("OutputGuardChecker.inspect_with_key() received None")
        return self._call_guard(text, api_key=api_key)

    def _call_guard(self, text: str, api_key: str) -> SafetyResult:
        delimiter = _generate_delimiter()
        user_content = (
            f"Evaluate the response between the delimiters.\n"
            f"START_DELIMITER_{delimiter}\n"
            f"{text}\n"
            f"END_DELIMITER_{delimiter}"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = self._llm.complete_with_temperature(
            messages, self._temperature, api_key=api_key
        )
        if result is None:
            return SafetyResult(safe=False, failure_mode="guard_unavailable")
        return _parse_guard_response(result.content)

    def close(self) -> None:
        # Do NOT close self._llm — the composition root owns it
        pass

    def __repr__(self) -> str:
        return f"OutputGuardChecker(llm={self._llm!r}, temperature={self._temperature!r})"
