"""
PatternChecker — detects structural injection tokens via compiled regex patterns.
OutputPatternChecker — detects jailbreak success indicators in LLM responses.

Both are stateless. Pattern sets are compiled once at construction.

Default injection patterns (input) target common prompt injection structures:
  - Role/instruction delimiters used to escape prompt boundaries
  - System prompt override attempts
  - Common jailbreak scaffolding tokens

Default output patterns target:
  - Jailbreak success indicators ("DAN mode enabled", "As DAN", etc.)
  - System prompt leakage indicators
  - Downstream injection attempts in responses
"""

from __future__ import annotations

import re

from carapex.core.types import SafetyResult
from carapex.safety.base import SafetyChecker


# Default input injection patterns
# These target structural tokens — not semantic content. Semantic evaluation
# is the input guard's job. Pattern matching is cheap and deterministic.
DEFAULT_INJECTION_PATTERNS: list[str] = [
    # Role/boundary delimiters
    r"(?i)\[INST\]",
    r"(?i)\[/INST\]",
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
    r"(?i)<\|system\|>",
    r"(?i)<\|user\|>",
    r"(?i)<\|assistant\|>",
    r"(?i)<<SYS>>",
    r"(?i)<</SYS>>",
    r"(?i)\[SYSTEM\]",
    # System prompt override tokens
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)disregard\s+(all\s+)?previous\s+instructions",
    r"(?i)forget\s+(all\s+)?previous\s+instructions",
    r"(?i)override\s+(all\s+)?previous\s+instructions",
    r"(?i)you\s+are\s+now\s+in\s+(developer|jailbreak|DAN)\s+mode",
    r"(?i)new\s+system\s+prompt\s*:",
    # Common jailbreak scaffolding
    r"(?i)\bDAN\b.*\bmode\b",
    r"(?i)do\s+anything\s+now",
    r"(?i)jailbreak(?:ed)?\s+mode",
    r"(?i)unrestricted\s+mode",
    # Prompt injection via data framing
    r"(?i)###\s*instruction",
    r"(?i)###\s*system",
]

# Default output patterns — jailbreak success indicators and leakage
DEFAULT_OUTPUT_PATTERNS: list[str] = [
    # Jailbreak success indicators
    r"(?i)(DAN|Developer)\s+mode\s+(enabled|activated|on)",
    r"(?i)I\s+am\s+now\s+(DAN|unrestricted|jailbroken)",
    r"(?i)as\s+DAN[,\s]",
    r"(?i)jailbreak\s+successful",
    r"(?i)I\s+have\s+no\s+restrictions",
    r"(?i)I\s+can\s+now\s+do\s+anything",
    # System prompt leakage
    r"(?i)my\s+system\s+prompt\s+is",
    r"(?i)my\s+instructions\s+are",
    r"(?i)I\s+was\s+instructed\s+to",
    r"(?i)the\s+system\s+prompt\s+(says|states|instructs)",
    # Downstream injection attempts
    r"(?i)\[INST\]",
    r"(?i)<\|im_start\|>",
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    """Compile a list of pattern strings. Raises ValueError on bad regex."""
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern {p!r}: {e}") from e
    return compiled


class PatternChecker(SafetyChecker):
    """Input-side pattern checker. Stateless."""

    name = "pattern"

    def __init__(self, patterns: list[str] | None = None) -> None:
        raw = patterns if patterns is not None else DEFAULT_INJECTION_PATTERNS
        self._patterns = _compile_patterns(raw)

    def inspect(self, text: str) -> SafetyResult:
        if text is None:
            raise ValueError("PatternChecker.inspect() received None")
        for pattern in self._patterns:
            m = pattern.search(text)
            if m:
                return SafetyResult(
                    safe=False,
                    failure_mode="safety_violation",
                    reason=f"Matched injection pattern: {pattern.pattern!r}",
                )
        return SafetyResult(safe=True)

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"PatternChecker(patterns={len(self._patterns)})"


class OutputPatternChecker(SafetyChecker):
    """Output-side pattern checker. Stateless."""

    name = "output_pattern"

    def __init__(self, patterns: list[str] | None = None) -> None:
        raw = patterns if patterns is not None else DEFAULT_OUTPUT_PATTERNS
        self._patterns = _compile_patterns(raw)

    def inspect(self, text: str) -> SafetyResult:
        if text is None:
            raise ValueError("OutputPatternChecker.inspect() received None")
        for pattern in self._patterns:
            m = pattern.search(text)
            if m:
                return SafetyResult(
                    safe=False,
                    failure_mode="safety_violation",
                    reason=f"Matched output pattern: {pattern.pattern!r}",
                )
        return SafetyResult(safe=True)

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"OutputPatternChecker(patterns={len(self._patterns)})"
