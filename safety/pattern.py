"""
safety/pattern.py
-----------------
Deterministic regex-based safety checker.
Runs on normalised text — encoding attacks already exposed by the normaliser.
Fast, no LLM needed. First layer in composite checker.

Governing principle:
    A pattern belongs here ONLY if there is no legitimate context in which
    it could appear in a user prompt. If any reasonable legitimate use exists,
    it belongs in the guard — not here.

    This set is intentionally minimal. Coverage is the guard's job.
    Certainty is this checker's job.

Built-in patterns fall into two categories:

    Prompt format injection markers — structural tokens from LLM prompt
    formats that have no place in natural user input:
        <s>          — Llama 2 BOS / start-of-sequence token
        <system>     — system tag variant used in some prompt formats
        [INST]       — instruction wrapper token
        [SYSTEM]     — system wrapper token
        ### Instruction
        ### System

    Explicit instruction-override phrases — combinations specific enough
    that no legitimate interpretation survives:
        ignore ... (instructions|directives|commands)
        (disregard|forget|bypass) ... (instructions|training|rules|guidelines|constraints)

Everything borderline — role-play, persona assignment, jailbreak terminology,
developer/unrestricted mode, urgency and coercion — is handled by the guard,
which has the semantic reasoning to evaluate intent and context.

Custom patterns:
    injection_patterns in config replaces DEFAULT_PATTERNS entirely.
    injection_patterns: null or [] — use defaults (security-first).
    Invalid patterns raise ConfigurationError at construction — they are
    not skipped. A pattern that fails to compile silently weakens the
    deterministic gate. This must fail at startup, not at runtime.
"""

import logging
import re
from typing import List, Optional

from .base import SafetyChecker, SafetyResult
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)

DEFAULT_PATTERNS = [
    # Prompt format injection markers — no legitimate use in user input.
    # <s> is the Llama 2 BOS token. <system> is used in other prompt formats.
    # Both are structural tokens with zero legitimate place in natural user input.
    r"<\s*s\s*>",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"\[SYSTEM\]",
    r"###\s*instruction",
    r"###\s*system",

    # Explicit instruction-override phrases — combination is unambiguous.
    r"ignore\s+.{0,30}(instructions|directives|commands)",
    r"(disregard|forget|bypass)\s+.{0,30}(instructions|training|rules|guidelines|constraints)",
]


class PatternSafetyChecker(SafetyChecker):
    """
    Scans normalised text against compiled regex patterns.

    Uses DEFAULT_PATTERNS unless custom patterns are provided in config.
    Custom patterns replace the defaults entirely — they do not extend them.

    Invalid patterns raise ConfigurationError at construction.
    A broken pattern silently weakens the deterministic safety gate — this
    must fail at startup, not at runtime. There is no silent skipping.
    """

    name = "pattern"

    def __init__(self, patterns: Optional[List[str]] = None):
        # None means "use defaults" — security-first.
        # Empty list is an explicit "no patterns" — respect the caller's intent.
        # Coercion of [] → None for the config path happens in config.from_dict().
        # See module docstring for rationale.
        raw_patterns = patterns if patterns is not None else DEFAULT_PATTERNS

        self._patterns: List[re.Pattern] = []

        for raw in raw_patterns:
            try:
                self._patterns.append(re.compile(raw, re.IGNORECASE | re.DOTALL))
            except re.error as e:
                # Invalid pattern is a configuration defect in a security-critical
                # component. Silent skipping would weaken the deterministic gate
                # without any signal at runtime. Fail at construction.
                raise ConfigurationError(
                    f"Invalid injection pattern {raw!r}: {e}. "
                    f"Fix or remove this pattern — invalid patterns are not skipped "
                    f"in a security boundary."
                ) from e

    def check(self, text: str) -> SafetyResult:
        for pattern in self._patterns:
            m = pattern.search(text)
            if m:
                return SafetyResult(
                    safe   = False,
                    reason = f"Injection pattern detected: '{m.group(0)}'",
                )
        return SafetyResult(safe=True)

    def __repr__(self) -> str:
        return f"PatternSafetyChecker(patterns={len(self._patterns)})"
