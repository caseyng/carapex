"""
Core data types for carapex.

All result and data contract types live here. No logic — only structure.
These types flow through the pipeline; nothing in this module imports
from other carapex modules.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UsageResult:
    """Token usage from one LLM call. All fields are zero when the provider
    returned no usage data — absent usage is not a failure condition."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class CompletionResult:
    """The successful result of one LLMProvider.complete() call.

    content is always a non-empty string — the invariant is enforced by the
    LLM implementation before constructing this object.
    usage is always present; zero-filled when the provider omitted it.
    """
    content: str
    finish_reason: str
    usage: UsageResult


# ---------------------------------------------------------------------------
# Safety layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyResult:
    """Return value of SafetyChecker.inspect().

    safe=True  -> checker found no issue.
    safe=False -> checker blocked; failure_mode identifies why.
    reason     -> human-readable explanation when available; null otherwise.
    """
    safe: bool
    failure_mode: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.safe and self.failure_mode is not None:
            raise ValueError("safe=True is incompatible with a non-null failure_mode")
        if not self.safe and self.failure_mode is None:
            raise ValueError("safe=False requires a non-null failure_mode")


@dataclass(frozen=True)
class ScriptResult:
    """Return value of ScriptChecker.inspect().

    Always safe=True — ScriptChecker is a detector, not a gate.
    detected_language: ISO code or None when detection failed.
    translation_needed: True when the guard must receive translated text.

    Not a SafetyResult subclass — safe is always True and is an invariant,
    not a constructor parameter. Translator reads translation_needed directly.
    """
    detected_language: str | None = None
    translation_needed: bool = True

    @property
    def safe(self) -> bool:
        return True

    @property
    def failure_mode(self) -> None:
        return None


@dataclass(frozen=True)
class NormaliserResult:
    """Return value of Normaliser.normalise().

    text:   the decoded output string.
    stable: True iff the output did not change on the final pass.
            False on max_passes exhaustion or cycle detection.
    """
    text: str
    stable: bool


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationResult:
    """The structured return value of every evaluate() call.

    Invariants (enforced at construction):
      safe=True  -> content is a non-null string, failure_mode is None.
      safe=False -> content is None, failure_mode is a non-null string.
    """
    safe: bool
    content: str | None = None
    failure_mode: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.safe:
            if self.content is None:
                raise ValueError("safe=True requires non-null content")
            if self.failure_mode is not None:
                raise ValueError("safe=True is incompatible with a non-null failure_mode")
        else:
            if self.content is not None:
                raise ValueError("safe=False requires null content")
            if self.failure_mode is None:
                raise ValueError("safe=False requires a non-null failure_mode")
