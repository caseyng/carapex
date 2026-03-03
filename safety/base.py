"""
safety/base.py
--------------
Abstract base, result type, and config for safety checkers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKER PIPELINE ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input pipeline (cheapest → most expensive):
    1. PatternChecker     — structural token detection, deterministic
    2. EntropyChecker     — statistical anomaly, deterministic
    3. ScriptChecker      — language/script detection, informational
    4. TranslationLayer   — non-English → English, LLM call at temp 0.0
    5. GuardChecker       — semantic evaluation on English, LLM at temp 0.1

Output pipeline:
    1. OutputPatternChecker  — pattern-based, always runs, no LLM
    2. OutputGuardChecker    — semantic evaluation, on by default, configurable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SafetyResult.failure_mode values:

    None                        — clean pass or genuine content violation
    "guard_unavailable"         — guard backend returned no response
    "guard_evaluation_corrupt"  — guard responded but output unparseable
    "translation_failed"        — language detection or translation call failed
    "entropy_exceeded"          — input entropy above threshold
    "normalisation_unstable"    — input did not stabilise (set by processor)
    "backend_unavailable"       — main LLM returned no response (set by processor)

failure_mode is None for genuine safety violations (pattern match, guard
refusal, entropy, output pattern) — reason carries the human-readable
explanation. failure_mode is set only for infrastructure and evaluation
integrity failures that are distinct from content-based refusals.

Note: "normalisation_unstable" and "backend_unavailable" are set directly
by processor.py, not by any SafetyChecker. They appear on Response objects
but never on SafetyResult objects. They are listed here as the complete
failure_mode vocabulary for callers branching on Response.failure_mode.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ADD A NEW SAFETY CHECKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STANDARD CHECKER (evaluates text, does not transform it):

1. Create a file in safety/  e.g. safety/my_checker.py
2. Subclass SafetyChecker
3. Set class attribute `name`
4. Implement check()
5. Wire into providers.py provide_input_checker() or provide_output_checker()

Safety checkers are composed in providers.py — not autodiscovered into the
pipeline automatically. This is intentional and different from backends,
audit backends, and decoders, which are autodiscovered.

Why: checker order is a security decision. An entropy checker placed after
a translation layer evaluates translated text, not the original. A guard
placed before a translation layer evaluates non-English input directly.
Autodiscovery cannot enforce ordering — a dropped file could silently land
in the wrong position in the pipeline. Explicit wiring in providers.py
makes the pipeline structure visible and auditable in one place.

Editing providers.py is required. This is the deliberate cost of explicit
control over a security-critical pipeline.

    class MyChecker(SafetyChecker):
        name = "my_checker"

        def check(self, text: str) -> SafetyResult:
            if "bad word" in text:
                return SafetyResult(safe=False, reason="Bad word found")
            return SafetyResult(safe=True)

TEXT-TRANSFORMING CHECKER (replaces working text on safe=True):

If your checker produces a transformed version of the text that subsequent
checkers should evaluate (e.g. a second translation or normalisation step),
subclass TextTransformingChecker instead.

The composite will:
    1. Call set_prior_result() with the preceding checker's SafetyResult
       before check() — use this to read metadata without re-computing it.
    2. Call get_output_text() after safe=True and switch working text
       for all subsequent checkers.

    class MyTransformer(TextTransformingChecker):
        name = "my_transformer"

        def __init__(self):
            self._output: Optional[str] = None

        def set_prior_result(self, result: SafetyResult) -> None:
            pass  # read prior result metadata here if needed

        def check(self, text: str) -> SafetyResult:
            self._output = text.upper()  # example transformation
            return SafetyResult(safe=True)

        def get_output_text(self) -> Optional[str]:
            return self._output

RULES (both types):
    - check() must always be implemented
    - Return SafetyResult for all operational outcomes
    - Allow internal bugs to propagate — CompositeSafetyChecker wraps them
      as PipelineInternalError
    - reason is returned directly to caller as refusal text — be specific
    - Set failure_mode only for infrastructure/evaluation failures
    - TextTransformingChecker instances are not reentrant — do not share
      instances across concurrent requests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SafetyResult:
    """
    Result of a safety check.

    safe         : True if text is safe to process, False otherwise.
    reason       : human-readable explanation when safe=False.
    failure_mode : machine-readable failure category.
                   None for genuine content violations and clean passes.
                   Set for infrastructure and evaluation integrity failures.
    """
    safe:         bool
    reason:       Optional[str] = None
    failure_mode: Optional[str] = None

    def __bool__(self) -> bool:
        return self.safe


@dataclass
class SafetyConfig:
    """
    Config for safety checkers.

    injection_patterns:
        None or [] — use built-in DEFAULT_PATTERNS (security-first default).
        [...]      — replace defaults entirely. Invalid patterns raise
                     ConfigurationError at construction — never skipped.

    entropy_threshold:
        Maximum allowed Shannon entropy (bits/char) for normalised input.
        Natural language: ~3.5–4.5. Dense technical: ~4.5–5.5. Encoded: >6.0.
        Default 5.8 — catches undecoded ciphers and binary-as-text injections
        without blocking legitimate technical content.
        None    — disable entropy gating entirely.
        0.0     — block all inputs above 0.0 bits/char (effectively everything
                  non-trivial). Not a disable signal — use None to disable.

    entropy_min_length:
        Minimum input length (chars) before entropy check applies.
        Entropy on very short strings is statistically meaningless.
        Default 50.

    output_guard_enabled:
        Whether OutputGuardChecker runs semantic evaluation on LLM responses.
        True  — full symmetric guarantee (default, recommended).
        False — pattern-only output checking (lower cost, weaker guarantee).
        Disabling this is a conscious reduction in protection — document why.
    """
    injection_patterns:   Optional[List[str]] = None
    entropy_threshold:    Optional[float]     = 5.8
    entropy_min_length:   int                 = 50
    output_guard_enabled: bool                = True

    def __post_init__(self):
        # Validation only — no coercion here.
        # injection_patterns normalisation ([] → None) happens in
        # config.from_dict() where config coercion belongs.
        if self.entropy_threshold is not None and self.entropy_threshold < 0:
            raise ValueError(
                f"entropy_threshold must be >= 0 or None, got {self.entropy_threshold}"
            )


class SafetyChecker(ABC):
    """
    Abstract base for safety checkers.
    See module docstring for full extension guide.
    """

    name: str  # identifier — set on every subclass

    @abstractmethod
    def check(self, text: str) -> SafetyResult:
        """
        Evaluate text for safety concerns.

        Return SafetyResult for all operational outcomes.
        Allow internal bugs to propagate — CompositeSafetyChecker converts
        them to PipelineInternalError.

        Args:
            text : normalised input prompt or LLM output

        Returns:
            SafetyResult(safe=True)  — text is safe to process
            SafetyResult(safe=False, reason="...", failure_mode=None)
                — genuine content violation
            SafetyResult(safe=False, reason="...", failure_mode="...")
                — infrastructure or evaluation integrity failure
        """
        ...

    def close(self) -> None:
        """Override to release resources held by this checker at shutdown."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TextTransformingChecker(SafetyChecker, ABC):
    """
    Subclass for checkers that transform the working text as a side effect
    of a successful check — translation being the canonical example.

    The composite checks isinstance(checker, TextTransformingChecker) and
    calls get_output_text() after a safe=True result to retrieve the
    transformed text, which then replaces the working text for subsequent
    checkers.

    Subclasses may also implement set_prior_result() to receive the
    SafetyResult from the immediately preceding checker — used by
    TranslationLayer to read ScriptResult.translation_needed without
    performing redundant language detection.

    Protocol:
        - check() runs as normal
        - get_output_text() returns transformed text after a safe=True result
        - set_prior_result() is called by the composite before check()
          if the prior checker produced a result; implementation is optional
    """

    def set_prior_result(self, result: SafetyResult) -> None:
        """
        Receive the SafetyResult from the preceding checker.
        Default: no-op. Override to consume prior checker metadata.
        """

    @abstractmethod
    def get_output_text(self) -> Optional[str]:
        """
        Return the transformed text from the last check() call.
        None if check() has not been called or transformation failed.
        """
        ...
