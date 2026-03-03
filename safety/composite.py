"""
safety/composite.py
-------------------
Composite safety checker — runs a list of checkers in order, short-circuits
on the first unsafe result.

Text transformation:
    Checkers that subclass TextTransformingChecker may replace the working
    text after a safe=True result. The composite calls get_output_text()
    and switches the working text for all subsequent checkers.

    This is how TranslationLayer passes English text to GuardChecker
    without the composite knowing about either concrete class.

Prior result passing:
    Before each TextTransformingChecker.check() call, the composite calls
    set_prior_result() with the preceding checker's SafetyResult. This lets
    TranslationLayer read ScriptResult.translation_needed without performing
    redundant language detection.

Exception handling:
    Checker exceptions propagate as PipelineInternalError.
    A crashing checker is a bug — not an operational failure.
    Bugs must not be silently converted to safe=True.
"""

import logging
from typing import List

from .base import SafetyChecker, TextTransformingChecker, SafetyResult
from ..exceptions import PipelineInternalError

logger = logging.getLogger(__name__)


class CompositeSafetyChecker(SafetyChecker):
    """
    Runs checkers in sequence. Short-circuits on first safe=False.
    Passes prior results to TextTransformingCheckers via set_prior_result().
    Switches working text when a TextTransformingChecker transforms it.
    Propagates checker exceptions as PipelineInternalError.
    """

    name = "composite"

    def __init__(self, checkers: List[SafetyChecker]):
        if not checkers:
            from ..exceptions import ConfigurationError
            raise ConfigurationError(
                "CompositeSafetyChecker requires at least one checker."
            )
        self._checkers = checkers

    def check(self, text: str) -> SafetyResult:
        working_text = text
        prior_result: SafetyResult | None = None

        for checker in self._checkers:
            # Pass prior result to checkers that implement the protocol.
            # The composite knows TextTransformingChecker (abstract type),
            # not TranslationLayer (concrete class) — DIP preserved.
            if prior_result is not None and isinstance(checker, TextTransformingChecker):
                checker.set_prior_result(prior_result)

            try:
                result = checker.check(working_text)
            except PipelineInternalError:
                raise
            except Exception as e:
                raise PipelineInternalError(
                    component=checker.name,
                    original=e,
                ) from e

            if not result.safe:
                # Propagate failure_mode from the failing checker explicitly.
                # Fall back to "safety_violation" only when the checker did
                # not set one — every documented failure path should set it,
                # but defensive fallback prevents silent misclassification.
                return SafetyResult(
                    safe         = False,
                    reason       = result.reason,
                    failure_mode = result.failure_mode or "safety_violation",
                )

            prior_result = result

            # If this checker transforms text, switch working text for
            # all subsequent checkers.
            if isinstance(checker, TextTransformingChecker):
                transformed = checker.get_output_text()
                if transformed is None:
                    # safe=True but no output text — this is a checker
                    # implementation defect. Fail closed rather than silently
                    # evaluate the wrong (original) text through subsequent layers.
                    raise PipelineInternalError(
                        component=checker.name,
                        original=RuntimeError(
                            f"{checker.name}.get_output_text() returned None after "
                            f"safe=True. TextTransformingChecker must return output "
                            f"text on success."
                        ),
                    )
                if transformed:
                    working_text = transformed
                else:
                    # Empty string — degenerate but not silently wrong.
                    # Log and continue with original text so pipeline completes.
                    logger.warning(
                        "%s returned empty string from get_output_text() after "
                        "safe=True — using original working text for subsequent checkers",
                        checker.name,
                    )

        return SafetyResult(safe=True, failure_mode=None)

    def __repr__(self) -> str:
        names = [type(c).__name__ for c in self._checkers]
        return f"CompositeSafetyChecker(checkers={names})"
