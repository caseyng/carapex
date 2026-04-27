"""
CheckerCoordinator — executes an ordered sequence of SafetyCheckers.

Stops at the first failure. Manages TextTransformingChecker handoff —
updates working text when a checker signals a transformation.

Two instances are constructed by the composition root:
  InputCoordinator  — runs the input checker sequence
  OutputCoordinator — runs the output checker sequence

A TextTransformingChecker is identified by the presence of get_output_text()
on the instance — no isinstance() check. The coordinator does not know
which concrete type it holds.
"""

from __future__ import annotations

import logging

from carapex.core.types import SafetyResult
from carapex.safety.base import SafetyChecker

log = logging.getLogger(__name__)


class CheckerCoordinator:
    """Runs an ordered checker sequence, stopping on first failure.

    An empty checker sequence is valid — inspect() returns safe=True immediately
    without emitting any check record. This is the correct behaviour when both
    output checkers are disabled via config.
    """

    def __init__(self, checkers: list[SafetyChecker]) -> None:
        self._checkers = list(checkers)

    def inspect(self, text: str) -> tuple[SafetyResult, str]:
        """Run all checkers in order. Delegates to inspect_with_prior_handoff()."""
        return self.inspect_with_prior_handoff(text)

    def inspect_with_prior_handoff(
        self, text: str
    ) -> tuple[SafetyResult, str]:
        """Run all checkers in order, stopping on first failure.

        Passes each checker's result to the next checker via set_prior_result()
        if that checker implements it (the ScriptChecker → Translator handoff).

        Returns (SafetyResult, working_text) where working_text reflects any
        TextTransformingChecker substitutions.
        """
        if text is None:
            raise ValueError("CheckerCoordinator received None text")

        if not self._checkers:
            return SafetyResult(safe=True), text

        working = text
        prior_result: SafetyResult | None = None

        for checker in self._checkers:
            if prior_result is not None and hasattr(checker, "set_prior_result"):
                checker.set_prior_result(prior_result)  # type: ignore[attr-defined]

            result = checker.inspect(working)
            prior_result = result

            if not result.safe:
                return result, working

            if hasattr(checker, "get_output_text"):
                working = checker.get_output_text()  # type: ignore[attr-defined]

        return SafetyResult(safe=True), working

    def close(self) -> None:
        """Close all child checkers."""
        for checker in self._checkers:
            try:
                checker.close()
            except Exception as e:  # noqa: BLE001
                log.warning("Checker %r raised during close(): %s", checker.name, e)

    def __repr__(self) -> str:
        names = [c.name for c in self._checkers]
        return f"CheckerCoordinator(checkers={names!r})"
