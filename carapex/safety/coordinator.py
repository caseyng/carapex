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
        """Run all checkers in order. Stop on first failure.

        Returns:
          (SafetyResult, working_text) where working_text is the text
          after any TextTransformingChecker transformations.
          On failure, working_text is the text at the point of failure.

        Raises ValueError if text is None.
        """
        if text is None:
            raise ValueError("CheckerCoordinator.inspect() received None text")

        if not self._checkers:
            return SafetyResult(safe=True), text

        working = text

        for checker in self._checkers:
            # TextTransformingChecker protocol: call set_prior_result() if present
            # The preceding checker's result is passed here. For the first checker
            # there is no prior result — we only call it when the attribute exists
            # and we have a prior result available.
            # (The ScriptChecker → Translator handoff is handled below.)

            result = self._call_checker(checker, working)

            if not result.safe:
                return result, working

            # TextTransformingChecker: update working text if checker transformed it
            if hasattr(checker, "get_output_text"):
                working = checker.get_output_text()

        return SafetyResult(safe=True), working

    def inspect_with_prior_handoff(
        self, text: str
    ) -> tuple[SafetyResult, str]:
        """Same as inspect() but handles the set_prior_result() TextTransformingChecker
        protocol correctly: passes the immediately preceding checker's result to the
        next checker if it implements set_prior_result().

        This is the method used by the input pipeline where ScriptChecker must
        hand its result to Translator before Translator's inspect() is called.
        """
        if text is None:
            raise ValueError("CheckerCoordinator received None text")

        if not self._checkers:
            return SafetyResult(safe=True), text

        working = text
        prior_result: SafetyResult | None = None

        for checker in self._checkers:
            # Hand prior result to TextTransformingChecker before calling inspect()
            if prior_result is not None and hasattr(checker, "set_prior_result"):
                checker.set_prior_result(prior_result)  # type: ignore[attr-defined]

            result = self._call_checker(checker, working)
            prior_result = result

            if not result.safe:
                return result, working

            # Update working text if this checker transformed it
            if hasattr(checker, "get_output_text"):
                working = checker.get_output_text()  # type: ignore[attr-defined]

        return SafetyResult(safe=True), working

    @staticmethod
    def _call_checker(checker: SafetyChecker, text: str) -> SafetyResult:
        """Call checker.inspect() and handle unexpected raises.

        inspect() should never raise per contract. If it does, this is a
        component bug — re-raise so the orchestrator can wrap it in
        PipelineInternalError.
        """
        return checker.inspect(text)

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
