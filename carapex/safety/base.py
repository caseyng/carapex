"""
SafetyChecker — the abstraction for all pipeline checkers.

Both input and output checkers implement this contract.

TextTransformingChecker is the optional extension for checkers that may
replace the working text (currently: Translator only).

To add a new checker:
  1. Create a file in carapex/safety/.
  2. Subclass SafetyChecker (or TextTransformingChecker if needed).
  3. Set a unique `name` class attribute.
  4. Implement inspect() and close().
  5. Update the composition root to insert the checker at the correct
     pipeline position. Position is a security decision — it cannot be
     determined by autodiscovery alone.

inspect() MUST NOT raise — return SafetyResult(safe=False, ...) instead.
close() MUST be idempotent and never raise.
close() MUST NOT call close() on injected LLMs — the composition root owns them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from carapex.core.types import SafetyResult


class SafetyChecker(ABC):
    """Abstract base for all pipeline safety checkers."""

    name: str  # unique class attribute

    @abstractmethod
    def inspect(self, text: str) -> SafetyResult:
        """Evaluate text and return a SafetyResult.

        Never raises on content or infrastructure failures — return
        SafetyResult(safe=False, failure_mode=...) instead.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any owned resources. Idempotent. Never raises.
        MUST NOT close injected dependencies."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class TextTransformingChecker(SafetyChecker, ABC):
    """SafetyChecker that may replace the working text after a safe=True result.

    The CheckerCoordinator identifies this behaviour by the presence of
    get_output_text() on the instance — no isinstance() check needed.

    Protocol:
      1. Coordinator calls set_prior_result() with the preceding checker's result.
      2. Coordinator calls inspect().
      3. If safe=True: coordinator calls get_output_text() and switches working text.
    """

    @abstractmethod
    def set_prior_result(self, result: SafetyResult) -> None:
        """Receive the preceding checker's SafetyResult before inspect() is called.

        Raises ValueError if called with None.
        """

    @abstractmethod
    def get_output_text(self) -> str:
        """Return the transformed working text after a safe=True inspect() result.

        Only valid to call after inspect() returned safe=True.
        """
