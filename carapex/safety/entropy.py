"""
EntropyChecker — rejects input whose Shannon entropy exceeds the configured threshold.

Default threshold: 5.8 bits/character (see §15 for rationale).
Strings below entropy_min_length (default 50 chars) skip the check unconditionally.
entropy_threshold=None disables the check entirely.
"""

from __future__ import annotations

import math

from carapex.core.types import SafetyResult
from carapex.safety.base import SafetyChecker


class EntropyChecker(SafetyChecker):
    """Stateless entropy-based input checker."""

    name = "entropy"

    def __init__(
        self,
        threshold: float | None = 5.8,
        min_length: int = 50,
    ) -> None:
        self._threshold = threshold
        self._min_length = min_length

    def inspect(self, text: str) -> SafetyResult:
        if text is None:
            raise ValueError("EntropyChecker.inspect() received None")

        # Disabled or too short → pass unconditionally
        if self._threshold is None:
            return SafetyResult(safe=True)
        if len(text) < self._min_length:
            return SafetyResult(safe=True)

        entropy = _shannon_entropy(text)

        if entropy > self._threshold:
            return SafetyResult(
                safe=False,
                failure_mode="entropy_exceeded",
                reason=(
                    f"Input entropy {entropy:.2f} bits/char exceeds "
                    f"threshold {self._threshold:.2f}"
                ),
            )
        return SafetyResult(safe=True)

    def close(self) -> None:
        pass  # stateless — nothing to release

    def __repr__(self) -> str:
        return (
            f"EntropyChecker(threshold={self._threshold!r}, "
            f"min_length={self._min_length!r})"
        )


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy in bits per character."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())
