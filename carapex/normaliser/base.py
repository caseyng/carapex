"""
Normaliser — decodes obfuscated or encoded input into canonical plain-text.

The Normaliser runs multiple passes over the input, applying all configured
Decoders in sequence each pass, until the output stabilises or max_passes
is exhausted.

Stability rule: the output of pass N is compared to the output of pass N-1.
  - Unchanged on final pass → stable=True.
  - Still changing on final pass → stable=False (normalisation_unstable).
  - Pass output matches any earlier pass → cycle detected → stable=False.

The LLM always receives the original prompt, not normalised text.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from carapex.core.types import NormaliserResult

log = logging.getLogger(__name__)


class Decoder(ABC):
    """Abstract base for all normaliser decoders.

    Contract:
      - decode() MUST be strictly reductive: it only moves text toward
        canonical plain form and cannot re-encode its output.
      - decode() MUST return the input unchanged when no transformation applies.
      - decode() MUST NOT raise on content — return input unchanged instead.
      - name MUST be a unique class attribute.
    """

    name: str

    @abstractmethod
    def decode(self, text: str) -> str:
        """Apply one decoding transformation. Returns decoded text or input unchanged."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class Normaliser:
    """Runs an ordered set of Decoders in passes until the output stabilises.

    Constructed with an immutable decoder list and max_passes limit.
    Stateless between calls — no mutable state after construction.
    """

    def __init__(self, decoders: list[Decoder], max_passes: int) -> None:
        if max_passes < 1:
            raise ValueError(f"max_passes must be >= 1, got {max_passes}")
        self._decoders = list(decoders)
        self._max_passes = max_passes

    def normalise(self, text: str) -> NormaliserResult:
        """Decode text to canonical form. Always returns a NormaliserResult.

        Never raises — all outcomes including cycle detection are in the result.
        """
        if text is None:
            raise ValueError("normalise() received None — input must be a string")

        if not self._decoders:
            return NormaliserResult(text=text, stable=True)

        seen: list[str] = [text]
        current = text

        for pass_num in range(1, self._max_passes + 1):
            after = self._apply_decoders(current)

            if after == current:
                # Output did not change on this pass → stable
                return NormaliserResult(text=after, stable=True)

            if after in seen:
                # Cycle detected — oscillating decoder set
                log.warning(
                    "Normaliser cycle detected on pass %d. "
                    "Decoder set is oscillating — check decoder configuration.",
                    pass_num,
                )
                return NormaliserResult(text=after, stable=False)

            seen.append(after)
            current = after

        # max_passes exhausted with output still changing
        log.debug(
            "Normaliser: max_passes=%d exhausted without convergence.", self._max_passes
        )
        return NormaliserResult(text=current, stable=False)

    def _apply_decoders(self, text: str) -> str:
        for decoder in self._decoders:
            try:
                text = decoder.decode(text)
            except Exception as e:  # noqa: BLE001 — never raise per Decoder contract
                log.debug("Decoder %r raised unexpectedly: %s — skipping", decoder.name, e)
        return text

    def __repr__(self) -> str:
        names = [d.name for d in self._decoders]
        return f"Normaliser(decoders={names!r}, max_passes={self._max_passes})"
