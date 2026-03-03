"""
normaliser/normaliser.py
------------------------
Normaliser orchestrator.

Runs configured decoders recursively until stable or max_passes reached.
Returns a NormaliserResult carrying both the text and a stability flag.

Stability semantics:
    stable=True  — output converged (equality check passed before max_passes)
    stable=False — max_passes exhausted without convergence, OR cycle detected

    Processor maps stable=False to safe=False, failure_mode="normalisation_unstable".
    An unstable input is either adversarially constructed or indicates a
    broken decoder. Neither should proceed through the safety pipeline.

Cycle detection:
    Tracks all previously seen values. If current output matches any prior
    value (not just the immediately preceding pass), the loop terminates
    and stable is set to False.

    With built-in decoders, oscillation cannot occur — all are strictly
    reductive. Cycle detection exists for custom decoder implementations
    that may violate this constraint. See normaliser/base.py.

Failure semantics:
    Decoder exceptions propagate as PipelineInternalError — a broken decoder
    must not silently degrade normalisation in a security pipeline.
"""

import logging
from dataclasses import dataclass
from typing import List

from .base import Decoder
from ..exceptions import PipelineInternalError

logger = logging.getLogger(__name__)


@dataclass
class NormaliserResult:
    """
    Result of Normaliser.normalise().

    text   : the normalised text (best available if unstable)
    stable : True if output converged cleanly.
             False if max_passes exhausted or cycle detected.
             Processor must treat stable=False as a security signal —
             map to safe=False, failure_mode="normalisation_unstable".
    """
    text:   str
    stable: bool


class Normaliser:
    """
    Runs decoders in sequence, recursively, until output stabilises.
    Returns NormaliserResult with stability flag.
    Raises PipelineInternalError if a decoder fails internally.
    """

    def __init__(self, decoders: List[Decoder], max_passes: int):
        """
        Args:
            decoders   : ordered list of Decoder instances
            max_passes : maximum recursive passes before stopping
        """
        self._decoders = decoders
        self._max      = max_passes

    def normalise(self, text: str) -> NormaliserResult:
        if not text:
            return NormaliserResult(text=text, stable=True)

        seen = {text}

        for _ in range(self._max):
            result = self._one_pass(text)

            if result == text:
                # Converged — output equals input, stable
                return NormaliserResult(text=result, stable=True)

            if result in seen:
                # Cycle detected — output matches a prior (non-adjacent) value.
                # This is a decoder design defect. Fail the stability check.
                logger.error(
                    "Normaliser cycle detected — a decoder is re-encoding its output. "
                    "This violates the strictly-reductive constraint. "
                    "Identify and fix the offending decoder. "
                    "Input will be treated as unstable."
                )
                return NormaliserResult(text=result, stable=False)

            seen.add(result)
            text = result

        # max_passes exhausted without convergence
        logger.error(
            "Normaliser did not converge after %d passes. "
            "Input may be adversarially constructed. "
            "Treating as unstable.",
            self._max,
        )
        return NormaliserResult(text=text, stable=False)

    def _one_pass(self, text: str) -> str:
        for decoder in self._decoders:
            try:
                text = decoder.decode(text)
            except PipelineInternalError:
                raise
            except Exception as e:
                raise PipelineInternalError(
                    component=f"Decoder:{decoder.name}",
                    original=e,
                ) from e
        return text

    def __repr__(self) -> str:
        names = [d.name for d in self._decoders]
        return f"Normaliser(decoders={names}, max_passes={self._max})"
