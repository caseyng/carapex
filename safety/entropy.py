"""
safety/entropy.py
-----------------
Shannon entropy-based anomaly detector.

Catches adversarial inputs that survive normalisation — custom ciphers,
ROT-N variants, undecoded binary-as-text, and other high-density encodings
that the decoder set does not cover.

Why entropy works here:
    Natural human language has characteristic information density.
    English prose: ~3.5–4.5 bits/char.
    Dense technical content (code, JSON, URLs): ~4.5–5.5 bits/char.
    Base64: ~6.0 bits/char.
    Random bytes / custom ciphers: ~6.5–8.0 bits/char.

    The normaliser runs first. If it decoded the content successfully,
    entropy drops to natural-language range. If high entropy persists
    after normalisation, the content was not decoded — either because
    no decoder covers the scheme, or because it is genuinely adversarial.

    This gate adds zero LLM cost and catches attack vectors that
    pattern matching cannot — schemes with no fixed structural markers.

Threshold:
    Default 5.8 bits/char. Above this threshold, no legitimate natural
    language prompt sits — even dense technical content rarely exceeds 5.5.
    Configurable via SafetyConfig.entropy_threshold.
    Disabled by setting entropy_threshold to None or 0.0.

Minimum length:
    Entropy on short strings is statistically unreliable — a 10-character
    string with two unusual characters will spike entropy without indicating
    anything. Default minimum: 50 characters. Configurable via
    SafetyConfig.entropy_min_length.
"""

import math
from collections import Counter

from .base import SafetyChecker, SafetyResult, SafetyConfig


def _shannon_entropy(text: str) -> float:
    """
    Compute Shannon entropy of text in bits per character.
    Returns 0.0 for empty or single-character strings.
    """
    if len(text) < 2:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


class EntropyChecker(SafetyChecker):
    """
    Blocks inputs whose Shannon entropy exceeds the configured threshold
    after normalisation.

    Stateless — no backend, no config beyond threshold values.
    Deterministic — same input always produces same result.
    """

    name = "entropy"

    def __init__(self, cfg: SafetyConfig):
        self._threshold  = cfg.entropy_threshold
        self._min_length = cfg.entropy_min_length

    def check(self, text: str) -> SafetyResult:
        """
        Compute entropy and compare to threshold.
        Returns safe=True if entropy gating is disabled or input is below threshold.
        Returns safe=False with failure_mode="entropy_exceeded" if above threshold.
        """
        # Disabled — explicit None check required.
        # `if not self._threshold` would also disable on 0.0, which is a
        # valid-looking misconfiguration that should not silently pass all inputs.
        if self._threshold is None:
            return SafetyResult(safe=True)

        # Too short for reliable measurement
        if len(text) < self._min_length:
            return SafetyResult(safe=True)

        entropy = _shannon_entropy(text)

        if entropy > self._threshold:
            return SafetyResult(
                safe         = False,
                reason       = (
                    f"Input entropy {entropy:.2f} bits/char exceeds threshold "
                    f"{self._threshold} bits/char. Content does not match "
                    f"natural language distribution — possible encoding attack."
                ),
                failure_mode = "entropy_exceeded",
            )

        return SafetyResult(safe=True)

    def __repr__(self) -> str:
        return (
            f"EntropyChecker("
            f"threshold={self._threshold}, "
            f"min_length={self._min_length})"
        )
