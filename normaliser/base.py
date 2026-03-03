"""
normaliser/base.py
------------------
Abstract base and config for input decoders.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ADD A NEW DECODER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create a file in normaliser/  e.g. normaliser/rot13.py
2. Subclass Decoder
3. Set class attribute `name` — registry key and config reference
4. Implement decode()
5. Add your decoder name to config.json normaliser.decoders list
6. Done — autodiscovered and runs in the order listed

EXAMPLE:

    class Rot13Decoder(Decoder):
        name = "rot13"

        def decode(self, text: str) -> str:
            import codecs
            return codecs.encode(text, "rot_13")

RULES:
    - decode() must always be implemented
    - decode() must never raise — return original text on any error
    - Decoders are stateless — no config, no __init__ needed
    - The Normaliser runs decoders in the order listed in config
    - Never edit existing files to register a new decoder

DECODER DESIGN CONSTRAINT — STRICTLY REDUCTIVE:

    Decoders MUST be strictly reductive. A decoder must only move text
    toward plain canonical form — it must never re-encode its output
    in a form that an earlier decoder in the pipeline would act on.

    Example of a violation: a decoder that URL-encodes its output after
    processing would create an oscillation cycle with the URL decoder.
    The Normaliser detects and breaks cycles (see normaliser.py), but
    a cycling decoder degrades normalisation quality — the pipeline
    terminates at max_passes without reaching a stable canonical form.

    All built-in decoders are strictly reductive. This constraint is
    enforced by convention, not by code — decoder authors must respect it.

    If oscillation is detected in production (logged at ERROR by the
    Normaliser), the root cause is a decoder that violates this constraint.
    Identify and fix the offending decoder. Do not increase max_passes
    to paper over the problem.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from ..exceptions import ConfigurationError


@dataclass
class NormaliserConfig:
    """
    Config for the Normaliser orchestrator.
    Decoders are referenced by name — order matters.
    """
    max_passes: int       = 5
    decoders:   List[str] = field(default_factory=lambda: [
        "whitespace",
        "unicode_escape",
        "html_entity",
        "url",
        "base64",
        "homoglyph",
    ])

    def __post_init__(self):
        if self.max_passes < 1:
            raise ConfigurationError(
                f"max_passes must be >= 1, got {self.max_passes}"
            )
        if not self.decoders:
            raise ConfigurationError("decoders list cannot be empty")


class Decoder(ABC):
    """
    Abstract base for all input decoders.
    See module docstring for full extension guide.
    """

    name: str  # registry key — must be set on every subclass

    @abstractmethod
    def decode(self, text: str) -> str:
        """
        Attempt to decode text.
        Returns decoded text, or original if not applicable.
        Must never raise — return original text on any error.
        Must be strictly reductive — see module docstring.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
