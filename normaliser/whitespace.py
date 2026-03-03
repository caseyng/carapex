"""
normaliser/whitespace.py
------------------------
Normalises whitespace obfuscation attacks.

Catches:
    - Zero-width characters (invisible unicode)
    - Excessive spacing between characters ("i g n o r e")
    - Mixed unicode whitespace variants
"""

import re
import unicodedata
from .base import Decoder

# Zero-width and invisible unicode characters
_ZERO_WIDTH = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u2060",  # word joiner
    "\u2061",  # function application
    "\u2062",  # invisible times
    "\u2063",  # invisible separator
    "\u2064",  # invisible plus
    "\ufeff",  # zero-width no-break space (BOM)
    "\u00ad",  # soft hyphen
]

# Collapse spaced-out words: "i g n o r e" → "ignore"
_SPACED_WORD = re.compile(r"\b(\w)((?:\s\w){2,})\b")


class WhitespaceDecoder(Decoder):
    """Remove zero-width characters and collapse spaced-out words."""

    name = "whitespace"

    def decode(self, text: str) -> str:
        # Remove zero-width characters
        for zw in _ZERO_WIDTH:
            text = text.replace(zw, "")

        # Normalise unicode whitespace variants to standard space
        text = "".join(
            " " if unicodedata.category(c) == "Zs" else c
            for c in text
        )

        # Collapse "i g n o r e" → "ignore"
        def _collapse(m):
            chars = m.group(0).replace(" ", "")
            return chars

        text = _SPACED_WORD.sub(_collapse, text)

        return text
