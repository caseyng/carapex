"""
normaliser/unicode_escape.py
-----------------------------
Decodes unicode escape sequences.

Catches:
    \u0069\u0067\u006E\u006F\u0072\u0065  →  ignore
    \x69\x67\x6e\x6f\x72\x65             →  ignore
"""

import re
from .base import Decoder

_UNICODE_ESC = re.compile(r"\\u([0-9a-fA-F]{4})")
_HEX_ESC     = re.compile(r"\\x([0-9a-fA-F]{2})")


class UnicodeEscapeDecoder(Decoder):
    """Decode \\uXXXX and \\xXX escape sequences."""

    name = "unicode_escape"

    def decode(self, text: str) -> str:
        try:
            text = _UNICODE_ESC.sub(
                lambda m: chr(int(m.group(1), 16)), text
            )
            text = _HEX_ESC.sub(
                lambda m: chr(int(m.group(1), 16)), text
            )
            return text
        except (ValueError, OverflowError):
            return text
