"""
normaliser/url.py
-----------------
Decodes URL percent-encoding.

Catches:
    %69%67%6E%6F%72%65  →  ignore
    ignore%20previous   →  ignore previous
"""

import urllib.parse
import re
from .base import Decoder

# Only decode if there are percent-encoded sequences worth checking
_HAS_PERCENT = re.compile(r"%[0-9a-fA-F]{2}")


class URLDecoder(Decoder):
    """Decode URL percent-encoded sequences."""

    name = "url"

    def decode(self, text: str) -> str:
        if _HAS_PERCENT.search(text):
            decoded = urllib.parse.unquote(text)
            if decoded != text:
                return decoded
        return text
