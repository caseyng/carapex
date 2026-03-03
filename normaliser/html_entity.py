"""
normaliser/html_entity.py
--------------------------
Decodes HTML entity encoding.

Catches:
    &#105;&#103;&#110;&#111;&#114;&#101;  →  ignore
    &lt;system&gt;                         →  <system>
    &amp;                                  →  &
"""

import html
import re
from .base import Decoder

# Detect presence of HTML entities worth decoding
_HAS_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


class HtmlEntityDecoder(Decoder):
    """Decode HTML entities using stdlib html.unescape."""

    name = "html_entity"

    def decode(self, text: str) -> str:
        if _HAS_ENTITY.search(text):
            return html.unescape(text)
        return text
