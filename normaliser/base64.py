"""
normaliser/base64.py
---------------------
Decodes base64 encoded content.

Catches:
    aWdub3JlIHByZXZpb3VzIGluc3RydWN0aW9ucw==  →  ignore previous instructions

Conservative approach:
    - Only decode if result is valid UTF-8 printable text
    - Ignore short strings (< 20 chars) — too many false positives
    - Ignore strings that don't look like base64 (wrong charset)
"""

import base64
import re
from .base import Decoder

# Base64 pattern — only attempt on strings that look like base64
_B64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")

# Minimum decoded length to consider valid
_MIN_DECODED_LENGTH = 8


class Base64Decoder(Decoder):
    """Decode base64 blobs that produce valid UTF-8 text."""

    name = "base64"

    def decode(self, text: str) -> str:
        try:
            # Check each whitespace-separated token
            tokens  = text.split()
            changed = False
            result  = []

            for token in tokens:
                decoded = self._try_decode(token)
                if decoded is not None:
                    result.append(decoded)
                    changed = True
                else:
                    result.append(token)

            return " ".join(result) if changed else text

        except (ValueError, UnicodeDecodeError):
            return text

    def _try_decode(self, token: str) -> str | None:
        """Return decoded string if token is valid base64, else None."""
        # Strip common base64 padding variants
        clean = token.strip().rstrip(".")

        if not _B64_PATTERN.match(clean):
            return None

        try:
            decoded_bytes = base64.b64decode(clean + "==")  # pad safely
            decoded_str   = decoded_bytes.decode("utf-8")

            if len(decoded_str) < _MIN_DECODED_LENGTH:
                return None

            # Must be mostly printable — reject binary-looking results
            printable = sum(1 for c in decoded_str if c.isprintable())
            if printable / len(decoded_str) < 0.85:
                return None

            return decoded_str

        except (ValueError, UnicodeDecodeError):
            return None
