"""
Built-in Decoder implementations.

All decoders are strictly reductive — they only move text toward canonical
plain form. None re-encode their output.

Registered decoders (applied in this order by default):
  unicode_escape  — \\uXXXX, \\UXXXXXXXX, \\xXX sequences
  html_entity     — &amp;, &#60;, &#x3C; etc.
  url_percent     — %XX percent-encoded bytes
  base64          — base64-encoded blocks (must look like base64)
  whitespace      — collapse runs of whitespace to single spaces
"""

from __future__ import annotations

import base64
import html
import logging
import re
import urllib.parse

from carapex.core.registry import register_decoder
from carapex.normaliser.base import Decoder

log = logging.getLogger(__name__)


@register_decoder
class UnicodeEscapeDecoder(Decoder):
    """Decodes Python-style Unicode escape sequences: \\uXXXX, \\UXXXXXXXX, \\xXX."""

    name = "unicode_escape"

    _PATTERN = re.compile(
        r"\\u[0-9A-Fa-f]{4}"
        r"|\\U[0-9A-Fa-f]{8}"
        r"|\\x[0-9A-Fa-f]{2}"
    )

    @staticmethod
    def _replace(m: re.Match[str]) -> str:
        try:
            return chr(int(m.group(0)[2:], 16))
        except (ValueError, OverflowError):
            return m.group(0)

    def decode(self, text: str) -> str:
        if not self._PATTERN.search(text):
            return text
        try:
            return self._PATTERN.sub(self._replace, text)
        except Exception:  # noqa: BLE001
            return text


@register_decoder
class HtmlEntityDecoder(Decoder):
    """Decodes HTML entities: &amp;, &#60;, &#x3C;, etc."""

    name = "html_entity"

    _PATTERN = re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")

    def decode(self, text: str) -> str:
        if not self._PATTERN.search(text):
            return text
        try:
            return html.unescape(text)
        except Exception:  # noqa: BLE001
            return text


@register_decoder
class UrlPercentDecoder(Decoder):
    """Decodes URL percent-encoded sequences: %XX and %uXXXX (non-standard Microsoft extension)."""

    name = "url_percent"

    _PATTERN = re.compile(r"%[0-9A-Fa-f]{2}|%u[0-9A-Fa-f]{4}", re.IGNORECASE)
    _PERCENT_U = re.compile(r"%u([0-9A-Fa-f]{4})", re.IGNORECASE)

    @staticmethod
    def _replace_percent_u(m: re.Match[str]) -> str:
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)

    def decode(self, text: str) -> str:
        if not self._PATTERN.search(text):
            return text
        try:
            # Decode %uXXXX first (not handled by urllib.parse.unquote)
            text = self._PERCENT_U.sub(self._replace_percent_u, text)
            # Decode standard %XX sequences
            return urllib.parse.unquote(text)
        except Exception:  # noqa: BLE001
            return text


@register_decoder
class Base64Decoder(Decoder):
    """Decodes base64-encoded blocks within text.

    Heuristic: a token is considered base64 if it is >=16 characters,
    consists only of base64 alphabet characters, and decodes to valid UTF-8.
    Short tokens are skipped to avoid false positives on legitimate words
    that happen to be valid base64 (e.g. "data", "user").
    """

    name = "base64"

    _TOKEN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

    def decode(self, text: str) -> str:
        def try_decode(m: re.Match[str]) -> str:
            token = m.group(0)
            # Pad to multiple of 4
            padded = token + "=" * ((-len(token)) % 4)
            try:
                decoded = base64.b64decode(padded).decode("utf-8")
                # Only substitute if the decoded text is printable-ish
                if decoded.isprintable() or "\n" in decoded:
                    return decoded
            except Exception:  # noqa: BLE001
                pass
            return token

        if not self._TOKEN.search(text):
            return text
        return self._TOKEN.sub(try_decode, text)


@register_decoder
class WhitespaceDecoder(Decoder):
    """Collapses runs of whitespace (spaces, tabs, newlines) to a single space.

    Strips leading and trailing whitespace. This is the final pass in the
    default decoder set — it canonicalises spacing after all other decodings.
    """

    name = "whitespace"

    def decode(self, text: str) -> str:
        return " ".join(text.split())
