"""Tests for Normaliser — convergence, cycle detection, decoder behaviour."""

import pytest

import carapex.normaliser  # noqa: F401 — triggers decoder registration
from carapex.normaliser.base import Decoder, Normaliser
from carapex.normaliser.decoders import (
    Base64Decoder,
    HtmlEntityDecoder,
    UrlPercentDecoder,
    UnicodeEscapeDecoder,
    WhitespaceDecoder,
)


class TestNormaliserConvergence:
    def test_stable_plain_text(self):
        n = Normaliser(decoders=[WhitespaceDecoder()], max_passes=5)
        result = n.normalise("hello world")
        assert result.stable is True
        assert result.text == "hello world"

    def test_single_pass_decode(self):
        n = Normaliser(decoders=[HtmlEntityDecoder()], max_passes=5)
        result = n.normalise("&lt;script&gt;")
        assert result.stable is True
        assert result.text == "<script>"

    def test_multi_pass_converges(self):
        # Double-encoded: &amp;lt; → &lt; → <
        n = Normaliser(decoders=[HtmlEntityDecoder()], max_passes=5)
        result = n.normalise("&amp;lt;")
        assert result.stable is True
        assert "<" in result.text

    def test_max_passes_exhausted_unstable(self):
        class OscillatingDecoder(Decoder):
            name = "_oscillating"
            def __init__(self):
                self._toggle = False
            def decode(self, text):
                self._toggle = not self._toggle
                return text + "x" if self._toggle else text[:-1]

        n = Normaliser(decoders=[OscillatingDecoder()], max_passes=3)
        result = n.normalise("abc")
        assert result.stable is False

    def test_empty_decoder_list(self):
        n = Normaliser(decoders=[], max_passes=5)
        result = n.normalise("anything")
        assert result.stable is True
        assert result.text == "anything"

    def test_empty_string_input(self):
        n = Normaliser(decoders=[WhitespaceDecoder()], max_passes=5)
        result = n.normalise("")
        assert result.stable is True
        assert result.text == ""

    def test_none_input_raises(self):
        n = Normaliser(decoders=[], max_passes=5)
        with pytest.raises(ValueError):
            n.normalise(None)  # type: ignore

    def test_max_passes_zero_raises(self):
        with pytest.raises(ValueError):
            Normaliser(decoders=[], max_passes=0)

    def test_cycle_detection(self):
        # Decoder that alternates between two states
        class CyclingDecoder(Decoder):
            name = "_cycling"
            _state = 0
            def decode(self, text):
                CyclingDecoder._state += 1
                return "A" if CyclingDecoder._state % 2 == 0 else "B"

        n = Normaliser(decoders=[CyclingDecoder()], max_passes=10)
        result = n.normalise("start")
        assert result.stable is False


class TestDecoders:
    def test_unicode_escape(self):
        d = UnicodeEscapeDecoder()
        assert d.decode(r"\u0048\u0065\u006C\u006C\u006F") == "Hello"

    def test_unicode_escape_passthrough(self):
        d = UnicodeEscapeDecoder()
        assert d.decode("plain text") == "plain text"

    def test_html_entity_named(self):
        d = HtmlEntityDecoder()
        assert d.decode("&amp;") == "&"
        assert d.decode("&lt;&gt;") == "<>"

    def test_html_entity_numeric(self):
        d = HtmlEntityDecoder()
        assert d.decode("&#72;&#101;&#108;&#108;&#111;") == "Hello"

    def test_html_entity_passthrough(self):
        d = HtmlEntityDecoder()
        assert d.decode("no entities here") == "no entities here"

    def test_url_percent(self):
        d = UrlPercentDecoder()
        assert d.decode("hello%20world") == "hello world"

    def test_url_percent_passthrough(self):
        d = UrlPercentDecoder()
        assert d.decode("no encoding") == "no encoding"

    def test_base64_decodes_block(self):
        import base64
        payload = "Hello from base64"
        encoded = base64.b64encode(payload.encode()).decode()
        d = Base64Decoder()
        result = d.decode(encoded)
        assert payload in result

    def test_base64_ignores_short_tokens(self):
        # "data" is valid base64 but too short (< 16 chars) — should not be decoded
        d = Base64Decoder()
        assert d.decode("data") == "data"

    def test_whitespace_collapses(self):
        d = WhitespaceDecoder()
        assert d.decode("hello   world\n\t!") == "hello world !"

    def test_whitespace_strips(self):
        d = WhitespaceDecoder()
        assert d.decode("  hello  ") == "hello"

    def test_decoder_never_raises_on_bad_input(self):
        # All decoders must return input unchanged on errors
        for cls in [UnicodeEscapeDecoder, HtmlEntityDecoder, UrlPercentDecoder,
                    Base64Decoder, WhitespaceDecoder]:
            d = cls()
            # Should not raise for any string
            result = d.decode("\x00\xff\xfe broken bytes string")
            assert isinstance(result, str)
