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


class TestUrlPercentDecoderExtended:
    """BUG-1 coverage: %uXXXX must be decoded, not silently skipped."""

    def test_percent_u_single_char(self):
        d = UrlPercentDecoder()
        assert d.decode("%u0048") == "H"

    def test_percent_u_full_word(self):
        d = UrlPercentDecoder()
        # "Hello" encoded as %uXXXX
        assert d.decode("%u0048%u0065%u006C%u006C%u006F") == "Hello"

    def test_percent_u_injection_token(self):
        # [INST] encoded as %uXXXX — the bypass vector
        d = UrlPercentDecoder()
        encoded = "%u005B%u0049%u004E%u0053%u0054%u005D"
        assert d.decode(encoded) == "[INST]"

    def test_mixed_percent_xx_and_percent_u(self):
        d = UrlPercentDecoder()
        # %20 (standard space) and %u0048 (H) mixed
        result = d.decode("hello%20%u0048ello")
        assert result == "hello Hello"

    def test_percent_u_uppercase(self):
        d = UrlPercentDecoder()
        assert d.decode("%U0048") == "H"

    def test_standard_percent_xx_still_works(self):
        d = UrlPercentDecoder()
        assert d.decode("hello%20world") == "hello world"

    def test_passthrough_no_encoding(self):
        d = UrlPercentDecoder()
        assert d.decode("plain text") == "plain text"

    def test_invalid_percent_u_passthrough(self):
        # %uXXXX where XXXX is not valid hex — should not crash
        d = UrlPercentDecoder()
        result = d.decode("%uZZZZ")
        assert isinstance(result, str)


class TestUnicodeEscapeDecoderExtended:
    def test_capital_U_eight_char(self):
        d = UnicodeEscapeDecoder()
        # \U0001F600 = 😀 (grinning face)
        assert d.decode(r"\U0001F600") == "\U0001F600"

    def test_lowercase_x_two_char(self):
        d = UnicodeEscapeDecoder()
        assert d.decode(r"\x48\x65\x6C\x6C\x6F") == "Hello"

    def test_mixed_escape_sequences(self):
        d = UnicodeEscapeDecoder()
        # H via \x48 and rest literal
        assert d.decode(r"\x48ello") == "Hello"

    def test_mixed_u_and_U(self):
        d = UnicodeEscapeDecoder()
        assert d.decode(r"H\U00000065") == "He"


class TestBase64DecoderExtended:
    def test_no_padding(self):
        import base64
        payload = "Hello from base64!"
        # base64 without padding (strip trailing =)
        encoded = base64.b64encode(payload.encode()).decode().rstrip("=")
        d = Base64Decoder()
        result = d.decode(encoded)
        assert "Hello" in result

    def test_double_padding(self):
        import base64
        # 1 byte → 2 base64 chars + "==" padding
        payload = b"X"
        encoded = base64.b64encode(payload).decode()
        assert encoded.endswith("==")
        # Pad to 16 chars minimum by repeating the encoded pattern
        long_encoded = (encoded.rstrip("=") * 8)[:16]
        padded = long_encoded + "=" * ((-len(long_encoded)) % 4)
        d = Base64Decoder()
        result = d.decode(padded)
        assert isinstance(result, str)

    def test_exactly_16_chars_decoded(self):
        import base64
        # Construct a payload that base64-encodes to exactly 16 chars
        # 12 bytes → 16 base64 chars (12 * 4/3 = 16)
        payload = b"123456789012"
        encoded = base64.b64encode(payload).decode()
        assert len(encoded) == 16
        d = Base64Decoder()
        result = d.decode(encoded)
        assert result != encoded  # was decoded

    def test_15_chars_not_decoded(self):
        # 15 chars < 16 minimum → left unchanged
        d = Base64Decoder()
        short = "MTIzNDU2Nzg5MA"  # 14 chars
        assert len(short) == 14
        assert d.decode(short) == short

    def test_non_utf8_output_left_unchanged(self):
        import base64
        # Bytes that decode to non-UTF-8 binary (high bytes)
        raw = bytes(range(200, 216))  # 16 bytes → 24 base64 chars
        encoded = base64.b64encode(raw).decode()
        d = Base64Decoder()
        result = d.decode(encoded)
        # Cannot decode to UTF-8 → original token left unchanged
        assert result == encoded


class TestNormalisedEndToEnd:
    def test_percent_u_encoded_injection_blocked(self):
        """BUG-1 end-to-end: %uXXXX-encoded [INST] must decode then be blocked."""
        import carapex.normaliser  # noqa: F401
        from carapex.core.registry import all_decoder_names, get_decoder
        from carapex.normaliser.base import Normaliser
        from carapex.safety.pattern import PatternChecker

        decoders = [get_decoder(n)() for n in all_decoder_names()]
        normaliser = Normaliser(decoders=decoders, max_passes=5)

        # [INST] encoded as %uXXXX
        encoded_inst = "%u005B%u0049%u004E%u0053%u0054%u005D"
        norm_result = normaliser.normalise(encoded_inst)
        assert norm_result.stable is True
        assert norm_result.text == "[INST]"

        # Pattern checker must now block it
        checker = PatternChecker()
        safety = checker.inspect(norm_result.text)
        assert safety.safe is False
        assert safety.failure_mode == "safety_violation"
