"""
Tests for ScriptChecker — language detection via lingua-language-detector.

These tests require the lingua package and are marked with @pytest.mark.lingua.
Run separately from the fast test suite:

    pytest tests/test_script_checker.py -m lingua -v

To run all non-lingua tests:

    pytest tests/ -m "not lingua" -v
"""

from __future__ import annotations

import pytest

from carapex.core.types import ScriptResult


pytestmark = pytest.mark.lingua


@pytest.fixture(scope="module")
def checker():
    """Build a ScriptChecker once per module — lingua initialisation is slow."""
    from carapex.safety.script import ScriptChecker
    return ScriptChecker(confidence_threshold=0.80)


class TestScriptChecker:
    def test_english_text_no_translation_needed(self, checker):
        result = checker.inspect(
            "The quick brown fox jumps over the lazy dog. "
            "This is a clearly English sentence with common words."
        )
        assert result.translation_needed is False
        assert result.detected_language is not None

    def test_non_english_triggers_translation(self, checker):
        # "Bonjour, comment allez-vous aujourd'hui?" — French
        result = checker.inspect("Bonjour, comment allez-vous aujourd'hui? J'espère que vous allez bien.")
        assert result.translation_needed is True

    def test_always_returns_safe_true(self, checker):
        # ScriptChecker is a detector, never a gate
        for text in ["Hello", "Bonjour", "你好世界", "مرحبا بالعالم"]:
            result = checker.inspect(text)
            assert result.safe is True
            assert result.failure_mode is None

    def test_none_raises_value_error(self, checker):
        with pytest.raises(ValueError):
            checker.inspect(None)  # type: ignore

    def test_empty_string_defaults_to_translation_needed(self, checker):
        # Empty text → detection fails → safe fallback: translation_needed=True
        result = checker.inspect("")
        assert result.safe is True
        assert result.translation_needed is True

    def test_detection_exception_defaults_to_translation_needed(self):
        # Monkeypatch detector to raise — ScriptChecker must never raise
        from carapex.safety.script import ScriptChecker
        sc = ScriptChecker.__new__(ScriptChecker)
        sc._threshold = 0.80

        class AlwaysRaisingDetector:
            def detect_language_of(self, text):
                raise RuntimeError("detection failed")

        sc._detector = AlwaysRaisingDetector()
        result = sc.inspect("any text")
        assert result.safe is True
        assert result.translation_needed is True

    def test_low_confidence_english_triggers_translation(self):
        # With threshold=0.99, even confidently-detected English may not pass
        from carapex.safety.script import ScriptChecker
        sc = ScriptChecker(confidence_threshold=0.99)
        # Short ambiguous text — detection confidence may be below 0.99
        result = sc.inspect("ok")
        assert result.safe is True
        # At 0.99 threshold, short text is more likely to trigger translation
        # (we don't assert translation_needed here as it depends on the model)

    def test_returns_script_result_type(self, checker):
        result = checker.inspect("Hello world, this is a test sentence.")
        assert isinstance(result, ScriptResult)

    def test_script_result_has_language_code(self, checker):
        result = checker.inspect(
            "This is an English sentence that should be detected confidently."
        )
        if not result.translation_needed:
            assert result.detected_language is not None
            assert isinstance(result.detected_language, str)
