"""Tests for Translator and CheckerCoordinator."""

import pytest

from carapex.core.types import SafetyResult, ScriptResult
from carapex.safety.coordinator import CheckerCoordinator
from carapex.safety.translator import Translator
from tests.conftest import StubLLM, UnavailableLLM


class TestTranslator:
    def _script_result(self, needed: bool) -> ScriptResult:
        return ScriptResult(
            detected_language=None if needed else "en",
            translation_needed=needed,
        )

    def test_no_translation_needed_passes_through(self):
        llm = StubLLM("should not be called")
        t = Translator(llm=llm, temperature=0.0)
        t.set_prior_result(self._script_result(needed=False))
        result = t.inspect("hello world")
        assert result.safe is True
        assert t.get_output_text() == "hello world"
        assert len(llm.calls) == 0

    def test_translation_succeeds(self):
        llm = StubLLM("Bonjour le monde translated to English")
        t = Translator(llm=llm, temperature=0.0)
        t.set_prior_result(self._script_result(needed=True))
        result = t.inspect("Bonjour le monde")
        assert result.safe is True
        assert t.get_output_text() == "Bonjour le monde translated to English"

    def test_llm_unavailable_fails(self):
        llm = UnavailableLLM()
        t = Translator(llm=llm, temperature=0.0)
        t.set_prior_result(self._script_result(needed=True))
        result = t.inspect("Bonjour")
        assert result.safe is False
        assert result.failure_mode == "translation_failed"

    def test_echo_response_fails(self):
        # LLM returns the input unchanged — must fail (§5)
        text = "Bonjour le monde"
        llm = StubLLM(text)
        t = Translator(llm=llm, temperature=0.0)
        t.set_prior_result(self._script_result(needed=True))
        result = t.inspect(text)
        assert result.safe is False
        assert result.failure_mode == "translation_failed"

    def test_empty_translation_fails(self):
        llm = StubLLM("   ")  # whitespace-only → stripped to empty
        t = Translator(llm=llm, temperature=0.0)
        t.set_prior_result(self._script_result(needed=True))
        result = t.inspect("Bonjour")
        assert result.safe is False
        assert result.failure_mode == "translation_failed"

    def test_inspect_without_set_prior_raises(self):
        llm = StubLLM()
        t = Translator(llm=llm, temperature=0.0)
        with pytest.raises(ValueError):
            t.inspect("anything")

    def test_does_not_close_injected_llm(self):
        """Translator.close() must not close the injected LLM."""
        closed = []
        class TrackingLLM(StubLLM):
            name = "_tracking"
            def close(self):
                closed.append(True)

        llm = TrackingLLM()
        t = Translator(llm=llm)
        t.close()
        assert closed == []  # LLM was NOT closed


class TestCheckerCoordinator:
    def test_empty_sequence_passes(self):
        coord = CheckerCoordinator([])
        result, text = coord.inspect("anything")
        assert result.safe is True
        assert text == "anything"

    def test_all_pass(self):
        from carapex.safety.entropy import EntropyChecker
        from carapex.safety.pattern import PatternChecker
        coord = CheckerCoordinator([EntropyChecker(), PatternChecker()])
        result, text = coord.inspect("What is the capital of France?")
        assert result.safe is True

    def test_stops_at_first_failure(self):
        call_order = []

        from carapex.safety.base import SafetyChecker

        class RecordingChecker(SafetyChecker):
            name = "_recording"
            def __init__(self, label, safe):
                self._label = label
                self._safe = safe
            def inspect(self, text):
                call_order.append(self._label)
                if self._safe:
                    return SafetyResult(safe=True)
                return SafetyResult(safe=False, failure_mode="safety_violation")
            def close(self): pass

        coord = CheckerCoordinator([
            RecordingChecker("first", safe=True),
            RecordingChecker("second", safe=False),
            RecordingChecker("third", safe=True),  # must NOT run
        ])
        result, _ = coord.inspect("text")
        assert result.safe is False
        assert call_order == ["first", "second"]
        assert "third" not in call_order

    def test_text_transforming_checker_updates_working_text(self):
        """Coordinator must call get_output_text() and update working text."""
        from carapex.core.types import ScriptResult
        from carapex.safety.base import SafetyChecker, TextTransformingChecker

        class UpperCaseTransformer(TextTransformingChecker):
            name = "_upper"
            def set_prior_result(self, result): pass
            def inspect(self, text):
                self._out = text.upper()
                return SafetyResult(safe=True)
            def get_output_text(self): return self._out
            def close(self): pass

        class RecordInput(SafetyChecker):
            name = "_record_input"
            def __init__(self): self.received = None
            def inspect(self, text):
                self.received = text
                return SafetyResult(safe=True)
            def close(self): pass

        recorder = RecordInput()
        coord = CheckerCoordinator([UpperCaseTransformer(), recorder])
        result, final_text = coord.inspect("hello")
        assert recorder.received == "HELLO"
        assert final_text == "HELLO"

    def test_none_input_raises(self):
        coord = CheckerCoordinator([])
        with pytest.raises(ValueError):
            coord.inspect(None)  # type: ignore
