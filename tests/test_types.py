"""Tests for core data types — invariant enforcement."""

import pytest

from carapex.core.types import (
    CompletionResult,
    EvaluationResult,
    SafetyResult,
    ScriptResult,
    UsageResult,
)


class TestSafetyResult:
    def test_safe_true_no_failure_mode(self):
        r = SafetyResult(safe=True)
        assert r.safe is True
        assert r.failure_mode is None

    def test_safe_false_requires_failure_mode(self):
        r = SafetyResult(safe=False, failure_mode="safety_violation")
        assert r.safe is False
        assert r.failure_mode == "safety_violation"

    def test_safe_true_with_failure_mode_raises(self):
        with pytest.raises(ValueError):
            SafetyResult(safe=True, failure_mode="something")

    def test_safe_false_without_failure_mode_raises(self):
        with pytest.raises(ValueError):
            SafetyResult(safe=False)

    def test_reason_present_on_false(self):
        r = SafetyResult(safe=False, failure_mode="safety_violation", reason="bad thing")
        assert r.reason == "bad thing"

    def test_reason_none_on_true(self):
        r = SafetyResult(safe=True)
        assert r.reason is None


class TestScriptResult:
    def test_always_safe_true(self):
        r = ScriptResult()
        assert r.safe is True
        assert r.failure_mode is None

    def test_translation_needed_default_true(self):
        r = ScriptResult()
        assert r.translation_needed is True

    def test_english_no_translation(self):
        r = ScriptResult(detected_language="en", translation_needed=False)
        assert r.safe is True
        assert r.translation_needed is False

    def test_cannot_set_safe_false(self):
        # __post_init__ forces safe=True regardless
        r = ScriptResult(detected_language=None, translation_needed=True)
        assert r.safe is True


class TestEvaluationResult:
    def test_safe_true_with_content(self):
        r = EvaluationResult(safe=True, content="response text")
        assert r.safe is True
        assert r.content == "response text"
        assert r.failure_mode is None

    def test_safe_false_with_failure_mode(self):
        r = EvaluationResult(safe=False, failure_mode="safety_violation")
        assert r.safe is False
        assert r.content is None

    def test_safe_true_without_content_raises(self):
        with pytest.raises(ValueError):
            EvaluationResult(safe=True)  # content is None

    def test_safe_false_without_failure_mode_raises(self):
        with pytest.raises(ValueError):
            EvaluationResult(safe=False)

    def test_safe_true_with_failure_mode_raises(self):
        with pytest.raises(ValueError):
            EvaluationResult(safe=True, content="x", failure_mode="something")

    def test_safe_false_with_content_raises(self):
        with pytest.raises(ValueError):
            EvaluationResult(safe=False, failure_mode="x", content="y")


class TestCompletionResult:
    def test_fields_present(self):
        usage = UsageResult(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        r = CompletionResult(content="hello", finish_reason="stop", usage=usage)
        assert r.content == "hello"
        assert r.finish_reason == "stop"
        assert r.usage.total_tokens == 15
