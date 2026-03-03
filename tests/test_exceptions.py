"""
tests/test_exceptions.py
-------------------------
Unit tests for the exception hierarchy.

Verifies that each exception:
- Is a subclass of CarapexError
- Carries the correct fields
- Produces a message that includes enough context to act on

In security code, exception type and message clarity are correctness
properties — a vague exception leads to incorrect handling.
"""

import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from carapex.exceptions import (
    CarapexError,
    ConfigurationError,
    BackendUnavailableError,
    PipelineInternalError,
    CarapexViolation,
    IntegrityFailure,
    NormalisationError,
    PluginNotFoundError,
)


class TestHierarchy(unittest.TestCase):
    """Every exception descends from CarapexError."""

    def test_configuration_error_is_carapex_error(self):
        self.assertTrue(issubclass(ConfigurationError, CarapexError))

    def test_backend_unavailable_is_carapex_error(self):
        self.assertTrue(issubclass(BackendUnavailableError, CarapexError))

    def test_pipeline_internal_error_is_carapex_error(self):
        self.assertTrue(issubclass(PipelineInternalError, CarapexError))

    def test_carapex_violation_is_carapex_error(self):
        self.assertTrue(issubclass(CarapexViolation, CarapexError))

    def test_integrity_failure_is_carapex_error(self):
        self.assertTrue(issubclass(IntegrityFailure, CarapexError))

    def test_normalisation_error_is_carapex_error(self):
        self.assertTrue(issubclass(NormalisationError, CarapexError))

    def test_plugin_not_found_is_carapex_error(self):
        self.assertTrue(issubclass(PluginNotFoundError, CarapexError))


class TestPipelineInternalError(unittest.TestCase):

    def test_carries_component_name(self):
        original = ValueError("something broke")
        e = PipelineInternalError("PatternChecker", original)
        self.assertEqual(e.component, "PatternChecker")

    def test_carries_original_exception(self):
        original = RuntimeError("decoder bug")
        e = PipelineInternalError("Base64Decoder", original)
        self.assertIs(e.original, original)

    def test_message_contains_component(self):
        e = PipelineInternalError("GuardChecker", ValueError("oops"))
        self.assertIn("GuardChecker", str(e))

    def test_message_contains_original_type(self):
        e = PipelineInternalError("GuardChecker", ValueError("oops"))
        self.assertIn("ValueError", str(e))


class TestCarapexViolation(unittest.TestCase):
    """Caller-raised on content refusal. Never raised internally."""

    def test_carries_reason(self):
        e = CarapexViolation("Role-play detected", "safety_violation")
        self.assertEqual(e.reason, "Role-play detected")

    def test_carries_failure_mode(self):
        e = CarapexViolation("High entropy", "entropy_exceeded")
        self.assertEqual(e.failure_mode, "entropy_exceeded")

    def test_message_contains_failure_mode(self):
        e = CarapexViolation("reason", "safety_violation")
        self.assertIn("safety_violation", str(e))

    def test_message_contains_reason(self):
        e = CarapexViolation("Role-play detected", "safety_violation")
        self.assertIn("Role-play detected", str(e))


class TestIntegrityFailure(unittest.TestCase):
    """Caller-raised when a shell component could not complete evaluation."""

    def test_carries_failure_mode(self):
        e = IntegrityFailure("guard_unavailable")
        self.assertEqual(e.failure_mode, "guard_unavailable")

    def test_message_contains_failure_mode(self):
        e = IntegrityFailure("guard_evaluation_corrupt")
        self.assertIn("guard_evaluation_corrupt", str(e))

    def test_all_integrity_failure_modes_accepted(self):
        modes = ["guard_unavailable", "guard_evaluation_corrupt", "translation_failed"]
        for mode in modes:
            e = IntegrityFailure(mode)
            self.assertEqual(e.failure_mode, mode)


class TestNormalisationError(unittest.TestCase):
    """Caller-raised when normalisation did not stabilise."""

    def test_carries_failure_mode(self):
        e = NormalisationError("normalisation_unstable")
        self.assertEqual(e.failure_mode, "normalisation_unstable")

    def test_message_contains_failure_mode(self):
        e = NormalisationError("normalisation_unstable")
        self.assertIn("normalisation_unstable", str(e))


class TestPluginNotFoundError(unittest.TestCase):

    def test_carries_family(self):
        e = PluginNotFoundError("backends", "unknown", ["openai_compatible", "llama_cpp"])
        self.assertEqual(e.family, "backends")

    def test_carries_name(self):
        e = PluginNotFoundError("backends", "unknown", [])
        self.assertEqual(e.name, "unknown")

    def test_carries_available(self):
        available = ["openai_compatible", "llama_cpp"]
        e = PluginNotFoundError("backends", "unknown", available)
        self.assertEqual(e.available, available)

    def test_message_contains_name(self):
        e = PluginNotFoundError("backends", "unknown", ["openai_compatible"])
        self.assertIn("unknown", str(e))

    def test_message_contains_available(self):
        e = PluginNotFoundError("backends", "unknown", ["openai_compatible"])
        self.assertIn("openai_compatible", str(e))


class TestCallerRaisedConventions(unittest.TestCase):
    """
    CarapexViolation, IntegrityFailure, and NormalisationError are never
    raised by carapex itself. This test documents the intended usage
    pattern and verifies the exceptions are raiseable in the expected way.
    """

    def test_caller_raises_carapex_violation_from_response(self):
        from carapex.processor import Response
        r = Response(
            output       = None,
            safe         = False,
            refusal      = "Role-play detected",
            failure_mode = "safety_violation",
            audit_id     = "abc",
        )
        with self.assertRaises(CarapexViolation) as ctx:
            if not r.safe and r.failure_mode == "safety_violation":
                raise CarapexViolation(r.refusal, r.failure_mode)
        self.assertEqual(ctx.exception.failure_mode, "safety_violation")

    def test_caller_raises_integrity_failure_from_response(self):
        from carapex.processor import Response
        r = Response(
            output       = None,
            safe         = False,
            refusal      = "Guard unavailable",
            failure_mode = "guard_unavailable",
            audit_id     = "abc",
        )
        with self.assertRaises(IntegrityFailure) as ctx:
            if r.failure_mode == "guard_unavailable":
                raise IntegrityFailure(r.failure_mode)
        self.assertEqual(ctx.exception.failure_mode, "guard_unavailable")

    def test_caller_raises_normalisation_error_from_response(self):
        from carapex.processor import Response
        r = Response(
            output       = None,
            safe         = False,
            refusal      = "Input unstable",
            failure_mode = "normalisation_unstable",
            audit_id     = "abc",
        )
        with self.assertRaises(NormalisationError) as ctx:
            if r.failure_mode == "normalisation_unstable":
                raise NormalisationError(r.failure_mode)
        self.assertEqual(ctx.exception.failure_mode, "normalisation_unstable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
