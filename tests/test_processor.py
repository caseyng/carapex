"""
tests/test_processor.py
------------------------
Unit tests for the Carapex processor.

No LLM required — all dependencies stubbed.
Tests cover: clean pass, input refusal, output refusal, backend failure,
normalisation instability, failure_mode contract, audit trail.
"""

import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from carapex.processor             import Carapex, Response
from carapex.safety.base           import SafetyResult
from carapex.normaliser.normaliser import Normaliser, NormaliserResult
from carapex.normaliser.url        import URLDecoder
from carapex.exceptions            import BackendUnavailableError


# ── Stubs ──────────────────────────────────────────────────────────────────

class _SafeBackend:
    name = "stub_safe"
    def chat(self, s, u, temperature=None): return "Safe response."
    def health_check(self): return True
    def close(self): pass
    def last_usage(self):
        return {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    def __repr__(self): return "SafeBackend()"

class _FailBackend:
    name = "stub_fail"
    def chat(self, s, u, temperature=None): return None
    def health_check(self): return False
    def close(self): pass
    def last_usage(self): return {}
    def __repr__(self): return "FailBackend()"

class _AlwaysSafe:
    name = "always_safe"
    def check(self, t): return SafetyResult(safe=True)
    def __repr__(self): return "AlwaysSafe()"

class _AlwaysUnsafe:
    name = "always_unsafe"
    def check(self, t): return SafetyResult(safe=False, reason="Test violation")
    def __repr__(self): return "AlwaysUnsafe()"

class _AlwaysUnsafeWithMode:
    name = "always_unsafe_with_mode"
    def __init__(self, failure_mode):
        self._mode = failure_mode
    def check(self, t):
        return SafetyResult(safe=False, reason="Shell failure", failure_mode=self._mode)
    def __repr__(self): return f"AlwaysUnsafeWithMode({self._mode!r})"

class _StubAudit:
    def __init__(self): self.events = []
    def log(self, e): self.events.append(e)
    def close(self): pass
    def __repr__(self): return "StubAudit()"

class _StableNormaliser:
    """Returns NormaliserResult(stable=True, text=text)."""
    def normalise(self, text): return NormaliserResult(text=text, stable=True)
    def __repr__(self): return "StableNormaliser()"

class _UnstableNormaliser:
    """Returns NormaliserResult(stable=False, text=text)."""
    def normalise(self, text): return NormaliserResult(text=text, stable=False)
    def __repr__(self): return "UnstableNormaliser()"


def _make(
    backend=None,
    input_checker=None,
    output_checker=None,
    normaliser=None,
):
    audit = _StubAudit()
    cp = Carapex(
        system_prompt  = "You are a helpful assistant.",
        backend        = backend        or _SafeBackend(),
        input_checker  = input_checker  or _AlwaysSafe(),
        output_checker = output_checker or _AlwaysSafe(),
        normaliser     = normaliser     or _StableNormaliser(),
        audit          = audit,
    )
    return cp, audit


# ── Clean pass ─────────────────────────────────────────────────────────────

class TestRunSuccess(unittest.TestCase):

    def test_returns_output(self):
        cp, _ = _make()
        r = cp.run("What is the capital of France?")
        self.assertTrue(r.safe)
        self.assertEqual(r.output, "Safe response.")
        self.assertIsNone(r.refusal)
        self.assertIsNone(r.failure_mode)

    def test_audit_id_present(self):
        cp, _ = _make()
        r = cp.run("Hello")
        self.assertIsNotNone(r.audit_id)
        # Full UUID — 32 hex chars + 4 hyphens = 36
        self.assertEqual(len(r.audit_id), 36)

    def test_token_counts_populated(self):
        cp, _ = _make()
        r = cp.run("Hello")
        self.assertEqual(r.tokens_in,    10)
        self.assertEqual(r.tokens_out,   20)
        self.assertEqual(r.tokens_total, 30)

    def test_failure_mode_none_on_clean_pass(self):
        cp, _ = _make()
        r = cp.run("Hello")
        self.assertIsNone(r.failure_mode)


# ── Input refusal ──────────────────────────────────────────────────────────

class TestInputRefusal(unittest.TestCase):

    def test_unsafe_input_returns_refusal(self):
        cp, _ = _make(input_checker=_AlwaysUnsafe())
        r = cp.run("bad prompt")
        self.assertFalse(r.safe)
        self.assertIsNone(r.output)
        self.assertEqual(r.refusal, "Test violation")

    def test_input_refusal_sets_failure_mode(self):
        cp, _ = _make(input_checker=_AlwaysUnsafe())
        r = cp.run("bad prompt")
        # No failure_mode from checker → processor defaults to "safety_violation"
        self.assertEqual(r.failure_mode, "safety_violation")

    def test_failure_mode_propagated_from_checker(self):
        cp, _ = _make(
            input_checker=_AlwaysUnsafeWithMode("entropy_exceeded")
        )
        r = cp.run("prompt")
        self.assertEqual(r.failure_mode, "entropy_exceeded")

    def test_guard_unavailable_propagates(self):
        cp, _ = _make(
            input_checker=_AlwaysUnsafeWithMode("guard_unavailable")
        )
        r = cp.run("prompt")
        self.assertEqual(r.failure_mode, "guard_unavailable")

    def test_guard_corrupt_propagates(self):
        cp, _ = _make(
            input_checker=_AlwaysUnsafeWithMode("guard_evaluation_corrupt")
        )
        r = cp.run("prompt")
        self.assertEqual(r.failure_mode, "guard_evaluation_corrupt")

    def test_translation_failed_propagates(self):
        cp, _ = _make(
            input_checker=_AlwaysUnsafeWithMode("translation_failed")
        )
        r = cp.run("prompt")
        self.assertEqual(r.failure_mode, "translation_failed")

    def test_output_not_checked_when_input_refused(self):
        cp, audit = _make(
            input_checker  = _AlwaysUnsafe(),
            output_checker = _AlwaysUnsafe(),
        )
        cp.run("bad")
        events = [e["event"] for e in audit.events]
        self.assertNotIn("output_safety_check", events)

    def test_run_never_raises_on_content_refusal(self):
        cp, _ = _make(input_checker=_AlwaysUnsafe())
        try:
            r = cp.run("bad")
            self.assertIsInstance(r, Response)
        except Exception as e:
            self.fail(f"run() raised unexpectedly: {e}")


# ── Output refusal ─────────────────────────────────────────────────────────

class TestOutputRefusal(unittest.TestCase):

    def test_unsafe_output_returns_refusal(self):
        cp, _ = _make(output_checker=_AlwaysUnsafe())
        r = cp.run("prompt")
        self.assertFalse(r.safe)
        self.assertIsNone(r.output)
        self.assertIsNotNone(r.refusal)

    def test_output_refusal_sets_failure_mode(self):
        cp, _ = _make(output_checker=_AlwaysUnsafe())
        r = cp.run("prompt")
        self.assertEqual(r.failure_mode, "safety_violation")

    def test_output_guard_failure_mode_propagates(self):
        cp, _ = _make(
            output_checker=_AlwaysUnsafeWithMode("guard_unavailable")
        )
        r = cp.run("prompt")
        self.assertEqual(r.failure_mode, "guard_unavailable")

    def test_token_counts_present_on_output_refusal(self):
        # Token counts must be captured before output checking
        cp, _ = _make(output_checker=_AlwaysUnsafe())
        r = cp.run("prompt")
        self.assertEqual(r.tokens_in,  10)
        self.assertEqual(r.tokens_out, 20)


# ── Normalisation instability ──────────────────────────────────────────────

class TestNormalisationInstability(unittest.TestCase):

    def test_unstable_input_returns_refusal(self):
        cp, _ = _make(normaliser=_UnstableNormaliser())
        r = cp.run("adversarial input")
        self.assertFalse(r.safe)
        self.assertIsNone(r.output)

    def test_unstable_input_sets_correct_failure_mode(self):
        cp, _ = _make(normaliser=_UnstableNormaliser())
        r = cp.run("adversarial input")
        self.assertEqual(r.failure_mode, "normalisation_unstable")

    def test_llm_not_called_when_normalisation_unstable(self):
        called = []
        class _TrackingBackend(_SafeBackend):
            def chat(self, s, u, temperature=None):
                called.append(True)
                return "response"
        cp, _ = _make(
            backend    = _TrackingBackend(),
            normaliser = _UnstableNormaliser(),
        )
        cp.run("prompt")
        self.assertEqual(len(called), 0)


# ── Backend failure ────────────────────────────────────────────────────────

class TestBackendFailure(unittest.TestCase):

    def test_backend_failure_returns_response(self):
        cp, _ = _make(backend=_FailBackend())
        r = cp.run("Hello")
        self.assertFalse(r.safe)
        self.assertIsNone(r.output)
        self.assertIsNotNone(r.refusal)

    def test_backend_failure_sets_failure_mode(self):
        cp, _ = _make(backend=_FailBackend())
        r = cp.run("Hello")
        self.assertEqual(r.failure_mode, "backend_unavailable")

    def test_health_check_healthy_returns_true(self):
        cp, _ = _make(backend=_SafeBackend())
        self.assertTrue(cp.health_check())

    def test_health_check_unhealthy_raises(self):
        cp, _ = _make(backend=_FailBackend())
        with self.assertRaises(BackendUnavailableError):
            cp.health_check()


# ── Audit trail ────────────────────────────────────────────────────────────

class TestAuditTrail(unittest.TestCase):

    def test_clean_pass_events_logged(self):
        cp, audit = _make()
        cp.run("Hello")
        events = [e["event"] for e in audit.events]
        self.assertIn("input_safety_check", events)
        self.assertIn("output_safety_check", events)
        self.assertIn("run_complete", events)

    def test_refusal_logged_as_safety_refused(self):
        cp, audit = _make(input_checker=_AlwaysUnsafe())
        cp.run("bad")
        events = [e["event"] for e in audit.events]
        self.assertIn("safety_refused", events)

    def test_audit_id_consistent_across_events(self):
        cp, audit = _make()
        cp.run("Hello")
        audit_ids = [
            e.get("audit_id") for e in audit.events
            if "audit_id" in e
        ]
        self.assertTrue(len(set(audit_ids)) == 1)

    def test_failure_mode_in_run_complete(self):
        cp, audit = _make()
        cp.run("Hello")
        complete = next(
            e for e in audit.events if e["event"] == "run_complete"
        )
        self.assertIn("failure_mode", complete)
        self.assertIsNone(complete["failure_mode"])

    def test_normalisation_logged(self):
        cp, audit = _make(
            normaliser=_StableNormaliser()
        )
        cp.run("hello")
        events = [e["event"] for e in audit.events]
        self.assertIn("input_normalised", events)

    def test_token_counts_in_run_complete(self):
        cp, audit = _make()
        cp.run("Hello")
        complete = next(
            e for e in audit.events if e["event"] == "run_complete"
        )
        self.assertEqual(complete.get("tokens_in"),    10)
        self.assertEqual(complete.get("tokens_out"),   20)
        self.assertEqual(complete.get("tokens_total"), 30)


# ── Response contract ──────────────────────────────────────────────────────

class TestResponseContract(unittest.TestCase):

    def test_safe_response_fields(self):
        r = Response(output="hi", safe=True, audit_id="abc")
        self.assertTrue(r.safe)
        self.assertIsNone(r.refusal)
        self.assertIsNone(r.failure_mode)

    def test_audit_id_default_is_none(self):
        r = Response(output="hi", safe=True)
        self.assertIsNone(r.audit_id)

    def test_unsafe_response_fields(self):
        r = Response(
            output       = None,
            safe         = False,
            refusal      = "Role-play detected.",
            failure_mode = "safety_violation",
            audit_id     = "abc",
        )
        self.assertFalse(r.safe)
        self.assertIsNone(r.output)
        self.assertEqual(r.failure_mode, "safety_violation")

    def test_all_failure_mode_values_accepted(self):
        """Every documented failure_mode value is a valid Response field."""
        modes = [
            None,
            "safety_violation",
            "entropy_exceeded",
            "guard_unavailable",
            "guard_evaluation_corrupt",
            "translation_failed",
            "normalisation_unstable",
            "backend_unavailable",
        ]
        for mode in modes:
            r = Response(output=None, safe=False, failure_mode=mode, audit_id="x")
            self.assertEqual(r.failure_mode, mode)


# ── Context manager ────────────────────────────────────────────────────────

class TestContextManager(unittest.TestCase):

    def test_context_manager_closes_cleanly(self):
        cp, _ = _make()
        with cp:
            r = cp.run("Hello")
        self.assertTrue(r.safe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
