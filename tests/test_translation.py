"""
tests/test_translation.py
--------------------------
Unit tests for ScriptChecker and TranslationLayer.

langdetect is required. Tests are skipped if not installed.
Install: pip install langdetect
"""

import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

from carapex.safety.base import SafetyResult


def _backend(response):
    class _B:
        def chat(self, s, u, temperature=None): return response
        def __repr__(self): return "StubBackend()"
    return _B()


# ── ScriptChecker ──────────────────────────────────────────────────────────

@unittest.skipUnless(LANGDETECT_AVAILABLE, "langdetect not installed")
class TestScriptChecker(unittest.TestCase):

    def setUp(self):
        from carapex.safety.script import ScriptChecker
        self.checker = ScriptChecker()

    def test_english_no_translation_needed(self):
        from carapex.safety.script import ScriptResult
        r = self.checker.check("Hello, how are you today?")
        self.assertIsInstance(r, ScriptResult)
        self.assertTrue(r.safe)
        self.assertFalse(r.translation_needed)
        self.assertEqual(r.detected_language, "en")

    def test_french_translation_needed(self):
        from carapex.safety.script import ScriptResult
        r = self.checker.check(
            "Bonjour, comment allez-vous? Je suis très bien merci."
        )
        self.assertTrue(r.safe)           # never blocks
        self.assertTrue(r.translation_needed)
        self.assertNotEqual(r.detected_language, "en")

    def test_german_translation_needed(self):
        r = self.checker.check(
            "Guten Morgen, wie geht es Ihnen? Ich bin sehr gut."
        )
        self.assertTrue(r.safe)
        self.assertTrue(r.translation_needed)

    def test_empty_text_treated_as_english(self):
        from carapex.safety.script import ScriptResult
        r = self.checker.check("")
        self.assertTrue(r.safe)
        self.assertFalse(r.translation_needed)

    def test_always_returns_safe_true(self):
        """ScriptChecker is a detector, not a gate — never blocks."""
        r = self.checker.check("Texto en español es muy interesante.")
        self.assertTrue(r.safe)

    def test_repr_informative(self):
        r = repr(self.checker)
        self.assertIn("ScriptChecker", r)


# ── TranslationLayer ───────────────────────────────────────────────────────

@unittest.skipUnless(LANGDETECT_AVAILABLE, "langdetect not installed")
class TestTranslationLayer(unittest.TestCase):

    def test_english_input_not_translated(self):
        """English input skips the translation call."""
        from carapex.safety.translation import TranslationLayer
        called = []
        class _Tracking:
            def chat(self, s, u, temperature=None):
                called.append(u)
                return "translated"
        tl = TranslationLayer(_Tracking())
        r = tl.check("Hello, how are you today?")
        self.assertTrue(r.safe)
        self.assertEqual(len(called), 0)

    def test_non_english_is_translated(self):
        """Non-English input calls the backend."""
        from carapex.safety.translation import TranslationLayer
        called = []
        class _Tracking:
            def chat(self, s, u, temperature=None):
                called.append(True)
                return "This is the translation."
        tl = TranslationLayer(_Tracking())
        r = tl.check("Bonjour, comment allez-vous? Je suis très bien merci.")
        self.assertTrue(r.safe)
        self.assertEqual(len(called), 1)

    def test_translation_stored_in_get_output_text(self):
        from carapex.safety.translation import TranslationLayer
        tl = TranslationLayer(_backend("English translation here."))
        tl.check("Bonjour, comment allez-vous? Je suis très bien merci.")
        translated = tl.get_output_text()
        self.assertIsNotNone(translated)

    def test_backend_failure_fails_closed(self):
        from carapex.safety.translation import TranslationLayer
        tl = TranslationLayer(_backend(None))
        r = tl.check("Bonjour, comment allez-vous? Je suis très bien merci.")
        self.assertFalse(r.safe)
        self.assertEqual(r.failure_mode, "translation_failed")

    def test_uses_temperature_zero(self):
        """Translation must use temperature 0.0 — transcription, not generation."""
        from carapex.safety.translation import TranslationLayer
        received = []
        class _Cap:
            def chat(self, s, u, temperature=None):
                received.append(temperature)
                return "English translation."
        tl = TranslationLayer(_Cap())
        tl.check("Bonjour, comment allez-vous? Je suis très bien merci.")
        if received:
            self.assertEqual(received[0], 0.0)

    def test_get_output_text_none_before_check(self):
        from carapex.safety.translation import TranslationLayer
        tl = TranslationLayer(_backend("translation"))
        self.assertIsNone(tl.get_output_text())

    def test_set_prior_result_reads_script_result(self):
        """set_prior_result with translation_needed=False skips backend call."""
        from carapex.safety.translation import TranslationLayer
        from carapex.safety.script import ScriptResult
        called = []
        class _Cap:
            def chat(self, s, u, temperature=None):
                called.append(True)
                return "translation"
        tl = TranslationLayer(_Cap())
        script_result = ScriptResult(safe=True, detected_language="en", translation_needed=False)
        tl.set_prior_result(script_result)
        r = tl.check("Hello, how are you?")
        self.assertTrue(r.safe)
        self.assertEqual(len(called), 0, "Backend should not be called when ScriptResult says no translation needed")

    def test_repr_informative(self):
        from carapex.safety.translation import TranslationLayer
        tl = TranslationLayer(_backend("x"))
        self.assertIn("TranslationLayer", repr(tl))


if __name__ == "__main__":
    unittest.main(verbosity=2)
