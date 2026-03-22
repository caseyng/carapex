"""
ScriptChecker — detects input language and signals whether translation is needed.

Uses lingua-language-detector (>=2.0) for robust detection on short texts.
Always returns safe=True — it is a detector, not a gate.

The confidence threshold defaults to 0.80 — the failure direction is deliberate:
under-confidence triggers an unnecessary translation call (cheap, safe);
over-confidence on a non-English prompt sends untranslated text to the input
guard, which is the worse outcome (§15 rationale).
"""

from __future__ import annotations

import logging

from carapex.core.types import ScriptResult
from carapex.safety.base import SafetyChecker

log = logging.getLogger(__name__)


class ScriptChecker(SafetyChecker):
    """Stateless between calls. Language detection library initialised once at construction."""

    name = "script"

    def __init__(self, confidence_threshold: float = 0.80) -> None:
        self._threshold = confidence_threshold
        self._detector = self._build_detector()

    @staticmethod
    def _build_detector() -> object:
        """Build the lingua detector. Raises ImportError if lingua is absent."""
        try:
            from lingua import LanguageDetectorBuilder  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "lingua-language-detector is required. "
                "Install it with: pip install lingua-language-detector"
            ) from e
        return (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )

    def inspect(self, text: str) -> ScriptResult:
        if text is None:
            raise ValueError("ScriptChecker.inspect() received None")

        try:
            from lingua import Language  # type: ignore[import]

            detected = self._detector.detect_language_of(text)  # type: ignore[attr-defined]

            if detected is None:
                return ScriptResult(detected_language=None, translation_needed=True)

            # compute_language_confidence(text, language) → float in [0, 1]
            confidence: float = self._detector.compute_language_confidence(  # type: ignore[attr-defined]
                text, detected
            )

            lang_code = detected.iso_code_639_1.name.lower()
            is_english = (detected == Language.ENGLISH)

            if is_english and confidence >= self._threshold:
                return ScriptResult(detected_language=lang_code, translation_needed=False)

            return ScriptResult(detected_language=lang_code, translation_needed=True)

        except Exception as e:  # noqa: BLE001 — never raise; translation_needed=True is the safe fallback
            log.debug(
                "ScriptChecker: language detection raised %s — defaulting to translation_needed=True", e
            )
            return ScriptResult(detected_language=None, translation_needed=True)

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"ScriptChecker(confidence_threshold={self._threshold!r})"
