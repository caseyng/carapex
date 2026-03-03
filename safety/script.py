"""
safety/script.py
----------------
Language and script detection — informational layer feeding TranslationLayer.

This checker does NOT block requests. Its role is to:
    1. Detect the language of the (normalised) input
    2. Tag the SafetyResult with detected language metadata
    3. Signal to TranslationLayer whether translation is needed

Why not a hard gate:
    Carapex is designed for multilingual deployments. Non-Latin script
    is not inherently suspicious — it is the user's language. The correct
    response to non-English input is translation for evaluation, not rejection.

    Rejection based on script alone would be:
        - A false security measure (attackers can write in English)
        - Discriminatory against legitimate non-English users
        - Inconsistent with the multilingual design goal

    The guard always evaluates English. TranslationLayer ensures it gets
    English. ScriptChecker enables that by detecting what language arrived.

Fail-closed on detection failure:
    If langdetect fails to identify the language (raises any exception during
    detection), the result signals translation_needed=True — TranslationLayer
    will attempt translation regardless. Better to translate unnecessarily than
    to pass unidentified content directly to the guard.

Dependency:
    langdetect — pure Python, offline, no tokens.
    Install: pip install langdetect
    Fixed seed (DetectorFactory.seed = 0) for deterministic results.
    If langdetect is not installed, raises ConfigurationError at construction.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .base import SafetyChecker, SafetyResult
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ScriptResult(SafetyResult):
    """
    Extended SafetyResult carrying language detection metadata.
    Always safe=True — ScriptChecker never blocks.
    TranslationLayer reads detected_language to decide whether to translate.
    """
    detected_language:  Optional[str] = None   # ISO 639-1 code e.g. "en", "fr", "zh"
    translation_needed: bool          = False   # True if not English / detection failed


class ScriptChecker(SafetyChecker):
    """
    Detects input language using langdetect.
    Always returns safe=True — never blocks.
    Passes language metadata to TranslationLayer via ScriptResult.
    """

    name = "script"

    # Languages considered "English" — no translation needed.
    # en only. Dialects (en-gb, en-au) are returned as "en" by langdetect.
    _ENGLISH = {"en"}

    def __init__(self):
        try:
            from langdetect import DetectorFactory
            DetectorFactory.seed = 0  # deterministic results
            self._detect = self._import_detect()
        except ImportError:
            raise ConfigurationError(
                "langdetect is required for multilingual support. "
                "Install it with: pip install langdetect"
            )

    @staticmethod
    def _import_detect():
        from langdetect import detect
        return detect

    def check(self, text: str) -> ScriptResult:
        """
        Detect language. Always returns safe=True.
        Sets translation_needed=True for non-English or undetectable input.
        """
        if not text or not text.strip():
            return ScriptResult(
                safe               = True,
                detected_language  = "en",
                translation_needed = False,
            )

        try:
            lang = self._detect(text)
            is_english = lang in self._ENGLISH

            if not is_english:
                logger.debug(
                    "Non-English input detected: lang=%s length=%d — "
                    "will translate before guard evaluation",
                    lang, len(text),
                )

            return ScriptResult(
                safe               = True,
                detected_language  = lang,
                translation_needed = not is_english,
            )

        except Exception:
            # Any exception during detection — treat as unknown, flag for translation.
            # Fail toward translation rather than toward bypassing it.
            logger.warning(
                "Language detection failed for input (length=%d) — "
                "flagging for translation to be safe",
                len(text),
            )
            return ScriptResult(
                safe               = True,
                detected_language  = None,
                translation_needed = True,
            )

    def __repr__(self) -> str:
        return "ScriptChecker()"
