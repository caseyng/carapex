"""
safety/output.py
----------------
Output safety checker — evaluates LLM response before returning to caller.

Symmetric to input safety. Catches:
    - Jailbreak success: model responded to adversarial input
    - Hallucinated instructions directed at caller
    - Harmful content generated despite safe input
    - Prompt injection in output targeting downstream systems

Does NOT attempt to detect factual hallucinations — that requires
ground truth and is out of scope for middleware.

Skipped automatically when input was refused — nothing to check.
"""

import re
from typing import List

from .base import SafetyChecker, SafetyResult

# Patterns that suggest the LLM was manipulated or is outputting
# instructions rather than responses
_OUTPUT_PATTERNS = [
    r"(as\s+an?\s+)?(unrestricted|jailbroken|DAN|uncensored)\s+(ai|model|version)",
    r"i\s+(am\s+now|have\s+been)\s+(freed|unlocked|unrestricted)",
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"my\s+new\s+instructions\s+are",
    r"system\s+prompt\s*:",
    r"<\s*system\s*>",
    r"\[INST\]",
]


class OutputSafetyChecker(SafetyChecker):
    """
    Checks LLM output for signs of jailbreak success or harmful generation.
    Pattern-based — fast, no additional LLM call needed.
    """

    name = "output"

    def __init__(self):
        self._patterns: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in _OUTPUT_PATTERNS
        ]

    def check(self, text: str) -> SafetyResult:
        """
        Scan LLM output for concerning patterns.
        Returns SafetyResult(safe=True) for None or empty input.
        Never raises on evaluation — propagates on internal bugs.
        """
        if not text:
            return SafetyResult(safe=True)

        for pattern in self._patterns:
            m = pattern.search(text)
            if m:
                return SafetyResult(
                    safe   = False,
                    reason = (
                        f"Output safety: response contains suspicious pattern "
                        f"'{m.group(0)}' — possible jailbreak success"
                    ),
                )
        return SafetyResult(safe=True)

    def __repr__(self) -> str:
        return f"OutputSafetyChecker(patterns={len(self._patterns)})"
