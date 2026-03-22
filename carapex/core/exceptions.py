"""
carapex exception hierarchy.

Rule: evaluate() raises only PipelineInternalError (component bug).
      All operational failures are returned in EvaluationResult.
      Programming errors (precondition violations) raise ValueError / TypeError.
      Configuration errors raise ConfigurationError at build() time.

CarapexViolation, IntegrityFailure, NormalisationError are *caller-raised* only.
They are never raised internally. They exist for callers who prefer
exception-based control flow over branching on EvaluationResult.failure_mode.
"""

from __future__ import annotations


class ConfigurationError(Exception):
    """Raised by build() when the supplied configuration is invalid.

    The error message always identifies the field that caused the failure.
    Never raised at runtime — all config validation occurs at build() time.
    """


class PipelineInternalError(Exception):
    """Raised by evaluate() when a component raises an unexpected exception.

    Indicates a bug in a component — not an operational failure.
    The Carapex instance MUST NOT be reused after this is raised.

    Attributes:
        component_name: the name of the component that raised.
        original: the original exception.
    """

    def __init__(self, component_name: str, original: Exception) -> None:
        self.component_name = component_name
        self.original = original
        super().__init__(
            f"Component '{component_name}' raised an unexpected exception: "
            f"{type(original).__name__}: {original}"
        )


# ---------------------------------------------------------------------------
# Caller-raised only — never raised by evaluate() internally
# ---------------------------------------------------------------------------

class CarapexViolation(Exception):
    """The prompt was evaluated and refused (content refusal).

    Callers MAY raise this from an EvaluationResult where failure_mode is
    'safety_violation' or 'entropy_exceeded'.
    Never raised by evaluate() itself.
    """


class IntegrityFailure(Exception):
    """A check component failed — the prompt was not fully evaluated.

    Callers MAY raise this from an EvaluationResult where failure_mode is
    'guard_unavailable', 'guard_evaluation_corrupt', or 'translation_failed'.
    Must not be treated as a content decision — do not retry without
    investigating the underlying component failure.
    Never raised by evaluate() itself.
    """


class NormalisationError(Exception):
    """The input did not converge during decoding.

    Callers MAY raise this from an EvaluationResult where failure_mode is
    'normalisation_unstable'.
    Never raised by evaluate() itself.
    """
