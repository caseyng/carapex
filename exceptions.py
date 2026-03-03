"""
exceptions.py
-------------
Exception hierarchy for carapex.

Rules:
    - Catch only what you can meaningfully handle.
    - Never catch Exception broadly.
    - Each exception carries enough context to act on.
    - Exception clarity is a security property — vague exceptions
      lead to incorrect handling in security-critical code paths.

Hierarchy:
    CarapexError
    ├── ConfigurationError       — bad config at startup, pipeline must not start
    ├── BackendUnavailableError  — backend unreachable or not ready
    ├── PipelineInternalError    — bug in a pipeline component
    │   carries: component: str, original: Exception
    ├── CarapexViolation        — for callers who want to raise on safety refusal
    │   carries: reason: str, failure_mode: str
    ├── IntegrityFailure         — for callers who want to raise on shell component failure
    │   carries: failure_mode: str
    │   covers:  guard_unavailable, guard_evaluation_corrupt,
    │            translation_failed, normalisation_unstable
    ├── NormalisationError       — for callers who want to raise on normalisation failure
    │   carries: failure_mode: str
    └── PluginNotFoundError      — registry resolve failed
        carries: family, name, available: list

Caller conventions:
    CarapexViolation, IntegrityFailure, and NormalisationError are never
    raised internally. run() always returns a Response. Callers raise these
    themselves if their architecture requires exception-based handling.

    Always read Response.failure_mode before deciding which exception to raise.
    failure_mode is the authoritative discriminator — do not infer from reason text.

    Example:
        r = cp.run(prompt)

        if r.failure_mode in (
            "guard_unavailable",
            "guard_evaluation_corrupt",
            "translation_failed",
        ):
            raise IntegrityFailure(r.failure_mode)

        if r.failure_mode == "normalisation_unstable":
            raise NormalisationError(r.failure_mode)

        if not r.safe:
            raise CarapexViolation(r.refusal, r.failure_mode)

Why distinct exception types matter in a security context:
    Conflating IntegrityFailure with CarapexViolation in a catch block
    means infrastructure failures are silently treated as safety refusals —
    or worse, retried as if the content was the problem. These are different
    operational conditions requiring different responses. The exception
    hierarchy enforces that distinction at the type level.
"""


class CarapexError(Exception):
    """Base for all carapex errors."""


class ConfigurationError(CarapexError):
    """
    Bad configuration at startup. Pipeline must not start.

    Raised by:
        - PatternSafetyChecker  — invalid regex pattern
        - CarapexConfig        — empty system_prompt
        - NormaliserConfig      — invalid max_passes, empty decoders
        - providers.py          — unknown decoder name in config
        - build()               — guard backend unreachable at startup
    """


class BackendUnavailableError(CarapexError):
    """Backend cannot be reached or is not ready."""


class PipelineInternalError(CarapexError):
    """
    A pipeline component failed internally — decoder bug, checker bug,
    impossible evaluation state.

    Distinct from operational failures (guard unavailable, network timeout).
    Never raised for expected domain failures or infrastructure outages.
    Carries the component name and original exception for diagnosis.
    """
    def __init__(self, component: str, original: Exception):
        self.component = component
        self.original  = original
        super().__init__(
            f"Internal failure in {component}: {type(original).__name__}: {original}"
        )


class CarapexViolation(CarapexError):
    """
    Raised by callers who want an exception on safety refusal.
    Never raised internally — run() always returns a Response.

    Only raise this when failure_mode is "safety_violation" or
    "entropy_exceeded" — genuine content-based refusals.
    Do not raise for infrastructure failures — use IntegrityFailure.

    Usage:
        r = cp.run(prompt)
        if not r.safe and r.failure_mode == "safety_violation":
            raise CarapexViolation(r.refusal, r.failure_mode)
    """
    def __init__(self, reason: str, failure_mode: str):
        self.reason       = reason
        self.failure_mode = failure_mode
        super().__init__(f"Carapex violation [{failure_mode}]: {reason}")


class IntegrityFailure(CarapexError):
    """
    Raised by callers who want an exception when a shell component could not
    complete evaluation. Never raised internally — failure_mode in Response
    is the primary signal.

    Covers any condition where carapex cannot guarantee the prompt was
    fully evaluated — guard unavailable, guard output corrupt, translation
    failed. These are infrastructure failures, not content refusals.

    The distinction matters: a CarapexViolation means the prompt was
    evaluated and found unsafe. An IntegrityFailure means evaluation could
    not be completed. These require different operational responses.

    Usage:
        r = cp.run(prompt)
        if r.failure_mode in (
            "guard_unavailable",
            "guard_evaluation_corrupt",
            "translation_failed",
        ):
            raise IntegrityFailure(r.failure_mode)
    """
    def __init__(self, failure_mode: str):
        self.failure_mode = failure_mode
        super().__init__(
            f"Carapex shell integrity failure: {failure_mode}. "
            f"Prompt evaluation could not be completed. "
            f"Investigate and restore the affected component."
        )


class NormalisationError(CarapexError):
    """
    Raised by callers who want an exception when input normalisation
    did not stabilise. Never raised internally.

    Unstable normalisation means the input did not converge to a canonical
    form within max_passes. This is either an adversarially constructed
    input designed to exhaust the normaliser, or a decoder design defect
    (oscillating custom decoder). Either way the input cannot be trusted
    to have been fully decoded before safety evaluation.

    Usage:
        r = cp.run(prompt)
        if r.failure_mode == "normalisation_unstable":
            raise NormalisationError(r.failure_mode)
    """
    def __init__(self, failure_mode: str):
        self.failure_mode = failure_mode
        super().__init__(
            f"Input normalisation did not stabilise: {failure_mode}. "
            f"Input may be adversarially constructed. "
            f"Check decoder composition if this occurs on legitimate input."
        )


class PluginNotFoundError(CarapexError):
    """Registry could not find a registered implementation."""
    def __init__(self, family: str, name: str, available: list):
        self.family    = family
        self.name      = name
        self.available = available
        super().__init__(
            f"No '{name}' registered in {family}. "
            f"Available: {available}"
        )
