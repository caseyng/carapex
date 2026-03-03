"""
carapex
--------
Hardened prompt execution boundary.
Security is the boundary itself — not a layer bolted on top.

Quick start:
    import carapex

    cp = carapex.build("config.json")
    r  = cp.run("Summarise this: ...")

    if r.failure_mode in (
        "guard_unavailable",
        "guard_evaluation_corrupt",
        "translation_failed",
    ):
        # Shell integrity failure — alert, do not retry as content problem
        alert_oncall(r.failure_mode)
        return service_unavailable()

    elif r.failure_mode == "normalisation_unstable":
        # Possible adversarial input — log as security event
        log_security_event(r)
        return service_unavailable()

    elif not r.safe:
        # Content refusal — safety_violation or entropy_exceeded
        return r.refusal

    else:
        return r.output

First time:
    python -m carapex --init        # write config.json
    python -m carapex --show-config # explain all keys
    python -m carapex --check       # health check

IMPORTANT — always branch on failure_mode before acting on safe.
safe=False covers content refusals AND infrastructure failures.
These are different operational conditions. Conflating them is dangerous.

Exception-based handling (optional):
    from carapex.exceptions import CarapexViolation, IntegrityFailure, NormalisationError

    r = cp.run(prompt)
    if r.failure_mode in ("guard_unavailable", "guard_evaluation_corrupt", "translation_failed"):
        raise IntegrityFailure(r.failure_mode)
    if r.failure_mode == "normalisation_unstable":
        raise NormalisationError(r.failure_mode)
    if not r.safe:
        raise CarapexViolation(r.refusal, r.failure_mode)
    return r.output
"""

from .processor  import Carapex, Response
from .exceptions import (
    CarapexError,
    ConfigurationError,
    BackendUnavailableError,
    PipelineInternalError,
    CarapexViolation,
    IntegrityFailure,
    NormalisationError,
    PluginNotFoundError,
)
from . import config


def build(cfg=None) -> Carapex:
    """
    Build a fully wired Carapex instance.

    Composition root — only place that wires providers together.
    Health checks guard backend at startup — under fail-closed semantics,
    an unreachable guard means every request is rejected. Catch at startup.

    Args:
        cfg : CarapexConfig | dict | path string | None (uses defaults)
    """
    from . import config as _config
    from .providers import (
        provide_backend,
        provide_normaliser,
        provide_input_checker,
        provide_output_checker,
        provide_audit,
    )
    from .exceptions import BackendUnavailableError

    if cfg is None:
        _cfg = _config.default()
    elif isinstance(cfg, str):
        _cfg = _config.load(cfg)
    elif isinstance(cfg, dict):
        _cfg = _config.from_dict(cfg)
    else:
        _cfg = cfg

    _backend       = provide_backend(_cfg.backend)
    _guard_backend = (
        provide_backend(_cfg.guard_backend)
        if _cfg.guard_backend
        else _backend
    )
    _normaliser     = provide_normaliser(_cfg.normaliser)
    _input_checker  = provide_input_checker(
        _cfg.safety, _guard_backend, debug=_cfg.debug
    )
    _output_checker = provide_output_checker(
        _cfg.safety, _guard_backend, debug=_cfg.debug
    )
    _audit = provide_audit(_cfg.audit)

    # Health check guard backend at startup.
    #
    # Guard unavailability is a startup blocker — under fail-closed semantics,
    # a dead guard means every request is rejected with no useful work done.
    # Surface this immediately rather than at first request.
    #
    # Main backend is NOT checked here by design. If it is down, run() returns
    # failure_mode="backend_unavailable" on every call — a clear operational signal.
    # This asymmetry is intentional: a dead main LLM is an infrastructure outage,
    # not a security failure. Operators should monitor backend_unavailable responses
    # in metrics/alerting rather than treating build() success as a main backend
    # health guarantee.
    if not _guard_backend.health_check():
        raise BackendUnavailableError(
            f"Guard backend {_guard_backend!r} is not ready. "
            f"Carapex cannot start — every request would be rejected. "
            f"Ensure the guard backend is running before starting."
        )

    return Carapex(
        system_prompt  = _cfg.system_prompt,
        backend        = _backend,
        input_checker  = _input_checker,
        output_checker = _output_checker,
        normaliser     = _normaliser,
        audit          = _audit,
        debug          = _cfg.debug,
        guard_backend  = _guard_backend if _guard_backend is not _backend else None,
    )


__all__ = [
    "Carapex",
    "Response",
    "build",
    "config",
    "CarapexError",
    "ConfigurationError",
    "BackendUnavailableError",
    "PipelineInternalError",
    "CarapexViolation",
    "IntegrityFailure",
    "NormalisationError",
    "PluginNotFoundError",
]
