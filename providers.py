"""
providers.py
------------
Explicit provider functions. Each owns construction for one component.

Edit ONLY when:
    - Adding a new component type to the pipeline
    - Changing how existing components are composed or ordered

Do NOT edit when:
    - Adding a new backend        → drop file in backends/
    - Adding a new audit backend  → drop file in audit/
    - Adding a new decoder        → drop file in normaliser/
    - Adding a new safety checker → drop file in safety/, wire here if needed

Rules:
    - Each provider receives only the config slice it needs
    - Each provider returns an abstraction, never a concrete class
    - No provider knows about other providers
    - build() in __init__.py is the only caller
"""

from .backends.base         import LLMBackend
from .safety.base           import SafetyChecker, SafetyConfig
from .audit.base            import AuditBackend
from .normaliser.base       import NormaliserConfig, Decoder
from .normaliser.normaliser import Normaliser


def provide_backend(raw: dict) -> LLMBackend:
    """
    Resolve and instantiate LLM backend by explicit "type" key.

    "type" is required in the backend config dict. It maps directly to the
    registered backend name (e.g. "openai_compatible", "llama_cpp").

    Raises ConfigurationError if "type" is absent or unrecognised.
    Shape-based inference was removed — it was fragile as the backend set grew
    and would silently select the wrong backend if two backends shared a field name.
    """
    from . import backends
    from .exceptions import ConfigurationError, PluginNotFoundError

    backend_type = raw.get("type")
    if not backend_type:
        raise ConfigurationError(
            "backend config requires a 'type' field. "
            f"Available backends: {backends.available()}. "
            "Example: {\"type\": \"openai_compatible\", \"base_url\": \"http://localhost:8080\"}"
        )

    try:
        cls = backends.resolve(backend_type)
    except PluginNotFoundError:
        raise ConfigurationError(
            f"Unknown backend type {backend_type!r}. "
            f"Available backends: {backends.available()}"
        )

    return cls.from_config(raw)


def provide_normaliser(cfg: NormaliserConfig) -> Normaliser:
    """
    Resolve decoder classes, instantiate each, return configured Normaliser.
    Decoder order follows cfg.decoders list.
    """
    from . import normaliser as norm_reg
    from .exceptions import ConfigurationError, PluginNotFoundError

    decoders: list[Decoder] = []
    for name in cfg.decoders:
        try:
            cls = norm_reg.resolve(name)
        except PluginNotFoundError as e:
            raise ConfigurationError(
                f"Unknown decoder {name!r} in normaliser.decoders. "
                f"Available: {norm_reg.available()}"
            ) from e
        decoders.append(cls())

    return Normaliser(decoders=decoders, max_passes=cfg.max_passes)


def provide_input_checker(
    cfg:     SafetyConfig,
    backend: LLMBackend,
    debug:   bool = False,
) -> SafetyChecker:
    """
    Build composite input safety checker.

    Pipeline order (cheapest → most expensive):
        1. PatternChecker    — structural token detection, deterministic, no LLM
        2. EntropyChecker    — statistical anomaly, deterministic, no LLM
        3. ScriptChecker     — language detection, deterministic, no LLM
        4. TranslationLayer  — non-English → English, LLM at temp 0.0
        5. GuardChecker      — semantic evaluation on English, LLM at temp 0.1

    Edit this function to change checker composition or order.
    """
    from .safety.pattern     import PatternSafetyChecker
    from .safety.entropy     import EntropyChecker
    from .safety.script      import ScriptChecker
    from .safety.translation import TranslationLayer
    from .safety.guard       import GuardSafetyChecker
    from .safety.composite   import CompositeSafetyChecker

    return CompositeSafetyChecker([
        PatternSafetyChecker(cfg.injection_patterns),
        EntropyChecker(cfg.entropy_threshold, cfg.entropy_min_length),
        ScriptChecker(),
        TranslationLayer(backend),
        GuardSafetyChecker(backend, debug=debug),
    ])


def provide_output_checker(
    cfg:     SafetyConfig,
    backend: LLMBackend,
    debug:   bool = False,
) -> SafetyChecker:
    """
    Build composite output safety checker.

    Pipeline:
        1. OutputPatternChecker — pattern-based, always runs, no LLM
        2. OutputGuardChecker   — semantic eval, on by default, configurable

    output_guard_enabled in SafetyConfig controls whether OutputGuardChecker
    is included. Default True — operators disable explicitly to reduce cost.
    Disabling is a conscious reduction in protection.
    """
    from .safety.output      import OutputSafetyChecker
    from .safety.output_guard import OutputGuardChecker
    from .safety.composite   import CompositeSafetyChecker

    checkers = [OutputSafetyChecker()]

    if cfg.output_guard_enabled:
        checkers.append(OutputGuardChecker(backend, debug=debug))

    return CompositeSafetyChecker(checkers)


def provide_audit(raw: dict) -> AuditBackend:
    """Resolve and instantiate audit backend."""
    from . import audit as audit_reg
    name = raw.get("backend", "file")
    cls  = audit_reg.resolve(name)
    return cls.from_config(raw)
