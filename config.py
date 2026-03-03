"""
config.py
---------
Configuration loader and discovery-driven config generation.

Responsibilities:
    - Load raw JSON from file into CarapexConfig
    - Generate config.json by asking registered components for their defaults
    - Explain config keys by asking registered components what they provide

What this file does NOT do:
    - Know about specific backend implementations
    - Know about specific audit backend implementations
    - Hardcode any config template
    - Import concrete backend or audit classes

When a new backend or audit backend is added:
    - config.py needs no changes
    - write_default() and explain() pick up new components automatically
    - The new component's default_config() defines its config keys
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .safety.base     import SafetyConfig
from .normaliser.base import NormaliserConfig
from .exceptions      import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class CarapexConfig:
    """
    Top-level config — typed data carrier only.

    backend and audit are raw dicts — providers own deserialisation.
    safety and normaliser are typed — they have no per-implementation
    variation, so early typing is safe and useful.
    """
    system_prompt : str                     = "You are a helpful assistant."
    backend       : dict                    = field(default_factory=dict)
    guard_backend : Optional[dict]          = None
    safety        : SafetyConfig            = field(default_factory=SafetyConfig)
    normaliser    : NormaliserConfig        = field(default_factory=NormaliserConfig)
    audit         : dict                    = field(default_factory=lambda: {"backend": "file"})
    debug         : bool                    = False

    def __post_init__(self):
        if not self.system_prompt or not self.system_prompt.strip():
            raise ConfigurationError("system_prompt cannot be empty")


def load(path: str) -> CarapexConfig:
    """
    Load config from JSON file.

    Raises:
        FileNotFoundError    : path does not exist
        json.JSONDecodeError : invalid JSON
        ConfigurationError   : config validation failed
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    return from_dict(raw)


def from_dict(raw: dict) -> CarapexConfig:
    """
    Build CarapexConfig from raw dict.
    backend and audit passed through as raw dicts — not deserialised here.
    safety and normaliser deserialised here — no per-implementation variation.
    """
    try:
        safety_raw = raw.get("safety") or {}
        # Treat empty injection_patterns list as None — security-first default.
        # An empty list is almost certainly a misconfiguration (no patterns = no
        # deterministic gate). Coercion belongs here, not in __post_init__.
        if safety_raw.get("injection_patterns") == []:
            safety_raw = {**safety_raw, "injection_patterns": None}
        return CarapexConfig(
            system_prompt = raw.get("system_prompt", "You are a helpful assistant."),
            backend       = raw.get("backend") or {},
            guard_backend = raw.get("guard_backend"),
            safety        = SafetyConfig(**safety_raw),
            normaliser    = NormaliserConfig(**(raw.get("normaliser") or {})),
            audit         = raw.get("audit") or {"backend": "file"},
            debug         = raw.get("debug", False),
        )
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e


def default() -> CarapexConfig:
    """Return fully default config. Useful for testing."""
    return CarapexConfig()


def _collect_backend_defaults() -> dict:
    """
    Ask the default backend (openai_compatible) for its config keys.
    Discovery-driven — config.py never imports concrete backend classes.
    """
    from . import backends
    from .exceptions import PluginNotFoundError
    try:
        cls = backends.resolve("openai_compatible")
        return cls.default_config()
    except PluginNotFoundError:
        logger.warning("openai_compatible backend not registered — backend config will be empty")
        return {}


def _collect_audit_defaults(name: str = "file") -> dict:
    """
    Ask the named audit backend for its config keys.
    Discovery-driven — config.py never imports concrete audit classes.
    """
    from . import audit as audit_reg
    from .exceptions import PluginNotFoundError
    try:
        cls = audit_reg.resolve(name)
        cfg = cls.default_config()
        cfg["backend"] = name
        return cfg
    except PluginNotFoundError:
        logger.warning("audit backend %r not registered — audit config will be minimal", name)
        return {"backend": name}


def write_default(path: str) -> None:
    """
    Write a default config.json to path.

    Discovery-driven — queries registered components for their defaults.
    Adding a new backend or audit backend requires no changes here.
    """
    import dataclasses

    backend_defaults  = _collect_backend_defaults()
    audit_defaults    = _collect_audit_defaults("file")

    normaliser_defaults = {
        f.name: (
            f.default
            if f.default is not dataclasses.MISSING
            else f.default_factory()
        )
        for f in dataclasses.fields(NormaliserConfig)
    }

    # Build safety section from SafetyConfig fields — discovery-driven.
    # Adding a field to SafetyConfig causes it to appear here automatically.
    safety_defaults = {}
    for f in dataclasses.fields(SafetyConfig):
        if f.default is not dataclasses.MISSING:
            safety_defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            safety_defaults[f.name] = f.default_factory()

    template = {
        "system_prompt": "You are a helpful assistant.",
        "debug"        : False,
        "backend"      : backend_defaults,
        "guard_backend": None,
        "normaliser"   : normaliser_defaults,
        "safety"       : safety_defaults,
        "audit"        : audit_defaults,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=4)


def explain() -> None:
    """
    Print all config keys with types and defaults.
    Discovery-driven — queries registered components.
    """
    from . import backends, audit as audit_reg

    print("\ncarapex configuration reference")
    print("─" * 60)

    _print_section("system_prompt", {
        "system_prompt": ("str", "You are a helpful assistant.",
                          "System prompt for the main LLM"),
        "debug"        : ("bool", "false",
                          "Log structural debug metadata (never sensitive values)"),
    })

    print("\n  [backend]")
    print("  'type' is required — it selects the backend implementation.")
    print("  Available types: " + ", ".join(f'"{n}"' for n in backends.available()))
    for name in backends.available():
        cls = backends.resolve(name)
        cfg = cls.default_config()
        print(f"\n    type = \"{name}\"")
        for key, val in cfg.items():
            if key == "type":
                continue
            print(f"      backend.{key:<30} default: {str(val)}")

    print("\n  [guard_backend]")
    print("  Separate backend for guard model (recommended for production).")
    print("  null = uses main backend (development only).")
    print("  Same keys as backend — requires 'type' field.")
    print("  WARNING: guard_backend: null means guard and main LLM share resources.")
    print("  A loaded main LLM may cause guard timeouts under load.")

    print("\n  [normaliser]")
    print(f"    normaliser.max_passes               default: 5")
    print(f"    normaliser.decoders                 default: [whitespace, unicode_escape, html_entity, url, base64, homoglyph]")

    print("\n  [safety]")
    print(f"    safety.injection_patterns           default: null (uses built-in patterns)")
    print(f"                                        null or [] = use built-in DEFAULT_PATTERNS")
    print(f"                                        [...] = replace defaults entirely")
    print(f"                                        Invalid patterns raise ConfigurationError at startup")
    print(f"    safety.entropy_threshold            default: null (entropy gating disabled)")
    print(f"                                        null = entropy gating disabled entirely")
    print(f"                                        0.0  = blocks everything non-trivial (not the same as null)")
    print(f"                                        Recommended range: 3.5–5.5 for most deployments")

    print("\n  [audit]")
    print(f"    audit.backend                       options: {audit_reg.available()}")
    for name in audit_reg.available():
        cls = audit_reg.resolve(name)
        cfg = cls.default_config()
        if cfg:
            for key, val in cfg.items():
                print(f"    audit.{key:<34} default: {str(val)}")
    print()


def _print_section(title: str, fields: dict) -> None:
    for key, (typ, default_val, desc) in fields.items():
        print(f"  {key:<40} {typ:<8} default: {str(default_val):<25} {desc}")
