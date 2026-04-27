"""
CarapexConfig — configuration container and I/O.

CarapexConfig.load(path)         deserialises a JSON or YAML file.
CarapexConfig.write_default(path) writes a complete config with all defaults.

Neither method is part of the Carapex instance lifecycle — both are
class-level operations used before build() is called.

Config objects do not travel past the composition root (build()). build()
reads all fields and passes extracted values to components directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carapex.core.exceptions import ConfigurationError


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML file. YAML is attempted only when PyYAML is present."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigurationError(f"Cannot read config file '{path}': {e}") from e

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import]
        except ImportError as e:
            raise ConfigurationError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            ) from e
        try:
            return yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in '{path}': {e}") from e
    else:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in '{path}': {e}") from e


@dataclass
class CarapexConfig:
    """Structured configuration for a Carapex instance.

    All fields mirror the §15 config schema exactly.
    Raw dicts are used for LLM / auditor / server sub-configs because
    each registered implementation may carry implementation-specific fields.
    build() is responsible for validating and extracting those fields.
    """

    # Required
    main_llm: dict[str, Any] = field(default_factory=dict)

    # Optional LLM roles — null means fall back to main_llm
    input_guard_llm: dict[str, Any] | None = None
    output_guard_llm: dict[str, Any] | None = None
    translator_llm: dict[str, Any] | None = None

    # Sub-config blocks
    safety: dict[str, Any] = field(default_factory=dict)
    normaliser: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] | None = None

    debug: bool = False

    # ------------------------------------------------------------------
    # Class-level I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "CarapexConfig":
        """Deserialise a JSON or YAML config file into CarapexConfig.

        Raises ConfigurationError if the file is missing, unreadable,
        syntactically invalid, or missing required top-level fields.
        """
        p = Path(path)
        raw = _load_yaml_or_json(p)

        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"Config file '{p}' must contain a JSON/YAML object at the top level."
            )

        if "main_llm" not in raw or not raw["main_llm"]:
            raise ConfigurationError("Config field 'main_llm' is required.")

        return cls(
            main_llm=raw.get("main_llm", {}),
            input_guard_llm=raw.get("input_guard_llm"),
            output_guard_llm=raw.get("output_guard_llm"),
            translator_llm=raw.get("translator_llm"),
            safety=raw.get("safety") or {},
            normaliser=raw.get("normaliser") or {},
            audit=raw.get("audit") or {},
            server=raw.get("server"),
            debug=bool(raw.get("debug", False)),
        )

    @classmethod
    def write_default(cls, path: str | Path) -> None:
        """Write a complete config file with defaults for all registered components.

        The written file is a valid starting point for operators — all fields
        are present with their default values and inline comments explaining
        each security-relevant field.
        """

        default = {
            "main_llm": {
                "type": "openai",
                "url": "https://api.openai.com",
                "model": "gpt-4o",
            },
            "input_guard_llm": None,
            "output_guard_llm": None,
            "translator_llm": None,
            "safety": {
                "injection_patterns": None,
                "entropy_threshold": 5.8,
                "entropy_min_length": 50,
                "script_confidence_threshold": 0.80,
                "output_guard_enabled": True,
                "input_guard_temperature": 0.1,
                "output_guard_temperature": 0.1,
                "translation_temperature": 0.0,
                "input_guard_system_prompt_path": None,
                "output_guard_system_prompt_path": None,
            },
            "normaliser": {
                "max_passes": 5,
                "decoders": None,
            },
            "audit": {
                "type": "file",
                "path": "carapex_audit.jsonl",
            },
            "server": {
                "type": "fastapi",
                "host": "127.0.0.1",
                "port": 8000,
                "workers": 1,
            },
            "debug": False,
        }

        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
            f.write("\n")
