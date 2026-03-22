"""
Registry and autodiscovery for carapex extension points.

Each extension category has its own registry dict keyed by the component's
class-level `name` attribute. Autodiscovery imports all .py files in the
appropriate package directory at import time.

Adding a new implementation:
  1. Create a .py file in the relevant package (backends/, safety/, audit/,
     normaliser/, server/).
  2. Set a unique `name` class attribute.
  3. The registry picks it up automatically — no other files need editing.

Duplicate names raise RuntimeError at import time (before any build() call).
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any


# Registry dicts: name → class
_llm_registry: dict[str, type] = {}
_checker_registry: dict[str, type] = {}
_decoder_registry: dict[str, type] = {}
_auditor_registry: dict[str, type] = {}
_server_registry: dict[str, type] = {}


def _register(registry: dict[str, type], cls: type) -> None:
    name: str = getattr(cls, "name", None)  # type: ignore[assignment]
    if name is None:
        raise RuntimeError(
            f"Component class {cls.__qualname__} has no 'name' class attribute. "
            "Every registered component must declare a unique 'name'."
        )
    if name in registry:
        raise RuntimeError(
            f"Duplicate component name '{name}': "
            f"{registry[name].__qualname__} and {cls.__qualname__} "
            "both declare the same name. Names must be unique within a registry."
        )
    registry[name] = cls


def register_llm(cls: type) -> type:
    """Class decorator: register an LLMProvider implementation."""
    _register(_llm_registry, cls)
    return cls


def register_checker(cls: type) -> type:
    """Class decorator: register a SafetyChecker implementation."""
    _register(_checker_registry, cls)
    return cls


def register_decoder(cls: type) -> type:
    """Class decorator: register a Decoder implementation."""
    _register(_decoder_registry, cls)
    return cls


def register_auditor(cls: type) -> type:
    """Class decorator: register an Auditor implementation."""
    _register(_auditor_registry, cls)
    return cls


def register_server(cls: type) -> type:
    """Class decorator: register a ServerBackend implementation."""
    _register(_server_registry, cls)
    return cls


def get_llm(name: str) -> type:
    return _get(_llm_registry, name, "LLMProvider")


def get_checker(name: str) -> type:
    return _get(_checker_registry, name, "SafetyChecker")


def get_decoder(name: str) -> type:
    return _get(_decoder_registry, name, "Decoder")


def get_auditor(name: str) -> type:
    return _get(_auditor_registry, name, "Auditor")


def get_server(name: str) -> type:
    return _get(_server_registry, name, "ServerBackend")


def _get(registry: dict[str, type], name: str, kind: str) -> type:
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"No {kind} registered under name '{name}'. "
            f"Available: {available}"
        )
    return registry[name]


def autodiscover(package: Any) -> None:
    """Import all submodules of a package to trigger registration decorators.

    Call once per package at import time. Idempotent — already-imported
    modules are skipped by Python's import machinery.
    """
    prefix = package.__name__ + "."
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, prefix):
        importlib.import_module(module_name)


def all_decoders() -> dict[str, type]:
    """Return a copy of the decoder registry. Used by Normaliser at build."""
    return dict(_decoder_registry)


def all_decoder_names() -> list[str]:
    """Return decoder names in registration order."""
    return list(_decoder_registry.keys())
