"""
backends/__init__.py
--------------------
Plugin registry for LLM backends.
Responsibility: store and retrieve LLMBackend classes by name only.
Construction and config belong in providers.py.
"""

import importlib
import pkgutil

from .base import LLMBackend
from ..exceptions import PluginNotFoundError

_registry: dict[str, type] = {}


def _register(name: str, cls: type) -> None:
    """
    Register a class under name. Raises on duplicate.
    Silent overwrite would mean last-import-wins — nondeterministic and
    undetectable until the wrong implementation is used at runtime.
    """
    if name in _registry:
        raise RuntimeError(
            f"Duplicate backend plugin name {name!r}: "
            f"registered by {_registry[name].__module__}, "
            f"conflict with {cls.__module__}. "
            f"Each plugin must have a unique name."
        )
    _registry[name] = cls


def _autodiscover() -> None:
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name in ("base",):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                try:
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, LLMBackend)
                        and attr is not LLMBackend
                        and hasattr(attr, "name")
                    ):
                        _register(attr.name, attr)
                except TypeError:
                    continue
        except ImportError as e:
            # Optional dependency not installed (e.g. llama-cpp-python for llama_cpp).
            # Log at debug so custom backend authors can diagnose missing deps.
            # Built-in backends with missing deps are expected — not an error.
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "Backend module %r skipped — ImportError: %s. "
                "If this is a custom backend, check its import dependencies.",
                module_name, e,
            )
            continue


_autodiscover()


def resolve(name: str) -> type:
    if name not in _registry:
        raise PluginNotFoundError("backends", name, list(_registry.keys()))
    return _registry[name]


def available() -> list:
    return list(_registry.keys())
