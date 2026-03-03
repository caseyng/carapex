"""
safety/__init__.py
------------------
Plugin registry for safety checkers.

Responsibility: store and retrieve SafetyChecker classes by name.
Nothing else. Construction and composition belong in providers.py.
"""

import importlib
import pkgutil

from .base import SafetyChecker, SafetyConfig, SafetyResult
from ..exceptions import PluginNotFoundError

_registry: dict[str, type] = {}


def _autodiscover() -> None:
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name in ("base",):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except ImportError:
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            try:
                if (
                    isinstance(attr, type)
                    and issubclass(attr, SafetyChecker)
                    and attr is not SafetyChecker
                    and hasattr(attr, "name")
                ):
                    _registry[attr.name] = attr
            except TypeError:
                continue


_autodiscover()


def resolve(name: str) -> type:
    """
    Return the SafetyChecker class registered under name.

    Raises:
        PluginNotFoundError : if name not registered
    """
    if name not in _registry:
        raise PluginNotFoundError("safety", name, list(_registry.keys()))
    return _registry[name]


def available() -> list:
    return list(_registry.keys())
