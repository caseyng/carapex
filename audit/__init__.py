"""
audit/__init__.py
-----------------
Plugin registry for audit backends.
Responsibility: store and retrieve AuditBackend classes by name only.
"""

import importlib
import pkgutil

from .base import AuditBackend
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
            f"Duplicate audit plugin name {name!r}: "
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
        except ImportError:
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            try:
                if (
                    isinstance(attr, type)
                    and issubclass(attr, AuditBackend)
                    and attr is not AuditBackend
                    and hasattr(attr, "name")
                ):
                    _register(attr.name, attr)
            except TypeError:
                continue


_autodiscover()


def resolve(name: str) -> type:
    if name not in _registry:
        raise PluginNotFoundError("audit", name, list(_registry.keys()))
    return _registry[name]


def available() -> list:
    return list(_registry.keys())
