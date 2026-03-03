"""
normaliser/__init__.py
-----------------------
Plugin registry for decoders.

Responsibility: store and retrieve Decoder classes by name.
Nothing else. Construction belongs in providers.py.
"""

import importlib
import pkgutil

from .base import Decoder, NormaliserConfig
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
            f"Duplicate normaliser plugin name {name!r}: "
            f"registered by {_registry[name].__module__}, "
            f"conflict with {cls.__module__}. "
            f"Each plugin must have a unique name."
        )
    _registry[name] = cls


def _autodiscover() -> None:
    _skip = {"base", "normaliser"}
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name in _skip:
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except ImportError:
            # Decoder has a missing optional dependency — skip it.
            # Built-in decoders have no optional deps; this guards custom ones.
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            try:
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Decoder)
                    and attr is not Decoder
                    and hasattr(attr, "name")
                ):
                    _register(attr.name, attr)
            except TypeError:
                continue


_autodiscover()


def resolve(name: str) -> type:
    """
    Return the Decoder class registered under name.

    Raises:
        PluginNotFoundError : if name not registered
    """
    if name not in _registry:
        raise PluginNotFoundError("normaliser", name, list(_registry.keys()))
    return _registry[name]


def available() -> list:
    return list(_registry.keys())
