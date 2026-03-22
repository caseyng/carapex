"""carapex.normaliser — Normaliser, Decoder ABC, and built-in decoders."""

from carapex.normaliser.base import Decoder, Normaliser
from carapex.core.registry import autodiscover
import carapex.normaliser as _pkg

autodiscover(_pkg)

__all__ = ["Decoder", "Normaliser"]
