"""carapex.server — ServerBackend ABC and built-in implementations."""

from carapex.server.base import ServerBackend, ChatHandler
from carapex.core.registry import autodiscover
import carapex.server as _pkg

autodiscover(_pkg)

__all__ = ["ServerBackend", "ChatHandler"]
