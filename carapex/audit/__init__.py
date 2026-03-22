"""carapex.audit — Auditor ABC and built-in implementations."""

from carapex.audit.base import Auditor
from carapex.core.registry import autodiscover
import carapex.audit as _pkg

autodiscover(_pkg)

__all__ = ["Auditor"]
