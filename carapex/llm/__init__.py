"""carapex.llm — LLMProvider ABC and built-in implementations."""

from carapex.llm.base import LLMProvider
from carapex.core.registry import autodiscover
import carapex.llm as _pkg

autodiscover(_pkg)

__all__ = ["LLMProvider"]
