"""
Auditor — the abstraction for audit log destinations.

Every evaluate() call produces a structured sequence of JSONL records.
The Auditor receives log() calls throughout — it must never raise and must
never block log() indefinitely.

To add a new auditor:
  1. Create a file in carapex/audit/.
  2. Subclass Auditor, set a unique `name` class attribute.
  3. Decorate with @register_auditor.
  4. Implement log() and close().

MUST NEVER:
  - Raise in log() — ever.
  - Log prompt content, response content, or API keys.
  - Block log() on slow I/O — buffer writes and flush asynchronously.

Thread safety: MUST be safe for concurrent log() calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Auditor(ABC):
    """Abstract base for all audit log implementations."""

    name: str

    @abstractmethod
    def log(self, event: str, data: dict[str, Any]) -> None:
        """Write one audit record. Never raises."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release resources. Idempotent. Never raises."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
