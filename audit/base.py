"""
audit/base.py
-------------
Abstract base and minimal config for audit backends.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ADD A NEW AUDIT BACKEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create a file in audit/  e.g. audit/sqlite_backend.py
2. Subclass AuditBackend
3. Set class attribute `name` — registry key
4. Implement log()
5. Done — autodiscovered, appears in config.json automatically

MINIMAL AUDIT BACKEND (no extra config):

    class MyAuditBackend(AuditBackend):
        name = "my_audit"

        def log(self, event: dict) -> None:
            ...  # write event somewhere, never raise

AUDIT BACKEND WITH EXTRA CONFIG:

    class SQLiteAuditBackend(AuditBackend):
        name = "sqlite"

        @classmethod
        def from_config(cls, raw: dict) -> "SQLiteAuditBackend":
            db_path = raw.get("db_path", "carapex.db")
            return cls(db_path)

        @classmethod
        def default_config(cls) -> dict:
            return {"db_path": "carapex.db"}

        def log(self, event: dict) -> None:
            ...

RULES:
    - log() must always be implemented
    - log() must never raise — audit must never crash the pipeline
    - from_config() — override only if you need config fields
    - default_config() — override only if you have extra fields
    - Never edit existing files to register a new audit backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from abc import ABC, abstractmethod
from typing import Optional



class AuditBackend(ABC):
    """
    Abstract base for all audit backends.
    See module docstring for full extension guide.
    """

    name: str  # registry key — must be set on every subclass

    @classmethod
    def from_config(cls, raw: dict) -> "AuditBackend":
        """
        Instantiate from raw config dict.
        Default: no-arg construction — works for backends with no config.
        Override if your backend needs config fields.
        """
        return cls()

    @classmethod
    def default_config(cls) -> dict:
        """
        Return config keys with defaults for config.json generation.
        Default: empty dict — no extra fields beyond backend name.
        Override to add your backend-specific fields.
        """
        return {}

    @abstractmethod
    def log(self, event: dict) -> None:
        """
        Record an audit event.
        Must be implemented — no universal default exists.
        Must never raise — audit must never crash the pipeline.

        Args:
            event : dict with at minimum an "event" key (str).
                    Timestamp is added by implementations.
        """
        ...

    def close(self) -> None:
        """Override to flush and release resources at shutdown."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
