"""
audit/null_backend.py
---------------------
Discards all events. Use for unit testing only.
Inherits from_config() and default_config() from base — no extra config.
"""

from .base import AuditBackend


class NullAuditBackend(AuditBackend):
    name = "null"

    def log(self, event: dict) -> None:
        pass

    def __repr__(self) -> str:
        return "NullAuditBackend()"
