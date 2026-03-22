"""
InMemoryAuditor — test-only auditor that stores records in memory.

NOT selectable via the 'type:' config field — it is not a valid deployment auditor.
Tests MUST use this rather than FileAuditor to avoid filesystem side effects.

Usage in tests:
    auditor = InMemoryAuditor()
    # ... run pipeline ...
    records = auditor.records  # list of dicts
    init_records = auditor.by_event("carapex_init")
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from carapex.audit.base import Auditor


class InMemoryAuditor(Auditor):
    """Stores audit records in memory for direct assertion in tests.

    Thread-safe. Deliberately not registered — cannot be selected via config.
    """

    name = "memory"  # not registered; exists for test use only

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False

    def log(self, event: str, data: dict[str, Any]) -> None:
        if self._closed:
            return
        record = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        with self._lock:
            if not self._closed:
                self._records.append(record)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def by_event(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("event") == event]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __repr__(self) -> str:
        return f"InMemoryAuditor(records={len(self._records)})"
