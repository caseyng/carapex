"""
FileAuditor — writes JSONL audit records to a file.

The default and required auditor for production deployments.
Thread-safe: a threading.Lock serialises writes so records are atomic per line.
Records are written synchronously with an immediate flush — no buffering
beyond the OS page cache. This keeps the implementation simple and ensures
records survive a crash after flush() returns.

Log failures are reported to stderr (never raise, never block).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from carapex.audit.base import Auditor
from carapex.core.registry import register_auditor

log = logging.getLogger(__name__)


@register_auditor
class FileAuditor(Auditor):
    """JSONL file auditor. Thread-safe."""

    name = "file"

    def __init__(self, path: str) -> None:
        self._path = Path(path).resolve()
        self._lock = threading.Lock()
        self._closed = False
        # Open in append mode — survives restart without losing prior records
        try:
            self._file = self._path.open("a", encoding="utf-8", buffering=1)
        except OSError as e:
            raise OSError(f"FileAuditor cannot open '{path}': {e}") from e

    def log(self, event: str, data: dict[str, Any]) -> None:
        if self._closed:
            return  # no-op after close (§2 Auditor lifecycle)

        record = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        line = json.dumps(record, ensure_ascii=False)

        try:
            with self._lock:
                if not self._closed:
                    self._file.write(line + "\n")
                    self._file.flush()
        except Exception as e:  # noqa: BLE001 — never raise
            print(f"[carapex audit] write failed: {e}", file=sys.stderr)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._file.flush()
                self._file.close()
            except Exception as e:  # noqa: BLE001
                print(f"[carapex audit] close failed: {e}", file=sys.stderr)

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "FileAuditor":
        path = raw.get("path")
        if not path:
            raise ValueError("FileAuditor config missing 'path'")
        return cls(path=path)

    def __repr__(self) -> str:
        return f"FileAuditor(path={str(self._path)!r})"
