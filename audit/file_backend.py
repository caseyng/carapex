"""
audit/file_backend.py
---------------------
JSONL file audit backend. Thread-safe via logging module.
"""

import json
import os
import logging
from datetime import datetime, timezone
from .base import AuditBackend

_DEFAULT_LOG_PATH = "carapex.audit.log"


class FileAuditBackend(AuditBackend):
    name = "file"

    @classmethod
    def from_config(cls, raw: dict) -> "FileAuditBackend":
        log_path = raw.get("log_path") or _DEFAULT_LOG_PATH
        return cls(log_path)

    @classmethod
    def default_config(cls) -> dict:
        return {"log_path": _DEFAULT_LOG_PATH}

    def __init__(self, log_path: str = _DEFAULT_LOG_PATH):
        os.makedirs(
            os.path.dirname(os.path.abspath(log_path)), exist_ok=True
        )
        self._log_path = log_path

        # Logger name must not include path separators — slashes are not
        # meaningful in Python's logging hierarchy (only dots are).
        # Using the raw path as a logger name with slashes creates a
        # malformed logger tree and risks duplicate FileHandler instances
        # on the same file if the path is referenced by relative and
        # absolute form in different calls.
        # Normalise to basename without extension for a clean hierarchy key.
        safe_name = os.path.basename(log_path).replace(".", "_")
        self._logger = logging.getLogger(f"carapex.audit.{safe_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def log(self, event: dict) -> None:
        record              = dict(event)
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            self._logger.info(
                json.dumps(record, ensure_ascii=False, default=str)
            )
        except OSError as e:
            # Disk full or handler closed — audit must never crash the pipeline.
            # Log to root logger (bypasses our FileHandler) so the signal reaches
            # whatever logging infrastructure the application has configured.
            # Silent loss of audit events in a security boundary is not acceptable.
            logging.getLogger().error(
                "Audit write failed — audit events are being lost. "
                "Check disk space and file permissions. Path: %r. Error: %s",
                self._log_path, e,
            )

    def close(self) -> None:
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

    def __repr__(self) -> str:
        return f"FileAuditBackend(path={self._log_path!r})"
