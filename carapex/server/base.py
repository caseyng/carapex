"""
ServerBackend — the abstraction for HTTP server implementations.

Exposes /v1/chat/completions and delegates each request to Carapex via ChatHandler.

The backend owns transport; Carapex owns request handling logic.
Headers dict passed to ChatHandler MUST have all keys normalised to lowercase.

To add a new server backend:
  1. Create a file in carapex/server/.
  2. Subclass ServerBackend, set a unique `name` class attribute.
  3. Decorate with @register_server.
  4. Implement serve() and close().

serve() MUST:
  - Block until shutdown.
  - Reject stream:true requests with HTTP 400 before invoking ChatHandler.
  - Normalise header keys to lowercase before passing to ChatHandler.
  - Never raise on individual request failures — return HTTP 500 instead.
  - Raise on startup failures (port conflict, bad config).

close() MUST be idempotent and never raise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

# ChatHandler: (request_body: dict, headers: dict[str, str]) → response_body: dict
ChatHandler = Callable[[dict[str, Any], dict[str, str]], dict[str, Any]]


class ServerBackend(ABC):
    """Abstract base for all HTTP server implementations."""

    name: str

    @abstractmethod
    def serve(self, handler: ChatHandler) -> None:
        """Start the HTTP server. Blocks until shutdown.

        Invokes handler(request, headers) for each incoming request.
        Headers dict has all keys normalised to lowercase.
        Raises on startup failure.
        """

    @abstractmethod
    def close(self) -> None:
        """Stop the server and release resources. Idempotent. Never raises."""

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "ServerBackend":
        return cls()  # type: ignore[call-arg]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
