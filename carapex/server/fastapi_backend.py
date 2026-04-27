"""
FastAPIBackend — HTTP server using FastAPI + uvicorn.

Registered as name="fastapi". The default ServerBackend.

Not production-hardened for high-concurrency deployments. For those,
supply a custom ServerBackend implementation.

Rejects stream:true before invoking ChatHandler (§3).
Normalises header keys to lowercase before passing to ChatHandler.
Returns HTTP 200 for all outcomes representable in EvaluationResult.
Returns HTTP 500 for PipelineInternalError.
Returns HTTP 400 for precondition violations (invalid_request_error).
"""

from __future__ import annotations

import logging
from typing import Any

from carapex.core.registry import register_server
from carapex.server.base import ChatHandler, ServerBackend

log = logging.getLogger(__name__)


@register_server
class FastAPIBackend(ServerBackend):
    """FastAPI + uvicorn HTTP server backend."""

    name = "fastapi"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        workers: int = 1,
    ) -> None:
        self._host = host
        self._port = port
        self._workers = workers
        self._server: Any = None  # uvicorn.Server, set in serve()

    def serve(self, handler: ChatHandler) -> None:
        try:
            import uvicorn  # type: ignore[import]
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
        except ImportError as e:
            raise ImportError(
                "fastapi and uvicorn are required for FastAPIBackend. "
                "Install them with: pip install fastapi uvicorn"
            ) from e

        from carapex.core.exceptions import PipelineInternalError

        app = FastAPI(title="carapex")

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> JSONResponse:
            # Reject streaming before invoking ChatHandler (§3)
            try:
                body: dict[str, Any] = await request.json()
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "Invalid JSON in request body",
                            "type": "invalid_request_error",
                            "code": "invalid_json",
                        }
                    },
                )

            if body.get("stream") is True:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "carapex does not support streaming",
                            "type": "invalid_request_error",
                            "code": "streaming_not_supported",
                        }
                    },
                )

            # Normalise header keys to lowercase
            headers = {k.lower(): v for k, v in request.headers.items()}

            try:
                response = handler(body, headers)
                return JSONResponse(status_code=200, content=response)
            except PipelineInternalError as e:
                log.error("PipelineInternalError: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": str(e),
                            "type": "internal_error",
                            "code": "pipeline_internal_error",
                        }
                    },
                )
            except (ValueError, TypeError) as e:
                # Precondition violation from evaluate()
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "code": "invalid_messages",
                        }
                    },
                )
            except Exception as e:  # noqa: BLE001
                log.exception("Unhandled error in ChatHandler: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Internal server error",
                            "type": "internal_error",
                            "code": "internal_error",
                        }
                    },
                )

        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            workers=self._workers,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.should_exit = True
            except Exception:  # noqa: BLE001
                pass

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "FastAPIBackend":
        return cls(
            host=raw.get("host", "127.0.0.1"),
            port=int(raw.get("port", 8000)),
            workers=int(raw.get("workers", 1)),
        )

    def __repr__(self) -> str:
        return f"FastAPIBackend(host={self._host!r}, port={self._port!r})"
