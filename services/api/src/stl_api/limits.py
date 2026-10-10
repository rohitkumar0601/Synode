"""Pure ASGI middleware bounding upload size while the body streams in."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_PATHS = frozenset({"/datasets", "/import-results"})


class UploadLimitMiddleware:
    """Answer 413 as soon as a request to an upload path exceeds ``max_mb`` megabytes.

    Counts body bytes as they arrive (so a missing or false Content-Length cannot bypass the
    limit), with a Content-Length fast path. Multipart is spooled by the parser before any
    handler runs, so this must happen below it.
    """

    def __init__(self, app, max_mb: int, paths: frozenset[str] | set[str] = DEFAULT_UPLOAD_PATHS):
        self.app = app
        self.max_mb = max_mb
        self.max_bytes = max_mb * 1024 * 1024
        self.paths = {p.rstrip("/") for p in paths}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") not in self.paths:
            await self.app(scope, receive, send)
            return
        body = json.dumps({"detail": f"upload exceeds STL_MAX_UPLOAD_MB={self.max_mb}"}).encode()
        state = {"rejected": False}

        async def reject() -> None:
            state["rejected"] = True
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})

        for name, value in scope["headers"]:
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                await reject()
                return

        received = 0

        async def counting_receive():
            nonlocal received
            if state["rejected"]:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    await reject()
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not state["rejected"]:
                await send(message)

        await self.app(scope, counting_receive, guarded_send)


class InternalErrorMiddleware:
    """Turn an unhandled exception into a JSON 500 below CORS so it keeps CORS headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            if started:
                raise
            logger.exception("unhandled error on %s", scope.get("path"))
            body = json.dumps({"detail": "internal server error"}).encode()
            await send({"type": "http.response.start", "status": 500,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
