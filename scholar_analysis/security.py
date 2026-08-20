"""AccessToken middleware for MCP SSE transport."""

from __future__ import annotations

import hmac
import logging
from collections.abc import Mapping

from starlette.datastructures import Headers

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Paths and methods that are part of the MCP SSE handshake and must be
# reachable without a Bearer token. Standard MCP SSE clients (Claude Desktop,
# Cline) open the SSE stream via GET /sse and post tool calls to /messages/.
# The actual tool invocations still go through token verification because
# they are POSTs, not the SSE handshake GETs.
_HANDSHAKE_PATHS: dict[str, set[str]] = {
    "/sse": {"GET"},
    "/messages/": {"GET"},
}


def extract_access_token(headers: Headers | Mapping[str, str]) -> str:
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return headers.get("x-scholar-analysis-token", "").strip()


def is_valid_access_token(expected: str, provided: str) -> bool:
    return bool(expected) and bool(provided) and hmac.compare_digest(expected, provided)


def is_handshake(path: str, method: str) -> bool:
    """Whether a request is part of the SSE handshake and should bypass auth."""
    methods = _HANDSHAKE_PATHS.get(path)
    if methods is None:
        return False
    return method.upper() in methods


class AccessTokenMiddleware:
    def __init__(self, app: ASGIApp, access_token: str):
        self._app = app
        self._access_token = access_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._access_token:
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        # Pass through SSE handshake endpoints without auth — see module docstring.
        if is_handshake(path, method):
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        provided = extract_access_token(headers)
        if is_valid_access_token(self._access_token, provided):
            await self._app(scope, receive, send)
            return

        logger.warning("MCP auth failure for %s %s", method, path)
        response = JSONResponse(
            {"error": "unauthorized", "message": "Authentication required"},
            status_code=401,
        )
        await response(scope, receive, send)
