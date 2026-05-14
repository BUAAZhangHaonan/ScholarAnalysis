"""AccessToken middleware for MCP SSE transport."""

from __future__ import annotations

import hmac
import logging
from collections.abc import Mapping

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


def extract_access_token(headers: Headers | Mapping[str, str]) -> str:
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return headers.get("x-scholar-analysis-token", "").strip()


def is_valid_access_token(expected: str, provided: str) -> bool:
    return bool(expected) and bool(provided) and hmac.compare_digest(expected, provided)


class AccessTokenMiddleware:
    def __init__(self, app: ASGIApp, access_token: str):
        self._app = app
        self._access_token = access_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._access_token:
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        provided = extract_access_token(headers)
        if is_valid_access_token(self._access_token, provided):
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        logger.warning("MCP auth failure for %s", path)
        response = JSONResponse(
            {"error": "unauthorized", "message": "Authentication required"},
            status_code=401,
        )
        await response(scope, receive, send)
