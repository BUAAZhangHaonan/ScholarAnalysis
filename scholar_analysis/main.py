"""CLI entry point: start the ScholarAnalysis MCP server."""

from __future__ import annotations

import logging


import uvicorn

from scholar_analysis.config import get_settings


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def validate_runtime_settings(settings) -> None:
    """Reject startup if the server would listen on a non-loopback interface
    without an access token. Listening on 0.0.0.0 without auth exposes the
    MCP endpoint to the entire LAN, which is almost never intended.
    """
    host = settings.host.strip().lower()
    is_loopback = host in _LOOPBACK_HOSTS

    if not is_loopback and not settings.access_token.strip():
        raise SystemExit(
            f"Refusing to start: host={settings.host!r} is not loopback and "
            "access_token is empty. Either set SCHOLAR_ANALYSIS_HOST=127.0.0.1, "
            "or configure SCHOLAR_ANALYSIS_ACCESS_TOKEN before binding to a "
            "public interface."
        )


def main() -> None:
    settings = get_settings()
    validate_runtime_settings(settings)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from scholar_analysis.mcp_server import create_mcp_sse_app

    app = create_mcp_sse_app(settings)
    logger = logging.getLogger("scholar_analysis")
    logger.info(
        "Starting ScholarAnalysis MCP on %s:%d",
        settings.host,
        settings.port,
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=300,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
