"""CLI entry point: start the ScholarAnalysis MCP server."""

from __future__ import annotations

import logging
import sys

import uvicorn

from scholar_analysis.config import get_settings


def main() -> None:
    settings = get_settings()
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
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
