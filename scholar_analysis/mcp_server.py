"""MCP server for ScholarAnalysis — 2 tools: get_paper_text, analyze_paper."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from functools import wraps

from scholar_analysis.config import Settings, get_settings
from scholar_analysis.cost import costed
from scholar_analysis.security import AccessTokenMiddleware

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Module-level orchestrator (lazily initialised)
_orchestrator = None
_pipeline_sem: asyncio.Semaphore | None = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from scholar_analysis.pipeline.orchestrator import Orchestrator

        logger.info("Initializing Orchestrator (first call)")
        try:
            _orchestrator = Orchestrator()
        except Exception as exc:
            logger.exception("Orchestrator initialization failed")
            raise RuntimeError(f"Orchestrator init failed: {exc}") from exc
        logger.info("Orchestrator initialized successfully")
    return _orchestrator


def _get_semaphore() -> asyncio.Semaphore:
    global _pipeline_sem
    if _pipeline_sem is None:
        _pipeline_sem = asyncio.Semaphore(get_settings().max_concurrent_pipelines)
    return _pipeline_sem


def safe_tool(func):
    """Wrap a tool function to catch exceptions and return structured errors.

    Exception details (may contain internal URLs/config) go to the server log
    only; the client gets a fixed message plus a log reference id.
    """

    @wraps(func)
    @costed
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception:
            ref = uuid.uuid4().hex[:8]
            logger.exception("Tool %s failed [ref=%s]", func.__name__, ref)
            return json.dumps(
                {
                    "status": "error",
                    "error_code": "INTERNAL_ERROR", "stage": "tool", "retryable": False,
                    "retry_after_seconds": None,
                    "error": (
                        f"Internal error in {func.__name__} [ref={ref}]. "
                        f"Details are in the server log; contact the administrator."
                    ),
                },
                ensure_ascii=False,
            )

    return wrapper


def create_mcp(settings: Settings | None = None) -> FastMCP:
    s = settings or get_settings()
    return FastMCP("ScholarAnalysis", host=s.host, port=s.port)


def create_mcp_sse_app(settings: Settings | None = None):
    s = settings or get_settings()
    logger.info(
        "Creating MCP SSE app: host=%s port=%d access_token=%s",
        s.host,
        s.port,
        "configured" if s.access_token else "NONE (unauthenticated!)",
    )
    inner_app = mcp.sse_app()

    from scholar_analysis.pipeline.temp_manager import cleanup_loop

    def _active_request_ids():
        orch = _orchestrator
        if orch is None:
            return set()
        return orch._tracker.active_ids

    @asynccontextmanager
    async def lifespan(app):
        cleanup_task = None
        if s.cleanup_interval_seconds > 0:
            cleanup_task = asyncio.create_task(
                cleanup_loop(
                    s.temp_dir,
                    s.cleanup_interval_seconds,
                    s.request_max_age_seconds,
                    active_request_ids=_active_request_ids,
                )
            )
            logger.info(
                "Started temp cleanup task: interval=%ss max_age=%ss dir=%s",
                s.cleanup_interval_seconds,
                s.request_max_age_seconds,
                s.temp_dir,
            )
        try:
            yield
        finally:
            if cleanup_task is not None:
                cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cleanup_task
                logger.info("Temp cleanup task stopped")
            # Close shared resources (httpx clients, model pool) if instantiated.
            orch = _orchestrator
            if orch is not None:
                with contextlib.suppress(Exception):
                    await orch.aclose()

    app = Starlette(routes=[Mount("/", app=inner_app)], lifespan=lifespan)

    if not s.access_token:
        logger.warning("No access token configured — MCP endpoint is UNAUTHENTICATED")

    if s.access_token:
        return AccessTokenMiddleware(app, s.access_token)
    return app


mcp = create_mcp()


async def _run_pipeline(method: str, **kwargs) -> str:
    from scholar_analysis.pipeline.errors import PipelineError, error_result
    sem = _get_semaphore()
    if method == "analyze_paper" and not kwargs.get("analysis_id"):
        kwargs["analysis_id"] = "analysis_"+uuid.uuid4().hex
    settings = get_settings()
    stage = "queue"
    try:
        async with asyncio.timeout(settings.request_max_age_seconds):
            try:
                await asyncio.wait_for(sem.acquire(), timeout=settings.queue_timeout_seconds)
            except TimeoutError:
                return json.dumps(error_result(uuid.uuid4().hex, PipelineError(
                    "QUEUE_TIMEOUT", "queue", "Service is busy; retry later.",
                    retryable=True, retry_after=5)), ensure_ascii=False)
            try:
                stage = "pipeline"
                result = await getattr(_get_orchestrator(), method)(**kwargs)
                return json.dumps(result, ensure_ascii=False)
            finally:
                sem.release()
    except TimeoutError:
        result = error_result(uuid.uuid4().hex, PipelineError(
            "REQUEST_TIMEOUT", stage, "Request exceeded its total deadline.",
            retryable=True))
        if kwargs.get("analysis_id"):
            result["analysis_id"] = kwargs["analysis_id"]
        return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool
async def get_paper_text(
    query: str = "", include_images: bool = False, pdf_url: str | None = None,
    document_id: str | None = None, offset: int = 0, limit_chars: int | None = None,
    find_text: str | None = None, lang: str = "", refresh: bool = False,
) -> str:
    """Read parsed paper Markdown with exact character/line/section locations.

    Supply one arXiv ID/URL or DOI in query, a public paper/PDF pdf_url, OR a
    previously returned document_id for a cache-only read. Publisher pages need
    a declared citation_pdf_url. No title search. offset/limit_chars page through
    the requested text mode; find_text locates literal text at/after offset.
    If limit_chars is omitted, ordinary reads return up to 64000 characters,
    find_text matches up to 4000; explicit limits are respected up to 64000.
    A missing literal returns match_status=no_match and no Markdown, not a full
    paper fallback. It does not establish that the paper lacks the concept.
    include_images retains image references, NOT image pixels or visual analysis.
    Coverage describes returned parsed text, never certifies PDF completeness.
    Use refresh with the original source to retry parsing or refresh a snapshot.
    """
    return await _run_pipeline(
        "get_paper_text", query=query, include_images=include_images, pdf_url=pdf_url,
        document_id=document_id, offset=offset, limit_chars=limit_chars,
        find_text=find_text, lang=lang, refresh=refresh,
    )


@mcp.tool()
@safe_tool
async def analyze_paper(
    query: str = "", question: str = "", language: str = "en",
    include_images: bool = False, pdf_url: str | None = None,
    document_id: str | None = None, lang: str = "",
    offset: int = 0, limit_chars: int | None = None, analysis_id: str | None = None,
) -> str:
    """Analyze a question using parsed paper text; this invokes a paid model.

    Input selection is identical to get_paper_text; offset/limit_chars can select a relevant passage. Returned analysis includes
    actual reading coverage and quoted evidence locations. Retained image
    references are not visually analyzed. language must be en or zh.
    Reuse analysis_id with identical inputs to retrieve/wait for the same paid task;
    a new analysis_id explicitly creates a new analysis. Previous failures are retained.
    """
    return await _run_pipeline(
        "analyze_paper", query=query, question=question, language=language,
        include_images=include_images, pdf_url=pdf_url, document_id=document_id, lang=lang,
        offset=offset, limit_chars=limit_chars, analysis_id=analysis_id,
    )
