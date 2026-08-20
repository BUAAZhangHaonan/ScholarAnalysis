"""MCP server for ScholarAnalysis — 2 tools: get_paper_text, analyze_paper."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from functools import wraps

from scholar_analysis.config import Settings, get_settings
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
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception:
            ref = uuid.uuid4().hex[:8]
            logger.exception("Tool %s failed [ref=%s]", func.__name__, ref)
            return json.dumps(
                {
                    "status": "error",
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
            # Close the shared httpx client used by the PostProcessor (if instantiated).
            orch = _orchestrator
            if orch is not None:
                with contextlib.suppress(Exception):
                    await orch._post_processor.aclose()

    app = Starlette(routes=[Mount("/", app=inner_app)], lifespan=lifespan)

    if not s.access_token:
        logger.warning("No access token configured — MCP endpoint is UNAUTHENTICATED")

    if s.access_token:
        return AccessTokenMiddleware(app, s.access_token)
    return app


mcp = create_mcp()


@mcp.tool()
@safe_tool
async def get_paper_text(
    query: str,
    include_images: bool = False,
) -> str:
    """获取论文的完整 Markdown 文本。

    下载并解析论文 PDF，返回 Markdown 格式文本。
    设置 include_images=true 可保留图片引用，供多模态模型使用。

    Args:
        query: arXiv ID (如 2402.01306) 或 arXiv URL。不支持标题搜索。
        include_images: True 保留图片引用(多模态模型用), False 纯文本(默认)

    Returns:
        JSON string with paper metadata and markdown text.
    """
    sem = _get_semaphore()
    async with sem:
        orch = _get_orchestrator()
        result = await orch.get_paper_text(query=query, include_images=include_images)
        return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool
async def analyze_paper(
    query: str,
    question: str,
    language: str = "en",
    include_images: bool = False,
) -> str:
    """下载解析论文后，使用 LLM 根据 question 提取相关内容。

    先下载并解析论文为 Markdown，再用 LLM 针对 question
    提取论文中相关的方法、结论和关键发现。
    避免将整篇论文塞入 Agent 上下文，只返回聚焦的分析结果。

    Args:
        query: arXiv ID (如 2402.01306) 或 arXiv URL。不支持标题搜索。
        question: 用户的分析问题/关注点
        language: "en" (English) 或 "zh" (中文), 默认 "en"
        include_images: True 保留图片引用, False 纯文本(默认)

    Returns:
        JSON string with focused analysis result.
    """
    sem = _get_semaphore()
    async with sem:
        orch = _get_orchestrator()
        result = await orch.analyze_paper(
            query=query,
            question=question,
            language=language,
            include_images=include_images,
        )
        return json.dumps(result, ensure_ascii=False)
