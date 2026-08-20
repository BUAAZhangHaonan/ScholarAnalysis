"""Pipeline orchestrator: download → parse → (optional LLM analysis)."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from scholar_analysis.clients.arxiv_mirror import ArxivMirrorClient, ArxivMirrorError
from scholar_analysis.clients.mineru import MinerUClient, extract_markdown
from scholar_analysis.config import get_settings
from scholar_analysis.llm.post_processor import PostProcessor
from scholar_analysis.pipeline.parse_cache import ParseCache
from scholar_analysis.pipeline.request_context import RequestTracker

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates paper download, parsing, and optional LLM analysis."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._tracker = RequestTracker(
            self._settings.temp_dir,
            max_age_seconds=self._settings.request_max_age_seconds,
        )
        self._arxiv = ArxivMirrorClient(
            base_url=self._settings.arxiv_mirror_base_url,
            timeout=self._settings.http_timeout,
        )
        self._mineru = MinerUClient.from_settings(self._settings)
        self._post_processor = PostProcessor()
        self._arxiv_data_dir = Path(self._settings.arxiv_mirror_data_dir)
        self._parse_sem = asyncio.Semaphore(self._settings.max_concurrent_parses)
        self._parse_cache = ParseCache(
            self._settings.parse_cache_dir,
            self._settings.parse_cache_max_bytes,
        )

    async def aclose(self) -> None:
        """Close shared resources (httpx clients, model pool)."""
        for closer in (
            self._arxiv.aclose,
            self._mineru.aclose,
            self._post_processor.aclose,
        ):
            try:
                await closer()
            except Exception:
                logger.exception("Error closing orchestrator resource %s", closer)

    async def _parse_with_cache(
        self, rid: str, versioned_id: str, pdf_path: Path
    ) -> dict[str, Any]:
        """Fetch MinerU parse result from cache, else parse and cache it."""
        cached = await self._parse_cache.get(versioned_id)
        if cached is not None:
            logger.info("[REQ %s] parse cache hit: %s", rid, versioned_id)
            return cached
        logger.info("[REQ %s] parse cache miss: %s", rid, versioned_id)
        async with self._parse_sem:
            result = await self._mineru.parse_pdf(pdf_path)
        await self._parse_cache.put(versioned_id, "", result)
        return result

    async def get_paper_text(
        self,
        query: str,
        include_images: bool = False,
    ) -> dict[str, Any]:
        """Download and parse a paper, return Markdown text."""
        timings: dict[str, float] = {}
        ctx = await self._tracker.create()
        rid = ctx.request_id
        logger.info(
            "[REQ %s] get_paper_text query=%s include_images=%s",
            rid,
            query,
            include_images,
        )

        try:
            # Step 1: Resolve + download PDF via arxiv_mirror
            ctx.status = "downloading"
            t0 = time.monotonic()

            async with asyncio.timeout(self._settings.request_max_age_seconds):
                info = await self._arxiv.resolve(query)
                logger.info(
                    "[REQ %s] Resolved: id=%s title=%s",
                    rid,
                    info.arxiv_id,
                    info.title[:80] if info.title else "(no title)",
                )

                asset = await self._arxiv.download(query)
                timings["download_s"] = round(time.monotonic() - t0, 2)
                logger.info(
                    "[REQ %s] Download complete in %.2fs (versioned_id=%s, path=%s, size=%d)",
                    rid,
                    timings["download_s"],
                    asset.versioned_id,
                    asset.local_path,
                    asset.file_size,
                )

                # Step 2: Parse PDF via MinerU
                ctx.status = "parsing"
                t1 = time.monotonic()

                pdf_path = self._arxiv_data_dir / asset.local_path
                if not pdf_path.exists():
                    raise ArxivMirrorError(
                        f"PDF file not found at {pdf_path}. "
                        f"arxiv_mirror data_dir may be misconfigured."
                    )

                parse_result = await self._parse_with_cache(
                    rid, asset.versioned_id, pdf_path
                )
                markdown = extract_markdown(parse_result, text_only=not include_images)

                timings["parse_s"] = round(time.monotonic() - t1, 2)
                logger.info(
                    "[REQ %s] Parsed text: %d chars in %.2fs",
                    rid,
                    len(markdown),
                    timings["parse_s"],
                )

                if not markdown:
                    raise RuntimeError(
                        f"MinerU parsing returned empty markdown for {asset.versioned_id}. "
                        f"Parse result keys: {list(parse_result.keys())}"
                    )

            ctx.status = "completed"
            timings["total_s"] = round(time.monotonic() - t0, 2)

            return {
                "request_id": rid,
                "paper": {
                    "arxiv_id": info.arxiv_id,
                    "versioned_id": asset.versioned_id,
                    "title": info.title,
                    "authors": info.authors,
                    "abstract": info.abstract,
                },
                "mode": "text_with_images" if include_images else "text_only",
                "status": "success",
                "markdown": markdown,
                "timing": timings,
            }
        except ArxivMirrorError as e:
            ctx.status = "failed"
            logger.error("[REQ %s] arxiv_mirror error: %s", rid, e)
            return _error_result(rid, str(e), timings)
        except TimeoutError:
            ctx.status = "failed"
            logger.error(
                "[REQ %s] get_paper_text timed out after %.0fs",
                rid,
                self._settings.request_max_age_seconds,
            )
            return _error_result(
                rid,
                f"Request {rid} timed out after {self._settings.request_max_age_seconds:.0f}s "
                f"(download + parse pipeline deadline exceeded)",
                timings,
            )
        except Exception as e:
            ctx.status = "failed"
            logger.exception("[REQ %s] get_paper_text failed", rid)
            return _error_result(rid, str(e), timings)
        finally:
            await self._tracker.remove(rid)

    async def analyze_paper(
        self,
        query: str,
        question: str,
        language: str = "en",
        include_images: bool = False,
    ) -> dict[str, Any]:
        """Download, parse, then run LLM-focused analysis on a paper."""
        timings: dict[str, float] = {}
        ctx = await self._tracker.create()
        rid = ctx.request_id
        logger.info(
            "[REQ %s] analyze_paper query=%s question=%s language=%s",
            rid,
            query,
            question[:80],
            language,
        )

        try:
            # Step 1: Resolve + download PDF via arxiv_mirror
            ctx.status = "downloading"
            t0 = time.monotonic()

            async with asyncio.timeout(self._settings.request_max_age_seconds):
                info = await self._arxiv.resolve(query)
                logger.info(
                    "[REQ %s] Resolved: id=%s title=%s",
                    rid,
                    info.arxiv_id,
                    info.title[:80] if info.title else "(no title)",
                )

                asset = await self._arxiv.download(query)
                timings["download_s"] = round(time.monotonic() - t0, 2)
                logger.info(
                    "[REQ %s] Download complete in %.2fs", rid, timings["download_s"]
                )

                # Step 2: Parse PDF via MinerU
                ctx.status = "parsing"
                t1 = time.monotonic()

                pdf_path = self._arxiv_data_dir / asset.local_path
                if not pdf_path.exists():
                    raise ArxivMirrorError(
                        f"PDF file not found at {pdf_path}. "
                        f"arxiv_mirror data_dir may be misconfigured."
                    )

                parse_result = await self._parse_with_cache(
                    rid, asset.versioned_id, pdf_path
                )
                markdown = extract_markdown(parse_result, text_only=not include_images)

                timings["parse_s"] = round(time.monotonic() - t1, 2)
                logger.info(
                    "[REQ %s] Parsed text: %d chars in %.2fs",
                    rid,
                    len(markdown),
                    timings["parse_s"],
                )

                if not markdown:
                    raise RuntimeError(
                        f"MinerU parsing returned empty markdown for {asset.versioned_id}. "
                        f"Parse result keys: {list(parse_result.keys())}"
                    )

                # Step 3: LLM analysis
                ctx.status = "processing"
                t2 = time.monotonic()
                analysis = await self._post_processor.extract(
                    markdown=markdown,
                    question=question,
                    language=language,
                )
                timings["llm_s"] = round(time.monotonic() - t2, 2)

            # Validate analysis shape
            required_keys = {"answer", "model_used", "backend", "token_usage"}
            missing = required_keys - set(analysis)
            if missing:
                raise RuntimeError(
                    f"PostProcessor.extract() returned dict missing keys: {missing}. "
                    f"Got keys: {list(analysis.keys())}"
                )

            ctx.status = "completed"
            timings["total_s"] = round(time.monotonic() - t0, 2)

            logger.info(
                "[REQ %s] Analysis complete in %.2fs (llm=%.2fs)",
                rid,
                timings["total_s"],
                timings["llm_s"],
            )

            return {
                "request_id": rid,
                "paper": {
                    "arxiv_id": info.arxiv_id,
                    "versioned_id": asset.versioned_id,
                    "title": info.title,
                },
                "mode": "text_with_images" if include_images else "text_only",
                "status": "success",
                "analysis": {
                    "question": question,
                    "answer": analysis["answer"],
                    "model_used": analysis["model_used"],
                    "backend": analysis["backend"],
                    "token_usage": analysis["token_usage"],
                    "truncated": analysis.get("truncated", False),
                },
                "timing": timings,
            }
        except ArxivMirrorError as e:
            ctx.status = "failed"
            logger.error("[REQ %s] arxiv_mirror error: %s", rid, e)
            return _error_result(rid, str(e), timings)
        except TimeoutError:
            ctx.status = "failed"
            logger.error(
                "[REQ %s] analyze_paper timed out after %.0fs",
                rid,
                self._settings.request_max_age_seconds,
            )
            return _error_result(
                rid,
                f"Request {rid} timed out after {self._settings.request_max_age_seconds:.0f}s "
                f"(download + parse + LLM pipeline deadline exceeded)",
                timings,
            )
        except Exception as e:
            ctx.status = "failed"
            logger.exception("[REQ %s] analyze_paper failed", rid)
            return _error_result(rid, str(e), timings)
        finally:
            await self._tracker.remove(rid)


def _error_result(
    request_id: str, error: str, timing: dict[str, float]
) -> dict[str, Any]:
    logger.error("[REQ %s] Returning error: %s", request_id, error)
    return {
        "request_id": request_id,
        "status": "error",
        "error": error,
        "timing": timing,
    }
