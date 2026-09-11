"""Shared download/parse path for bounded reads and optional focused analysis."""
from __future__ import annotations
import asyncio
import logging
import re
import time
import uuid
from weakref import WeakValueDictionary
from pathlib import Path
from typing import Any
import httpx
from scholar_analysis.clients.arxiv_mirror import ArxivMirrorClient, ArxivMirrorError
from scholar_analysis.clients.mineru import MinerUClient, MinerUError, extract_markdown
from scholar_analysis.config import get_settings
from scholar_analysis.cost import costed, cost_scope
from scholar_analysis.llm.post_processor import PostProcessor
from scholar_analysis.pipeline.errors import PipelineError, error_result
from scholar_analysis.pipeline.identifiers import normalize
from scholar_analysis.pipeline.parse_cache import ParseCache
from scholar_analysis.pipeline.analysis_cache import AnalysisCache
from scholar_analysis.pipeline.request_context import RequestTracker

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self._settings = get_settings()
        self._tracker = RequestTracker(self._settings.temp_dir, self._settings.request_max_age_seconds)
        self._arxiv = ArxivMirrorClient(self._settings.arxiv_mirror_base_url, self._settings.http_timeout)
        self._mineru = MinerUClient.from_settings(self._settings)
        self._post_processor = PostProcessor()
        self._arxiv_data_dir = Path(self._settings.arxiv_mirror_data_dir)
        self._parse_sem = asyncio.Semaphore(self._settings.max_concurrent_parses)
        self._parse_cache = ParseCache(self._settings.parse_cache_dir, self._settings.parse_cache_max_bytes)
        self._inflight: dict[tuple, asyncio.Task] = {}
        self._parse_locks = WeakValueDictionary()
        self._analysis_tasks = {}
        self._analysis_cache = AnalysisCache(Path(self._settings.parse_cache_dir)/"analyses", self._settings.parse_cache_max_bytes)

    async def aclose(self):
        for _, task in list(self._analysis_tasks.values()):
            task.cancel()
        await asyncio.gather(*(task for _, task in self._analysis_tasks.values()), return_exceptions=True)
        for task in list(self._inflight.values()):
            task.cancel()
        await asyncio.gather(*self._inflight.values(), return_exceptions=True)
        for obj in (self._arxiv, self._mineru, self._post_processor):
            await obj.aclose()

    async def _parse_with_cache(self, rid, versioned_id, pdf_path, *, lang="", refresh=False):
        lock = self._parse_locks.setdefault((versioned_id, lang), asyncio.Lock())
        async with lock:
            cached = None if refresh else await self._parse_cache.get(versioned_id, lang)
            if cached is not None:
                return cached
            async with self._parse_sem:
                result = await self._mineru.parse_pdf(pdf_path, lang_list=lang)
            if not extract_markdown(result).strip():
                raise PipelineError("PARSE_EMPTY", "parse", "Parser returned no usable Markdown.", retryable=True)
            await self._parse_cache.put(versioned_id, lang, result)
            return result

    async def _load(self, query="", pdf_url=None, document_id=None, *, lang="", refresh=False):
        if document_id:
            if query or pdf_url or refresh:
                raise PipelineError("INVALID_INPUT", "input", "document_id is a cache-only read; do not combine it with a source or refresh.")
            cached = await self._parse_cache.get(document_id, lang)
            if cached is None:
                raise PipelineError("DOCUMENT_NOT_CACHED", "cache", "Document is absent or expired. Read the original source again.")
            return cached, document_id, True
        kind, value = normalize(query, pdf_url)
        key = value if kind == "arxiv" else await self._parse_cache.document_key(value)
        cached = None if refresh else await self._parse_cache.get(key, lang)
        if cached is not None:
            return cached, cached.get("_paper", {}).get("versioned_id") or key, True
        flight_key = (key, lang)
        task = self._inflight.get(flight_key)
        shared = task is not None
        if task is None:
            task = asyncio.create_task(self._fetch(kind, value, key, lang, refresh))
            self._inflight[flight_key] = task
            def done(t):
                self._inflight.pop(flight_key, None)
                if not t.cancelled():
                    t.exception()  # retrieve errors even if all callers disconnected
            task.add_done_callback(done)
        result, document = await asyncio.shield(task)
        return result, document, shared

    async def _fetch(self, kind, value, key, lang, refresh):
        # This context belongs to the shared work, not to any single waiter.
        ctx = await self._tracker.create()
        try:
            async with asyncio.timeout(self._settings.request_max_age_seconds):
                cached = None if refresh else await self._parse_cache.get(key, lang)
                if cached is not None:
                    return cached, cached.get("_paper", {}).get("versioned_id") or key
                if kind == "url":
                    ctx.status = "pdf_download"
                    async with self._parse_sem:
                        result = await self._mineru.parse_from_url(value, ctx.temp_dir, lang_list=lang)
                    document = key
                    paper = {"arxiv_id": None, "versioned_id": None, "title": "", "authors": [], "abstract": ""}
                else:
                    ctx.status = "mirror_resolve"
                    info = await self._arxiv.resolve(value)
                    document = info.versioned_id
                    result = None if refresh else await self._parse_cache.get(document, lang)
                    if result is None:
                        ctx.status = "mirror_download"
                        asset = await self._arxiv.download(value)
                        document = asset.versioned_id
                        path = (self._arxiv_data_dir / asset.local_path).resolve()
                        if not path.is_relative_to(self._arxiv_data_dir.resolve()) or not path.is_file():
                            raise PipelineError("PDF_FILE_UNAVAILABLE", "mirror_download", "Mirror PDF is not available in the configured shared data directory.", retryable=True)
                        ctx.status = "parse"
                        result = await self._parse_with_cache(ctx.request_id, document, path, lang=lang, refresh=refresh)
                    paper = {"arxiv_id": info.arxiv_id, "versioned_id": document, "title": info.title,
                             "authors": info.authors, "abstract": info.abstract}
                    result.setdefault("_source", {"original_url": "https://arxiv.org/abs/" + value,
                                                   "final_url": "https://arxiv.org/pdf/" + document,
                                                   "document_type": "pdf"})
                if not isinstance(result, dict) or not extract_markdown(result).strip():
                    raise PipelineError("PARSE_EMPTY", "parse", "Parser returned no usable Markdown.", retryable=True)
                result["_paper"] = paper
                result.setdefault("_revision", uuid.uuid4().hex)
                await self._parse_cache.put(document, lang, result)
                if key != document:
                    await self._parse_cache.put(key, lang, result)
                return result, document
        except PipelineError:
            raise
        except TimeoutError as exc:
            raise PipelineError("PIPELINE_TIMEOUT", ctx.status, "Document request exceeded its deadline.", retryable=True) from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise PipelineError("MIRROR_HTTP_ERROR", ctx.status, f"arXiv mirror returned HTTP {code}.",
                                retryable=code in (429, 500, 502, 503, 504), details={"http_status": code}) from exc
        except httpx.RequestError as exc:
            raise PipelineError("MIRROR_NETWORK_ERROR", ctx.status, "Unable to reach the arXiv mirror.", retryable=True) from exc
        except ArxivMirrorError as exc:
            raise PipelineError("MIRROR_OPERATION_FAILED", ctx.status,
                                "arXiv mirror could not resolve or download the document.",
                                retryable=ctx.status == "mirror_download") from exc
        except MinerUError as exc:
            raise PipelineError("PARSE_FAILED", "parse", "PDF parsing failed on the configured parser endpoints.", retryable=True) from exc
        finally:
            await self._tracker.remove(ctx.request_id)

    @costed
    async def get_paper_text(self, query="", include_images=False, *, pdf_url=None,
                             document_id=None, offset=0, limit_chars=64000,
                             find_text=None, lang="", refresh=False):
        rid, start = uuid.uuid4().hex, time.monotonic()
        try:
            if isinstance(offset, bool) or offset < 0 or not 1 <= limit_chars <= 64000:
                raise PipelineError("INVALID_RANGE", "input", "offset must be nonnegative and limit_chars between 1 and 64000.")
            result, document, cache_hit = await self._load(query, pdf_url, document_id, lang=lang, refresh=refresh)
            markdown = extract_markdown(result, text_only=not include_images)
            body = read_slice(markdown, offset, limit_chars, find_text)
            return {"request_id": rid, "status": "success", "paper": result.get("_paper", {"arxiv_id": document, "versioned_id": document}),
                    "document_id": document, "document_revision": result.get("_revision") or "legacy:"+document,
                    "source": result.get("_source") or ({"original_url": "https://arxiv.org/abs/"+document,
                        "final_url": "https://arxiv.org/pdf/"+document, "document_type": "pdf"} if not document.startswith("doc_") else {}), "cache_hit": cache_hit,
                    "mode": "image_references" if include_images else "text_only",
                    "image_capability": "References only; images are not fetched or visually interpreted.",
                    "parse_coverage": "Parser Markdown only; PDF page coverage/completeness is not certified.",
                    **body, "timing": {"total_s": round(time.monotonic()-start, 3)}}
        except PipelineError as exc:
            return error_result(rid, exc, {"total_s": round(time.monotonic()-start, 3)})
        except Exception:
            logger.exception("Document read failed [request_id=%s]", rid)
            return error_result(rid, PipelineError("INTERNAL_ERROR", "read", "Document read failed; use request_id to inspect the server log."))

    @costed
    async def analyze_paper(self, query, question, language="en", include_images=False,
                            *, pdf_url=None, document_id=None, lang="", offset=0,
                            limit_chars=None, analysis_id=None):
        key = analysis_id or "analysis_"+uuid.uuid4().hex
        args = dict(query=query, question=question, language=language, include_images=include_images,
                    pdf_url=pdf_url, document_id=document_id, lang=lang, offset=offset, limit_chars=limit_chars)
        try:
            saved = await self._analysis_cache.get(key)
            active = self._analysis_tasks.get(key)
            if saved is not None:
                if saved["input"] != args:
                    raise PipelineError("ANALYSIS_ID_CONFLICT", "input", "analysis_id is already bound to different inputs.")
                if active is None:
                    return {**saved["result"], "analysis_reused": True, "cache_hit": True}
            reused = active is not None
            if active is not None and active[0] != args:
                raise PipelineError("ANALYSIS_ID_CONFLICT", "input", "analysis_id is already bound to different inputs.")
            if active is None:
                task = asyncio.create_task(self._saved_analysis(key, args))
                self._analysis_tasks[key] = (args, task)
                def done(t):
                    self._analysis_tasks.pop(key, None)
                    if not t.cancelled():
                        t.exception()
                task.add_done_callback(done)
            else:
                task = active[1]
            result = await asyncio.shield(task)
            return {**result, "analysis_reused": reused, **({"cache_hit": True} if reused else {})}
        except PipelineError as exc:
            return {**error_result(uuid.uuid4().hex, exc), "analysis_id": key}

    async def _saved_analysis(self, key, args):
        with cost_scope() as ledger:
            pending = {**error_result(key, PipelineError(
                "ANALYSIS_OUTCOME_UNRESOLVED", "analysis",
                "The prior task did not record a final result. It will not be regenerated automatically; inspect its outcome before using a new analysis_id.")),
                "analysis_id": key, "original_cost": None}
            await self._analysis_cache.put(key, {"input": args, "result": pending, "state": "running"})
            result = await self._analyze_once(**args)
            result["analysis_id"] = key
            result["original_cost"] = ledger.report()
            await self._analysis_cache.put(key, {"input": args, "result": result, "state": "finished"})
            return result

    async def _analyze_once(self, query, question, language="en", include_images=False,
                            *, pdf_url=None, document_id=None, lang="", offset=0, limit_chars=None):
        rid, start = uuid.uuid4().hex, time.monotonic()
        try:
            if not question.strip() or language not in ("en", "zh"):
                raise PipelineError("INVALID_INPUT", "input", "Provide a question and language en or zh.")
            async with asyncio.timeout(self._settings.request_max_age_seconds):
                result, document, cached = await self._load(query, pdf_url, document_id, lang=lang)
                markdown = extract_markdown(result, text_only=not include_images)
                if offset < 0 or offset > len(markdown) or (limit_chars is not None and not 1 <= limit_chars <= 64000):
                    raise PipelineError("INVALID_RANGE", "input", "Select a valid offset and optional limit_chars from 1 to 64000.")
                excerpt = markdown[offset:offset+limit_chars] if limit_chars is not None else markdown[offset:]
                analysis = await self._post_processor.extract(excerpt, question, language,
                                                              source_offset=offset, total_chars=len(markdown))
            return {"request_id": rid, "status": "success", "paper": result.get("_paper", {}),
                    "document_id": document, "document_revision": result.get("_revision") or "legacy:"+document,
                    "source": result.get("_source") or ({"original_url": "https://arxiv.org/abs/"+document,
                        "final_url": "https://arxiv.org/pdf/"+document, "document_type": "pdf"} if not document.startswith("doc_") else {}), "cache_hit": cached,
                    "analysis": {"question": question, **analysis},
                    "image_capability": "Text analysis only; retained image references are not visual evidence.",
                    "timing": {"total_s": round(time.monotonic()-start, 3)}}
        except PipelineError as exc:
            return error_result(rid, exc)
        except TimeoutError:
            return error_result(rid, PipelineError("ANALYSIS_TIMEOUT", "analysis", "Analysis deadline exceeded.", retryable=True))
        except Exception:
            logger.exception("Analysis failed [request_id=%s]", rid)
            return error_result(rid, PipelineError("ANALYSIS_FAILED", "analysis", "Analysis failed; use request_id to inspect the server log."))

def read_slice(text: str, offset: int, limit: int, find_text: str | None = None) -> dict:
    if offset > len(text):
        raise PipelineError("INVALID_RANGE", "input", "offset exceeds document length.")
    matches = []
    if find_text is not None:
        if not find_text:
            raise PipelineError("INVALID_INPUT", "input", "find_text must not be empty.")
        pos = text.find(find_text, offset)
        while pos >= 0 and len(matches) < 20:
            matches.append({"start": pos, "end": pos+len(find_text),
                            "line": text.count("\n", 0, pos)+1})
            pos = text.find(find_text, pos+max(1, len(find_text)))
        if matches:
            offset = max(offset, matches[0]["start"] - min(400, limit//4))
    end = min(len(text), offset+limit)
    headings = []
    current = None
    for match in re.finditer(r"(?m)^#{1,6} +(.+)$", text):
        entry = {"title": match[1], "start": match.start(), "line": text.count("\n", 0, match.start())+1}
        if match.start() <= offset:
            current = entry
        elif match.start() < end:
            headings.append(entry)
    if current:
        headings.insert(0, current)
    return {"markdown": text[offset:end], "range": {"start": offset, "end": end},
            "total_chars": len(text), "next_offset": end if end < len(text) else None,
            "eof": end >= len(text), "truncated": offset != 0 or end < len(text),
            "coverage": "full_parsed_text" if offset == 0 and end == len(text) else "partial_parsed_text",
            "line_range": {"start": text.count("\n", 0, offset)+1, "end": text.count("\n", 0, end)+1},
            "sections": headings, "matches": matches, "page_numbers_available": False,
            "locator_basis": "Unicode character offsets and Markdown lines in the requested image mode"}
