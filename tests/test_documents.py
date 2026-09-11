"""Offline contract tests: no service, model, credentials or shared cache used."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import httpx
from scholar_analysis.clients.mineru import MinerUClient
from scholar_analysis.config import Settings
from scholar_analysis.pipeline.errors import PipelineError
from scholar_analysis.pipeline.identifiers import normalize
from scholar_analysis.pipeline.orchestrator import Orchestrator, read_slice
from scholar_analysis.pipeline.parse_cache import ParseCache

def parsed(text="# Methods\nA method.\n\n## Table 1\n| ours | 40.44 |\n\n$$a+b=c$$"):
    return {"results": {"paper": {"md_content": text}}}

class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = ParseCache(self.tmp.name, 10**7)
    async def asyncTearDown(self):
        self.tmp.cleanup()
    async def test_empty_not_cached_and_bad_cache_recovers(self):
        with self.assertRaises(ValueError):
            await self.cache.put("2401.00001v1", "", {"results": {}})
        Path(self.tmp.name, "2401.00001v1__default.json").write_text('{"results": {}}')
        self.assertIsNone(await self.cache.get("2401.00001v1"))
        await self.cache.put("2401.00001v1", "", parsed())
        self.assertIsNotNone(await self.cache.get("2401.00001v1"))
    async def test_atomic_concurrent_writes_and_legacy_id(self):
        await asyncio.gather(*(self.cache.put("hep-th/9901001v2", "", parsed(str(i))) for i in range(20)))
        self.assertIsNotNone(await self.cache.get("hep-th/9901001v2"))
        self.assertFalse(list(Path(self.tmp.name).glob("*.tmp")))
    async def test_url_catalog_stable_and_distinct(self):
        keys = await asyncio.gather(*(self.cache.document_key("https://papers.example/a.pdf") for _ in range(10)))
        self.assertEqual(len(set(keys)), 1)
        self.assertNotEqual(keys[0], await self.cache.document_key("https://papers.example/b.pdf"))
    async def test_language_is_escaped(self):
        await self.cache.put("2401.00001v1", "../en", parsed())
        self.assertIsNotNone(await self.cache.get("2401.00001v1", "../en"))

class URLTests(unittest.IsolatedAsyncioTestCase):
    async def run_url(self, handler, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            client = MinerUClient([("https://parser.example", "", "")])
            client._download_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
            client.parse_pdf = AsyncMock(return_value=parsed())
            try:
                result = await client.parse_from_url("https://doi.org/10.1234/test", Path(d), **kwargs)
                self.assertFalse(list(Path(d).glob("*.pdf")))
                return result, client.parse_pdf.await_count
            finally:
                await client.aclose()
    async def test_doi_declared_pdf_and_redirect(self):
        seen = []
        def handler(req):
            seen.append(str(req.url))
            if req.url.host == "doi.org":
                return httpx.Response(302, headers={"location": "https://aclanthology.org/example/"})
            if req.url.path.endswith(".pdf"):
                return httpx.Response(200, content=b"%PDF-1.4 fixture", headers={"content-type": "application/pdf"})
            return httpx.Response(200, text='<meta name="citation_pdf_url" content="/example.pdf">')
        result, calls = await self.run_url(handler)
        self.assertEqual(calls, 1)
        self.assertEqual(result["_source"]["final_url"], "https://aclanthology.org/example.pdf")
        self.assertEqual(len(seen), 3)
    async def test_nonpdf_does_not_parse(self):
        with self.assertRaises(PipelineError) as e:
            await self.run_url(lambda _: httpx.Response(200, text="<html>paywall</html>"))
        self.assertEqual(e.exception.code, "PDF_LINK_REQUIRED")
    async def test_pdf_size_is_bounded(self):
        with self.assertRaises(PipelineError) as e:
            await self.run_url(lambda _: httpx.Response(200, content=b"%PDF-"+b"x"*100), max_bytes=20)
        self.assertEqual(e.exception.code, "PDF_TOO_LARGE")
    async def test_parser_empty_200_tries_next_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "a.pdf")
            path.write_bytes(b"%PDF-fixture")
            client = MinerUClient([("https://one.example", "", ""), ("https://two.example", "", "")])
            client._clients = [
                httpx.AsyncClient(base_url="https://one.example", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"error": "empty"}))),
                httpx.AsyncClient(base_url="https://two.example", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=parsed()))),
            ]
            try:
                self.assertIn("results", await client.parse_pdf(path))
            finally:
                await client.aclose()

class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings = Settings(_env_file=None, temp_dir=str(root/"tmp"), parse_cache_dir=str(root/"cache"),
                                 arxiv_mirror_data_dir=str(root), mineru_base_url="https://parser.example")
        self.arxiv = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(
            arxiv_id="2401.00001", versioned_id="2401.00001v2", title="Paper", authors=[], abstract="")),
            download=AsyncMock(return_value=SimpleNamespace(versioned_id="2401.00001v2", local_path="a.pdf")),
            aclose=AsyncMock())
        (root/"a.pdf").write_bytes(b"%PDF-fixture")
        self.mineru = SimpleNamespace(parse_pdf=AsyncMock(return_value=parsed()),
                                     parse_from_url=AsyncMock(return_value=parsed()), aclose=AsyncMock())
        with patch("scholar_analysis.pipeline.orchestrator.get_settings", return_value=self.settings), \
             patch("scholar_analysis.pipeline.orchestrator.ArxivMirrorClient", return_value=self.arxiv), \
             patch("scholar_analysis.pipeline.orchestrator.MinerUClient.from_settings", return_value=self.mineru):
            self.orch = Orchestrator()
    async def asyncTearDown(self):
        await self.orch.aclose()
        self.tmp.cleanup()
    async def test_warm_versioned_cache_skips_mirror(self):
        await self.orch._parse_cache.put("2401.00001v2", "", parsed())
        self.arxiv.resolve.side_effect = RuntimeError("mirror unavailable")
        result = await self.orch.get_paper_text("2401.00001v2")
        self.assertEqual(result["status"], "success")
        self.arxiv.resolve.assert_not_awaited()
    async def test_concurrent_same_document_single_flight(self):
        results = await asyncio.gather(*(self.orch.get_paper_text("2401.00001") for _ in range(10)))
        self.assertTrue(all(r["status"] == "success" for r in results))
        self.assertEqual(self.arxiv.download.await_count, 1)
        self.assertEqual(self.mineru.parse_pdf.await_count, 1)
    async def test_cancelling_waiter_does_not_cancel_shared_work(self):
        started, finish = asyncio.Event(), asyncio.Event()
        async def parse(*args, **kwargs):
            started.set()
            await finish.wait()
            return parsed()
        self.mineru.parse_pdf.side_effect = parse
        first = asyncio.create_task(self.orch.get_paper_text("2401.00001"))
        await started.wait()
        second = asyncio.create_task(self.orch.get_paper_text("2401.00001"))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        finish.set()
        self.assertEqual((await second)["status"], "success")
        self.assertEqual(self.mineru.parse_pdf.await_count, 1)
        self.assertEqual(self.orch._tracker.active_count, 0)
    async def test_pdf_input_and_cached_pagination(self):
        first = await self.orch.get_paper_text(pdf_url="https://aclanthology.org/a.pdf", limit_chars=10)
        second = await self.orch.get_paper_text(document_id=first["document_id"], offset=10, limit_chars=10)
        self.assertEqual(first["status"], "success")
        self.assertEqual(second["range"]["start"], 10)
        self.assertEqual(self.mineru.parse_from_url.await_count, 1)
        self.arxiv.resolve.assert_not_awaited()
    async def test_mirror_502_stage_is_download_not_parser(self):
        self.arxiv.download.side_effect = httpx.HTTPStatusError("bad", request=httpx.Request("POST", "https://mirror/resolve-and-download"), response=httpx.Response(502))
        result = await self.orch.get_paper_text("2401.00001")
        self.assertEqual(result["error_code"], "MIRROR_HTTP_ERROR")
        self.assertEqual(result["stage"], "mirror_download")
        self.mineru.parse_pdf.assert_not_awaited()
    async def test_refresh_reparses_but_new_version_not_old_cache(self):
        await self.orch.get_paper_text("2401.00001")
        await self.orch.get_paper_text("2401.00001", refresh=True)
        self.assertEqual(self.mineru.parse_pdf.await_count, 2)
        self.assertIsNone(await self.orch._parse_cache.get("2401.00001v1"))
    async def test_invalid_range_fails_before_download(self):
        result = await self.orch.get_paper_text("2401.00001", limit_chars=64001)
        self.assertEqual(result["error_code"], "INVALID_RANGE")
        self.arxiv.resolve.assert_not_awaited()

class LocationTests(unittest.TestCase):
    def test_pages_reconstruct_text_and_keep_table_formula(self):
        text = "# Methods\n| ours | 40.44 |\n\n$$a+b=c$$\n"
        chunks = [read_slice(text, n, 8)["markdown"] for n in range(0, len(text), 8)]
        self.assertEqual("".join(chunks), text)
        match = read_slice(text, 0, 30, "40.44")
        self.assertEqual(text[match["matches"][0]["start"]:match["matches"][0]["end"]], "40.44")
        self.assertEqual(match["sections"][0]["title"], "Methods")
        self.assertFalse(match["page_numbers_available"])
    def test_identifiers(self):
        for query in ("2401.00001v2", "https://arxiv.org/pdf/2401.00001v2.pdf", "hep-th/9901001v2"):
            self.assertEqual(normalize(query)[0], "arxiv")
        self.assertEqual(normalize("10.1234/abc"), ("url", "https://doi.org/10.1234/abc"))
        with self.assertRaises(PipelineError):
            normalize("A paper title")
