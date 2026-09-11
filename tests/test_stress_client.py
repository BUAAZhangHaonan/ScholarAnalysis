"""Client-only auth regression fixtures; no external endpoint or real secret."""
import asyncio
import importlib.util
import logging
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import httpx

spec = importlib.util.spec_from_file_location("sa_stress_client", Path(__file__).resolve().parents[1]/"scripts/stress_test.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

class HeaderTests(unittest.TestCase):
    def test_loads_env_file_when_shell_token_not_exported(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            path = Path(d,"service.env")
            path.write_text("SCHOLAR_ANALYSIS_ACCESS_TOKEN=fixture-file-token\n")
            self.assertEqual(client.client_headers(path)["Authorization"], "Bearer fixture-file-token")
    def test_shell_token_has_priority(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"SCHOLAR_ANALYSIS_ACCESS_TOKEN":"fixture-shell"}, clear=True):
            path = Path(d,"service.env")
            path.write_text("SCHOLAR_ANALYSIS_ACCESS_TOKEN=fixture-file\n")
            self.assertEqual(client.client_headers(path)["Authorization"], "Bearer fixture-shell")
    def test_missing_token_fails_before_session(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(client.ClientFailure) as exc:
                client.client_headers(Path(d,"absent.env"))
            self.assertEqual(exc.exception.code, "AUTH_TOKEN_MISSING")
            self.assertEqual(client.client_headers(Path(d,"absent.env"), no_auth=True), {})

class SSEStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"event: endpoint\ndata: /messages/?session_id=offline\n\n"
        await asyncio.Event().wait()

class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_401_exits_promptly_even_though_sse_get_is_200(self):
        seen = []
        def handler(request):
            seen.append((request.method, request.url.path))
            if request.method == "GET":
                return httpx.Response(200, headers={"content-type":"text/event-stream"}, stream=SSEStream())
            return httpx.Response(401, json={"error":"unauthorized"})
        def factory(**kwargs):
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
        started = time.monotonic()
        with patch("mcp.shared._httpx_utils.create_mcp_http_client", side_effect=factory), \
             self.assertLogs("mcp.client.sse", level=logging.ERROR):
            with self.assertRaises(client.ClientFailure) as exc:
                await client.call_mcp("https://offline.example", {"Authorization":"Bearer fixture-wrong"}, 2,
                                      "get_paper_text", {"query":"1409.1556v1"})
        self.assertEqual(exc.exception.code, "AUTH_REJECTED")
        self.assertLess(time.monotonic()-started, 1)
        self.assertIn(("GET","/sse"), seen)
        self.assertIn(("POST","/messages/"), seen)

    async def test_stalled_session_has_total_deadline(self):
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, headers={"content-type":"text/event-stream"}, stream=SSEStream())
            return httpx.Response(202)
        def factory(**kwargs):
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
        with patch("mcp.shared._httpx_utils.create_mcp_http_client", side_effect=factory):
            with self.assertRaises(client.ClientFailure) as exc:
                await client.call_mcp("https://offline.example", {}, .05, "get_paper_text", {})
        self.assertEqual(exc.exception.code, "MCP_CLIENT_TIMEOUT")


class CostRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **updates):
        from types import SimpleNamespace
        values = dict(env_file=None, no_auth=True, url="https://offline.example",
                      timeout=1, pdf_url=None, query="1409.1556v1", analysis=True,
                      analysis_run_id="recoverable", concurrency=1, requests=1,
                      distinct_analysis_ids=False, question="Short fixture question",
                      language="en", output=None)
        values.update(updates)
        return SimpleNamespace(**values)

    def warmup(self):
        return {"status":"success", "document_id":"fixture-paper", "document_revision":"v1",
                "markdown":"text", "total_chars":4, "range":{"start":0,"end":4},
                "source":{"final_url":"https://offline.example/paper.pdf"},
                "cost":{"schema_version":"mcp.cost.v1", "model_calls":0, "attempts":[],
                        "complete":True, "lower_cny":"0", "upper_cny":"0", "total_cny":"0"}}

    async def fail_analysis(self, dispatched):
        from unittest.mock import AsyncMock
        calls = AsyncMock(side_effect=[self.warmup(), client.ClientFailure(
            "FIXTURE_CONNECTION_LOST", "fixture", tool_dispatched=dispatched)])
        with patch.object(client, "call_mcp", calls), patch.object(client, "record_request_start"):
            return await client.live(self.args())

    async def test_before_tool_dispatch_failure_is_known_zero(self):
        result = await self.fail_analysis(False)
        self.assertEqual(result["summary"]["provider_model_calls"], 0)
        self.assertEqual(result["summary"]["provider_cost_upper_cny"], "0")
        self.assertEqual(result["summary"]["unknown_calls"], 0)
        self.assertEqual(result["requests"][0]["analysis_id"], "stress_recoverable")

    async def test_missing_analysis_receipt_is_unknown_not_free(self):
        result = await self.fail_analysis(True)
        self.assertEqual(result["summary"]["provider_model_calls"], 0)
        self.assertFalse(result["summary"]["provider_model_calls_complete"])
        self.assertIsNone(result["summary"]["provider_cost_upper_cny"])
        self.assertEqual(result["summary"]["unknown_calls"], 1)
        self.assertEqual(result["summary"]["missing_analysis_receipts"], ["stress_recoverable"])

    async def test_rerun_preserves_ids_and_journals_before_call(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            args = self.args(output=Path(d, "report.json"), requests=2, distinct_analysis_ids=True)
            async def call(url, headers, timeout, tool, values):
                if tool == "get_paper_text":
                    return self.warmup()
                journal = Path(str(args.output)+".requests.jsonl").read_text()
                self.assertIn(values["analysis_id"], journal)
                seen.append(values["analysis_id"])
                raise client.ClientFailure("LOST", "fixture", tool_dispatched=True)
            with patch.object(client, "call_mcp", side_effect=call), patch("sys.stderr"):
                a = await client.live(args)
                b = await client.live(args)
        self.assertEqual(seen, ["stress_recoverable_0", "stress_recoverable_1"]*2)
        self.assertEqual(a["summary"]["missing_analysis_receipts"], b["summary"]["missing_analysis_receipts"])

    async def test_analysis_requires_stable_batch_id(self):
        with self.assertRaisesRegex(ValueError, "analysis-run-id"):
            await client.live(self.args(analysis_run_id=None))

    async def test_transport_marks_timeout_after_tool_dispatch(self):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def sse(*args, **kwargs):
            yield None, None
        class Session:
            def __init__(self, *args):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def initialize(self):
                return None
            async def call_tool(self, *args):
                await asyncio.Event().wait()
        with patch("mcp.client.sse.sse_client", sse), patch("mcp.ClientSession", Session):
            with self.assertRaises(client.ClientFailure) as exc:
                await client.call_mcp("https://offline.example", {}, .02, "analyze_paper", {})
        self.assertTrue(exc.exception.tool_dispatched)

    def test_zero_incremental_reuse_does_not_resolve_missing_original_receipt(self):
        missing = {"analysis_id":"same", "analysis_receipt_missing":True}
        reused = {"analysis_id":"same", "cost":self.warmup()["cost"]}
        summary = client.summarize_costs([reused["cost"]], [missing, reused])
        self.assertIsNone(summary["provider_cost_upper_cny"])
        self.assertEqual(summary["missing_analysis_receipts"], ["same"])
