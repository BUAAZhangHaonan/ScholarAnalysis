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
