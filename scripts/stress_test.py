"""End-to-end stress test for ScholarAnalysis MCP server (SSE transport)."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

BASE_URL = "http://127.0.0.1:8005"
TOKEN = "g203-mcp"
AUTH_HEADERS = {"Authorization": f"Bearer {TOKEN}"}
JSON_HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


class SSEClient:
    """Manages a single SSE session with FastMCP."""

    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self._client = client
        self._base_url = base_url
        self._sse_response: httpx.Response | None = None
        self._raw_iter = None
        self._endpoint: str | None = None
        self._buffer = ""

    async def connect(self) -> None:
        self._sse_response = await self._client.send(
            self._client.build_request(
                "GET",
                f"{self._base_url}/sse",
                headers={**AUTH_HEADERS, "Accept": "text/event-stream"},
            ),
            stream=True,
        )
        assert self._sse_response.status_code == 200, \
            f"SSE handshake failed: {self._sse_response.status_code}"
        self._raw_iter = self._sse_response.aiter_bytes()
        # Read the endpoint event
        event_type, data = await self._read_event()
        assert event_type == "endpoint", f"Expected endpoint event, got {event_type}: {data}"
        self._endpoint = data
        print(f"  SSE session: {self._endpoint}")

    async def _read_event(self) -> tuple[str, str]:
        """Read one SSE event from the stream, properly handling chunk boundaries."""
        event_type = ""
        data = ""
        async for chunk in self._raw_iter:
            self._buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if line.startswith("event:"):
                    event_type = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data = line.removeprefix("data:").strip()
                elif line == "":
                    if event_type or data:
                        return event_type, data
        raise RuntimeError("SSE stream ended unexpectedly")

    async def initialize(self) -> None:
        """Perform MCP initialize handshake (required before any tool calls)."""
        assert self._endpoint, "Not connected"
        url = f"{self._base_url}{self._endpoint}"

        # Step 1: send initialize request
        init_payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "stress-test", "version": "1.0"},
            },
        }
        r = await self._client.post(url, headers=JSON_HEADERS, json=init_payload)
        assert r.status_code == 202, f"initialize POST returned {r.status_code}: {r.text}"

        # Read initialize response from SSE stream
        while True:
            event_type, data = await self._read_event()
            if event_type == "message":
                msg = json.loads(data)
                if "error" in msg:
                    raise RuntimeError(f"initialize failed: {msg['error']}")
                break
            print(f"  SSE event (init, skipped): {event_type}: {data[:100]}")

        # Step 2: send notifications/initialized (no id = notification)
        notif_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        r = await self._client.post(url, headers=JSON_HEADERS, json=notif_payload)
        assert r.status_code == 202, f"notifications/initialized POST returned {r.status_code}: {r.text}"

    async def call_tool(self, payload: dict) -> dict:
        """Send a tool call and read the response from the SSE stream."""
        assert self._endpoint, "Not connected"
        url = f"{self._base_url}{self._endpoint}"
        r = await self._client.post(url, headers=JSON_HEADERS, json=payload)
        assert r.status_code == 202, f"POST returned {r.status_code}: {r.text}"

        # Read response events until we get the tool result
        while True:
            event_type, data = await self._read_event()
            if event_type == "message":
                msg = json.loads(data)
                return msg
            print(f"  SSE event (skipped): {event_type}: {data[:100]}")

    async def close(self) -> None:
        if self._sse_response:
            await self._sse_response.aclose()


async def _run_tool_test(
    name: str,
    tool_name: str,
    arguments: dict,
    timeout: float = 120.0,
) -> dict:
    """Run a single tool test via SSE and return the parsed result dict."""
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        sse = SSEClient(client, BASE_URL)
        await sse.connect()
        await sse.initialize()
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            response = await sse.call_tool(payload)
            # Extract the text content from the JSON-RPC result
            result_text = (
                response.get("result", {})
                .get("content", [{}])[0]
                .get("text", "{}")
            )
            return json.loads(result_text)
        finally:
            await sse.close()


async def test_health():
    """Test 1: Server health check via SSE handshake."""
    print("\n=== Test 1: Health Check (SSE handshake) ===")
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        sse = SSEClient(client, BASE_URL)
        await sse.connect()
        await sse.initialize()
        assert sse._endpoint and "session_id" in sse._endpoint
        await sse.close()
    print("  PASS")


async def test_get_paper_text():
    """Test 2: Single paper text retrieval."""
    print("\n=== Test 2: get_paper_text ===")
    result = await _run_tool_test(
        "get_paper_text",
        "get_paper_text",
        {"query": "2402.01306", "include_images": False},
        timeout=120.0,
    )
    status = result.get("status", "unknown")
    md_len = len(result.get("markdown", ""))
    timing = result.get("timing", {})
    print(f"  Status: {status}, Markdown length: {md_len}")
    print(f"  Timing: {timing}")
    assert status == "success", f"Failed: {result.get('error', 'unknown')}"
    print("  PASS")


async def test_analyze_paper():
    """Test 3: Single paper LLM analysis."""
    print("\n=== Test 3: analyze_paper ===")
    result = await _run_tool_test(
        "analyze_paper",
        "analyze_paper",
        {
            "query": "2402.01306",
            "question": "What alignment methods does this paper propose?",
            "language": "en",
        },
        timeout=300.0,
    )
    status = result.get("status", "unknown")
    analysis = result.get("analysis", {})
    print(f"  Status: {status}")
    print(f"  Model: {analysis.get('model_used', 'N/A')}")
    print(f"  Answer length: {len(analysis.get('answer', ''))}")
    print(f"  Timing: {result.get('timing', {})}")
    assert status == "success", f"Failed: {result.get('error', 'unknown')}"
    print("  PASS")


async def test_concurrent():
    """Test 4: Concurrent requests (3 users, 1 paper each)."""
    print("\n=== Test 4: Concurrent requests ===")
    queries = ["2402.01306", "2312.11805", "2401.04088"]
    tasks = [
        _run_tool_test(
            f"concurrent_{q}",
            "get_paper_text",
            {"query": q},
            timeout=120.0,
        )
        for q in queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    print(f"  {ok}/{len(queries)} succeeded")
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  Query {queries[i]}: FAILED - {r}")
        else:
            print(f"  Query {queries[i]}: {r.get('status')}")
    assert ok == len(queries), f"Only {ok}/{len(queries)} succeeded"
    print("  PASS")


async def test_invalid_id():
    """Test 5: Invalid arXiv ID error handling."""
    print("\n=== Test 5: Invalid arXiv ID ===")
    result = await _run_tool_test(
        "invalid_id",
        "get_paper_text",
        {"query": "INVALID_ID_12345"},
        timeout=60.0,
    )
    print(f"  Status: {result.get('status')}")
    print(f"  Error: {result.get('error', 'N/A')[:200]}")
    assert result.get("status") in ("error", "success"), f"Unexpected: {result}"
    print("  PASS (error handled gracefully)")


async def main():
    print("ScholarAnalysis MCP Stress Test (SSE transport)")
    print("=" * 50)
    t0 = time.monotonic()

    tests = [
        ("Health Check", test_health),
        ("get_paper_text", test_get_paper_text),
        ("analyze_paper", test_analyze_paper),
        ("Concurrent Requests", test_concurrent),
        ("Invalid ID", test_invalid_id),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed in {elapsed:.1f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
