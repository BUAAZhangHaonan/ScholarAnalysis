#!/usr/bin/env python3
"""Bounded offline or explicit live MCP acceptance; never prints credentials."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered)-1, max(0, int((len(ordered)-1)*fraction+0.5)))]

def check_cost(value):
    cost = value.get("cost")
    if not isinstance(cost, dict) or cost.get("schema_version") != "mcp.cost.v1":
        raise AssertionError("Missing mcp.cost.v1")
    if cost["model_calls"] == 0 and cost["total_cny"] is not None:
        assert Decimal(cost["total_cny"]) == 0
    if cost["complete"]:
        assert cost["lower_cny"] is not None and cost["upper_cny"] is not None
        if Decimal(cost["lower_cny"]) == Decimal(cost["upper_cny"]):
            assert cost["total_cny"] is not None
        else:
            assert cost["total_cny"] is None
    else:
        assert cost["total_cny"] is None

class ClientFailure(RuntimeError):
    def __init__(self, code, message, *, tool_dispatched=False):
        super().__init__(message)
        self.code = code
        self.tool_dispatched = tool_dispatched

def client_headers(env_file=None, no_auth=False):
    """Use the deployment's configuration without printing or changing secrets."""
    if no_auth:
        return {}
    token = os.environ.get("SCHOLAR_ANALYSIS_ACCESS_TOKEN", "").strip()
    if not token:
        from scholar_analysis.config import Settings
        path = env_file or Path(__file__).resolve().parents[1]/".env"
        token = Settings(_env_file=path).access_token.strip()
    if not token:
        raise ClientFailure("AUTH_TOKEN_MISSING",
            "No client token configured. Set SCHOLAR_ANALYSIS_ACCESS_TOKEN or --env-file. "
            "Use --no-auth only for a deliberately unauthenticated endpoint.")
    return {"Authorization": "Bearer "+token}

async def call_mcp(base_url, headers, timeout, tool, arguments):
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    rejected = asyncio.Event()
    tool_dispatched = False

    async def check_auth(response):
        if response.status_code in (401, 403):
            rejected.set()

    def client_factory(**kwargs):
        client = create_mcp_http_client(**kwargs)
        client.event_hooks.setdefault("response", []).append(check_auth)
        return client

    async def exchange():
        nonlocal tool_dispatched
        async with sse_client(base_url.rstrip("/")+"/sse", headers=headers,
                              timeout=min(30, timeout), sse_read_timeout=timeout,
                              httpx_client_factory=client_factory) as (read, write):
            async with ClientSession(read, write) as session:
                # The SDK logs POST failures but may leave initialization waiting.
                await asyncio.wait_for(session.initialize(), timeout=min(30, timeout))
                tool_dispatched = True
                result = await session.call_tool(tool, arguments)
                if result.isError:
                    raise ClientFailure("MCP_TOOL_ERROR", "MCP tool wrapper returned an error.")
                text = next((b.text for b in result.content if b.type == "text"), "")
                value = json.loads(text)
                check_cost(value)
                return value

    request = asyncio.create_task(exchange())
    auth_wait = asyncio.create_task(rejected.wait())
    try:
        async with asyncio.timeout(timeout):
            done, _ = await asyncio.wait((request, auth_wait), return_when=asyncio.FIRST_COMPLETED)
            if rejected.is_set():
                raise ClientFailure("AUTH_REJECTED",
                    "MCP returned HTTP 401/403. Verify the client environment or --env-file; the service token was not changed.")
            return await request
    except TimeoutError as exc:
        raise ClientFailure("MCP_CLIENT_TIMEOUT", "MCP client session exceeded its deadline.",
                            tool_dispatched=tool_dispatched) from exc
    except Exception as exc:
        if isinstance(exc, ClientFailure):
            exc.tool_dispatched = tool_dispatched
            raise
        raise ClientFailure("MCP_EXCHANGE_FAILED", "MCP exchange failed; inspect the request receipt.",
                            tool_dispatched=tool_dispatched) from exc
    finally:
        for task in (request, auth_wait):
            if not task.done():
                task.cancel()
        await asyncio.gather(request, auth_wait, return_exceptions=True)

def summarize_costs(costs, results):
    """Count observed attempts; missing receipts are not invented provider calls."""
    attempts = {}
    for cost in costs:
        for attempt in cost["attempts"]:
            key = (attempt.get("request_id"), attempt["started_at"], attempt["model_requested"])
            attempts[key] = attempt
    received = {r["analysis_id"] for r in results
                if r.get("analysis_id") and (r.get("cost") or {}).get("attempts")}
    missing = {r["analysis_id"] for r in results
               if r.get("analysis_receipt_missing") and r["analysis_id"] not in received}
    low = sum((Decimal(a["lower_cny"] or "0") for a in attempts.values()), Decimal(0))
    unmetered = sum(a["upper_cny"] is None for a in attempts.values())
    high = None if missing or unmetered else sum(
        (Decimal(a["upper_cny"]) for a in attempts.values()), Decimal(0))
    return {"provider_model_calls":len(attempts), "provider_model_calls_complete":not bool(missing),
            "provider_cost_lower_cny":str(low),
            "provider_cost_upper_cny":str(high) if high is not None else None,
            "unknown_calls":unmetered+len(missing),
            "missing_analysis_receipts":sorted(missing)}

def record_request_start(args, index, analysis_id):
    event = {"event":"analysis_request_start", "index":index, "analysis_id":analysis_id}
    line = json.dumps(event, ensure_ascii=False)+"\n"
    if args.output:
        path = Path(str(args.output)+".requests.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
    print(line, end="", file=sys.stderr, flush=True)

async def live(args):
    if args.analysis and not getattr(args, "analysis_run_id", None):
        raise ValueError("Live analysis requires --analysis-run-id for recoverable requests.")
    try:
        headers = client_headers(args.env_file, args.no_auth)
    except ClientFailure as exc:
        return {"mode":"live", "passed":False, "error_code":exc.code, "error":str(exc), "requests":[]}
    async def call(tool, arguments):
        return await call_mcp(args.url, headers, args.timeout, tool, arguments)

    source = {"pdf_url": args.pdf_url} if args.pdf_url else {"query": args.query}
    warm_start = time.monotonic()
    try:
        first = await call("get_paper_text", {**source, "limit_chars": 1000})
    except Exception as exc:
        return {"mode":"live", "passed":False, "requests":[],
                "warmup":{"status":"error", "error_code":getattr(exc, "code", type(exc).__name__),
                          "seconds":round(time.monotonic()-warm_start, 3)},
                "error":str(exc) if isinstance(exc, ClientFailure) else "MCP warmup failed."}
    warmup = {"seconds": round(time.monotonic()-warm_start, 3), "status": first.get("status"),
              "error_code": first.get("error_code"), "stage": first.get("stage"),
              "total_chars": first.get("total_chars"), "source": first.get("source"), "cost": first["cost"]}
    if first.get("status") != "success":
        return {"mode":"live", "warmup":warmup, "passed":False, "requests":[]}
    assert first["markdown"] and first["source"].get("final_url")
    assert first["range"]["start"] == 0
    assert first["range"]["end"] == len(first["markdown"])
    document = first["document_id"]
    analysis_key = "stress_"+args.analysis_run_id if args.analysis else None
    gate = asyncio.Semaphore(args.concurrency)
    costs = []
    async def one(index):
        async with gate:
            started = time.monotonic()
            analysis_id = analysis_key+("_"+str(index) if args.distinct_analysis_ids else "") if args.analysis else None
            receipt = None
            invoked = False
            try:
                if args.analysis:
                    values = {"document_id": document, "question":args.question, "language":args.language,
                              "analysis_id":analysis_id}
                    tool = "analyze_paper"
                    record_request_start(args, index, analysis_id)
                else:
                    values = {"document_id":document, "offset":min(1000, first["total_chars"]), "limit_chars":1000}
                    tool = "get_paper_text"
                invoked = True
                result = await call(tool, values)
                costs.append(result["cost"])
                receipt = result["cost"]
                status = result.get("status")
                if status == "success":
                    if args.analysis:
                        assert result["analysis"]["answer"].strip()
                        assert result["analysis"]["finish_reason"] == "stop"
                        assert "coverage" in result["analysis"]
                        assert "original_cost" in result
                    else:
                        assert result["cache_hit"]
                        assert result["document_revision"] == first["document_revision"]
                        assert result["range"]["start"] == values["offset"]
                        assert result["cost"]["model_calls"] == 0
                return {"index": index, "status":status, "error_code":result.get("error_code"),
                        "stage":result.get("stage"), "request_id":result.get("request_id"),
                        "analysis_id":analysis_id or result.get("analysis_id"), "analysis_reused":result.get("analysis_reused"),
                        "cache_hit":result.get("cache_hit"), "seconds":round(time.monotonic()-started, 3),
                        "answer_chars":len(result.get("analysis",{}).get("answer","")),
                        "evidence_count":len(result.get("analysis",{}).get("evidence",[])),
                        "cost":result["cost"]}
            except Exception as exc:
                dispatched = getattr(exc, "tool_dispatched", invoked)
                return {"index":index, "status":"error", "error_code":getattr(exc, "code", type(exc).__name__),
                        "analysis_id":analysis_id, "tool_dispatched":dispatched,
                        "analysis_receipt_missing":bool(args.analysis and dispatched and receipt is None),
                        "cost":receipt, "seconds":round(time.monotonic()-started, 3)}
    results = await asyncio.gather(*(one(i) for i in range(args.requests)))
    counts = Counter(r["status"] for r in results)
    elapsed = [r["seconds"] for r in results]
    passed = counts["success"] == args.requests
    if args.analysis and not args.distinct_analysis_ids and passed:
        passed = len({r["request_id"] for r in results}) == 1
    return {"mode":"live", "warmup":warmup, "passed":passed, "requests":results,
            "summary":{"requests":len(results), "success":counts["success"],
                       "degraded":counts["degraded"]+counts["partial"], "failed":counts["error"],
                       "p50_seconds":round(statistics.median(elapsed), 3), "p95_seconds":percentile(elapsed,.95),
                       **summarize_costs(costs, results)}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--url", help="MCP server base URL, without /sse")
    parser.add_argument("--env-file", type=Path,
                        help="Client .env file; defaults to this repository's .env. Environment token wins.")
    parser.add_argument("--no-auth", action="store_true",
                        help="Explicitly use an unauthenticated test endpoint; never changes server configuration.")
    parser.add_argument("--query", default="2402.01306")
    parser.add_argument("--pdf-url")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--allow-paid-analysis", action="store_true")
    parser.add_argument("--distinct-analysis-ids", action="store_true")
    parser.add_argument("--analysis-run-id", help="Stable logical batch ID; reuse it with the same input when recovering analysis.")
    parser.add_argument("--question", default="Explain one central method in under 100 words, with one exact supporting quote.")
    parser.add_argument("--language", choices=("en","zh"), default="en")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64 or not 1 <= args.requests <= 1000:
        parser.error("concurrency must be 1..64 and requests 1..1000; do not use account capacity as worker count")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.live:
        if not args.url:
            parser.error("--live requires --url")
        if args.analysis and not args.allow_paid_analysis:
            parser.error("Live analysis requires explicit --allow-paid-analysis")
        if args.analysis and not args.analysis_run_id:
            parser.error("Live analysis requires --analysis-run-id; reuse it to recover the same logical tasks")
        report = asyncio.run(live(args))
    else:
        completed = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-q"],
                                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        report = {"mode":"offline", "passed":completed.returncode == 0,
                  "provider_model_calls":0, "provider_cost_cny":"0.00000000",
                  "test_output":completed.stdout+completed.stderr}
    data = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data+"\n", encoding="utf-8")
    print(data)
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
