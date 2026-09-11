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
import uuid

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

async def live(args):
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    token = os.environ.get("SCHOLAR_ANALYSIS_ACCESS_TOKEN", "")
    headers = {"Authorization": "Bearer "+token} if token else {}
    async def call(tool, arguments):
        async with sse_client(args.url.rstrip("/")+"/sse", headers=headers,
                              timeout=30, sse_read_timeout=args.timeout) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
                if result.isError:
                    raise RuntimeError("MCP transport/tool wrapper returned an error")
                text = next((b.text for b in result.content if b.type == "text"), "")
                value = json.loads(text)
                check_cost(value)
                return value

    source = {"pdf_url": args.pdf_url} if args.pdf_url else {"query": args.query}
    warm_start = time.monotonic()
    first = await call("get_paper_text", {**source, "limit_chars": 1000})
    warmup = {"seconds": round(time.monotonic()-warm_start, 3), "status": first.get("status"),
              "error_code": first.get("error_code"), "stage": first.get("stage"),
              "total_chars": first.get("total_chars"), "source": first.get("source"), "cost": first["cost"]}
    if first.get("status") != "success":
        return {"mode":"live", "warmup":warmup, "passed":False, "requests":[]}
    assert first["markdown"] and first["source"].get("final_url")
    assert first["range"]["start"] == 0
    assert first["range"]["end"] == len(first["markdown"])
    document = first["document_id"]
    analysis_key = "stress_"+uuid.uuid4().hex
    gate = asyncio.Semaphore(args.concurrency)
    costs = []
    async def one(index):
        async with gate:
            started = time.monotonic()
            try:
                if args.analysis:
                    values = {"document_id": document, "question":args.question, "language":args.language,
                              "analysis_id":analysis_key+("_"+str(index) if args.distinct_analysis_ids else "")}
                    tool = "analyze_paper"
                else:
                    values = {"document_id":document, "offset":min(1000, first["total_chars"]), "limit_chars":1000}
                    tool = "get_paper_text"
                result = await call(tool, values)
                costs.append(result["cost"])
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
                        "analysis_id":result.get("analysis_id"), "analysis_reused":result.get("analysis_reused"),
                        "cache_hit":result.get("cache_hit"), "seconds":round(time.monotonic()-started, 3),
                        "answer_chars":len(result.get("analysis",{}).get("answer","")),
                        "evidence_count":len(result.get("analysis",{}).get("evidence",[])),
                        "cost":result["cost"]}
            except Exception as exc:
                return {"index":index, "status":"error", "error_code":type(exc).__name__,
                        "seconds":round(time.monotonic()-started, 3)}
    results = await asyncio.gather(*(one(i) for i in range(args.requests)))
    # Sum actual incremental attempts once, including unknown/failed requests.
    attempts = {}
    for cost in costs:
        for attempt in cost["attempts"]:
            key = (attempt.get("request_id"), attempt["started_at"], attempt["model_requested"])
            attempts[key] = attempt
    low = sum((Decimal(a["lower_cny"] or "0") for a in attempts.values()), Decimal(0))
    high = None if any(a["upper_cny"] is None for a in attempts.values()) else sum((Decimal(a["upper_cny"]) for a in attempts.values()), Decimal(0))
    counts = Counter(r["status"] for r in results)
    elapsed = [r["seconds"] for r in results]
    passed = counts["success"] == args.requests
    if args.analysis and not args.distinct_analysis_ids and passed:
        passed = len({r["request_id"] for r in results}) == 1
    return {"mode":"live", "warmup":warmup, "passed":passed, "requests":results,
            "summary":{"requests":len(results), "success":counts["success"],
                       "degraded":counts["degraded"]+counts["partial"], "failed":counts["error"],
                       "p50_seconds":round(statistics.median(elapsed), 3), "p95_seconds":percentile(elapsed,.95),
                       "provider_model_calls":len(attempts), "provider_cost_lower_cny":str(low),
                       "provider_cost_upper_cny":str(high) if high is not None else None,
                       "unknown_calls":sum(a["upper_cny"] is None for a in attempts.values())}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--url", help="MCP server base URL, without /sse")
    parser.add_argument("--query", default="2402.01306")
    parser.add_argument("--pdf-url")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--allow-paid-analysis", action="store_true")
    parser.add_argument("--distinct-analysis-ids", action="store_true")
    parser.add_argument("--question", default="Explain one central method in under 100 words, with one exact supporting quote.")
    parser.add_argument("--language", choices=("en","zh"), default="en")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64 or not 1 <= args.requests <= 1000:
        parser.error("concurrency must be 1..64 and requests 1..1000; do not use account capacity as worker count")
    if args.live:
        if not args.url:
            parser.error("--live requires --url")
        if args.analysis and not args.allow_paid_analysis:
            parser.error("Live analysis requires explicit --allow-paid-analysis")
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
