"""Request-local provider API usage estimates; never a provider invoice."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from functools import wraps
from zoneinfo import ZoneInfo
import json

_ACTIVE = ContextVar("scholar_cost", default=None)
PRICE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
PRICE_VERSION = "2026-09-12"
MILLION = Decimal(1000000)
D = Decimal

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def _stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

def _peak(at):
    local = at.astimezone(ZoneInfo("Asia/Shanghai"))
    hour = local.hour + local.minute / 60 + local.second / 3600
    return local.weekday() < 5 and (9 <= hour < 12 or 14 <= hour < 18)

def tariffs(start, finish):
    a, b = _stamp(start), _stamp(finish)
    if b < a:
        return {False, True}
    kinds = {_peak(a), _peak(b)}
    # Exact interval boundaries, including weekends; endpoints alone are insufficient.
    day = a.astimezone(ZoneInfo("Asia/Shanghai")).date()
    last = b.astimezone(ZoneInfo("Asia/Shanghai")).date()
    if (last-day).days > 31:
        return {False, True}
    while day <= last:
        for hour in (0, 9, 12, 14, 18):
            at = datetime(day.year, day.month, day.day, hour, tzinfo=ZoneInfo("Asia/Shanghai"))
            if a <= at <= b:
                kinds.add(_peak(at))
        day += timedelta(days=1)
    return kinds

def _decimal(value, *, upper=False):
    return format(value.quantize(D("0.00000001"), rounding=ROUND_CEILING if upper else ROUND_FLOOR), "f") if value is not None else None

def usage_fields(raw):
    raw = raw if isinstance(raw, dict) else {}
    def number(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    details = raw.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    return {
        "input_tokens": number(raw.get("prompt_tokens")),
        "cache_hit_tokens": number(raw.get("prompt_cache_hit_tokens", details.get("cached_tokens"))),
        "cache_miss_tokens": number(raw.get("prompt_cache_miss_tokens")),
        "output_tokens": number(raw.get("completion_tokens")),
    }

def estimate(attempt):
    model = attempt.get("model_returned") or attempt["model_requested"]
    if model == "deepseek-flash":
        base = (D(".02"), D("1"), D("4"))
    elif model == "deepseek-v4-pro":
        base = (D(".15"), D("4.5"), D("13.5"))
    else:
        return D(0), None
    usage = attempt["usage"]
    n, hit, miss, out = (usage[k] for k in ("input_tokens", "cache_hit_tokens", "cache_miss_tokens", "output_tokens"))
    if n is None or out is None:
        return D(0), None
    if (hit is not None and hit > n) or (miss is not None and miss > n):
        return D(0), None
    if hit is not None and miss is None and hit <= n:
        miss = n-hit
    if miss is not None and hit is None and miss <= n:
        hit = n-miss
    if hit is not None and miss is not None and hit+miss != n:
        return D(0), None
    estimates = []
    for peak in tariffs(attempt["started_at"], attempt["finished_at"]):
        h, m, o = (v*(2 if peak else 1) for v in base)
        if hit is None or miss is None:
            estimates.extend(((D(n)*h+D(out)*o)/MILLION, (D(n)*m+D(out)*o)/MILLION))
        else:
            estimates.append((D(hit)*h+D(miss)*m+D(out)*o)/MILLION)
    return min(estimates), max(estimates)

class CostLedger:
    def __init__(self):
        self.attempts = []

    def start(self, model, stage="analysis"):
        item = {"stage": stage, "model_requested": model, "model_returned": None,
                "request_id": None, "status": "unknown", "started_at": utc_now(),
                "finished_at": None, "usage": usage_fields(None), "lower_cny": None,
                "upper_cny": None, "error_code": None}
        self.attempts.append(item)
        return item

    def finish(self, item, *, status, data=None, error_code=None):
        data = data if isinstance(data, dict) else {}
        item.update(status=status, finished_at=utc_now(), model_returned=data.get("model"),
                    request_id=data.get("id"), usage=usage_fields(data.get("usage")),
                    error_code=error_code)
        low, high = estimate(item)
        item.update(lower_cny=_decimal(low), upper_cny=_decimal(high, upper=True))

    def report(self, reused=False):
        items = self.attempts
        lower = sum((D(a["lower_cny"] or "0") for a in items), D(0))
        upper = None if any(a["upper_cny"] is None for a in items) else sum((D(a["upper_cny"]) for a in items), D(0))
        complete = upper is not None
        return {"schema_version": "mcp.cost.v1", "currency": "CNY", "scope": "provider_api_only",
                "basis": "estimated_usage" if items else "no_paid_calls",
                "total_cny": _decimal(lower) if complete and upper == lower else None,
                "lower_cny": _decimal(lower), "upper_cny": _decimal(upper, upper=True), "complete": complete,
                "model_calls": len(items), "unknown_calls": sum(a["upper_cny"] is None for a in items),
                "attempts": [dict(a) for a in items], "reused_result": reused,
                "pricing_source": PRICE_SOURCE, "pricing_version": PRICE_VERSION,
                "pricing_note": "Usage estimate, not invoice; crossing tariff intervals and missing cache breakdown retain bounds."}

@contextmanager
def cost_scope():
    ledger = _ACTIVE.get()
    token = None
    if ledger is None:
        ledger = CostLedger()
        token = _ACTIVE.set(ledger)
    try:
        yield ledger
    finally:
        if token is not None:
            _ACTIVE.reset(token)

def current_ledger():
    return _ACTIVE.get()

def costed(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        with cost_scope() as ledger:
            result = await func(*args, **kwargs)
            if isinstance(result, str):
                value = json.loads(result)
                value["cost"] = ledger.report(reused=value.get("cache_hit", False) and not ledger.attempts)
                return json.dumps(value, ensure_ascii=False)
            result["cost"] = ledger.report(reused=result.get("cache_hit", False) and not ledger.attempts)
            return result
    return wrapped
