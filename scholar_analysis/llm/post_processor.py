"""Focused text analysis with explicit reading coverage and per-attempt accounting."""
from __future__ import annotations
import asyncio
from datetime import date
import json
import logging
import time
import httpx
from scholar_analysis.config import get_settings
from scholar_analysis.cost import cost_scope
from scholar_analysis.llm.backends import deepseek as deepseek_backend
from scholar_analysis.llm.model_pool import ModelPool
from scholar_analysis.llm.prompt_budget import PromptBudget
from scholar_analysis.llm.prompt_loader import load_prompt
from scholar_analysis.pipeline.errors import PipelineError

logger = logging.getLogger(__name__)

class PostProcessorError(PipelineError):
    def __init__(self, message, code="ANALYSIS_OUTPUT_INVALID"):
        super().__init__(code, "analysis", message)

def _parse_answer(content, source, base_offset=0):
    try:
        value = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise PostProcessorError("Analysis did not return the required JSON answer and source quotes.") from exc
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise PostProcessorError("Analysis answer is missing.")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or len(evidence) > 8:
        raise PostProcessorError("Analysis evidence must be a list of at most eight exact quotes.")
    located = []
    for quote in evidence:
        if not isinstance(quote, str) or not quote.strip():
            raise PostProcessorError("Evidence quote must be nonempty text.")
        starts = []
        at = source.find(quote)
        while at >= 0:
            starts.append(at)
            at = source.find(quote, at+max(1, len(quote)))
        if not starts:
            raise PostProcessorError("An evidence quote does not match the text actually supplied to the model.")
        located.append({"quote": quote, "locations": [
            {"start": pos+base_offset, "end": pos+base_offset+len(quote),
             "line_in_excerpt": source.count("\n", 0, pos)+1} for pos in starts],
             "match_status": "unique" if len(starts) == 1 else "multiple"})
    return value["answer"].strip(), located

class PostProcessor:
    def __init__(self):
        self._client = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(timeout=300.0, trust_env=False)
        return self._client

    async def aclose(self):
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def extract(self, markdown, question, language="en", max_attempts=2, *, source_offset=0, total_chars=None):
        settings = get_settings()
        pool = await ModelPool.get(settings)
        template = load_prompt("extract_focus", language, prompts_dir=settings.prompts_dir)
        max_attempts = min(3, max(1, max_attempts or 1))
        # No backend/model switching. A sent request of unknown outcome is never retried.
        entry = await pool.acquire(timeout=30.0)
        success = None
        try:
            budget = PromptBudget(entry.context_tokens, settings.response_headroom_tokens)
            system = template.system.format(current_date=date.today().isoformat())
            overhead = budget.estimate_text(system+question+template.user) + 2000
            limit = budget.max_input_tokens-overhead
            if limit <= 0:
                raise PostProcessorError("Question leaves no room for paper text.", "ANALYSIS_INPUT_TOO_LONG")
            content = budget.truncate_text(markdown, limit)
            if not content.strip():
                raise PostProcessorError("No paper text fits the model context.", "ANALYSIS_INPUT_TOO_LONG")
            coverage = {
                "start": source_offset, "end": source_offset+len(content),
                "total_chars": total_chars if total_chars is not None else len(markdown),
                "basis": "parsed Markdown characters; PDF completeness is not certified",
                "truncated": len(content) < len(markdown) or source_offset > 0 or (total_chars is not None and len(content) < total_chars),
            }
            user = template.user.format(user_question=question, paper_content=content,
                                        coverage=json.dumps(coverage, ensure_ascii=False))
            payload = deepseek_backend.build_payload(model=entry.model, system_msg=system, user_msg=user,
                      max_tokens=settings.response_headroom_tokens, thinking=settings.deepseek_thinking)
            payload["response_format"] = {"type": "json_object"}
            client = await self._get_client()
            started = time.monotonic()
            with cost_scope() as ledger:
                for number in range(max_attempts):
                    record = ledger.start(entry.model)
                    response_data = None
                    try:
                        resp = await client.post(entry.base_url, headers={"Authorization": "Bearer "+entry.api_key}, json=payload)
                        try:
                            response_data = resp.json()
                        except ValueError:
                            response_data = None
                        if resp.status_code == 429:
                            ledger.finish(record, status="error", data=response_data, error_code="LLM_RATE_LIMIT")
                            if number+1 < max_attempts:
                                await asyncio.sleep(min(2**number, 4))
                                continue
                            raise PipelineError("LLM_RATE_LIMIT", "analysis", "Analysis model is rate limited.", retryable=True, retry_after=5)
                        if not resp.is_success:
                            ledger.finish(record, status="error", data=response_data, error_code="LLM_HTTP_ERROR")
                            raise PipelineError("LLM_HTTP_ERROR", "analysis", f"Analysis provider returned HTTP {resp.status_code}.",
                                                retryable=resp.status_code >= 500)
                        if not isinstance(response_data, dict) or response_data.get("error"):
                            raise PostProcessorError("Analysis provider returned an invalid response.", "LLM_RESPONSE_INVALID")
                        try:
                            choice = response_data["choices"][0]
                            final = choice["message"].get("content")
                        except (KeyError, IndexError, TypeError) as exc:
                            raise PostProcessorError("Analysis response has no final answer.", "LLM_RESPONSE_INVALID") from exc
                        finish = choice.get("finish_reason")
                        if finish != "stop":
                            raise PostProcessorError("Analysis did not finish normally; partial content is not certified as an answer.",
                                                     "LLM_OUTPUT_INCOMPLETE")
                        if not isinstance(final, str) or not final.strip():
                            raise PostProcessorError("Analysis returned no final content; reasoning is not an answer.", "LLM_FINAL_EMPTY")
                        cleaned = final.strip()
                        if cleaned.startswith("```") and cleaned.endswith("```"):
                            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                        answer, evidence = _parse_answer(cleaned, content, source_offset)
                        ledger.finish(record, status="success", data=response_data)
                        usage = record["usage"]
                        success = True
                        return {"answer": answer, "evidence": evidence,
                                "evidence_status": "quotes_matched_to_supplied_text" if evidence else "no_quotes",
                                "evidence_limit": "Exact matching validates attribution, not whether a quote entails the conclusion.",
                                "model_used": response_data.get("model") or entry.model, "backend": entry.backend,
                                "token_usage": {"input": usage["input_tokens"], "output": usage["output_tokens"]},
                                "finish_reason": finish, "truncated": coverage["truncated"], "coverage": coverage,
                                "attempts": [dict(a) for a in ledger.attempts],
                                "llm_seconds": round(time.monotonic()-started, 3)}
                    except asyncio.CancelledError:
                        if record["finished_at"] is None:
                            ledger.finish(record, status="cancelled", data=response_data, error_code="LLM_CANCELLED")
                        raise
                    except (httpx.RequestError, TimeoutError) as exc:
                        if record["finished_at"] is None:
                            ledger.finish(record, status="unknown", data=response_data, error_code="LLM_OUTCOME_UNKNOWN")
                        raise PipelineError("LLM_OUTCOME_UNKNOWN", "analysis", "A model request was sent but its result is unknown; no automatic retry was issued.") from exc
                    except Exception as exc:
                        if record["finished_at"] is None:
                            ledger.finish(record, status="error", data=response_data,
                                          error_code=getattr(exc, "code", "LLM_RESPONSE_INVALID"))
                        raise
        except Exception:
            success = False
            raise
        finally:
            # CancelledError is a BaseException; release exactly once even then.
            pool.release(entry, success=success)
