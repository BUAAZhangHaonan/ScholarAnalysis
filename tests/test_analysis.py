import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import httpx
from scholar_analysis.config import Settings
from scholar_analysis.cost import CostLedger, cost_scope, tariffs
from scholar_analysis.llm.post_processor import PostProcessor, _parse_answer
from scholar_analysis.llm.prompt_budget import PromptBudget
from scholar_analysis.llm.prompt_loader import load_prompt
from scholar_analysis.pipeline.errors import PipelineError

class CostTests(unittest.TestCase):
    def item(self, usage, start="2026-09-12T10:00:00+08:00", finish="2026-09-12T10:00:01+08:00"):
        ledger = CostLedger()
        item = ledger.start("deepseek-flash")
        ledger.finish(item, status="success", data={"usage": usage})
        item.update(started_at=start, finished_at=finish)
        from scholar_analysis.cost import estimate, _decimal
        low, high = estimate(item)
        item.update(lower_cny=_decimal(low), upper_cny=_decimal(high))
        return ledger
    def test_offpeak_known_breakdown(self):
        cost = self.item({"prompt_tokens":1000,"prompt_cache_hit_tokens":200,"prompt_cache_miss_tokens":800,"completion_tokens":100}).report()
        self.assertEqual(cost["total_cny"], "0.00120400")
        self.assertTrue(cost["complete"])
    def test_unknown_usage_not_zero(self):
        cost = self.item({}).report()
        self.assertIsNone(cost["total_cny"])
        self.assertIsNone(cost["upper_cny"])
        self.assertEqual(cost["unknown_calls"], 1)
    def test_missing_breakdown_has_range(self):
        cost = self.item({"prompt_tokens":1000,"completion_tokens":100}).report()
        self.assertEqual(cost["lower_cny"], "0.00042000")
        self.assertEqual(cost["upper_cny"], "0.00140000")
        self.assertFalse(cost["complete"])
    def test_crossing_peak_boundary_preserves_range(self):
        ledger = self.item({"prompt_tokens":1000,"prompt_cache_hit_tokens":0,"completion_tokens":100},
                           "2026-09-14T08:59:00+08:00", "2026-09-14T09:01:00+08:00")
        cost = ledger.report()
        self.assertEqual(cost["lower_cny"], "0.00140000")
        self.assertEqual(cost["upper_cny"], "0.00280000")
    def test_no_paid_calls_and_cached_read(self):
        cost = CostLedger().report(reused=True)
        self.assertEqual(cost["total_cny"], "0.00000000")
        self.assertEqual(cost["model_calls"], 0)
        self.assertTrue(cost["reused_result"])

class AnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = Settings(_env_file=None, deepseek_api_key="offline-fixture", model_pool_cooldown_seconds=0)
        self.entry = SimpleNamespace(model="deepseek-flash", backend="deepseek", context_tokens=256000,
                                      base_url="https://model.example/completions", api_key="offline-fixture")
        self.pool = SimpleNamespace(acquire=AsyncMock(return_value=self.entry), release=lambda entry, success: self.releases.append(success))
        self.releases = []
        self.processor = PostProcessor()
        self.patches = [
            patch("scholar_analysis.llm.post_processor.get_settings", return_value=self.settings),
            patch("scholar_analysis.llm.post_processor.ModelPool.get", AsyncMock(return_value=self.pool)),
        ]
        for p in self.patches:
            p.start()
    async def asyncTearDown(self):
        await self.processor.aclose()
        for p in self.patches:
            p.stop()
    def response(self, **changes):
        value = {"id":"offline-call", "model":"deepseek-flash", "choices":[{"finish_reason":"stop",
                 "message":{"content":json.dumps({"answer":"The method uses a gate.","evidence":["uses a gate"]})}}],
                 "usage":{"prompt_tokens":100,"prompt_cache_hit_tokens":0,"completion_tokens":20}}
        value.update(changes)
        return value
    def mock(self, handler):
        self.processor._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async def test_final_answer_quotes_and_cost(self):
        captured = []
        def handler(req):
            captured.append(json.loads(req.content))
            return httpx.Response(200, json=self.response())
        self.mock(handler)
        with cost_scope() as ledger:
            result = await self.processor.extract("The method uses a gate.", "How?")
            self.assertEqual(result["evidence"][0]["locations"][0]["start"], 11)
            self.assertEqual(ledger.report()["model_calls"], 1)
        self.assertEqual(self.releases, [True])
        self.assertEqual(captured[0]["model"], "deepseek-flash")
        self.assertIn("Actual input coverage", captured[0]["messages"][1]["content"])
    async def test_reasoning_is_never_final_answer(self):
        self.mock(lambda _: httpx.Response(200, json=self.response(choices=[{"finish_reason":"stop","message":{"content":"","reasoning_content":"private reasoning"}}])))
        with cost_scope() as ledger:
            with self.assertRaises(PipelineError) as exc:
                await self.processor.extract("Text.", "How?")
            self.assertEqual(exc.exception.code, "LLM_FINAL_EMPTY")
            self.assertEqual(ledger.report()["model_calls"], 1)
            self.assertNotIn("private reasoning", json.dumps(ledger.report()))
        self.assertEqual(self.releases, [False])
    async def test_length_finish_not_success(self):
        self.mock(lambda _: httpx.Response(200, json=self.response(choices=[{"finish_reason":"length","message":{"content":"partial"}}])))
        with self.assertRaises(PipelineError) as exc:
            await self.processor.extract("Text.", "How?")
        self.assertEqual(exc.exception.code, "LLM_OUTPUT_INCOMPLETE")
    async def test_unknown_outcome_is_not_retried(self):
        count = 0
        def handler(req):
            nonlocal count
            count += 1
            raise httpx.ReadTimeout("offline timeout", request=req)
        self.mock(handler)
        with cost_scope() as ledger:
            with self.assertRaises(PipelineError) as exc:
                await self.processor.extract("Text.", "How?")
            self.assertEqual(exc.exception.code, "LLM_OUTCOME_UNKNOWN")
            self.assertEqual(ledger.report()["unknown_calls"], 1)
        self.assertEqual(count, 1)
    async def test_cancel_releases_slot_and_preserves_unknown_attempt(self):
        started = asyncio.Event()
        async def handler(req):
            started.set()
            await asyncio.Event().wait()
        self.mock(handler)
        with cost_scope() as ledger:
            task = asyncio.create_task(self.processor.extract("Text.", "How?"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(ledger.report()["attempts"][0]["status"], "cancelled")
        self.assertEqual(self.releases, [None])
    async def test_429_then_success_accounts_both(self):
        replies = [httpx.Response(429, json={"usage":{"prompt_tokens":1,"prompt_cache_hit_tokens":0,"completion_tokens":0}}),
                   httpx.Response(200, json=self.response())]
        self.mock(lambda _: replies.pop(0))
        with cost_scope() as ledger, patch("scholar_analysis.llm.post_processor.asyncio.sleep", new=AsyncMock()):
            result = await self.processor.extract("The method uses a gate.", "How?")
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(ledger.report()["model_calls"], 2)
    async def test_supplied_excerpt_offsets_are_original_offsets(self):
        self.mock(lambda _: httpx.Response(200, json=self.response()))
        result = await self.processor.extract("The method uses a gate.", "How?", source_offset=200, total_chars=1000)
        self.assertEqual(result["evidence"][0]["locations"][0]["start"], 211)
        self.assertTrue(result["coverage"]["truncated"])
    async def test_bad_quote_rejected_without_retry(self):
        self.mock(lambda _: httpx.Response(200, json=self.response()))
        with self.assertRaises(PipelineError):
            await self.processor.extract("Another paragraph.", "How?")

class PromptTests(unittest.TestCase):
    def test_packaged_templates_independent_of_cwd(self):
        import os
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                self.assertIn("{coverage}", load_prompt("extract_focus", "zh").user)
                self.assertIn("{coverage}", load_prompt("extract_focus", "en", prompts_dir="prompts").user)
            finally:
                os.chdir(previous)
    def test_budget_does_not_split_formula_block(self):
        text = "Intro.\n\n$$" + "x"*100 + "$$\n\nEnd."
        short = PromptBudget().truncate_text(text, 30)
        self.assertEqual(short, "Intro.")
    def test_evidence_matching_does_not_certify_entailment(self):
        answer, evidence = _parse_answer(json.dumps({"answer":"Interpretation","evidence":["observed"]}), "observed twice, observed again")
        self.assertEqual(evidence[0]["match_status"], "multiple")
        self.assertEqual(len(evidence[0]["locations"]), 2)
