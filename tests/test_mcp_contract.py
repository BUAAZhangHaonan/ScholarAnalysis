import asyncio
import importlib
import json
import unittest
from unittest.mock import AsyncMock, patch
from scholar_analysis.config import Settings

class MCPContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = Settings(_env_file=None, queue_timeout_seconds=.01, request_max_age_seconds=1)
        with patch("scholar_analysis.config.get_settings", return_value=self.settings):
            self.module = importlib.import_module("scholar_analysis.mcp_server")
        self.p = patch.object(self.module, "get_settings", return_value=self.settings)
        self.p.start()
        self.module._pipeline_sem = asyncio.Semaphore(1)
    async def asyncTearDown(self):
        self.p.stop()
        self.module._pipeline_sem = None
    async def test_queue_timeout_typed_and_zero_cost(self):
        await self.module._pipeline_sem.acquire()
        result = json.loads(await self.module.get_paper_text(query="2401.00001"))
        self.assertEqual(result["error_code"], "QUEUE_TIMEOUT")
        self.assertEqual(result["cost"]["model_calls"], 0)
        self.assertEqual(self.module._pipeline_sem._value, 0)
    async def test_new_parameters_forward_and_release(self):
        orch = AsyncMock()
        orch.get_paper_text.return_value = {"status":"success","cache_hit":True,"markdown":"slice"}
        with patch.object(self.module, "_get_orchestrator", return_value=orch):
            result = json.loads(await self.module.get_paper_text(document_id="doc_test", offset=2, limit_chars=10, find_text="method"))
        self.assertEqual(orch.get_paper_text.call_args.kwargs["offset"], 2)
        self.assertEqual(self.module._pipeline_sem._value, 1)
        self.assertTrue(result["cost"]["reused_result"])
    async def test_tool_exception_has_public_contract(self):
        with patch.object(self.module, "_get_orchestrator", side_effect=RuntimeError("internal fixture")):
            result = json.loads(await self.module.get_paper_text(query="2401.00001"))
        self.assertEqual(result["error_code"], "INTERNAL_ERROR")
        self.assertNotIn("internal fixture", result["error"])
        self.assertEqual(result["cost"]["model_calls"], 0)
