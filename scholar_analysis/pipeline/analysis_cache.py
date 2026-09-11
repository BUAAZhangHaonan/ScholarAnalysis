"""Idempotent analysis results, keyed by an explicit client retry identifier."""
import asyncio
import json
import re
import tempfile
import threading
from pathlib import Path
from scholar_analysis.pipeline.errors import PipelineError

class AnalysisCache:
    def __init__(self, directory, max_bytes):
        self.root = Path(directory)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def path(self, key):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", key):
            raise PipelineError("INVALID_ANALYSIS_ID", "input", "analysis_id must contain 1–100 letters, digits, underscores or hyphens.")
        return self.root / (key+".json")

    async def get(self, key):
        path = self.path(key)
        def read():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict) or not isinstance(value.get("result"), dict):
                    raise ValueError("invalid result")
                return value
            except FileNotFoundError:
                return None
            except (ValueError, OSError) as exc:
                # A broken paid-result record must not silently trigger another charge.
                raise PipelineError("ANALYSIS_CACHE_UNREADABLE", "cache", "Saved analysis cannot be read; inspect the record before issuing another paid task.") from exc
        return await asyncio.to_thread(read)

    async def put(self, key, value):
        path = self.path(key)
        def write():
            self.root.mkdir(parents=True, exist_ok=True)
            if not path.exists() and sum(p.stat().st_size for p in self.root.glob("*.json")) >= self.max_bytes:
                raise PipelineError("ANALYSIS_CACHE_FULL", "cache", "Analysis receipt storage is full; no new paid task was started.")
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                                 suffix=".tmp", delete=False) as f:
                    tmp = Path(f.name)
                    json.dump(value, f, ensure_ascii=False)
                tmp.replace(path)
            finally:
                if tmp:
                    tmp.unlink(missing_ok=True)
        def locked_write():
            with self._lock:
                write()
        await asyncio.to_thread(locked_write)
