"""Priority-ordered LLM model pool with per-model concurrency control."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from scholar_analysis.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelPoolEntry:
    """One model slot in the concurrent pool."""

    backend: str
    model: str
    api_key: str
    base_url: str
    max_concurrent: int
    context_tokens: int
    semaphore: asyncio.Semaphore
    last_error_time: float = 0.0
    error_count: int = 0
    disabled: bool = False

    @property
    def in_cooldown(self) -> bool:
        if self.disabled:
            return True
        cooldown = get_settings().model_pool_cooldown_seconds
        return (time.monotonic() - self.last_error_time) < cooldown

    def disable(self) -> None:
        """Permanently disable this entry (e.g. auth failure). Never returns to acquire."""
        self.disabled = True
        logger.error(
            "Model %s (%s) PERMANENTLY DISABLED — will not be acquired again",
            self.model,
            self.backend,
        )

    def record_error(self) -> None:
        self.error_count += 1
        self.last_error_time = time.monotonic()
        logger.warning(
            "Model %s (%s) error_count=%d, cooldown for %.0fs",
            self.model,
            self.backend,
            self.error_count,
            get_settings().model_pool_cooldown_seconds,
        )

    def record_success(self) -> None:
        if self.error_count > 0:
            logger.info(
                "Model %s (%s) recovered from %d errors",
                self.model,
                self.backend,
                self.error_count,
            )
        self.error_count = 0


class ModelPool:
    """Priority-ordered pool of LLM models with per-model concurrency control."""

    _instance: ModelPool | None = None
    _lock: asyncio.Lock | None = None

    def __init__(self, entries: list[ModelPoolEntry]) -> None:
        self._entries = entries

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def get(cls, settings: Settings) -> ModelPool:
        if cls._instance is None:
            async with cls._get_lock():
                if cls._instance is None:
                    cls._instance = cls._build(settings)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    async def acquire(self, timeout: float = 30.0) -> ModelPoolEntry:
        """Acquire the first available (not in cooldown, semaphore free) model.

        Raises RuntimeError with explicit reason if no model is configured or
        all models are in cooldown.
        """
        if not self._entries:
            raise RuntimeError(
                "ModelPool is empty — no LLM API keys configured. "
                "Set SCHOLAR_ANALYSIS_DEEPSEEK_API_KEY, SCHOLAR_ANALYSIS_BIGMODEL_API_KEY, "
                "or SCHOLAR_ANALYSIS_QWEN_BASE_URL in .env"
            )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for entry in self._entries:
                if entry.in_cooldown:
                    continue
                try:
                    await asyncio.wait_for(entry.semaphore.acquire(), timeout=0.1)
                    logger.debug("Acquired model %s (%s)", entry.model, entry.backend)
                    return entry
                except asyncio.TimeoutError:
                    continue
            await asyncio.sleep(0.1)

        cooldown_models = [
            f"{e.model}(errors={e.error_count})" for e in self._entries if e.in_cooldown
        ]
        busy_models = [f"{e.model}" for e in self._entries if not e.in_cooldown]
        raise RuntimeError(
            f"ModelPool: no model available within {timeout}s timeout. "
            f"Cooldown: {cooldown_models}, Busy: {busy_models}"
        )

    def release(self, entry: ModelPoolEntry, *, success: bool | None) -> None:
        if success is True:
            entry.record_success()
        elif success is False:
            entry.record_error()
        entry.semaphore.release()

    def disable(self, entry: ModelPoolEntry) -> None:
        """Permanently remove an entry from rotation (auth-dead API keys)."""
        entry.disable()

    @property
    def entries(self) -> list[ModelPoolEntry]:
        return list(self._entries)

    @staticmethod
    def _build(settings: Settings) -> ModelPool:
        entries: list[ModelPoolEntry] = []

        # Primary: explicitly configured DeepSeek model (Flash by default)
        if settings.deepseek_api_key.strip():
            entries.append(
                ModelPoolEntry(
                    backend="deepseek",
                    model=settings.deepseek_model,
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                    max_concurrent=settings.deepseek_max_concurrent,
                    context_tokens=settings.deepseek_context_tokens,
                    semaphore=asyncio.Semaphore(settings.deepseek_max_concurrent),
                )
            )

        # One explicitly configured model per deployment. Account peak capacity
        # is not a worker default; Pro is selected through deepseek_model only.
        if not entries:
            logger.error(
                "ModelPool built with ZERO entries — no LLM backends configured. "
                "analyze_paper will always fail."
            )
        else:
            logger.info(
                "ModelPool initialised with %d models: %s",
                len(entries),
                ", ".join(
                    f"{e.model}({e.max_concurrent}, ctx={e.context_tokens})"
                    for e in entries
                ),
            )
        return ModelPool(entries)
