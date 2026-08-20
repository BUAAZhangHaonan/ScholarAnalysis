"""Orchestrate LLM-based paper analysis: prompt → model → response."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date
from typing import Any

import httpx

from scholar_analysis.config import get_settings
from scholar_analysis.llm.backends import deepseek as deepseek_backend
from scholar_analysis.llm.backends import glm as glm_backend
from scholar_analysis.llm.backends import qwen as qwen_backend
from scholar_analysis.llm.model_pool import ModelPool, ModelPoolEntry
from scholar_analysis.llm.prompt_budget import PromptBudget
from scholar_analysis.llm.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_TOTAL_LLM_CALLS = 6  # hard budget per extract() invocation across all backends


class PostProcessorError(Exception):
    pass


class LLMRetryableError(Exception):
    """Errors that are worth retrying (timeout, 429, 5xx)."""

    pass


class LLMFatalError(Exception):
    """Errors that should not be retried (auth, business error, bad request)."""

    pass


class LLMAuthError(LLMFatalError):
    """401/403 from a backend — its API key is dead; disable the pool entry."""


class LLMBudgetExhausted(PostProcessorError):
    """Total LLM call budget for this extract() exhausted."""


def _build_payload(
    entry: ModelPoolEntry,
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    settings: Any,
) -> dict[str, Any]:
    if entry.backend == "deepseek":
        return deepseek_backend.build_payload(
            model=entry.model,
            system_msg=system_msg,
            user_msg=user_msg,
            max_tokens=max_tokens,
            thinking=settings.deepseek_thinking,
        )
    if entry.backend == "glm":
        return glm_backend.build_payload(
            model=entry.model,
            system_msg=system_msg,
            user_msg=user_msg,
            max_tokens=max_tokens,
        )
    if entry.backend == "qwen":
        return qwen_backend.build_payload(
            model=entry.model,
            system_msg=system_msg,
            user_msg=user_msg,
            max_tokens=max_tokens,
        )
    raise PostProcessorError(f"Unknown backend: {entry.backend}")


async def _call_llm(
    entry: ModelPoolEntry,
    payload: dict[str, Any],
    client: httpx.AsyncClient,
    call_counter: list[int] | None = None,
) -> tuple[str, dict[str, int]]:
    """Call the LLM API with retries for transient errors only.

    Business errors (HTTP 200 with error body, HTTP 4xx) are raised
    immediately without retry — retrying them is pointless.
    ``call_counter`` is a one-element list counting total LLM HTTP calls made
    by the enclosing extract(); once the global budget is hit, raise
    LLMBudgetExhausted so no further backend is attempted.
    """
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        if call_counter is not None:
            if call_counter[0] >= _MAX_TOTAL_LLM_CALLS:
                raise LLMBudgetExhausted(
                    f"LLM budget exhausted: {call_counter[0]} calls made, "
                    f"cap is {_MAX_TOTAL_LLM_CALLS}. Aborting before calling {entry.model}."
                )
            call_counter[0] += 1
        try:
            call_start = time.monotonic()
            headers: dict[str, str] = {}
            if entry.api_key.strip():
                headers["Authorization"] = f"Bearer {entry.api_key.strip()}"
            resp = await client.post(
                entry.base_url,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            elapsed = time.monotonic() - call_start
            logger.info(
                "[LLM] %s responded in %.2fs (attempt %d/%d)",
                entry.model,
                elapsed,
                attempt + 1,
                _MAX_RETRIES + 1,
            )

            # Business errors returned with HTTP 200 — fatal, do not retry
            if "error" in data and "choices" not in data:
                err = data["error"]
                msg = (
                    err.get("message", str(err)) if isinstance(err, dict) else str(err)
                )
                code = (
                    err.get("code", "unknown") if isinstance(err, dict) else "unknown"
                )
                logger.error(
                    "[LLM] %s business error (code=%s): %s",
                    entry.model,
                    code,
                    msg,
                )
                raise LLMFatalError(
                    f"Business error from {entry.model} (code={code}): {msg}"
                )

            message = data["choices"][0]["message"]
            content = message.get("content")
            if not content:
                # DeepSeek thinking mode may put the real answer in reasoning_content
                reasoning = message.get("reasoning_content")
                if reasoning:
                    logger.warning(
                        "[LLM] %s returned empty content but has reasoning_content (%d chars), using it",
                        entry.model,
                        len(reasoning),
                    )
                    content = reasoning
                else:
                    raise LLMFatalError(
                        f"LLM {entry.model} returned empty content and no reasoning_content. "
                        f"Message keys: {list(message.keys())}"
                    )

            usage = data.get("usage", {})
            token_usage = {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }

            logger.info(
                "[LLM] %s token usage: input=%d, output=%d",
                entry.model,
                token_usage["input"],
                token_usage["output"],
            )

            stripped = content.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1]
            if stripped.endswith("```"):
                stripped = stripped.rsplit("```", 1)[0]
            stripped = stripped.strip()

            if not stripped:
                logger.error(
                    "[LLM] %s content became empty after fence stripping (raw=%d chars)",
                    entry.model,
                    len(content),
                )
                raise LLMFatalError(
                    f"LLM {entry.model} content is empty after stripping code fences"
                )

            return stripped, token_usage

        except LLMFatalError:
            raise
        except httpx.TimeoutException as exc:
            last_error = exc
            logger.warning(
                "[LLM] %s timeout on attempt %d/%d",
                entry.model,
                attempt + 1,
                _MAX_RETRIES + 1,
            )
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                wait = 2**attempt
                logger.warning(
                    "[LLM] %s HTTP %d on attempt %d, retrying in %ds",
                    entry.model,
                    status,
                    attempt + 1,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            # Non-retryable HTTP errors (401, 403, 400, etc.) — fatal
            if status in (401, 403):
                raise LLMAuthError(
                    f"Auth error from {entry.model} (HTTP {status}): "
                    f"check API key configuration"
                ) from exc
            raise LLMFatalError(
                f"HTTP {status} from {entry.model}: {exc.response.text[:500]}"
            ) from exc
        except (KeyError, IndexError) as exc:
            raise LLMFatalError(
                f"Malformed response from {entry.model}: {exc}. "
                f"Response structure does not match expected format — do not retry."
            ) from exc
        except Exception as exc:
            last_error = exc
            logger.error("[LLM] %s unexpected error: %s", entry.model, exc)

        if attempt < _MAX_RETRIES:
            wait = 2**attempt
            logger.info("[LLM] Waiting %ds before retry", wait)
            await asyncio.sleep(wait)

    raise LLMRetryableError(
        f"All {_MAX_RETRIES + 1} attempts exhausted for {entry.model}: {last_error}"
    )


class PostProcessor:
    """Orchestrate focused paper analysis via LLM, with model-pool fallback."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create a long-lived AsyncClient shared across LLM calls."""
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=300.0,
                        trust_env=False,
                    )
        return self._client

    async def aclose(self) -> None:
        """Close the shared HTTP client. Safe to call multiple times."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def extract(
        self,
        markdown: str,
        question: str,
        language: str = "en",
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Extract focused content from paper markdown based on a question.

        Tries multiple entries from the model pool in priority order. On a
        retryable failure (LLMRetryableError), the entry is released with
        success=False (triggers cooldown) and the next entry is attempted.
        Fatal errors (LLMFatalError) on a given entry also fall through to the
        next entry, since a different backend may accept the same request.

        Args:
            markdown: Paper markdown text.
            question: User's analysis question.
            language: Prompt language ("en" / "zh").
            max_attempts: Override for total attempts across the pool. Defaults
                to len(pool.entries) * 2 so each backend can be retried twice.
        """
        settings = get_settings()
        pool = await ModelPool.get(settings)

        template = load_prompt(
            "extract_focus", language, prompts_dir=settings.prompts_dir
        )

        today = date.today().isoformat()
        system_msg = template.system.format(current_date=today)

        if max_attempts is None:
            max_attempts = max(1, len(pool.entries) * 2)

        # Pre-flight: question must fit even in the largest-context entry.
        if pool.entries:
            widest = max(pool.entries, key=lambda e: e.context_tokens)
            probe_budget = PromptBudget(
                model_context_tokens=widest.context_tokens,
                response_headroom_tokens=settings.response_headroom_tokens,
            )
            if (
                probe_budget.max_input_tokens
                - probe_budget.estimate_text(question)
                - 2000
                <= 0
            ):
                raise PostProcessorError(
                    f"Question too long ({probe_budget.estimate_text(question)} est. tokens): "
                    f"leaves no room for paper content in any configured model "
                    f"(widest context {widest.context_tokens} tokens). Shorten the question."
                )

        start = time.monotonic()
        client = await self._get_client()

        call_counter = [0]
        tried_entries: list[str] = []
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                entry = await pool.acquire(timeout=30.0)
            except RuntimeError as exc:
                last_error = exc
                logger.error("[LLM] ModelPool acquire failed: %s", exc)
                break

            entry_label = f"{entry.backend}:{entry.model}"
            tried_entries.append(entry_label)

            try:
                budget = PromptBudget(
                    model_context_tokens=entry.context_tokens,
                    response_headroom_tokens=settings.response_headroom_tokens,
                )
                max_input = (
                    budget.max_input_tokens - budget.estimate_text(question) - 2000
                )
                truncated = budget.truncate_text(markdown, max_input)
                was_truncated = len(truncated) < len(markdown)

                if was_truncated:
                    original_tokens = budget.estimate_text(markdown)
                    truncated_tokens = budget.estimate_text(truncated)
                    logger.warning(
                        "[LLM] Paper truncated for %s: %d → %d tokens (budget=%d)",
                        entry.model,
                        original_tokens,
                        truncated_tokens,
                        max_input,
                    )

                user_msg = template.user.format(
                    user_question=question,
                    paper_content=truncated,
                )

                payload = _build_payload(
                    entry,
                    system_msg,
                    user_msg,
                    settings.response_headroom_tokens,
                    settings,
                )
                content, token_usage = await _call_llm(
                    entry, payload, client, call_counter
                )
                elapsed = time.monotonic() - start

                pool.release(entry, success=True)
                logger.info(
                    "[LLM] Analysis complete on %s after %d attempt(s): %.2fs, truncated=%s",
                    entry_label,
                    attempt + 1,
                    elapsed,
                    was_truncated,
                )
                return {
                    "answer": content,
                    "model_used": entry.model,
                    "backend": entry.backend,
                    "token_usage": token_usage,
                    "llm_seconds": round(elapsed, 2),
                    "truncated": was_truncated,
                    "attempts": attempt + 1,
                }
            except LLMBudgetExhausted:
                pool.release(entry, success=False)
                raise
            except LLMAuthError as exc:
                # Dead API key: release and permanently disable this entry.
                last_error = exc
                pool.release(entry, success=False)
                pool.disable(entry)
                logger.warning(
                    "[LLM] %s auth failure on attempt %d, entry permanently disabled: %s",
                    entry_label,
                    attempt + 1,
                    exc,
                )
            except LLMFatalError as exc:
                # Fatal on this entry: don't retry THIS entry, but try the next backend.
                last_error = exc
                pool.release(entry, success=False)
                logger.warning(
                    "[LLM] %s fatal error on attempt %d, trying next backend: %s",
                    entry_label,
                    attempt + 1,
                    exc,
                )
            except LLMRetryableError as exc:
                # Retries inside this entry exhausted — release with cooldown and try next.
                last_error = exc
                pool.release(entry, success=False)
                logger.warning(
                    "[LLM] %s exhausted retries on attempt %d, trying next backend: %s",
                    entry_label,
                    attempt + 1,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                pool.release(entry, success=False)
                logger.error(
                    "[LLM] %s unexpected error on attempt %d, trying next backend: %s",
                    entry_label,
                    attempt + 1,
                    exc,
                )

        raise PostProcessorError(
            f"All LLM backends failed after {max_attempts} attempt(s). "
            f"Tried: {tried_entries}. Last error: {last_error}"
        ) from last_error
