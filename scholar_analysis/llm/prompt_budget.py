"""Token estimation and truncation for LLM prompt budgeting."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class PromptBudget:
    """Conservative prompt budget estimator.

    Uses ceil(UTF-8 bytes / 2) as a rough token estimate.
    """

    model_context_tokens: int = 256_000
    response_headroom_tokens: int = 16_000
    tool_headroom_tokens: int = 4_000
    message_overhead_tokens: int = 24

    @property
    def max_input_tokens(self) -> int:
        return max(
            1,
            self.model_context_tokens
            - self.response_headroom_tokens
            - self.tool_headroom_tokens,
        )

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, ceil(len(text.encode("utf-8")) / 2))

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            total += self._message_overhead(message)
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.estimate_text(content)
            else:
                total += self.estimate_text(str(content))
        return total

    def _message_overhead(self, message: dict[str, Any]) -> int:
        return self.message_overhead_tokens + self.estimate_text(
            str(message.get("role", ""))
        )

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self.estimate_text(text) <= max_tokens:
            return text
        truncated = text
        while truncated and self.estimate_text(truncated) > max_tokens:
            new_length = max(1, int(len(truncated) * 0.8))
            if new_length >= len(truncated):
                new_length = len(truncated) - 1
            truncated = truncated[:new_length]
        return truncated
