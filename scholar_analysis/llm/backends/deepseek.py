"""DeepSeek payload: thinking mode is always explicit, including disabled."""

from __future__ import annotations

from typing import Any


def build_payload(
    *,
    model: str,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 16_000,
    thinking: bool = False,
) -> dict[str, Any]:
    """Build a DeepSeek chat completion payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    return payload
