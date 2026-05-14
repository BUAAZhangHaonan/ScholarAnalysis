"""DeepSeek backend — builds request payloads with thinking mode support."""

from __future__ import annotations

from typing import Any


def build_payload(
    *,
    model: str,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 16_000,
    thinking: bool = True,
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
    if thinking:
        payload["thinking"] = {"type": "enabled"}
    else:
        payload["thinking"] = {"type": "disabled"}
    return payload
