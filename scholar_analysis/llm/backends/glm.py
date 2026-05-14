"""GLM backend — builds request payloads for Zhipu AI models."""

from __future__ import annotations

from typing import Any


def build_payload(
    *,
    model: str,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 16_000,
) -> dict[str, Any]:
    """Build a GLM chat completion payload."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
