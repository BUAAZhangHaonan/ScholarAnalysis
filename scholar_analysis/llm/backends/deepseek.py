"""DeepSeek backend — builds request payloads.

The non-standard `thinking` field is only emitted when explicitly requested
via the `thinking=True` flag (defaults to False). DeepSeek's official chat
completions API does not accept this field on most models; only enable it if
you are targeting a reasoner-class model that documents it.
"""

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
    if thinking:
        payload["thinking"] = {"type": "enabled"}
    return payload
