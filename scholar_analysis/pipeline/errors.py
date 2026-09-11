"""Public, stage-specific failures without leaking upstream credentials or URLs."""
from __future__ import annotations
from typing import Any

class PipelineError(RuntimeError):
    def __init__(self, code: str, stage: str, message: str, *, retryable: bool = False,
                 retry_after: float | None = None, details: dict | None = None):
        super().__init__(message)
        self.code, self.stage = code, stage
        self.retryable, self.retry_after = retryable, retry_after
        self.details = details or {}

def error_result(request_id: str, exc: PipelineError, timing: dict | None = None) -> dict[str, Any]:
    return {"request_id": request_id, "status": "error", "error": str(exc),
            "error_code": exc.code, "stage": exc.stage, "retryable": exc.retryable,
            "retry_after_seconds": exc.retry_after, "details": exc.details, "timing": timing or {}}
