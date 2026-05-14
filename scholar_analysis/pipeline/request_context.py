"""Per-request context and tracker for request isolation."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    request_id: str
    created_at: float
    temp_dir: Path
    status: str = "pending"  # pending → downloading → parsing → processing → completed/failed


class RequestTracker:
    """Tracks active requests and cleans up expired temp directories."""

    def __init__(self, base_temp_dir: str | Path, max_age_seconds: float = 1800.0):
        self._base_dir = Path(base_temp_dir)
        self._max_age = max_age_seconds
        self._contexts: dict[str, RequestContext] = {}

    async def create(self) -> RequestContext:
        request_id = uuid.uuid4().hex[:16]
        temp_dir = self._base_dir / request_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        ctx = RequestContext(
            request_id=request_id,
            created_at=time.monotonic(),
            temp_dir=temp_dir,
        )
        self._contexts[request_id] = ctx
        logger.info("[REQ %s] Created, temp_dir=%s", request_id, temp_dir)
        return ctx

    async def remove(self, request_id: str) -> None:
        ctx = self._contexts.pop(request_id, None)
        if ctx is None:
            logger.warning("[REQ %s] remove() called but context not found", request_id)
            return
        if ctx.temp_dir.exists():
            try:
                shutil.rmtree(ctx.temp_dir)
                logger.info("[REQ %s] Cleaned up temp_dir=%s", request_id, ctx.temp_dir)
            except OSError as exc:
                logger.error("[REQ %s] Failed to clean temp_dir=%s: %s", request_id, ctx.temp_dir, exc)

    async def cleanup_expired(self) -> int:
        """Remove contexts older than max_age. Returns count cleaned."""
        now = time.monotonic()
        expired = [
            rid
            for rid, ctx in self._contexts.items()
            if now - ctx.created_at > self._max_age
        ]
        for rid in expired:
            await self.remove(rid)
            logger.warning("[REQ %s] Expired, cleaned up", rid)
        return len(expired)

    @property
    def active_count(self) -> int:
        return len(self._contexts)
