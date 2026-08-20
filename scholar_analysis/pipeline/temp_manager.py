"""Background task to clean up stale temp directories."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import Callable, Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


async def cleanup_loop(
    base_dir: str | Path,
    interval_seconds: float = 300.0,
    max_age_seconds: float = 1800.0,
    active_request_ids: Callable[[], Iterable[str]] | None = None,
) -> None:
    """Periodically scan base_dir and remove subdirs older than max_age.

    Directories whose name matches an active request id are never removed,
    regardless of age (active check takes priority over age check).
    """

    def _active_ids() -> set[str]:
        if active_request_ids is None:
            return set()
        try:
            return set(active_request_ids())
        except Exception as exc:
            logger.error("Temp cleanup: failed to read active request ids: %s", exc)
            return set()

    base = Path(base_dir)
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            if not base.exists():
                continue
            active = _active_ids()
            removed = 0
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                try:
                    if child.name in active:
                        continue
                    mtime = child.stat().st_mtime
                    if time.time() - mtime > max_age_seconds:
                        shutil.rmtree(child)
                        removed += 1
                        logger.info(
                            "Temp cleanup: removed %s (age=%.0fs)",
                            child,
                            time.time() - mtime,
                        )
                except OSError as exc:
                    logger.error("Temp cleanup: failed to process %s: %s", child, exc)
                except Exception as exc:
                    logger.error("Temp cleanup: unexpected error on %s: %s", child, exc)
            if removed:
                logger.info("Temp cleanup: removed %d expired dirs", removed)
        except Exception as e:
            logger.error("Temp cleanup error: %s", e)
