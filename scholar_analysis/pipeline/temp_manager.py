"""Background task to clean up stale temp directories."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


async def cleanup_loop(
    base_dir: str | Path,
    interval_seconds: float = 300.0,
    max_age_seconds: float = 1800.0,
) -> None:
    """Periodically scan base_dir and remove subdirs older than max_age."""
    base = Path(base_dir)
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            if not base.exists():
                continue
            removed = 0
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                try:
                    mtime = child.stat().st_mtime
                    if time.time() - mtime > max_age_seconds:
                        shutil.rmtree(child)
                        removed += 1
                        logger.info("Temp cleanup: removed %s (age=%.0fs)", child, time.time() - mtime)
                except OSError as exc:
                    logger.error("Temp cleanup: failed to process %s: %s", child, exc)
                except Exception as exc:
                    logger.error("Temp cleanup: unexpected error on %s: %s", child, exc)
            if removed:
                logger.info("Temp cleanup: removed %d expired dirs", removed)
        except Exception as e:
            logger.error("Temp cleanup error: %s", e)
