"""Disk cache for MinerU parse results — pure acceleration layer.

Keyed by versioned arXiv id + MinerU lang hint. Stores the full
/file_parse response JSON (including image references); text_only
stripping stays in extract_markdown per-request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[0-9]{4,5}\.[0-9]{4,5}(v[0-9]+)?$")


def _filename(versioned_id: str, lang: str) -> Path | None:
    """Return cache file path, or None if versioned_id fails whitelist."""
    if not _SAFE_ID.match(versioned_id):
        logger.warning("parse cache: unsafe versioned_id %r -> miss", versioned_id)
        return None
    return Path(f"{versioned_id}__{lang or 'default'}.json")


class ParseCache:
    """Simple per-file JSON cache with size-bounded eviction (by mtime)."""

    def __init__(self, cache_dir: str, max_bytes: int) -> None:
        self._dir = Path(cache_dir)
        self._max_bytes = max_bytes

    async def get(self, versioned_id: str, lang: str = "") -> dict | None:
        name = _filename(versioned_id, lang)
        if name is None:
            return None
        return await asyncio.to_thread(self._get_sync, self._dir / name)

    @staticmethod
    def _get_sync(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"cache entry is {type(data).__name__}, not dict")
            return data
        except FileNotFoundError:
            return None
        except Exception:
            logger.warning("parse cache: failed to read %s, treating as miss", path)
            return None

    async def put(self, versioned_id: str, lang: str, result: dict) -> None:
        name = _filename(versioned_id, lang)
        if name is None:
            return
        await asyncio.to_thread(self._put_sync, self._dir / name, result)

    def _put_sync(self, path: Path, result: dict) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            self._evict()
        except Exception:
            logger.warning("parse cache: failed to write %s", path, exc_info=True)

    def _evict(self) -> None:
        files = [p for p in self._dir.iterdir() if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        if total <= self._max_bytes:
            return
        for p in sorted(files, key=lambda p: p.stat().st_mtime):
            if total <= self._max_bytes:
                break
            size = p.stat().st_size
            try:
                p.unlink()
                total -= size
                logger.info("parse cache: evicted %s (%d bytes)", p.name, size)
            except OSError:
                logger.warning("parse cache: failed to evict %s", p)
