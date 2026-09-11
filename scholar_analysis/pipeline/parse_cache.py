"""Validated, atomic parse cache and a small URL-to-document catalog."""
from __future__ import annotations
import asyncio
import json
import logging
import sqlite3
import tempfile
import uuid
from contextlib import closing
from pathlib import Path
from urllib.parse import quote
from scholar_analysis.clients.mineru import extract_markdown

logger = logging.getLogger(__name__)

def _filename(key: str, lang: str) -> Path:
    # Reversible escaping; no content hashing. Old arXiv filenames remain readable.
    encoded = quote(key, safe=".")
    hint = quote(lang or "default", safe="")
    if not key or len(encoded) > 180 or len(hint) > 40:
        raise ValueError("Invalid cache document key or language hint")
    return Path(f"{encoded}__{hint}.json")

class ParseCache:
    def __init__(self, cache_dir: str, max_bytes: int):
        self._dir = Path(cache_dir)
        self._max_bytes = max_bytes

    async def document_key(self, alias: str) -> str:
        return await asyncio.to_thread(self._document_key, alias)

    def _document_key(self, alias: str) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._dir / "catalog.sqlite", timeout=30)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS documents (alias TEXT PRIMARY KEY, document_id TEXT NOT NULL)")
            key = "doc_" + uuid.uuid4().hex
            db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?)", (alias, key))
            db.commit()
            return db.execute("SELECT document_id FROM documents WHERE alias=?", (alias,)).fetchone()[0]

    async def get(self, key: str, lang: str = "") -> dict | None:
        return await asyncio.to_thread(self._get_sync, self._dir / _filename(key, lang))

    @staticmethod
    def _get_sync(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not extract_markdown(data).strip():
                raise ValueError("Cached parse has no usable Markdown")
            return data
        except FileNotFoundError:
            return None
        except (ValueError, TypeError, OSError):
            logger.warning("Invalid parse cache entry; treating as miss: %s", path.name)
            return None

    async def put(self, key: str, lang: str, result: dict) -> None:
        if not isinstance(result, dict) or not extract_markdown(result).strip():
            raise ValueError("Refusing to cache a parse without usable Markdown")
        await asyncio.to_thread(self._put_sync, self._dir / _filename(key, lang), result)

    def _put_sync(self, path: Path, result: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._dir,
                                             prefix="parse-", suffix=".tmp", delete=False) as f:
                tmp = Path(f.name)
                json.dump(result, f, ensure_ascii=False)
            tmp.replace(path)
            self._evict()
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    async def invalidate(self, key: str, lang: str = "") -> None:
        await asyncio.to_thread((self._dir / _filename(key, lang)).unlink, missing_ok=True)

    def _evict(self) -> None:
        files = []
        for p in self._dir.glob("*.json"):
            try:
                st = p.stat()
                files.append((p, st.st_mtime, st.st_size))
            except FileNotFoundError:
                pass
        total = sum(size for _, _, size in files)
        for path, _, size in sorted(files, key=lambda x: x[1]):
            if total <= self._max_bytes:
                break
            try:
                path.unlink(missing_ok=True)
                total -= size
            except OSError:
                logger.warning("Unable to evict cache entry %s", path.name)
