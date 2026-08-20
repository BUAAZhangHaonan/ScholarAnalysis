"""Client for the arxiv_mirror REST API (resolve, download, get parsed text)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass


import httpx

logger = logging.getLogger(__name__)


@dataclass
class PaperInfo:
    arxiv_id: str
    versioned_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]


@dataclass
class AssetStatus:
    versioned_id: str
    download_status: str
    local_path: str
    file_size: int = 0


class ArxivMirrorError(Exception):
    """Raised when the arxiv_mirror API returns a business-level error."""


class ArxivMirrorClient:
    """Thin async wrapper around the arxiv_mirror REST API."""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self._base_url = base_url
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        """Long-lived shared client (connection reuse across requests)."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                trust_env=False,
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()

    async def resolve(self, query: str) -> PaperInfo:
        """Resolve arXiv ID / URL to paper metadata.

        Raises ArxivMirrorError on business-level errors (not found, etc.).
        Raises httpx.HTTPError on network/transport failures (propagates to caller).
        """
        c = self.http
        r = await c.post("/resolve", json={"query": query})
        r.raise_for_status()
        data = r.json()

        if data.get("error"):
            raise ArxivMirrorError(f"Resolve failed for {query!r}: {data['error']}")

        state = data.get("state", "")
        if state == "not_found":
            raise ArxivMirrorError(
                f"Paper not found for query {query!r}. "
                f"Only arXiv IDs (e.g. 2402.01306) and arXiv URLs are supported."
            )

        result = data.get("result") or data
        arxiv_id = result.get("arxiv_id", "") or ""

        if not arxiv_id:
            raise ArxivMirrorError(
                f"Resolve returned empty arxiv_id for query {query!r}. State: {state}"
            )

        return PaperInfo(
            arxiv_id=arxiv_id,
            versioned_id=result.get("versioned_id", arxiv_id),
            title=result.get("title") or "",
            authors=result.get("authors") or [],
            abstract=result.get("abstract") or "",
            categories=result.get("categories") or [],
        )

    async def download(self, query: str) -> AssetStatus:
        """Resolve and download a paper PDF. Polls until download completes.

        Returns AssetStatus with local_path for filesystem access.

        Raises ArxivMirrorError on business-level errors.
        Raises httpx.HTTPError on network/transport failures.
        """
        c = self.http
        r = await c.post("/resolve-and-download", json={"query": query})
        r.raise_for_status()
        data = r.json()
        versioned_id = data.get("versioned_id", "")
        if not versioned_id:
            raise ArxivMirrorError(
                f"No versioned_id in download response for {query!r}"
            )

        logger.info(
            "[arxiv_mirror] Download started for %s (versioned_id=%s)",
            query,
            versioned_id,
        )

        dl_status = ""
        local_path = data.get("local_path", "")
        file_size = data.get("file_size", 0)
        max_polls = 60  # 60 * 5s = 300s max wait for download

        for attempt in range(max_polls):
            try:
                r = await c.get(f"/asset/{versioned_id}")
                r.raise_for_status()
                status = r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 500:
                    logger.warning(
                        "[arxiv_mirror] Poll %d/%d for %s: 500 (server still processing), retrying",
                        attempt + 1,
                        max_polls,
                        versioned_id,
                    )
                    await asyncio.sleep(5)
                    continue
                raise
            except httpx.RequestError as e:
                logger.warning(
                    "[arxiv_mirror] Poll %d/%d for %s: connection error (%s), retrying",
                    attempt + 1,
                    max_polls,
                    versioned_id,
                    e,
                )
                await asyncio.sleep(5)
                continue

            dl_status = status.get("download_status", "")

            if not local_path:
                local_path = status.get("local_path", "")
            if not file_size:
                file_size = status.get("file_size", 0)

            logger.info(
                "[arxiv_mirror] Poll %d/%d for %s: download=%s",
                attempt + 1,
                max_polls,
                versioned_id,
                dl_status,
            )

            if dl_status == "failed":
                raise ArxivMirrorError(
                    f"Download failed for {query!r} (versioned_id={versioned_id}): "
                    f"server reported download_status=failed"
                )

            if dl_status == "completed":
                if not local_path:
                    raise ArxivMirrorError(
                        f"Download completed but no local_path for {query!r} "
                        f"(versioned_id={versioned_id})"
                    )
                return AssetStatus(
                    versioned_id=versioned_id,
                    download_status=dl_status,
                    local_path=local_path,
                    file_size=file_size,
                )

            await asyncio.sleep(5)

        raise ArxivMirrorError(
            f"Download timed out for {query!r} (versioned_id={versioned_id}): "
            f"{max_polls} polls over {max_polls * 5}s, "
            f"last download_status={dl_status}"
        )
