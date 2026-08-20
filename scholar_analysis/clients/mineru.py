"""Client for the MinerU PDF-to-Markdown API.

Supports multiple endpoints with priority-ordered fallback. Each endpoint can
have its own BasicAuth credentials (or none).
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MinerUError(RuntimeError):
    """Raised when all MinerU endpoints fail."""


class MinerUClient:
    """Async client for MinerU file_parse endpoint with multi-endpoint fallback."""

    def __init__(
        self,
        endpoints: list[tuple[str, str, str]] | None = None,
        timeout: float = 600.0,
    ) -> None:
        """endpoints is a list of (url, username, password) tuples in priority order.

        For an endpoint that does not require auth, pass ("", "") as credentials.
        """
        if not endpoints:
            raise ValueError(
                "MinerUClient requires at least one endpoint. "
                "Configure SCHOLAR_ANALYSIS_MINERU_ENDPOINTS or SCHOLAR_ANALYSIS_MINERU_BASE_URL."
            )
        self._endpoints: list[tuple[str, str, str]] = [
            (url.rstrip("/"), user or "", pwd or "") for (url, user, pwd) in endpoints
        ]
        self._timeout = timeout
        # Long-lived clients for connection reuse (index-aligned with _endpoints).
        self._clients: list[httpx.AsyncClient | None] = [None] * len(self._endpoints)
        self._download_client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        for c in self._clients:
            if c is not None and not c.is_closed:
                await c.aclose()
        if self._download_client is not None and not self._download_client.is_closed:
            await self._download_client.aclose()

    @classmethod
    def from_settings(cls, settings: Any = None) -> "MinerUClient":
        if settings is None:
            from scholar_analysis.config import get_settings

            settings = get_settings()

        if not settings.mineru_endpoints_list:
            raise ValueError(
                "No MinerU endpoints configured. Set SCHOLAR_ANALYSIS_MINERU_ENDPOINTS "
                "(comma-separated URLs) or SCHOLAR_ANALYSIS_MINERU_BASE_URL (legacy single URL)."
            )

        triples: list[tuple[str, str, str]] = []
        for url, (user, pwd) in zip(
            settings.mineru_endpoints_list, settings.mineru_creds_list, strict=False
        ):
            triples.append((url, user, pwd))

        if not triples:
            raise ValueError("MinerU endpoint list resolved to empty.")
        return cls(endpoints=triples, timeout=settings.http_timeout)

    def _client_for(self, idx: int) -> httpx.AsyncClient:
        """Return the (lazily created) long-lived client for endpoint idx."""
        c = self._clients[idx]
        if c is None or c.is_closed:
            url, user, pwd = self._endpoints[idx]
            auth: httpx.Auth | None = None
            if user or pwd:
                auth = httpx.BasicAuth(user, pwd)
            c = httpx.AsyncClient(
                base_url=url,
                auth=auth,
                timeout=self._timeout,
                trust_env=False,
            )
            self._clients[idx] = c
        return c

    def _get_download_client(self) -> httpx.AsyncClient:
        c = self._download_client
        if c is None or c.is_closed:
            c = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                trust_env=False,
            )
            self._download_client = c
        return c

    async def parse_pdf(
        self,
        pdf_path: Path,
        *,
        text_only: bool = True,
        lang_list: str = "",
    ) -> dict[str, Any]:
        """Upload a local PDF to MinerU, trying endpoints in order until one succeeds.

        Args:
            pdf_path: Path to the local PDF file.
            text_only: If True, strip image references from output.
            lang_list: Optional language hint for MinerU.

        Returns:
            MinerU JSON result dict.

        Raises:
            MinerUError: If every endpoint failed.
        """
        last_exc: Exception | None = None
        for idx, (url, user, _pwd) in enumerate(self._endpoints, start=1):
            label = f"{url} (auth={'yes' if user else 'no'})"
            try:
                c = self._client_for(idx - 1)
                with open(pdf_path, "rb") as f:
                    r = await c.post(
                        "/file_parse",
                        files={"files": (pdf_path.name, f, "application/pdf")},
                        data={
                            "backend": "hybrid-auto-engine",
                            "return_md": "true",
                            "formula_enable": "true",
                            "table_enable": "true",
                            **({"lang_list": lang_list} if lang_list else {}),
                        },
                    )
                r.raise_for_status()
                result = r.json()
                logger.info(
                    "[MinerU] endpoint %d/%d succeeded: %s",
                    idx,
                    len(self._endpoints),
                    label,
                )
                return result
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                last_exc = exc
                if 400 <= status < 500 and status not in (401, 403, 429):
                    # Deterministic client-side failure (e.g. 413 payload too large):
                    # every endpoint would reject the same PDF — don't re-parse.
                    raise MinerUError(
                        f"MinerU endpoint {label} returned deterministic HTTP {status}; "
                        f"not trying remaining endpoints"
                    ) from exc
                logger.warning(
                    "[MinerU] endpoint %d/%d %s returned HTTP %d; trying next",
                    idx,
                    len(self._endpoints),
                    label,
                    status,
                )
            except ValueError as exc:
                # r.json() failing (JSONDecodeError) — endpoint returned non-JSON
                last_exc = exc
                logger.warning(
                    "[MinerU] endpoint %d/%d %s returned non-JSON body: %s; trying next",
                    idx,
                    len(self._endpoints),
                    label,
                    exc,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "[MinerU] endpoint %d/%d %s connection error: %s; trying next",
                    idx,
                    len(self._endpoints),
                    label,
                    exc,
                )

        raise MinerUError(
            f"All MinerU endpoints failed ({len(self._endpoints)} tried). Last error: {last_exc}"
        ) from last_exc

    async def parse_from_url(
        self,
        url: str,
        temp_dir: Path,
        *,
        text_only: bool = True,
        lang_list: str = "",
    ) -> dict[str, Any]:
        """Download PDF from URL, upload to MinerU, return parsed result.

        The temp PDF is cleaned up after parsing.
        """
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = temp_dir / f"{uuid.uuid4().hex}.pdf"

        try:
            dl = self._get_download_client()
            r = await dl.get(url)
            r.raise_for_status()
            tmp_path.write_bytes(r.content)

            logger.info(
                "Downloaded PDF to %s (%d bytes)", tmp_path, tmp_path.stat().st_size
            )
            return await self.parse_pdf(
                tmp_path, text_only=text_only, lang_list=lang_list
            )
        finally:
            tmp_path.unlink(missing_ok=True)


_IMAGE_REF_RE = re.compile(r"!\[(?:[^\]]|\][^(])*\]\([^)]+\)")


def extract_markdown(parse_result: dict[str, Any], *, text_only: bool = True) -> str:
    """Extract concatenated Markdown text from MinerU parse result."""
    results = parse_result.get("results", {})
    if not isinstance(results, dict):
        logger.warning(
            "MinerU parse result has unexpected 'results' type=%s (expected dict); "
            "top-level keys: %s",
            type(results).__name__,
            list(parse_result.keys()),
        )
        return ""

    parts: list[str] = []
    for v in results.values():
        if not isinstance(v, dict):
            continue
        md = v.get("md_content", "")
        if md:
            parts.append(md)

    if not parts:
        logger.warning(
            "MinerU parse result contained no md_content (results keys: %s)",
            list(results.keys()),
        )
        return ""

    text = "\n\n".join(parts)

    if text_only:
        # Remove image references: ![alt](path) — alt text may itself contain ']'
        text = _IMAGE_REF_RE.sub("", text)
        # Clean up excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
