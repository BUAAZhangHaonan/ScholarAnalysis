"""Client for the MinerU PDF-to-Markdown API.

Supports multiple endpoints with priority-ordered fallback. Each endpoint can
have its own BasicAuth credentials (or none).
"""

from __future__ import annotations

import logging
import re
import uuid
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from scholar_analysis.pipeline.errors import PipelineError
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
                if not isinstance(result, dict) or not extract_markdown(result).strip():
                    raise ValueError("MinerU returned no usable Markdown")
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
                if status in (400, 404, 422):
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
        self, url: str, temp_dir: Path, *, text_only: bool = True,
        lang_list: str = "", max_bytes: int = 50 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Read a direct PDF or a landing page's declared citation_pdf_url.

        Redirects and one explicit publisher PDF link are followed. This does
        not infer a PDF from arbitrary links or claim paywalled text was read.
        """
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = temp_dir / f"{uuid.uuid4().hex}.pdf"
        original_url, current = url, url
        try:
            for step in range(2):
                parsed = urlsplit(current)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                    raise PipelineError("INVALID_PDF_URL", "pdf_download", "Use an HTTP(S) URL without credentials.")
                dl = self._get_download_client()
                try:
                    async with dl.stream("GET", current, headers={"Accept": "application/pdf,text/html;q=0.8"}) as r:
                        r.raise_for_status()
                        final_url = str(r.url)
                        content = bytearray()
                        content_type = r.headers.get("content-type", "").lower()
                        limit = max_bytes if "text/html" not in content_type else min(max_bytes, 2 * 1024 * 1024)
                        async for chunk in r.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > limit:
                                raise PipelineError("PDF_TOO_LARGE", "pdf_download", "Document exceeds the configured download size limit.")
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    raise PipelineError("PDF_HTTP_ERROR", "pdf_download", f"Paper server returned HTTP {status}.",
                                        retryable=status in (429, 500, 502, 503, 504),
                                        details={"http_status": status}) from exc
                except httpx.RequestError as exc:
                    raise PipelineError("PDF_NETWORK_ERROR", "pdf_download", "Paper download failed.",
                                        retryable=True) from exc
                if bytes(content[:1024]).lstrip().startswith(b"%PDF-"):
                    tmp_path.write_bytes(content)
                    result = await self.parse_pdf(tmp_path, text_only=text_only, lang_list=lang_list)
                    result["_source"] = {"original_url": original_url, "final_url": final_url,
                                         "document_type": "pdf", "download_bytes": len(content)}
                    return result
                if step == 0:
                    links = _PDFLinks()
                    links.feed(bytes(content).decode("utf-8", errors="replace"))
                    if links.pdf_url:
                        current = urljoin(final_url, links.pdf_url)
                        continue
                raise PipelineError("PDF_LINK_REQUIRED", "pdf_download",
                                    "No PDF body or declared publisher PDF link was available; provide a direct accessible PDF URL.")
            raise AssertionError("unreachable")
        finally:
            tmp_path.unlink(missing_ok=True)


class _PDFLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdf_url = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == "meta" and attrs.get("name", "").lower() == "citation_pdf_url":
            self.pdf_url = attrs.get("content") or self.pdf_url
        elif tag.lower() == "link" and attrs.get("type", "").lower() == "application/pdf":
            self.pdf_url = self.pdf_url or attrs.get("href")


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
        if isinstance(md, str) and md.strip():
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
