"""Client for the MinerU PDF-to-Markdown API."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MinerUClient:
    """Async client for MinerU file_parse endpoint."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: float = 600.0,
    ):
        self._base_url = base_url
        self._auth = httpx.BasicAuth(username, password)
        self._timeout = timeout

    def _mineru_client(self, **kwargs) -> httpx.AsyncClient:
        defaults = dict(
            base_url=self._base_url,
            auth=self._auth,
            timeout=self._timeout,
            trust_env=False,
        )
        defaults.update(kwargs)
        return httpx.AsyncClient(**defaults)

    def _download_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            trust_env=False,
        )

    async def parse_pdf(
        self,
        pdf_path: Path,
        *,
        text_only: bool = True,
        lang_list: str = "",
    ) -> dict[str, Any]:
        """Upload a local PDF to MinerU and return the parsed result.

        Args:
            pdf_path: Path to the local PDF file.
            text_only: If True, strip image references from output.
            lang_list: Optional language hint for MinerU.

        Returns:
            MinerU JSON result dict.
        """
        async with self._mineru_client() as c:
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

        return result

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
            async with self._download_client() as dl:
                r = await dl.get(url)
                r.raise_for_status()
                tmp_path.write_bytes(r.content)

            logger.info("Downloaded PDF to %s (%d bytes)", tmp_path, tmp_path.stat().st_size)
            return await self.parse_pdf(tmp_path, text_only=text_only, lang_list=lang_list)
        finally:
            tmp_path.unlink(missing_ok=True)


def extract_markdown(parse_result: dict[str, Any], *, text_only: bool = True) -> str:
    """Extract concatenated Markdown text from MinerU parse result."""
    results = parse_result.get("results", {})
    if not isinstance(results, dict):
        return ""

    parts: list[str] = []
    for v in results.values():
        if not isinstance(v, dict):
            continue
        md = v.get("md_content", "")
        if md:
            parts.append(md)

    text = "\n\n".join(parts)

    if text_only:
        import re
        # Remove image references: ![alt](path)
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        # Clean up excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
