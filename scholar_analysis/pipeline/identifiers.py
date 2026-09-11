"""Normalize supported identifiers without contacting a resolver."""
from __future__ import annotations
import re
from urllib.parse import urlsplit, urlunsplit
from scholar_analysis.pipeline.errors import PipelineError

_ARXIV = re.compile(r"^(?:[0-9]{4}\.[0-9]{4,5}|[a-zA-Z][a-zA-Z0-9.-]*/[0-9]{7})(?:v[1-9][0-9]*)?$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$", re.I)

def normalize(query: str = "", pdf_url: str | None = None) -> tuple[str, str]:
    if bool(query.strip()) == bool(pdf_url):
        raise PipelineError("INVALID_INPUT", "input", "Provide exactly one of query or pdf_url.")
    value = (pdf_url or query).strip()
    if value.lower().startswith("doi:"):
        value = value[4:].strip()
    if _DOI.fullmatch(value):
        return "url", "https://doi.org/" + value
    arxiv = value.removeprefix("arxiv:").removeprefix("arXiv:")
    if _ARXIV.fullmatch(arxiv):
        return "arxiv", arxiv
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise PipelineError("UNSUPPORTED_IDENTIFIER", "input", "Use an arXiv ID, DOI, or an HTTP(S) paper/PDF URL.")
    host = parsed.hostname.lower()
    if host in ("arxiv.org", "www.arxiv.org", "export.arxiv.org"):
        match = re.match(r"^/(?:abs|pdf)/(.+?)(?:\.pdf)?/?$", parsed.path)
        if match and _ARXIV.fullmatch(match[1]):
            return "arxiv", match[1]
    return "url", urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.query, ""))
