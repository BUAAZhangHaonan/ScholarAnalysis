# ScholarAnalysis

MCP paper reading and optional focused analysis. arXiv uses the existing mirror;
public PDF and DOI/publisher pages use their declared PDF links and the existing MinerU parser.

## Install and run

Python 3.11 or later is required.

```bash
python -m venv venv
venv/bin/pip install -e .
cp .env.example .env
# Set service credentials and upstream endpoints locally; never commit .env.
venv/bin/python -m scholar_analysis.main
```

The SSE endpoint is `/sse`. Clients authenticate with an Authorization Bearer
header. A production systemd template is in `scripts/scholar-analysis.service`;
review its paths and environment before installing it.

## Read a document

`get_paper_text` accepts exactly one source: `query` (arXiv ID/URL or DOI),
`pdf_url` (direct PDF or a page declaring citation_pdf_url), or a returned
`document_id` for cache-only reading. It does not search paper titles or bypass paywalls.

```json
{"pdf_url":"https://aclanthology.org/2024.findings-acl.212.pdf","limit_chars":1000}
```

Continue with the returned document ID and next offset:

```json
{"document_id":"doc_from_previous_response","offset":1000,"limit_chars":64000}
```

`find_text` performs literal search from offset and returns matching character/line
locations with surrounding text. `lang` is an optional parser hint; reuse it on
subsequent reads. `refresh=true` with the original source explicitly reparses.
Requests for the same source share in-flight work. Valid cached versions are
readable without the mirror. An unversioned arXiv query may return its cached
version; use refresh to request an updated snapshot.

The response preserves Markdown, source original/final URLs, document revision,
range, total_chars, next_offset, eof, section headings and line positions.
Positions refer to Unicode characters in the requested text mode, not PDF pages.
`eof` only means the returned slice reaches the end; full coverage also requires
range.start=0. Parse coverage is not a certification of PDF completeness.
Tables and formula text supplied by MinerU are retained. Image references can be
preserved with include_images; no image pixels are fetched or interpreted.

## Optional paid analysis

`analyze_paper` adds question, language (en/zh), optional offset/limit_chars and
analysis_id. An explicit range can select a relevant method or appendix.
If the model context cannot hold all input, only complete Markdown blocks are
supplied and the actual coverage is disclosed. Unread text cannot support a
claim that the full paper lacks a result.

```json
{"document_id":"doc_from_previous_response","question":"Explain the method and its assumptions in one paragraph.","language":"en","analysis_id":"my-review-001"}
```

Reuse analysis_id with identical inputs to wait for or retrieve the same task.
Results, failures and unresolved interrupted tasks are retained; a new ID
explicitly authorizes a new task. Cached replies expose original_cost separately
and incur zero new model cost. Analysis receipts are not silently evicted to
permit duplicate generation; when receipt storage is full, new tasks fail before
a model call. Clearing that storage also clears the idempotency history.

The response includes the final answer, exact evidence quotes and matched source
positions. Matching proves attribution only, not scientific entailment.
Reasoning-only or length-stopped responses are errors. Text analysis does not
claim to inspect images. Unknown sent outcomes are not automatically retried;
bounded HTTP 429 retry stays on the same configured model.

Default: `deepseek-flash`. Select `deepseek-v4-pro` explicitly through
SCHOLAR_ANALYSIS_DEEPSEEK_MODEL if wanted. There is no automatic provider/model
fallback. Thinking false/true is sent explicitly as disabled/enabled.
Account concurrency capacities are not worker defaults; size model and PDF
concurrency from measured service capacity.

## Errors, usage and storage

All tool successes, operation errors and cache reads include `mcp.cost.v1`:
CNY provider API usage estimates, per-attempt records, unknown usage and price
ranges. These are not invoices and exclude parsing/hosting/GPU costs. Tariff
source/version is included; peak crossings or missing cache breakdown keep a
range. Missing usage is never counted as zero.

Errors include error_code, stage, retryable and retry_after_seconds. For example,
mirror_download HTTP 502 is a mirror failure, not a MinerU error. Both queue wait
and total request duration are bounded. Raw upstream diagnostics remain in the
server log.

Parse cache files are validated and written atomically. The URL catalog contains
source identities, not hashes. Analysis records retain question, result and cost;
restrict access to the configured cache directory. Packaged YAML prompts work
outside the repository; an explicit prompts_dir may override them.

## Validation

Offline, no network or model calls:

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python scripts/stress_test.py
```

Explicit live PDF and cached-concurrency acceptance:

```bash
venv/bin/python scripts/stress_test.py --live --url http://127.0.0.1:8005 \
  --pdf-url https://aclanthology.org/2024.findings-acl.212.pdf --requests 4 --concurrency 4
```

The CLI reads SCHOLAR_ANALYSIS_ACCESS_TOKEN from the shell first, then this
repository's .env (or --env-file). It never prints credentials. Missing credentials
fail before connecting; HTTP 401/403 ends the session promptly. --no-auth is only
for deliberately unauthenticated test endpoints. Live model calls additionally require --analysis --allow-paid-analysis.
By default analysis requests share one ID to test idempotency; add
--distinct-analysis-ids for separately authorized parallel generations.
Reports include semantic checks, p50/p95 latency, success/degraded/failure counts,
unique model attempts and cost bounds. See [the recorded validation](docs/VALIDATION_20260912.md).
