# ScholarAnalysis functional validation — 2026-09-12

This record separates offline checks, real PDF extraction and model/service deployment.
No production configuration or service restart was performed by this implementation task.

## Completed

- Offline unittest suite: 43 tests at this checkpoint, including invalid parse cache
  recovery, DOI/publisher PDF handoff, same-document single-flight, waiter cancellation,
  exact character/line/section reads, mirror/download 502 attribution, bounded queue,
  final-answer/finish-reason checks, quote matching, unknown usage, peak/offpeak bounds,
  analysis task idempotency and retention of unresolved tasks.
- No offline test uses a live endpoint, paid model or shared production cache.
- Real public PDF: https://aclanthology.org/2024.findings-acl.212.pdf.
  Downloaded PDF: 903408 bytes; parser Markdown: 55367 Unicode characters.
  Initial bounded read: 1000 characters, 70.595 seconds.
- Four concurrent cache-only next-page reads: all successful, same document revision,
  0.003–0.004 seconds each, no repeated parsing and no model calls.
- The real PDF check used an isolated temporary cache and request directory, both
  cleaned afterward. Provider model API cost: zero; this excludes parser/hosting costs.
- Authenticated SSE/CLI acceptance on an isolated loopback server: four real
  cache reads, 4 success / 0 degraded / 0 failed, p50 0.143s / p95 0.156s.
  No model or upstream parsing calls. This exercises the actual MCP schema and CLI.
- Wheel and sdist both contain the packaged prompts, cost and document modules;
  importing directly from the wheel outside the checkout and loading Chinese
  prompts succeeded. Builds used the existing base setuptools 78.1.1 in temporary
  directories; no dependency was installed or production package changed.
- Authorized live analysis reused the existing 1409.1556v1 parser cache
  (https://arxiv.org/abs/1409.1556v1), with only characters 0–6000 of 39064 supplied.
  Four concurrent calls sharing one analysis_id all succeeded and made one model
  request; three shared its result. A later persisted read made zero new calls.
  Actual requested/returned model: deepseek-flash; thinking disabled explicitly.
  finish_reason=stop; 1712 input tokens (0 hit, 1712 miss), 232 output tokens.
  Provider API usage estimate: 0.00264000 CNY, known offpeak tariff, 1.485s for the
  four calls plus persisted read. This is not an invoice.
  The 506-character final answer explained the paper's controlled depth comparison.
  Three exact evidence quotes were located in the supplied excerpt, and partial
  coverage was explicit. The prompt requested one supporting quote; the model
  supplied three evidence entries, within the tool's eight-quote contract.
- These checks show accessible text and correct transport/cache behavior.
  They do not certify every equation, table or PDF page was correctly extracted.

## Reproducible commands

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python scripts/stress_test.py --output /tmp/scholar-analysis-offline.json
venv/bin/python scripts/stress_test.py --live --url http://127.0.0.1:8005 \
  --pdf-url https://aclanthology.org/2024.findings-acl.212.pdf --requests 4 --concurrency 4 \
  --output /tmp/scholar-analysis-read-live.json
```

To repeat paid analysis after authorization, use --analysis --allow-paid-analysis.
Shared-ID mode tests one generation; --distinct-analysis-ids tests separate model
requests. Credentials are read only from the environment and are absent from reports.

## Deployment checks still owned by the coordinator

- Set the configured model alias to deepseek-flash; keep the chosen thinking mode
  explicit. Do not raise PDF parsing concurrency to the account model capacity.
- If an installed systemd unit still names the removed top-level prompts directory
  in ReadWritePaths, remove that obsolete entry before restarting.
- Restart the service, verify real advertised MCP parameters and authenticated calls.
- Confirm production calls preserve the same usage, range and evidence contracts; the isolated live analysis above already passed.

## Production client authentication correction

The first coordinator-run live CLI inherited no shell access token, while systemd
loaded the production service token from EnvironmentFile. GET /sse was allowed by
the handshake policy, but POST /messages returned 401. The SDK logged the failed
POST while initialization kept waiting. This was a test-client credential-loading
and failure-propagation defect, not a changed production token.

The CLI now reads the shell token first, then the repository .env or --env-file.
It fails before connecting when credentials are absent (unless --no-auth was
explicitly requested), watches HTTP 401/403, and bounds the whole MCP session.
No production credential or service restart was needed for this correction.

Five client-only regression tests passed: .env loading without exported variables,
shell precedence, missing-token behavior, GET 200 plus POST 401 terminating
promptly, and a stalled-session deadline. Tests use fixture credentials and mocked
HTTP transport.

The corrected CLI was run against the already-restarted production service:
1409.1556v1, 16 requests, concurrency 8. Results: 16 successful, 0 degraded,
0 failed; p50 0.172s, p95 0.217s; zero model calls and zero provider API cost.
