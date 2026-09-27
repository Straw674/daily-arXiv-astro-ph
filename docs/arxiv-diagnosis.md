# arXiv HTTP diagnosis

Use the manually triggered **arXiv HTTP diagnosis** workflow to investigate API
failures on a GitHub-hosted runner. Push the workflow to the default branch first,
then select it under Actions and choose **Run workflow**. No API secrets are needed;
the existing optional repository variable `EMAIL` supplies the contact address.

Wait for other daily or diagnostic runs to finish, or cancel stuck runs, before
starting. Diagnostic concurrency serializes diagnostic runs only; it does not
serialize the daily workflow. arXiv's limits apply across machines under your
control: https://info.arxiv.org/help/api/tou.html.

The script makes three RSS requests and at most five API requests, separated by
10 seconds, without automatic retries or redirects. A 429 response or any
`Retry-After` header stops collection. It selects up to 100 sorted, unique new or
cross-listed IDs from the current RSS feeds. With no IDs, API probes are skipped.

API comparisons use the installed arxiv library's URL serializer:

| Case | IDs | max_results | Headers |
|---|---|---|---|
| single-current | First ID | 1 | Current fetcher headers |
| single-legacy | First ID | 1 | Headers before e5b269b |
| single-current-page500 | First ID | 500 | Current fetcher headers |
| batch-current | Selected IDs | 500 | Current fetcher headers |
| batch-legacy | Selected IDs | 500 | Headers before e5b269b |

Read the run summary and download `arxiv-diagnostics-<run_id>` from Artifacts.
`report.json` includes timestamps, commit, package versions, exact URLs, selected
request/response headers, timing, and the first 8 KiB of each response body.
Individual probe JSON files remain available if later collection fails. Artifacts
expire after seven days. The contact address appears in recorded request headers.
A green workflow means collection completed; HTTP errors are diagnostic results.

Interpret comparisons as evidence, not proof: requests run sequentially and
server state can change between them.

- All API probes fail while RSS succeeds: inspect body previews and request IDs
  for endpoint rejection details. A status code alone cannot establish an IP ban.
- Single-ID probes succeed but batch probes fail: investigate batch size or query
  filtering, then confirm with a smaller batch before changing production.
- Only legacy or current headers succeed: isolate User-Agent and Accept separately
  in a follow-up comparison before changing production headers.
- Only the page-size comparison differs: investigate the max_results parameter.
- Everything succeeds: the original failure may be transient or runner-specific;
  compare against the failed run's time and environment.

For a local comparison, run `uv sync --locked` and then
`.venv/bin/python scripts/diagnose_arxiv.py`, with no simultaneous Actions run.
The same report is written to `diagnostics/`. All run configuration is defined in
the script's immutable module-level `CONFIG`; there are no command-line options.

Production retries are limited to three retries for transient HTTP statuses
(429, 500, 502, 503, 504) and transport failures. The waits are 5, 15, and 45
seconds unless `Retry-After` requires longer, with a 60-second per-wait budget.
If the server requests more than 60 seconds, collection fails instead of retrying
before the permitted time. HTTP 406 fails immediately with response diagnostics.
The arxiv client's additional retry layer is disabled, and fetch failures propagate
to fail the workflow instead of being reported as an empty paper list.

Persistent
endpoint rejection may require arXiv support; include UTC timestamps, request URLs,
response details, and the fact that the client runs on GitHub-hosted runners.
