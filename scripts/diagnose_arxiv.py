"""Collect bounded HTTP diagnostics without the production retry machinery."""

import json
import os
import platform
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

import arxiv
import requests

CONFIG = MappingProxyType(
    {
        "categories": ("astro-ph.GA", "astro-ph.CO", "astro-ph.IM"),
        "email": os.getenv("EMAIL") or "qx24@mails.tsinghua.edu.cn",
        "interval": 10,
        "timeout": 30,
        "max_ids": 100,
        "body_limit": 8192,
        "output": Path("diagnostics"),
        "summary": os.getenv("GITHUB_STEP_SUMMARY"),
        "commit": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
    }
)


def rss_ids(content):
    root = ET.fromstring(content)
    return [
        item.findtext("link").split("/abs/")[-1].strip()
        for item in root.findall("./channel/item")
        if item.findtext("{http://arxiv.org/schemas/atom}announce_type")
        in ("new", "cross")
        and "/abs/" in (item.findtext("link") or "")
    ]


def api_cases(ids, email):
    legacy = {"User-Agent": "arxiv.py/2.1.3", "Accept": "*/*"}
    current = {
        "User-Agent": f"arxiv.py/2.1.3 daily-arxiv-astro-ph/1.0 (mailto:{email})",
        "Accept": "application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
    }
    return (
        ("single-current", ids[:1], 1, current),
        ("single-legacy", ids[:1], 1, legacy),
        ("single-current-page500", ids[:1], 500, current),
        ("batch-current", ids, 500, current),
        ("batch-legacy", ids, 500, legacy),
    )


def probe(name, url, headers, config):
    started = time.monotonic()
    record = {
        "case": name,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "status": None,
    }
    content = b""
    try:
        # A fresh session avoids cookies carrying over between comparisons.
        # requests defaults to zero retries; redirects are recorded, not followed.
        with requests.Session() as session:
            response = session.get(
                url, headers=headers, timeout=config["timeout"], allow_redirects=False
            )
            content = response.content
            record.update(
                status=response.status_code,
                request_headers={
                    key: response.request.headers[key]
                    for key in ("User-Agent", "Accept", "Accept-Encoding")
                },
                response_headers={
                    key: value
                    for key, value in response.headers.items()
                    if key.lower()
                    in {
                        "date",
                        "server",
                        "content-type",
                        "content-length",
                        "retry-after",
                        "via",
                        "x-cache",
                        "x-served-by",
                        "x-request-id",
                        "cf-ray",
                        "location",
                    }
                },
                retry_after=response.headers.get("Retry-After"),
                body_bytes=len(content),
                body_preview=content[: config["body_limit"]].decode("utf-8", "replace"),
            )
            if response.status_code == 200 and "api/query" in url:
                try:
                    root = ET.fromstring(content)
                    record["atom_feed"] = (
                        root.tag == "{http://www.w3.org/2005/Atom}feed"
                    )
                    record["entry_count"] = len(
                        root.findall("{http://www.w3.org/2005/Atom}entry")
                    )
                except ET.ParseError as error:
                    record["parse_error"] = str(error)
    except requests.RequestException as error:
        record["error"] = str(error)
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (config["output"] / f"{name}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(
        f"{name}: status={record['status']} elapsed={record['elapsed_seconds']}s",
        flush=True,
    )
    return record, content


def should_stop(record):
    return record["status"] == 429 or bool(record.get("retry_after"))


def run(config):
    config["output"].mkdir(parents=True, exist_ok=True)
    records = []
    ids = set()
    notes = []
    for category in config["categories"]:
        if records:
            time.sleep(config["interval"])
        record, content = probe(
            f"rss-{category}",
            f"https://rss.arxiv.org/rss/{category}",
            {
                "User-Agent": f"daily-arxiv-astro-ph/1.0 (mailto:{config['email']})",
                "Accept": "application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
            },
            config,
        )
        records.append(record)
        if should_stop(record):
            notes.append(
                "Stopped on rate limiting or Retry-After; no further probes sent."
            )
            break
        if record["status"] == 200:
            try:
                ids.update(rss_ids(content))
            except ET.ParseError as error:
                notes.append(f"RSS parse error for {category}: {error}")

    selected_ids = sorted(ids)[: config["max_ids"]]
    if selected_ids and not should_stop(records[-1]):
        for name, case_ids, page_size, headers in api_cases(
            selected_ids, config["email"]
        ):
            time.sleep(config["interval"])
            # Use the installed library's own URL serializer to reproduce production.
            client = arxiv.Client(page_size=page_size)
            try:
                url = client._format_url(arxiv.Search(id_list=case_ids), 0, page_size)
            finally:
                client._session.close()
            record, _ = probe(name, url, headers, config)
            records.append(record)
            if should_stop(record):
                notes.append(
                    "Stopped on rate limiting or Retry-After; no further probes sent."
                )
                break
    elif not selected_ids:
        notes.append(
            "No RSS IDs available; API comparisons skipped. Run on an announcement day."
        )

    report = {
        "commit": config["commit"],
        "run_id": config["run_id"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("arxiv", "requests", "urllib3")},
        "ids": selected_ids,
        "rss_unique_ids": len(ids),
        "notes": notes,
        "probes": records,
    }
    (config["output"] / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    summary = "# arXiv HTTP diagnostics\n\n| Case | HTTP status | Seconds | Atom entries |\n|---|---|---|---|\n"
    for record in records:
        summary += f"| {record['case']} | {record['status']} | {record['elapsed_seconds']} | {record.get('entry_count', '')} |\n"
    summary += "\n" + "\n".join(notes)
    summary += "\n\nDownload the diagnostics artifact for request headers, response headers, and body previews. A successful workflow means collection completed, not that arXiv requests succeeded.\n"
    (config["output"] / "summary.md").write_text(summary, encoding="utf-8")
    if config["summary"]:
        with Path(config["summary"]).open("a", encoding="utf-8") as output:
            output.write(summary)


if __name__ == "__main__":
    run(CONFIG)
