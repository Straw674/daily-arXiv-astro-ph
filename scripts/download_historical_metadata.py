import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CONFIG = MappingProxyType(
    {
        "dataset": "secemp9/arxiv-complete",
        "revision": "cee894837962fede5612cccf2a4c7cacf49b4c3a",
        "source_path": "metadata/train-00000-of-00001.parquet",
        "source_bytes": 1643927530,
        "source_sha256": "7f7bea788af318a3f0fed1c446303c725d339c2cb189ede4412aeadd0ab9bf1f",
        "start_date": "2020-01-01",
        "end_date_exclusive": "2026-01-01",
        "primary_categories": ("astro-ph.GA", "astro-ph.CO"),
        "output_path": Path(__file__).resolve().parents[1]
        / "data/arxiv_2020_2025_primary_ga_co.jsonl",
        "batch_rows": 32768,
        "download_chunk_bytes": 8 * 1024 * 1024,
        "log_interval_bytes": 256 * 1024 * 1024,
        "timeout_seconds": 90,
        "download_attempts": 3,
        "retry_delay_seconds": 5,
    }
)

logger = logging.getLogger(__name__)


def download_metadata(config, destination):
    url = (
        f"https://huggingface.co/datasets/{config['dataset']}/resolve/"
        f"{config['revision']}/{config['source_path']}"
    )
    for attempt in range(config["download_attempts"]):
        offset = destination.stat().st_size if destination.exists() else 0
        if offset == config["source_bytes"]:
            break
        request = Request(url, headers={"Range": f"bytes={offset}-"})
        try:
            with urlopen(request, timeout=config["timeout_seconds"]) as response:
                if response.status != 206:
                    offset = 0
                elif not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("Unexpected download range")
                next_log = offset + config["log_interval_bytes"]
                with destination.open("ab" if offset else "wb") as target:
                    while chunk := response.read(config["download_chunk_bytes"]):
                        target.write(chunk)
                        offset += len(chunk)
                        if offset >= next_log:
                            logger.info("Downloaded %d metadata bytes", offset)
                            next_log += config["log_interval_bytes"]
            if destination.stat().st_size != config["source_bytes"]:
                raise ValueError("Incomplete metadata download")
            break
        except Exception:
            if attempt + 1 == config["download_attempts"]:
                raise
            logger.warning("Download interrupted; retrying", exc_info=True)
            time.sleep(config["retry_delay_seconds"])
    with destination.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != config["source_sha256"]:
        raise ValueError("Metadata SHA-256 mismatch")
    logger.info("Metadata download verified: %d bytes", destination.stat().st_size)


def select_papers(source_path, config):
    columns = (
        "paper_id",
        "title",
        "abstract",
        "categories",
        "primary_category",
        "first_version_date",
    )
    parquet = pq.ParquetFile(source_path)
    papers = []
    scanned = 0
    for batch in parquet.iter_batches(
        batch_size=config["batch_rows"], columns=list(columns)
    ):
        scanned += batch.num_rows
        primary = batch.column(batch.schema.get_field_index("primary_category"))
        selected = batch.filter(
            pc.is_in(primary, value_set=pa.array(config["primary_categories"]))
        )
        for row in selected.to_pylist():
            submitted_date = str(row["first_version_date"])[:10]
            if (
                not config["start_date"]
                <= submitted_date
                < config["end_date_exclusive"]
            ):
                continue
            papers.append(
                {
                    "id": row["paper_id"],
                    "title": row["title"],
                    "summary": row["abstract"],
                    "categories": row["categories"].split(),
                    "primary_category": row["primary_category"],
                    "submitted_date": submitted_date,
                }
            )
    if scanned != parquet.metadata.num_rows:
        raise ValueError("Not all metadata rows were scanned")
    if not papers or len({paper["id"] for paper in papers}) != len(papers):
        raise ValueError("Empty selection or duplicate paper IDs")
    if any(
        not paper["title"].strip()
        or not paper["summary"].strip()
        or paper["primary_category"] not in paper["categories"]
        for paper in papers
    ):
        raise ValueError("Invalid title, abstract, or category")
    logger.info("Scanned %d rows; selected %d unique papers", scanned, len(papers))
    return sorted(
        papers, key=lambda paper: (paper["submitted_date"], paper["id"])
    ), scanned


def save_papers(papers, scanned, config, temporary_directory):
    staged = temporary_directory / "papers.jsonl"
    with staged.open("w", encoding="utf-8") as target:
        for paper in papers:
            target.write(json.dumps(paper, ensure_ascii=False) + "\n")
    with staged.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    counts = Counter(
        (paper["submitted_date"][:4], paper["primary_category"]) for paper in papers
    )
    manifest = {
        "dataset": config["dataset"],
        "revision": config["revision"],
        "source_path": config["source_path"],
        "source_sha256": config["source_sha256"],
        "date_field": "first_version_date",
        "start_date": config["start_date"],
        "end_date_exclusive": config["end_date_exclusive"],
        "primary_categories": config["primary_categories"],
        "source_rows_scanned": scanned,
        "paper_count": len(papers),
        "output_bytes": staged.stat().st_size,
        "output_sha256": digest,
        "abstract_field": "summary",
        "counts_by_year_and_primary_category": [
            {"year": year, "primary_category": category, "paper_count": count}
            for (year, category), count in sorted(counts.items())
        ],
    }
    staged_manifest = temporary_directory / "manifest.json"
    staged_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output = config["output_path"]
    staged.replace(output)
    staged_manifest.replace(output.with_suffix(".manifest.json"))
    logger.info(
        "Saved %d papers (%d bytes) to %s", len(papers), output.stat().st_size, output
    )
    for (year, category), count in sorted(counts.items()):
        logger.info("%s %s: %d papers", year, category, count)


def main(config=CONFIG):
    config["output_path"].parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading metadata from dataset revision %s", config["revision"])
    with TemporaryDirectory(prefix="arxiv-metadata-") as directory:
        temporary_directory = Path(directory)
        source = temporary_directory / "metadata.parquet"
        download_metadata(config, source)
        logger.info("Selecting papers by first submission date and primary category")
        papers, scanned = select_papers(source, config)
        save_papers(papers, scanned, config, temporary_directory)
    logger.info("Download and selection complete; temporary metadata removed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
