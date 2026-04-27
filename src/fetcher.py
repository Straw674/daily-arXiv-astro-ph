import logging
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List

import arxiv

logger = logging.getLogger(__name__)


def _paper_version(paper: arxiv.Result) -> int:
    """Extract version number from short id, e.g. 2601.00001v2 -> 2."""
    short_id = paper.get_short_id()
    if "v" not in short_id:
        return 1
    try:
        return int(short_id.rsplit("v", 1)[1])
    except (ValueError, IndexError):
        return 1


def _format_date_distribution(results: List[arxiv.Result]) -> str:
    """Format updated.date distribution for logging."""
    if not results:
        return "{}"
    dist = Counter(p.updated.date().isoformat() for p in results)
    ordered = sorted(dist.items(), reverse=True)
    return "{" + ", ".join(f"{d}: {n}" for d, n in ordered) + "}"


def fetch_papers_for_category(category: str) -> List[arxiv.Result]:
    """Fetch the latest batch of papers for a specific category using RSS feed."""
    url = f"https://rss.arxiv.org/rss/{category}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
    except Exception as e:
        logger.error("Failed to fetch RSS for %s: %s", category, e)
        return []

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.error("Failed to parse RSS XML for %s: %s", category, e)
        return []

    channel = root.find("channel")
    if channel is None:
        logger.info("[%s] No channel found in RSS.", category)
        return []

    items = channel.findall("item")
    new_ids = []

    for item in items:
        # Check announce_type
        announce_type_elem = item.find("{http://arxiv.org/schemas/atom}announce_type")
        if announce_type_elem is not None and announce_type_elem.text in [
            "new",
            "cross",
        ]:
            link_elem = item.find("link")
            if link_elem is not None and link_elem.text:
                # link is typically https://arxiv.org/abs/2604.22105
                paper_id = link_elem.text.strip().split("/")[-1]
                new_ids.append(paper_id)

    if not new_ids:
        logger.info(
            "[%s] No 'new' or 'cross' submissions found in RSS today.", category
        )
        return []

    logger.info(
        "[%s] Found %d 'new'/'cross' submissions in RSS feed.", category, len(new_ids)
    )

    # Fetch full metadata via arXiv API
    search = arxiv.Search(id_list=new_ids)
    client = arxiv.Client()
    try:
        results = list(client.results(search))
    except Exception as e:
        logger.error("[%s] Failed to fetch metadata from arXiv API: %s", category, e)
        return []

    for p in results:
        logger.info(
            f"Fetched [{category}]: ID: {p.get_short_id()} | Updated: {p.updated.date()} | Version: {_paper_version(p)} | Primary Category: {getattr(p, 'primary_category', 'N/A')}"
        )

    return results


def fetch_papers(categories: List[str]) -> List[arxiv.Result]:
    """Fetch the latest papers for all given categories and deduplicate."""
    all_papers = []
    seen_ids = set()
    for cat in categories:
        logger.info(f"Fetching arXiv papers for category: {cat}")
        papers = fetch_papers_for_category(cat)
        logger.info("[%s] papers returned after filtering: %d", cat, len(papers))
        for p in papers:
            if p.get_short_id() not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p.get_short_id())

    logger.info(f"Fetched {len(all_papers)} unique papers across all categories.")
    return all_papers
