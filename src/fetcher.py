import logging
import requests
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import arxiv

logger = logging.getLogger(__name__)

class TimeoutSession(requests.Session):
    def request(self, *args, **kwargs):
        # Default to 30 seconds timeout if not explicitly set
        kwargs.setdefault('timeout', 30)
        return super().request(*args, **kwargs)

class CustomRetry(Retry):
    def get_backoff_time(self):
        retry_count = len(self.history)
        # 1st retry: 5s, 2nd: 15s, 3rd: 45s, 4th: 135s, 5th: 405s, 6th: 1215s (20 mins)
        return 5 * (3 ** retry_count)

    def increment(self, method=None, url=None, response=None, error=None, _pool=None, _stacktrace=None):
        retry_count = len(self.history)
        backoff = self.get_backoff_time()

        if response:
            status = response.status
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                logger.warning(f"Request failed with status {status}. Server requested Retry-After: {retry_after}s. Retrying attempt {retry_count + 1}...")
            else:
                logger.warning(f"Request failed with status {status}. Backing off for {backoff}s before retry {retry_count + 1}...")
        elif error:
            logger.warning(f"Request encountered error: {error}. Backing off for {backoff}s before retry {retry_count + 1}...")
            
        return super().increment(method, url, response, error, _pool, _stacktrace)

def get_robust_session() -> requests.Session:
    session = TimeoutSession()
    retry_strategy = CustomRetry(
        total=8,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

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


def _fetch_ids_from_rss(category: str) -> List[str]:
    """Fetch the latest arXiv IDs for a specific category using RSS feed."""
    url = f"https://rss.arxiv.org/rss/{category}"
    try:
        session = get_robust_session()
        response = session.get(url, headers={"User-Agent": "Mozilla/5.0 (daily-arxiv-astro-ph)"}, timeout=30)
        response.raise_for_status()
        xml_data = response.content
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
    return new_ids


def fetch_papers(categories: List[str]) -> List[arxiv.Result]:
    """Fetch the latest papers for all given categories and deduplicate."""
    unique_ids = set()
    for cat in categories:
        logger.info(f"Fetching arXiv IDs via RSS for category: {cat}")
        cat_ids = _fetch_ids_from_rss(cat)
        unique_ids.update(cat_ids)

    if not unique_ids:
        logger.info("No new papers found across any categories.")
        return []

    id_list = list(unique_ids)
    logger.info(
        f"Fetched {len(id_list)} unique paper IDs across all categories in total. Querying arXiv API..."
    )

    # Fetch full metadata via arXiv API in one single batch request
    search = arxiv.Search(id_list=id_list)

    # Use custom client with higher base delay and a custom requests session
    # to handle exponential backoff and Retry-After headers automatically.
    client = arxiv.Client(page_size=500, delay_seconds=10.0, num_retries=3)
    client._session = get_robust_session()

    try:
        results = list(client.results(search))
    except Exception as e:
        logger.error("Failed to fetch metadata from arXiv API: %s", e)
        return []

    for p in results:
        logger.info(
            f"Fetched arXiv metadata: ID: {p.get_short_id()} | Updated: {p.updated.date()} | Version: {_paper_version(p)} | Primary Category: {getattr(p, 'primary_category', 'N/A')}"
        )

    logger.info(
        f"Successfully fetched {len(results)} fully detailed papers from arXiv API."
    )
    return results
