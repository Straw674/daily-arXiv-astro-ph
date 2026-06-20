import logging
import requests
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import arxiv
import urllib.request
from bs4 import BeautifulSoup
from datetime import datetime

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
                # link is typically https://arxiv.org/abs/2604.22105 or https://arxiv.org/abs/astro-ph/9605168
                link_text = link_elem.text.strip()
                if "/abs/" in link_text:
                    paper_id = link_text.split("/abs/")[-1]
                else:
                    paper_id = link_text.split("/")[-1]
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

    return fetch_papers_by_ids(id_list)


def fetch_papers_for_date(categories: List[str], target_date_str: str) -> List[arxiv.Result]:
    """Fetch papers for a specific date using the pastweek page."""
    try:
        dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        formatted_date = dt.strftime("%d %b %Y")
        if formatted_date.startswith("0"):
            formatted_date = formatted_date[1:]
    except ValueError:
        logger.error(f"Invalid date format: {target_date_str}. Expected YYYY-MM-DD.")
        return []

    unique_ids = set()
    for cat in categories:
        logger.info(f"Scraping pastweek page for {cat} on {formatted_date}")
        url = f"https://arxiv.org/list/{cat}/pastweek?show=2000"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            continue

        soup = BeautifulSoup(html, 'html.parser')
        h3_tags = soup.find_all('h3')
        
        for h3 in h3_tags:
            if formatted_date in h3.text:
                current = h3.next_sibling
                while current and current.name != 'h3':
                    if current.name == 'dt':
                        dt_tag = current
                        descriptor = dt_tag.find('span', class_='descriptor')
                        is_replace = False
                        if descriptor and 'replaced' in descriptor.text.lower():
                            is_replace = True
                            
                        a_tag = dt_tag.find('a', title='Abstract')
                        if a_tag:
                            paper_id = a_tag.text.replace('arXiv:', '').strip()
                            if not is_replace:
                                unique_ids.add(paper_id)
                    current = current.next_sibling
                break

    if not unique_ids:
        logger.info(f"No papers found on pastweek page for date {target_date_str}.")
        return []
        
    return fetch_papers_by_ids(list(unique_ids))




def fetch_papers_by_ids(id_list: List[str]) -> List[arxiv.Result]:
    """Fetch the latest papers for a specific list of IDs."""
    fixed_ids = []
    for pid in id_list:
        pid = pid.strip()
        # Fix broken 7-digit old IDs from the log
        if "." not in pid and len(pid) == 7 and pid.isdigit():
            fixed_ids.append(f"astro-ph/{pid}")
        else:
            fixed_ids.append(pid)

    logger.info(
        f"Fetching {len(fixed_ids)} unique paper IDs directly via arXiv API..."
    )

    search = arxiv.Search(id_list=fixed_ids)
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
