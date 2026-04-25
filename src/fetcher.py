import arxiv
import logging
from typing import List

logger = logging.getLogger(__name__)


def fetch_papers_for_category(category: str) -> List[arxiv.Result]:
    """Fetch the latest batch of papers for a specific category."""
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    results = list(client.results(search))
    if not results:
        return []

    # We only want the most recent batch. The first result dictates the latest updated date.
    latest_date = results[0].updated.date()
    return [p for p in results if p.updated.date() == latest_date]


def fetch_papers(categories: List[str]) -> List[arxiv.Result]:
    """Fetch the latest papers for all given categories and deduplicate."""
    all_papers = []
    seen_ids = set()
    for cat in categories:
        logger.info(f"Fetching arXiv papers for category: {cat}")
        papers = fetch_papers_for_category(cat)
        for p in papers:
            if p.get_short_id() not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p.get_short_id())

    logger.info(f"Fetched {len(all_papers)} unique papers across all categories.")
    return all_papers
