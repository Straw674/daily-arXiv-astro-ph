import os
import asyncio
import logging
from datetime import datetime, timezone
import json
import argparse
from collections import defaultdict
from dotenv import load_dotenv

from openai import AsyncOpenAI, OpenAI

from fetcher import fetch_papers
from llm import enhance_papers_concurrently, generate_daily_topics
from renderer import render_daily_markdown, render_readme
from embedding import get_embeddings_in_batches, compute_knn_scores

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_step1(categories, fetched_jsonl_path, force_regen):
    if force_regen:
        logger.info("Force regeneration enabled. Removing existing files.")
        if os.path.exists(fetched_jsonl_path):
            os.remove(fetched_jsonl_path)

    if os.path.exists(fetched_jsonl_path):
        logger.info(f"{fetched_jsonl_path} exists. Skipping Step 1 (Fetch).")
        return

    logger.info("Fetching new papers from arXiv...")
    papers = fetch_papers(categories)
    if not papers:
        logger.info("No new papers found today.")
        return

    with open(fetched_jsonl_path, "w", encoding="utf-8") as f:
        for p in papers:
            p_dict = {
                "id": p.get_short_id(),
                "title": p.title,
                "summary": p.summary,
                "url": p.entry_id,
                "pdf_url": p.pdf_url,
                "categories": p.categories,
                "primary_category": getattr(p, "primary_category", None),
                "updated_date": p.updated.date().isoformat() if p.updated else None,
            }
            f.write(json.dumps(p_dict, ensure_ascii=False) + "\n")
    logger.info(f"Step 1 complete. Fetched data saved to {fetched_jsonl_path}")


async def run_step2(fetched_jsonl_path, jsonl_path, model_name, language, force_regen):
    if force_regen:
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)

    if os.path.exists(jsonl_path):
        logger.info(f"{jsonl_path} exists. Skipping Step 2 (LLM generation).")
        return

    if not os.path.exists(fetched_jsonl_path):
        logger.error(f"{fetched_jsonl_path} not found. Please run Step 1 first.")
        return

    logger.info("Loading fetched data from file...")
    with open(fetched_jsonl_path, "r", encoding="utf-8") as f:
        papers = [json.loads(line) for line in f]

    if not papers:
        logger.info("No papers to process.")
        return

    logger.info("Generating topics...")
    client = AsyncOpenAI()
    topics = await generate_daily_topics(client, papers, model_name)

    logger.info("Enhancing papers with LLM...")
    concurrency_limit = int(os.getenv("CONCURRENCY_LIMIT") or 5)

    enhanced_data = await enhance_papers_concurrently(
        client, papers, model_name, language, topics, concurrency=concurrency_limit
    )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in enhanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Step 2 complete. Enhanced data saved to {jsonl_path}")


def load_zotero_embeddings(zotero_emb_path):
    if not os.path.exists(zotero_emb_path):
        return None
    try:
        with open(zotero_emb_path, "r", encoding="utf-8") as f:
            zotero_data = json.load(f)
        embs = [
            p.get("embedding")
            for p in zotero_data.get("papers", [])
            if "embedding" in p
        ]
        return embs if embs else None
    except Exception as e:
        logger.error(f"Failed to load Zotero embeddings: {e}")
        return None


def score_papers_with_zotero(enhanced_data, zotero_embs):
    emb_model = os.getenv("EMBEDDING_MODEL_NAME") or "text-embedding-v4"
    logger.info(f"Computing embeddings for arXiv papers with {emb_model}...")

    client = OpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=(
            os.getenv("EMBEDDING_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )

    texts_to_embed = [
        f"{p.get('title', '')}\n{p.get('summary', '')}" for p in enhanced_data
    ]

    try:
        paper_embs = get_embeddings_in_batches(
            client, texts_to_embed, emb_model, batch_size=10
        )
        scores = compute_knn_scores(paper_embs, zotero_embs, top_k=5)
        for item, score in zip(enhanced_data, scores):
            item["similarity_score"] = score
    except Exception as e:
        logger.error(f"kNN scoring failed: {e}")
        for item in enhanced_data:
            item["similarity_score"] = 0.0


def group_and_sort_papers(enhanced_data):
    grouped_data = defaultdict(list)
    for item in enhanced_data:
        grouped_data[item.get("topic", "Others")].append(item)

    for topic in grouped_data:
        grouped_data[topic].sort(
            key=lambda x: x.get("similarity_score", 0.0), reverse=True
        )

    sorted_topics = sorted(
        grouped_data.keys(),
        key=lambda t: (
            sum(x.get("similarity_score", 0.0) for x in grouped_data[t][:3])
            / len(grouped_data[t][:3])
        )
        if t != "Others"
        else -float("inf"),
        reverse=True,
    )

    ordered_grouped_data = [
        {"topic": t, "papers": grouped_data[t]}
        for t in sorted_topics
        if grouped_data[t]
    ]
    return ordered_grouped_data


def run_step3(
    jsonl_path, md_path, readme_path, data_dir, today_str, force_regen, zotero_emb_path
):
    if not os.path.exists(jsonl_path):
        logger.error(f"{jsonl_path} not found. Please run Step 2 first.")
        return

    if force_regen and os.path.exists(md_path):
        os.remove(md_path)

    logger.info("Loading data from file...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        enhanced_data = [json.loads(line) for line in f]

    logger.info(f"Loading Zotero embeddings from {zotero_emb_path}...")
    zotero_embs = load_zotero_embeddings(zotero_emb_path)

    if zotero_embs:
        score_papers_with_zotero(enhanced_data, zotero_embs)
    else:
        logger.warning(
            f"Zotero embeddings not found or invalid at {zotero_emb_path}. Setting all scores to 0.0."
        )
        for item in enhanced_data:
            item["similarity_score"] = 0.0

    logger.info("Grouping and sorting data...")
    ordered_grouped_data = group_and_sort_papers(enhanced_data)

    logger.info("Rendering daily markdown...")
    render_daily_markdown(ordered_grouped_data, today_str, md_path)

    logger.info("Updating README...")
    render_readme(data_dir, "data", readme_path)
    logger.info("Step 3 complete.")


async def main():
    parser = argparse.ArgumentParser(description="Daily arXiv Summarizer")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        help="1: Fetch, 2: LLM, 3: Embed and Render. If omitted, runs all sequentially.",
    )
    args = parser.parse_args()

    load_dotenv()

    categories = (
        os.getenv("CATEGORIES") or "astro-ph.GA, astro-ph.CO, astro-ph.IM"
    ).split(",")
    categories = [c.strip() for c in categories]
    model_name = os.getenv("MODEL_NAME") or "deepseek-chat"
    language = os.getenv("LANGUAGE") or "中文"
    output_root = os.getenv("OUTPUT_ROOT") or "dist"
    data_dir = os.path.join(output_root, "data")
    force_regen = os.getenv("FORCE_REGEN") == "true"
    zotero_emb_path = os.getenv("ZOTERO_EMB_PATH") or "zotero/zotero_embeddings.json"

    os.makedirs(data_dir, exist_ok=True)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fetched_jsonl_path = os.path.join(data_dir, f"{today_str}_fetched.jsonl")
    jsonl_path = os.path.join(data_dir, f"{today_str}.jsonl")
    md_path = os.path.join(data_dir, f"{today_str}.md")
    readme_path = os.path.join(output_root, "README.md")

    if args.step in [1, None]:
        run_step1(categories, fetched_jsonl_path, force_regen)

    if args.step in [2, None]:
        await run_step2(
            fetched_jsonl_path, jsonl_path, model_name, language, force_regen
        )

    if args.step in [3, None]:
        run_step3(
            jsonl_path,
            md_path,
            readme_path,
            data_dir,
            today_str,
            force_regen,
            zotero_emb_path,
        )


if __name__ == "__main__":
    asyncio.run(main())
