import os
import json
import logging
import argparse
from dotenv import load_dotenv

import bibtexparser
from openai import OpenAI

from embedding import get_embeddings_in_batches

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Process Zotero .bib file to generate embeddings."
    )
    parser.add_argument(
        "--bib",
        type=str,
        default=None,
        help="Path to the Zotero .bib file. Defaults to ZOTERO_BIB_PATH env var or 'zotero_test.bib'.",
    )
    args = parser.parse_args()

    load_dotenv()

    bib_path = args.bib or os.getenv("ZOTERO_BIB_PATH") or "zotero/zotero_all.bib"
    emb_path = os.getenv("ZOTERO_EMB_PATH") or "zotero/zotero_embeddings.json"

    if not os.path.exists(bib_path):
        logger.error(f"Zotero bib file not found: {bib_path}")
        return

    logger.info(f"Parsing {bib_path}...")
    with open(bib_path, "r", encoding="utf-8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    entries = bib_database.entries
    logger.info(f"Found {len(entries)} entries in the bib file.")

    texts_to_embed = []
    metadata = []

    for entry in entries:
        title = entry.get("title", "").replace("{", "").replace("}", "").strip()
        abstract = entry.get("abstract", "").replace("{", "").replace("}", "").strip()

        if not title and not abstract:
            continue

        text = f"{title}\n{abstract}"
        texts_to_embed.append(text)
        metadata.append({"id": entry.get("ID", ""), "title": title})

    if not texts_to_embed:
        logger.warning("No valid entries with title/abstract found.")
        return

    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        logger.error("EMBEDDING_API_KEY environment variable is not set.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url=(
            os.getenv("EMBEDDING_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )
    emb_model = os.getenv("EMBEDDING_MODEL_NAME") or "text-embedding-v4"

    logger.info(f"Computing embeddings with {emb_model}...")
    try:
        all_embeddings = get_embeddings_in_batches(
            client, texts_to_embed, emb_model, batch_size=10
        )
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        return

    # Save to file
    os.makedirs(os.path.dirname(emb_path) or ".", exist_ok=True)

    zotero_data = {"model": emb_model, "papers": []}

    for meta, emb in zip(metadata, all_embeddings):
        zotero_data["papers"].append(
            {"id": meta["id"], "title": meta["title"], "embedding": emb}
        )

    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(zotero_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully saved {len(all_embeddings)} embeddings to {emb_path}")


if __name__ == "__main__":
    main()
