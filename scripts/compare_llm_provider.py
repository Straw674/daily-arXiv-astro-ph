import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# Add src to python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedding import compute_knn_scores, get_embeddings_in_batches
from fetcher import fetch_papers, fetch_papers_for_date
from llm import _sanitize_json_string, get_system_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Top-Level Constants & Evaluation Configuration
# ==============================================================================

DATE: str | None = None  # None for today UTC, or specific date "YYYY-MM-DD"
TOP_N: int = 1
LANGUAGE: str = "中文"
CATEGORIES: List[str] = ["astro-ph.GA", "astro-ph.CO", "astro-ph.IM"]
ZOTERO_EMB_PATH: str = "zotero/zotero_embeddings.json"
DEFAULT_TOPICS: List[str] = [
    "General Astrophysics",
    "Cosmology",
    "Stars and Exoplanets",
    "Galaxies",
    "Others",
]

USD_TO_CNY_RATE: float = 6.74

# Centralized Model Configurations and Pricing (USD per 1M tokens)
# Note: For Chinese domestic models priced natively in CNY (e.g. Qwen),
# we divide by USD_TO_CNY_RATE so that cost_cny is exact and consistent.
MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "gemini-3.5-flash-lite": {
        "model_name": "gemini-3.5-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"),
        "pricing": {"input": 0.30, "output": 2.50},  # $0.30 in / $2.50 out per 1M
        "extra_kwargs": {"reasoning_effort": "high"},
        "enabled": True,
    },
    "gemini-3.7-flash": {
        "model_name": "gemini-3.7-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"),
        "pricing": {
            "input": 0.75,
            "output": 3.75,
        },  # $0.75 in / $3.75 out per 1M (promo rate)
        "extra_kwargs": {"reasoning_effort": "high"},
        "enabled": True,
    },
    "deepseek-v4-flash": {
        "model_name": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": ("DEEPSEEK_API_KEY",),
        # Valley / Off-peak (Cache Miss): $0.22 in / $0.66 out per 1M tokens
        # Peak: $0.44 in / $1.32 out per 1M tokens
        # Off-peak Cache Hit: $0.007 in per 1M tokens
        # Abstract summarization prompt has >80% unique content, so Cache Miss pricing is used.
        "pricing": {"input": 0.22, "output": 0.66},
        "extra_kwargs": {},
        "enabled": True,
    },
    "deepseek-v4-pro": {
        "model_name": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key_env": ("DEEPSEEK_API_KEY",),
        # Valley / Off-peak (Cache Miss): $0.66 in / $1.98 out per 1M tokens
        # Peak: $1.32 in / $3.96 out per 1M tokens
        # Off-peak Cache Hit: $0.022 in per 1M tokens
        "pricing": {"input": 0.66, "output": 1.98},
        "extra_kwargs": {},
        "enabled": True,
    },
    "qwen-plus": {
        "model_name": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "EMBEDDING_API_KEY"),
        # Native CNY: ¥0.80 in / ¥2.00 out per 1M tokens
        "pricing": {
            "input": 0.80 / USD_TO_CNY_RATE,
            "output": 2.00 / USD_TO_CNY_RATE,
        },
        "extra_kwargs": {},
        "enabled": True,
    },
    "qwen3.6-27b": {
        "model_name": "qwen3.6-27b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "EMBEDDING_API_KEY"),
        # Native CNY: ¥3.00 in / ¥18.00 out per 1M tokens
        "pricing": {
            "input": 3.00 / USD_TO_CNY_RATE,
            "output": 18.00 / USD_TO_CNY_RATE,
        },
        "extra_kwargs": {"extra_body": {"enable_thinking": True}},
        "enabled": True,
    },
}


# ==============================================================================
# Pure Utility & Calculation Functions
# ==============================================================================


def calculate_usage_and_cost(
    usage: Any, pricing: Dict[str, float], usd_to_cny: float = USD_TO_CNY_RATE
) -> Dict[str, Any]:
    """Pure function to compute token breakdowns and estimated costs."""
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = (
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        if usage
        else 0
    )

    # Thinking token handling across different providers:
    # 1. Gemini OpenAI-compatible API: total_tokens includes thinking tokens, but completion_tokens does not.
    # 2. OpenAI / DeepSeek: completion_tokens often already includes reasoning_tokens.
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0
    )
    unaccounted_tokens = max(0, total_tokens - prompt_tokens - completion_tokens)
    thinking_tokens = max(reasoning_tokens, unaccounted_tokens)

    # Billed output tokens = completion_tokens + any extra thinking tokens not counted in completion_tokens
    billed_output_tokens = completion_tokens + unaccounted_tokens

    cost_usd = (prompt_tokens * pricing.get("input", 0.0) / 1e6) + (
        billed_output_tokens * pricing.get("output", 0.0) / 1e6
    )
    cost_cny = cost_usd * usd_to_cny

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "billed_output_tokens": billed_output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_cny": round(cost_cny, 4),
    }


def create_model_client(model_config: Dict[str, Any]) -> AsyncOpenAI | None:
    """Instantiate an AsyncOpenAI client based on configured environment variables."""
    api_key_env = model_config.get("api_key_env")
    if isinstance(api_key_env, (list, tuple)):
        api_key = next((os.getenv(k) for k in api_key_env if os.getenv(k)), None)
    elif isinstance(api_key_env, str):
        api_key = os.getenv(api_key_env)
    else:
        api_key = None

    if not api_key:
        logger.warning(
            f"API key not found for {model_config['model_name']} (checked {api_key_env}). Skipping."
        )
        return None

    return AsyncOpenAI(api_key=api_key, base_url=model_config.get("base_url"))


async def call_model_summary(
    client: AsyncOpenAI,
    model_config: Dict[str, Any],
    paper: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, Any]:
    """Calls a single model with summarization prompt and returns parsed results."""
    model_name = model_config["model_name"]
    extra_kwargs = model_config.get("extra_kwargs", {})
    pricing = model_config.get("pricing", {"input": 0.0, "output": 0.0})

    prompt = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('summary', '')}"
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            stream=False,
            **extra_kwargs,
        )
        elapsed = time.time() - start_time
        content = response.choices[0].message.content or ""
        parsed = json.loads(_sanitize_json_string(content))

        cost_info = calculate_usage_and_cost(response.usage, pricing)

        return {
            "success": True,
            "elapsed_seconds": round(elapsed, 2),
            **cost_info,
            "topic": parsed.get("topic", ""),
            "background_knowledge": parsed.get("background_knowledge", ""),
            "contribution": parsed.get("contribution", ""),
            "raw_response": content,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error invoking {model_name} for paper {paper.get('id')}: {e}")
        return {
            "success": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }


# ==============================================================================
# Data Loading & Zotero Scoring
# ==============================================================================


def load_zotero_embs(zotero_path: str) -> List[List[float]]:
    if not os.path.exists(zotero_path):
        logger.warning(f"Zotero embeddings not found at {zotero_path}")
        return []
    with open(zotero_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [p["embedding"] for p in data.get("papers", []) if "embedding" in p]


def get_top_papers(
    papers: List[Dict[str, Any]],
    zotero_embs: List[List[float]],
    top_n: int = 1,
) -> List[Dict[str, Any]]:
    if not zotero_embs:
        logger.warning("No Zotero embeddings found, taking the first N papers.")
        return papers[:top_n]

    emb_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    emb_base_url = (
        os.getenv("EMBEDDING_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    emb_model = os.getenv("EMBEDDING_MODEL_NAME") or "text-embedding-v4"
    knn_top_k = int(os.getenv("KNN_TOP_K") or 10)

    logger.info(f"Computing embeddings for {len(papers)} papers using {emb_model}...")
    client = OpenAI(api_key=emb_api_key, base_url=emb_base_url)
    texts = [f"{p.get('title', '')}\n{p.get('summary', '')}" for p in papers]

    paper_embs = get_embeddings_in_batches(client, texts, emb_model, batch_size=10)
    scores = compute_knn_scores(paper_embs, zotero_embs, top_k=knn_top_k)

    for p, score in zip(papers, scores):
        p["similarity_score"] = score

    sorted_papers = sorted(
        papers, key=lambda x: x.get("similarity_score", 0.0), reverse=True
    )
    return sorted_papers[:top_n]


# ==============================================================================
# Reporting
# ==============================================================================


def save_markdown_report(
    results: List[Dict[str, Any]], output_md_path: str, date_str: str
):
    lines = []
    lines.append(f"# Model Comparison Report - {date_str}\n")
    lines.append(
        "> 本次测试针对 Top 文章，对比各主流大模型在论文总结（分类、前置科普与核心贡献）任务上的生成效果、耗时与成本。\n"
    )

    for idx, item in enumerate(results, 1):
        paper = item["paper"]
        lines.append(f"## Paper {idx}: {paper.get('title')}\n")
        lines.append(f"- **arXiv ID**: [{paper.get('id')}]({paper.get('url')})")
        lines.append(f"- **Categories**: {', '.join(paper.get('categories', []))}")
        lines.append(
            f"- **Zotero Similarity Score**: `{paper.get('similarity_score', 0.0):.4f}`\n"
        )
        lines.append(
            f"<details><summary><b>查看原摘要 (Abstract)</b></summary>\n\n{paper.get('summary')}\n\n</details>\n"
        )

        for model_key, res in item["models"].items():
            lines.append(f"### 🤖 模型：`{model_key}`")
            if not res.get("success"):
                lines.append(f"❌ **执行失败**: `{res.get('error')}`\n")
                continue

            cost_str = ""
            if "cost_usd" in res:
                cost_str = f" | **Cost**: ${res['cost_usd']:.6f} (¥{res.get('cost_cny', 0.0):.4f})"

            token_details = (
                f"Input {res['prompt_tokens']}, Output {res['completion_tokens']}"
            )
            if res.get("thinking_tokens", 0) > 0:
                token_details += f" (+ {res['thinking_tokens']} thinking tokens, Total Output Billed {res.get('billed_output_tokens')})"

            lines.append(
                f"- **耗时**: {res['elapsed_seconds']}s | "
                f"**Tokens**: {token_details}{cost_str}\n"
            )
            if res.get("topic"):
                lines.append(f"#### 🏷️ 主题分类 (`topic`)\n{res['topic']}\n")
            lines.append("#### 🔭 领域背景与前置科普 (`background_knowledge`)")
            lines.append(f"{res['background_knowledge']}\n")
            lines.append("#### 🎯 核心工作与科学发现 (`contribution`)")
            lines.append(f"{res['contribution']}\n")
            lines.append("---\n")

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Comparison report saved to {output_md_path}")


# ==============================================================================
# Main Orchestration
# ==============================================================================


async def evaluate_paper_across_models(
    paper: Dict[str, Any],
    active_models: List[tuple[str, Dict[str, Any], AsyncOpenAI]],
    system_prompt: str,
) -> Dict[str, Any]:
    """Issues concurrent summary requests across all active models for a single paper."""
    logger.info(
        f"Evaluating paper {paper.get('id')} across {len(active_models)} models concurrently..."
    )
    tasks = [
        call_model_summary(client, config, paper, system_prompt)
        for _, config, client in active_models
    ]
    responses = await asyncio.gather(*tasks)

    results: Dict[str, Any] = {}
    for (key, _, _), res in zip(active_models, responses):
        results[key] = res
        if res.get("success"):
            logger.info(
                f"  -> [{key}] finished in {res['elapsed_seconds']}s "
                f"(tokens: {res['prompt_tokens']} in / {res['completion_tokens']} out, "
                f"cost: ${res['cost_usd']:.6f} / ¥{res['cost_cny']:.4f})"
            )
        else:
            logger.error(f"  -> [{key}] failed: {res.get('error')}")

    return {"paper": paper, "models": results}


async def main():
    load_dotenv()

    today_str = DATE if DATE else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dist_dir = Path(__file__).resolve().parent.parent / "dist" / "data"
    fetched_jsonl = dist_dir / f"{today_str}_fetched.jsonl"

    papers: List[Dict[str, Any]] = []
    if fetched_jsonl.exists():
        logger.info(f"Reading already fetched papers from {fetched_jsonl}")
        with open(fetched_jsonl, "r", encoding="utf-8") as f:
            papers = [json.loads(line) for line in f]
    else:
        logger.info(
            f"Fetching arXiv papers for date {today_str} and categories {CATEGORIES}..."
        )
        fetched_raw = (
            fetch_papers_for_date(CATEGORIES, today_str)
            if DATE
            else fetch_papers(CATEGORIES)
        )
        for p in fetched_raw:
            papers.append(
                {
                    "id": p.get_short_id(),
                    "title": p.title,
                    "summary": p.summary,
                    "url": p.entry_id,
                    "pdf_url": p.pdf_url,
                    "categories": p.categories,
                    "primary_category": getattr(p, "primary_category", None),
                    "updated_date": (
                        p.updated.date().isoformat() if p.updated else None
                    ),
                }
            )

    if not papers:
        logger.error("No papers found to evaluate.")
        return

    logger.info(f"Loaded {len(papers)} papers in total.")

    # Select Top N papers via Zotero kNN scoring
    zotero_emb_path = Path(__file__).resolve().parent.parent / ZOTERO_EMB_PATH
    zotero_embs = load_zotero_embs(str(zotero_emb_path))
    top_papers = get_top_papers(papers, zotero_embs, top_n=TOP_N)

    logger.info(f"Top {len(top_papers)} paper(s) selected:")
    for idx, p in enumerate(top_papers, 1):
        score = p.get("similarity_score", 0.0)
        logger.info(f"  {idx}. [{p['id']}] (score: {score:.4f}) {p['title'][:60]}...")

    # Initialize active models
    active_models: List[tuple[str, Dict[str, Any], AsyncOpenAI]] = []
    for key, config in MODELS_CONFIG.items():
        if not config.get("enabled", True):
            continue
        client = create_model_client(config)
        if client:
            active_models.append((key, config, client))

    if not active_models:
        logger.error("No active model clients could be initialized.")
        return

    system_prompt = get_system_prompt(LANGUAGE, DEFAULT_TOPICS)

    eval_results = []
    for p_idx, paper in enumerate(top_papers, 1):
        logger.info(f"\n--- Testing Paper {p_idx}/{len(top_papers)}: {paper['id']} ---")
        paper_res = await evaluate_paper_across_models(
            paper, active_models, system_prompt
        )
        eval_results.append(paper_res)

    output_dir = Path(__file__).resolve().parent.parent / "dist" / "eval"
    output_report_path = output_dir / f"model_comparison_{today_str}.md"
    save_markdown_report(eval_results, str(output_report_path), today_str)
    print(
        f"\n✅ All comparisons complete! View the full report at:\n{output_report_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
