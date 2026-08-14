import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_summary_prompt(language: str = "中文") -> str:
    """Prompt for evaluating model summarization performance."""
    return (
        f"你是一位严谨、专业的天文领域研究者，精通文献阅读和信息提取。\n"
        f"你的任务是基于提供的文献摘要以及你的领域世界知识，输出一段结构化的 {language} 总结。\n"
        f"请务必返回一个合法的 JSON 对象，包含以下两个字段：\n"
        f"1. `background_knowledge`：跳出这篇论文的具体细节，从宏观的视角使用 {language} 详细介绍该子领域的基础物理图像、整体研究概况或核心范式，为读者提供一个宽泛且充实的背景科普与前置知识储备。\n"
        f"2. `contribution`：使用 {language} 清晰、忠实于原摘要地阐述论文的具体核心工作与主要科学发现。\n"
        f"排版与翻译指引：\n"
        f"- 【严禁使用 Markdown 标题】：不要在 `background_knowledge` 或 `contribution` 等字段内部自行添加 `#`, `##`, `###` 等 Markdown 标题（例如 `### 背景知识` 或 `#### 核心发现`）。外部渲染模板已经自带了各个章节的标题，如果你自己加上会导致最终排版出现重复与混乱。只输出纯粹的正文段落或无序列表即可。\n"
        f"- 遇到专业名词时，如果有合适的 {language} 翻译，请使用翻译并在首次出现时用括号标注英文原词；如果没有通用翻译，请直接保持英文原词。\n"
        f"- 【标点与空格规范】：请遵循规范的中文学术排版美学。必须使用标准全角中文标点（例如使用中文双引号 “ ” 而非英文引号或单引号）；在 {language} 字符与英文字母、数字之间自然地保留一个半角空格（例如 `利用 DESI-DR1 巡天样本` 而非 `利用DESI-DR1巡天样本`）。\n"
        f"- 如果输出内容包含多个要点或逻辑层次，请优先采用分段或无序列表（Bullet points）进行组织，避免将所有内容挤在臃肿的一整段中。\n"
        f"- 在表示数学乘号时，请严格使用符号 `×` 或字母 `x`，绝对不要使用星号 `*`，以免引起 Markdown 语法解析为斜体。\n\n"
        f"请输出格式如下的 JSON：\n"
        "{\n"
        '  "background_knowledge": "...",\n'
        '  "contribution": "..."\n'
        "}"
    )


async def call_model_summary(
    client: AsyncOpenAI,
    model_name: str,
    paper: Dict[str, Any],
    extra_kwargs: Dict[str, Any],
    language: str = "中文",
) -> Dict[str, Any]:
    prompt = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('summary', '')}"
    system_prompt = get_summary_prompt(language)

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
        content = response.choices[0].message.content
        parsed = json.loads(content)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return {
            "success": True,
            "elapsed_seconds": round(elapsed, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "background_knowledge": parsed.get("background_knowledge", ""),
            "contribution": parsed.get("contribution", ""),
            "raw_response": content,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error invoking {model_name} for {paper.get('id')}: {e}")
        return {
            "success": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }


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

    emb_api_key = os.getenv("EMBEDDING_API_KEY")
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


def save_markdown_report(
    results: List[Dict[str, Any]], output_md_path: str, date_str: str
):
    lines = []
    lines.append(f"# Model Comparison Report - {date_str}\n")
    lines.append(
        "> 本次测试针对 Top 文章，对比各主流大模型（DeepSeek、Gemini、Qwen 等）生成文献背景科普与核心贡献的效果。\n"
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

            lines.append(
                f"- **耗时**: {res['elapsed_seconds']}s | "
                f"**Tokens**: Input {res['prompt_tokens']}, Output {res['completion_tokens']}\n"
            )
            lines.append("#### 🔭 领域背景与前置科普 (`background_knowledge`)")
            lines.append(f"{res['background_knowledge']}\n")
            lines.append("#### 🎯 核心工作与科学发现 (`contribution`)")
            lines.append(f"{res['contribution']}\n")
            lines.append("---\n")

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Comparison report saved to {output_md_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepSeek Pro/Flash, Gemini Flash/Flash-Lite, and Qwen Plus on arXiv papers"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string (YYYY-MM-DD), defaults to today UTC",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top papers to test (default: 1)",
    )
    args = parser.parse_args()

    load_dotenv()

    today_str = (
        args.date if args.date else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    dist_dir = Path(__file__).resolve().parent.parent / "dist" / "data"
    fetched_jsonl = dist_dir / f"{today_str}_fetched.jsonl"

    papers: List[Dict[str, Any]] = []
    if fetched_jsonl.exists():
        logger.info(f"Reading already fetched papers from {fetched_jsonl}")
        with open(fetched_jsonl, "r", encoding="utf-8") as f:
            papers = [json.loads(line) for line in f]
    else:
        categories = (
            os.getenv("CATEGORIES") or "astro-ph.GA, astro-ph.CO, astro-ph.IM"
        ).split(",")
        categories = [c.strip() for c in categories]
        logger.info(
            f"Fetching arXiv papers for date {today_str} and categories {categories}..."
        )
        fetched_raw = (
            fetch_papers_for_date(categories, today_str)
            if args.date
            else fetch_papers(categories)
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
                    "updated_date": p.updated.date().isoformat() if p.updated else None,
                }
            )

    if not papers:
        logger.error("No papers found to evaluate.")
        return

    logger.info(f"Loaded {len(papers)} papers in total.")

    # Select Top N papers via Zotero kNN scoring
    zotero_emb_path = Path(__file__).resolve().parent.parent / (
        os.getenv("ZOTERO_EMB_PATH") or "zotero/zotero_embeddings.json"
    )
    zotero_embs = load_zotero_embs(str(zotero_emb_path))
    top_papers = get_top_papers(papers, zotero_embs, top_n=args.top_n)

    logger.info(f"Top {len(top_papers)} paper(s) selected:")
    for idx, p in enumerate(top_papers, 1):
        score = p.get("similarity_score", 0.0)
        logger.info(f"  {idx}. [{p['id']}] (score: {score:.4f}) {p['title'][:60]}...")

    # API Keys & Endpoints
    gemini_api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or (
            os.getenv("OPENAI_API_KEY")
            if "googleapis" in (os.getenv("OPENAI_BASE_URL") or "")
            else None
        )
    )
    gemini_base_url = os.getenv("GEMINI_BASE_URL") or (
        os.getenv("OPENAI_BASE_URL")
        if "googleapis" in (os.getenv("OPENAI_BASE_URL") or "")
        else "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    ds_api_key = os.getenv("DEEPSEEK_API_KEY") or (
        os.getenv("OPENAI_API_KEY")
        if "deepseek" in (os.getenv("OPENAI_BASE_URL") or "")
        else None
    )
    ds_base_url = (
        os.getenv("DEEPSEEK_BASE_URL")
        or (
            os.getenv("OPENAI_BASE_URL")
            if "deepseek" in (os.getenv("OPENAI_BASE_URL") or "")
            else "https://api.deepseek.com"
        )
    ).rstrip("/")

    qwen_api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
    )
    qwen_base_url = (
        os.getenv("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    models_to_test = []

    # 1. Gemini Models
    if gemini_api_key:
        gemini_client = AsyncOpenAI(api_key=gemini_api_key, base_url=gemini_base_url)
        models_to_test.append(
            {
                "key": "gemini-3.5-flash-lite",
                "model_name": "gemini-3.5-flash-lite",
                "client": gemini_client,
                "extra_kwargs": {},
            }
        )
        models_to_test.append(
            {
                "key": "gemini-3.7-flash",
                "model_name": "gemini-3.7-flash",
                "client": gemini_client,
                "extra_kwargs": {},
            }
        )

    # 2. DeepSeek Models
    if ds_api_key:
        ds_client = AsyncOpenAI(api_key=ds_api_key, base_url=ds_base_url)
        models_to_test.append(
            {
                "key": "deepseek-v4-flash",
                "model_name": "deepseek-v4-flash",
                "client": ds_client,
                "extra_kwargs": {
                    "reasoning_effort": os.getenv("LLM_REASONING_EFFORT") or "max",
                },
            }
        )
        models_to_test.append(
            {
                "key": "deepseek-v4-pro",
                "model_name": "deepseek-v4-pro",
                "client": ds_client,
                "extra_kwargs": {
                    "reasoning_effort": os.getenv("LLM_REASONING_EFFORT") or "max",
                },
            }
        )

    # 3. Qwen Model
    if qwen_api_key:
        models_to_test.append(
            {
                "key": "qwen-plus",
                "model_name": "qwen-plus",
                "client": AsyncOpenAI(api_key=qwen_api_key, base_url=qwen_base_url),
                "extra_kwargs": {},
            }
        )

    if not models_to_test:
        logger.error(
            "No models configured. Please provide GEMINI_API_KEY, OPENAI_API_KEY, or EMBEDDING_API_KEY in .env."
        )
        return

    eval_results = []
    for p_idx, paper in enumerate(top_papers, 1):
        logger.info(f"\n--- Testing Paper {p_idx}/{len(top_papers)}: {paper['id']} ---")
        paper_res = {"paper": paper, "models": {}}

        for m in models_to_test:
            logger.info(f"Calling model: {m['key']}...")
            res = await call_model_summary(
                client=m["client"],
                model_name=m["model_name"],
                paper=paper,
                extra_kwargs=m["extra_kwargs"],
            )
            paper_res["models"][m["key"]] = res
            if res.get("success"):
                logger.info(
                    f"  -> {m['key']} finished in {res['elapsed_seconds']}s (tokens: {res['prompt_tokens']} in / {res['completion_tokens']} out)"
                )
            else:
                logger.error(f"  -> {m['key']} failed: {res.get('error')}")

        eval_results.append(paper_res)

    output_dir = Path(__file__).resolve().parent.parent / "dist" / "eval"
    output_report_path = output_dir / f"model_comparison_{today_str}.md"
    save_markdown_report(eval_results, str(output_report_path), today_str)
    print(f"\n✅ All tests complete! View the full report at:\n{output_report_path}")


if __name__ == "__main__":
    asyncio.run(main())
