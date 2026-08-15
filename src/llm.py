import asyncio
import json
import logging
import os
import re
from typing import Any, Dict

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# JSON only allows these escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
# LLMs sometimes produce invalid ones like \_ (from LaTeX habits).
_VALID_JSON_ESCAPES = frozenset('"\\bfnrtu/')


def _sanitize_json_string(raw: str) -> str:
    """Strip markdown code fences and fix invalid JSON escape sequences."""
    s = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        s = match.group(1).strip()

    def _replace_invalid_escape(m: re.Match) -> str:
        char = m.group(1)
        if char in _VALID_JSON_ESCAPES:
            return m.group(0)  # keep valid escapes
        return char  # drop the backslash

    return re.sub(r"\\(.)", _replace_invalid_escape, s)


class PaperSummary(BaseModel):
    topic: str = Field(
        description="从系统提供的子领域列表中，选择一个最适合该论文的主题分类。"
    )
    background_knowledge: str = Field(
        description="脱离论文的具体细节，从更宏观的视角详细介绍该子领域的基础范式、整体概况或核心物理图像，为读者提供足够充分的背景科普与前置知识储备。"
    )
    contribution: str = Field(
        description="清晰、准确、忠实于摘要地陈述论文得出的主要科学发现或核心工作。"
    )


def get_system_prompt(language: str, topics: list[str]) -> str:
    topics_str = ", ".join([f'"{t}"' for t in topics])
    return (
        f"你是一位严谨专业的天文领域学者。请基于论文标题与摘要，结合领域专业知识，输出结构化的 {language} JSON 总结：\n\n"
        f"字段要求：\n"
        f"1. `topic`：从以下候选列表中选择最匹配的一个：[{topics_str}]。若无合适匹配请选 'Others'。\n"
        f"2. `background_knowledge`：跳出论文具体细节，从宏观视角介绍该子领域的基础物理图像、研究范式或背景科普，为读者提供前置知识。\n"
        f"3. `contribution`：忠实于摘要，清晰陈述论文的核心工作与主要科学发现。\n\n"
        f"排版与行文规范（特别是针对中文输出）：\n"
        f"1. 中英文及数字空格：中文字符与西文字符（英文单词、字母）、阿拉伯数字之间必须保留一个半角空格（例如：“利用 JWST 观测”、“在 2026 年”）；中文字符与全角标点符号之间不加空格。\n"
        f"2. 标点符号规范：中文文本中必须使用全角标点符号（，。！？：；“”‘’（）等），严禁在中文语境中混用半角英文标点；引用使用中文全角引号“”，省略号使用……，破折号使用——。\n"
        f"3. 术语与行文：天文学科专业术语应规范、准确、地道，英文缩写（如 AGN、CMB、JWST 等）保持标准大小写；语言精炼流畅，避免机器翻译腔与生硬句式。\n"
        f"4. 格式限制：字段内容请直接输出正文段落或无序列表，不要包含 Markdown 标题（如 `#`、`##`）。"
    )


async def enhance_paper(
    client: AsyncOpenAI,
    paper: dict,
    sem: asyncio.Semaphore,
    model_name: str,
    language: str,
    topics: list[str],
) -> Dict[str, Any]:
    MAX_RETRIES = 5
    extra_kwargs: Dict[str, Any] = {}
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT")
    if reasoning_effort:
        extra_kwargs["reasoning_effort"] = reasoning_effort

    async with sem:
        prompt = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"
        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt == 0:
                    logger.info(f"Processing LLM for {paper['id']}...")
                else:
                    logger.info(
                        f"Processing LLM for {paper['id']} (Retry {attempt}/{MAX_RETRIES})..."
                    )
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": get_system_prompt(language, topics),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    stream=False,
                    **extra_kwargs,
                )
                content = response.choices[0].message.content
                parsed = json.loads(_sanitize_json_string(content))
                result = PaperSummary.model_validate(parsed)

                return {
                    "id": paper["id"],
                    "title": paper["title"],
                    "url": paper["url"],
                    "pdf_url": paper["pdf_url"],
                    "categories": paper["categories"],
                    "topic": result.topic,
                    "background_knowledge": result.background_knowledge,
                    "contribution": result.contribution,
                    "summary": paper["summary"],
                }
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Error processing {paper['id']} on attempt {attempt + 1}: {e}. Retrying..."
                    )
                    # Exponential backoff with ceiling
                    await asyncio.sleep(min(30, (2**attempt) * 2))
                    continue

                logger.error(
                    f"Final error processing {paper['id']} after {MAX_RETRIES} retries: {e}"
                )
                return {
                    "id": paper["id"],
                    "title": paper["title"],
                    "url": paper["url"],
                    "pdf_url": paper["pdf_url"],
                    "categories": paper["categories"],
                    "topic": "Others",
                    "background_knowledge": f"Failed when generating background knowledge. {e}",
                    "contribution": f"Failed when generating contribution. {e}",
                    "summary": paper["summary"],
                }


async def enhance_papers_concurrently(
    client: AsyncOpenAI,
    papers: list[dict],
    model_name: str,
    language: str,
    topics: list[str],
    concurrency: int = 5,
) -> list[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        enhance_paper(client, paper, sem, model_name, language, topics)
        for paper in papers
    ]
    results = await asyncio.gather(*tasks)
    return results


async def generate_daily_topics(
    client: AsyncOpenAI, papers: list[dict], model_name: str
) -> list[str]:
    """Generates a list of topics based on the titles of today's papers."""

    custom_groups = os.getenv("CUSTOM_GROUPS")

    if custom_groups:
        topics = [t.strip() for t in custom_groups.split(",") if t.strip()]
        if "Others" not in topics:
            topics.append("Others")
        logger.info(f"Using custom groups from environment: {topics}")
        return topics

    titles = [f"- {p['title']}" for p in papers]
    titles_str = "\n".join(titles)

    prompt = (
        "Based on the following paper titles from today's arXiv updates, "
        "generate a list of 4 to 8 distinct, broad research topics that best categorize them. "
        "The topics should be broad enough to encompass most papers, but specific enough to be useful "
        "(e.g., 'Exoplanets', 'Galaxy Clusters', etc.). "
        "Also, ALWAYS include exactly one topic named 'Others' for papers that do not fit nicely into the generated groups.\n\n"
        "Return ONLY a valid JSON object with a single key 'topics' mapping to a list of strings. Do not use Markdown formatting.\n\n"
        f"Paper Titles:\n{titles_str}"
    )

    MAX_RETRIES = 5
    extra_kwargs: Dict[str, Any] = {}
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT")
    if reasoning_effort:
        extra_kwargs["reasoning_effort"] = reasoning_effort

    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt == 0:
                logger.info("Generating dynamic daily topics...")
            else:
                logger.info(
                    f"Generating dynamic daily topics (Retry {attempt}/{MAX_RETRIES})..."
                )
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                stream=False,
                **extra_kwargs,
            )
            content = response.choices[0].message.content
            parsed = json.loads(_sanitize_json_string(content))
            topics = parsed.get("topics", ["General Astrophysics", "Others"])
            if "Others" not in topics:
                topics.append("Others")
            logger.info(f"Generated topics: {topics}")
            return topics
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"Failed to generate daily topics on attempt {attempt + 1}: {e}. Retrying..."
                )
                await asyncio.sleep(min(30, (2**attempt) * 2))
                continue

            logger.error(
                f"Failed to generate daily topics after {MAX_RETRIES} retries: {e}"
            )
            return [
                "General Astrophysics",
                "Cosmology",
                "Stars and Exoplanets",
                "Galaxies",
                "Others",
            ]
