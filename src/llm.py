import asyncio
import json
import logging
import os
import re
from typing import Any, Dict

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _sanitize_json_string(raw: str) -> str:
    """Strip markdown code fences and fix unescaped LaTeX/control escapes in JSON strings."""
    s = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        s = match.group(1).strip()

    def _replace_invalid_escape(m: re.Match) -> str:
        rest = s[m.end() : m.end() + 6]
        char = m.group(1)
        # Keep valid standard JSON escapes
        if char in ('"', "\\", "/"):
            return m.group(0)
        # Keep unicode escape \uXXXX
        if char == "u" and re.match(r"^[0-9a-fA-F]{4}", rest):
            return m.group(0)
        # Keep standard newline/carriage-return/tab ONLY when not followed by alpha (e.g. \n-, \n\n, \t" etc.)
        if char in ("n", "r", "t") and (not rest or not rest[0].isalpha()):
            return m.group(0)
        # Otherwise it's a LaTeX command (\text, \theta, \rm, \beta, \frac, etc.) or LaTeX escape (\_, \%, etc.)
        # Escape the backslash so json.loads produces a literal backslash
        return r"\\" + char

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
        f"你是一位严谨专业的天文学者。请基于论文标题与摘要，输出结构化的 {language} JSON 总结：\n\n"
        f"【字段要求】\n"
        f"1. `topic`：从候选列表中选择最匹配的一个：[{topics_str}]（若无合适则选 'Others'）。\n"
        f"2. `background_knowledge`：宏观介绍该子领域的基础物理图像、研究范式或背景科普，提供充分前置知识。\n"
        f"3. `contribution`：忠实于摘要，精炼陈述论文的核心工作与主要科学发现（多要点使用 `- ` 无序列表，适度 `**加粗**` 核心结论，严禁输出 `#` 标题）。\n\n"
        f"【排版与 LaTeX 规范】\n"
        f"1. 空格与标点：中文与英文/数字之间保留一个半角空格；中文文本使用全角标点（，。！？：“”等）。\n"
        f"2. LaTeX 适用范围：仅对含希腊字母、上下标、物理变量及复合表达式（如 $z > 3$、$M_\\odot$、$\\Lambda\\text{{CDM}}$、$10^5\\ M_\\odot$）使用行内公式 $...$。普通整数、年份、简单百分比、常见单位（如 15 K、10 Myr、200 km/h）、简单分子（如 CO、HI）及英文缩写（如 AGN、JWST、3D）一律写为纯文本，严禁套用公式。\n"
        f"3. 公式边界空格（最高优先级）：行内公式与两侧任何字符（含中文、全角标点及全角括号）之间必须保留一个半角空格。例如：`红移 $z > 3$ 的样本`、`（ $\\Delta\\chi^2 \\approx -4$ ）`、`结果为 $\\alpha = 0.5$ ，表明`（严禁出现 `（$z$）` 或 `$z$，`）。\n"
        f"4. 公式语法规范：公式内部首尾紧贴美元符号（如 $z > 3$ 正确）；严禁使用过时的 `\\rm`（统一使用 `\\mathrm{{...}}` 或 `\\text{{...}}`）；命令参数必须用花括号包裹（如 `\\text{{M}}`）；希腊字母必须使用 LaTeX 命令（如 `\\alpha`，禁止裸写 Unicode `$α$`）；严禁在公式外层添加 Markdown 反引号。"
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
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    parsed = parsed[0]
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
