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
        f"你是一位严谨专业的天文领域学者。请基于论文标题与摘要，结合领域专业知识，输出结构化的 {language} JSON 总结：\n\n"
        f"字段要求：\n"
        f"1. `topic`：从以下候选列表中选择最匹配的一个：[{topics_str}]。若无合适匹配请选 'Others'。\n"
        f"2. `background_knowledge`：跳出论文具体细节，从宏观视角介绍该子领域的基础物理图像、研究范式或背景科普，为读者提供前置知识。\n"
        f"3. `contribution`：忠实于摘要，清晰陈述论文的核心工作与主要科学发现。\n\n"
        f"排版、Markdown 与行文规范（特别是针对中文输出）：\n"
        f"1. 中英文及数字空格：中文字符与西文字符（英文单词、字母）、阿拉伯数字之间必须保留一个半角空格（例如：“利用 JWST 观测”、“在 2026 年”）；中文字符与全角标点符号之间不加空格，全角括号内部两端不留多余空格。\n"
        f"2. 标点符号规范：中文文本中必须使用全角标点符号（，。！？：；“”‘’（）等），严禁在中文语境中混用半角英文标点；引用使用中文全角引号“”，省略号使用……，破折号使用——。\n"
        f"3. 数学公式与物理量（LaTeX 规范）：\n"
        f"   - 适用范围：仅对含有希腊字母、数学符号、上下标、物理变量及复合表达式（如红移 $z$、恒星质量 $M_\\odot$、$\\Lambda\\text{{CDM}}$ 宇宙学模型、谱指数 $n_s$、$10^5\\ M_\\odot$ 等）使用标准的行内公式 $...$；\n"
        f"   - 严禁滥用：普通的纯数字、年份、样本数量、简单百分比及纯英文缩写（如 31 个源、2026 年、30%、AGN、JWST、3D 等）直接写为纯文本，严禁套用 LaTeX 公式（严禁写成 $31$、$30\\%$、$\\text{{AGN}}$）；\n"
        f"   - 排版与边界：行内公式前后与中文字符或全角标点之间必须保留一个半角空格（例如“红移 $z > 3$ 的样本”）；公式内部首尾紧贴美元符号（如 $z > 3$ 正确，$ z > 3 $ 错误）；严禁在公式外层添加 Markdown 反引号；\n"
        f"4. Markdown 结构与强调：\n"
        f"   - 严禁在字段内容内部输出任何 Markdown 标题（如 `#`、`##`、`###` 等）。\n"
        f"   - 若需列举多个要点，请使用标准的 `- ` 无序列表。\n"
        f"   - 可对关键物理量、核心结论适度使用 `**加粗**` 强调，避免整段大面积加粗。\n"
        f"5. 术语与行文：天文学科专业术语应规范、准确、地道，英文缩写（如 AGN、CMB、JWST 等）保持标准大小写；语言精炼流畅，避免机器翻译腔与生硬句式。"
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
