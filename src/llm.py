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
        f"【排版要求】\n"
        f"请遵循良好的排版规范，数学公式和专业符号使用标准 LaTeX 格式（行内公式使用 $...$）。"
    )


def create_llm_client(model_name: str | None = None) -> tuple[AsyncOpenAI, str]:
    """Create an AsyncOpenAI client configured for the given model_name or environment settings.

    Automatically resolves base_url and API key based on the model family:
      - Qwen / DashScope (e.g. qwen3.8-flash, qwen-plus) -> DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL
      - Google Gemini (e.g. gemini-3.5-flash-lite)        -> GEMINI_API_KEY / GEMINI_BASE_URL
      - DeepSeek (e.g. deepseek-v4-flash)                -> DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL
      - OpenAI / Generic                                 -> OPENAI_API_KEY / OPENAI_BASE_URL
    """
    model = model_name or os.getenv("MODEL_NAME") or "qwen3.8-flash"
    model_lower = model.lower()

    # Generic or custom override if explicitly provided
    custom_base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    custom_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    if custom_base_url and custom_api_key:
        logger.info(
            f"Using custom LLM configuration: base_url={custom_base_url}, model={model}"
        )
        return AsyncOpenAI(api_key=custom_api_key, base_url=custom_base_url), model

    if model_lower.startswith("qwen") or "dashscope" in model_lower:
        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        base_url = (
            os.getenv("DASHSCOPE_BASE_URL")
            or os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        key_source = "DASHSCOPE_API_KEY"
    elif model_lower.startswith("gemini"):
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        base_url = (
            os.getenv("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        key_source = "GEMINI_API_KEY"
    elif model_lower.startswith("deepseek"):
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        key_source = "DEEPSEEK_API_KEY"
    else:
        # OpenAI or general fallback
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
        )
        base_url = os.getenv("OPENAI_BASE_URL") or None
        key_source = "OPENAI_API_KEY"

    if not api_key:
        raise ValueError(
            f"Missing API key for model '{model}'. "
            f"Please set {key_source} (or OPENAI_API_KEY) in your environment/.env file."
        )

    logger.info(
        f"Initialized LLM client for model '{model}' with endpoint '{base_url or 'default'}'."
    )
    return AsyncOpenAI(api_key=api_key, base_url=base_url), model


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
