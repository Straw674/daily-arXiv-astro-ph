import asyncio
import logging
import re
from typing import Dict, Any
from pydantic import BaseModel, Field
import json
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# JSON only allows these escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
# LLMs sometimes produce invalid ones like \_ (from LaTeX habits).
_VALID_JSON_ESCAPES = frozenset('"\\bfnrtu/')


def _sanitize_json_string(raw: str) -> str:
    """Fix invalid JSON escape sequences by removing the backslash."""

    def _replace_invalid_escape(m: re.Match) -> str:
        char = m.group(1)
        if char in _VALID_JSON_ESCAPES:
            return m.group(0)  # keep valid escapes
        return char  # drop the backslash

    return re.sub(r"\\(.)", _replace_invalid_escape, raw)


class PaperSummary(BaseModel):
    topic: str = Field(
        description="从系统提供的子领域列表中，选择一个最适合该论文的主题分类。"
    )
    toc_summary: str = Field(
        description="Use a very short, simple, and highly readable English phrase (under 10 words) to act as a Table of Contents entry. It should immediately convey the core topic without complex academic jargon. This must be in English regardless of the target language."
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
        f"你是一位严谨、专业的天文领域研究者，精通文献阅读和信息提取。\n"
        f"你的任务是基于提供的文献摘要以及你的领域世界知识，输出一段结构化的 {language} 总结。\n"
        f"请务必返回一个合法的 JSON 对象，包含以下四个字段：\n"
        f"1. `topic`: 必须从以下预定义的列表中选择一个最适合该论文的主题：[{topics_str}]。如果确实不属于任何一个，选择 'Others'（如果列表中有的话，或者自己选择最接近的一个，但强烈建议遵循列表）。注意：为了避免碎片化，只要有一点相关性，请尽量归入提供的主题之一。\n"
        f"2. `toc_summary`: Use a very short, simple, and highly readable English phrase (under 10 words) to act as a Table of Contents entry. It should immediately convey the core topic without complex academic jargon. This MUST be in English regardless of the {language} variable.\n"
        f"3. `background_knowledge`：跳出这篇论文的具体细节，从宏观的视角使用 {language} 详细介绍该子领域的基础物理图像、整体研究概况或核心范式，为读者提供一个宽泛且充实的背景科普与前置知识储备。\n"
        f"4. `contribution`：使用 {language} 清晰、忠实于原摘要地阐述论文的具体核心工作与主要科学发现。\n"
        f"排版与翻译指引：\n"
        f"- 【严禁使用 Markdown 标题】：不要在 `background_knowledge` 或 `contribution` 等字段内部自行添加 `#`, `##`, `###` 等 Markdown 标题（例如 `### 背景知识` 或 `#### 核心发现`）。外部渲染模板已经自带了各个章节的标题，如果你自己加上会导致最终排版出现重复与混乱。只输出纯粹的正文段落或无序列表即可。\n"
        f"- 遇到专业名词时，如果有合适的 {language} 翻译，请使用翻译并在首次出现时用括号标注英文原词；如果没有通用翻译，请直接保持英文原词。\n"
        f"- 请遵循通用的排版美学，在 {language} 字符与英文字母、数字之间自然地保留一个半角空格。\n"
        f"- 如果输出内容包含多个要点或逻辑层次，请优先采用分段或无序列表（Bullet points）进行组织，避免将所有内容挤在臃肿的一整段中。\n"
        f"- 在表示数学乘号时，请严格使用符号 `×` 或字母 `x`，绝对不要使用星号 `*`，以免引起 Markdown 语法解析为斜体。\n\n"
        f"请输出格式如下的 JSON：\n"
        "{\n"
        '  "topic": "...",\n'
        '  "toc_summary": "...",\n'
        '  "background_knowledge": "...",\n'
        '  "contribution": "..."\n'
        "}"
    )


async def enhance_paper(
    client: AsyncOpenAI,
    paper: dict,
    sem: asyncio.Semaphore,
    model_name: str,
    language: str,
    topics: list[str],
) -> Dict[str, Any]:
    async with sem:
        prompt = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"
        try:
            logger.info(f"Processing LLM for {paper['id']}...")
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": get_system_prompt(language, topics)},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
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
                "toc_summary": result.toc_summary,
                "background_knowledge": result.background_knowledge,
                "contribution": result.contribution,
                "summary": paper["summary"],
            }
        except Exception as e:
            logger.error(f"Error processing {paper['id']}: {e}")
            return {
                "id": paper["id"],
                "title": paper["title"],
                "url": paper["url"],
                "pdf_url": paper["pdf_url"],
                "categories": paper["categories"],
                "topic": "Others",
                "toc_summary": "Failed to generate summary.",
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

    try:
        logger.info("Generating dynamic daily topics...")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        content = response.choices[0].message.content
        parsed = json.loads(_sanitize_json_string(content))
        topics = parsed.get("topics", ["General Astrophysics", "Others"])
        if "Others" not in topics:
            topics.append("Others")
        logger.info(f"Generated topics: {topics}")
        return topics
    except Exception as e:
        logger.error(f"Failed to generate daily topics: {e}")
        return [
            "General Astrophysics",
            "Cosmology",
            "Stars and Exoplanets",
            "Galaxies",
            "Others",
        ]
