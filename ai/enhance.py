import os
import json
import sys

import dotenv
import argparse

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists(".env"):
    dotenv.load_dotenv()
# template = open("template.txt", "r").read()
# system = open("system.txt", "r").read()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    language = os.environ.get("LANGUAGE", "Chinese")

    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_data.append(item)

    data = unique_data

    # llm = ChatOpenAI(model=model_name).with_structured_output(
    #     Structure, method="function_calling"
    # )
    # print("Connect to:", model_name, file=sys.stderr)
    # prompt_template = ChatPromptTemplate.from_messages([
    #     SystemMessagePromptTemplate.from_template(system),
    #     HumanMessagePromptTemplate.from_template(template=template),
    # ])

    # print(prompt_template)

    # chain = prompt_template | llm

    # for idx, d in enumerate(data):
    #     try:
    #         response: Structure = chain.invoke({
    #             "content": d["summary"],
    #         })
    #         d["AI"] = response.model_dump()
    #     except Exception as e:
    #         print(f"{d['id']} has an error: {e}", file=sys.stderr)
    #         d["AI"] = {
    #             "tldr": f"{e}",
    #             "background": "Error",
    #             "data": "Error",
    #             "method": "Error",
    #             "result": "Error",
    #         }
    #     with open(
    #         args.data.replace(".jsonl", f"_AI_enhanced_{language}.jsonl"), "a"
    #     ) as f:
    #         f.write(json.dumps(d) + "\n")

    #     print(f"Finished {idx + 1}/{len(data)}", file=sys.stderr)

    llm = ChatOpenAI(model=model_name)
    print("Connect to:", model_name, file=sys.stderr)

    fields = {
        "tldr": "请用一句话高度概括这篇论文的核心贡献，明确指出研究对象和主要发现。",
        "background": "请阐述该研究的宏观科学背景或者科学主题。你的回答会将作为四个并列部分（background、data、method 或者 result）中的一个，所以不要输出和其他三个部分相关的信息。",
        "data": "请详细说明研究中使用的（观测或者 simulation）数据来源。你的回答会将作为四个并列部分（background、data、method 或者 result）中的一个，所以不要输出和其他三个部分相关的信息。",
        "method": "请描述研究采用的核心分析方法、物理模型或技术手段。你的回答会将作为四个并列部分（background、data、method 或者 result）中的一个，所以不要输出和其他三个部分相关的信息。",
        "result": "请清晰、准确地陈述论文得出的主要科学发现或核心结论。你的回答会将作为四个并列部分（background、data、method 或者 result）中的一个，所以不要输出和其他三个部分相关的信息。",
    }

    system = "你是一位天文领域的研究者。请基于摘要对以下问题给出简短、凝练的一句话回答（不要使用换行或者列表）。使用中文回答。只返回答案内容，不要包含其他说明。不要使用 markdown 的数学公式语法。在回答中省略主语，也就是不要用“该研究”“该论文”开头。"

    for idx, d in enumerate(data):
        try:
            ai_result = {}

            for field_name, field_prompt in fields.items():
                prompt = (
                    system
                    + f"""

{field_prompt}

摘要内容：
{d["summary"]}"""
                )

                response = llm.invoke(prompt)
                ai_result[field_name] = response.content.strip()

            d["AI"] = ai_result

        except Exception as e:
            print(f"{d['id']} has an error: {e}", file=sys.stderr)
            d["AI"] = {
                "tldr": f"{e}",
                "background": "Error",
                "data": "Error",
                "method": "Error",
                "result": "Error",
            }

        with open(
            args.data.replace(".jsonl", f"_AI_enhanced_{language}.jsonl"), "a"
        ) as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"Finished {idx + 1}/{len(data)}", file=sys.stderr)


if __name__ == "__main__":
    main()
