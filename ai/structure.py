from pydantic import BaseModel, Field


class Structure(BaseModel):
    """
    对一篇天文学论文的标题和摘要进行结构化总结。
    所有内容都必须基于原文，并使用严谨、客观的语言。
    """
    tldr: str = Field(
        description="对论文核心贡献的一句话高度概括，明确指出研究对象和主要发现。"
    )
    background: str = Field(
        description="阐述该研究的宏观科学背景、所要解决的具体科学问题或旨在验证的科学假设。"
    )
    data: str = Field(
        description="详细说明研究中使用的（观测或者 simulation）数据来源。"
    )
    method: str = Field( # 建议使用复数 methods
        description="描述研究采用的核心分析方法、物理模型或技术手段。"
    )
    result: str = Field( # 建议使用复数 results
        description="清晰、准确地陈述论文得出的主要科学发现或核心结论。"
    )