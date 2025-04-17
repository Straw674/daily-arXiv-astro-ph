from pydantic import BaseModel, Field


class Structure(BaseModel):
    tldr: str = Field(
        description="Summarize the abstract of an astronomy paper in one concise sentence that captures the essence of the article."
    )
    background: str = Field(
        description="Extract the research background information from the abstract. Background information typically refers to the current state of the relevant field, the shortcomings of existing research, and the significance of the study, which are presented to introduce the research question. When extracting such information, carefully analyze which parts of the abstract serve as groundwork for the study."
    )
    data: str = Field(
        description="Extract the information about which data are used in this paper from the abstract."
    )
    method: str = Field(
        description="Extract the information about which method are used in this paper from the abstract."
    )
    result: str = Field(
        description="Extract the information about the research result of this paper from the abstract."
    )
