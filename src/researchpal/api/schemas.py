from pydantic import BaseModel, ConfigDict, Field


class AskHttpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)


class AskHttpResponse(BaseModel):
    question: str
    answer: str
