from pydantic import BaseModel, ConfigDict


class AskHttpResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str


class HealthResponse(BaseModel):
    status: str
