from pydantic import BaseModel, ConfigDict

from researchpal.models import Citation


class AskHttpResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    status: str
