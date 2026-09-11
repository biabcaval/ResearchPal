import json
import logging
from typing import Any

from google.genai import errors as genai_errors
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.exceptions import ModelError
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from pydantic import ValidationError

from researchpal.agent.errors import ModelUnavailableError
from researchpal.config import Settings, get_settings
from researchpal.models import (
    AskRequest,
    AskResponse,
    ExtractedSection,
    RetrievedDocument,
    ToolResult,
)
from researchpal.tools import VectorStore
from researchpal.tools.extract_section import ExtractSectionTool
from researchpal.tools.search_documents import SearchDocumentsTool

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """
You are a research assistant for academic papers.
Answer in Portuguese and use only evidence returned by the available tools.
Use search_documents for semantic evidence and extract_section when the user
asks about an abstract, introduction, or conclusion.
When calling search_documents, pass the query in English so retrieval can
compare English text with the original English papers. If the user asked in
Portuguese, rewrite the search query into English first. The user-facing
answer stays in Portuguese.
Never invent facts, citations, paper contents, or section contents.
If the tools return no evidence or fail, clearly state that there is not enough
evidence to answer safely. Mention the paper identifiers used when possible.
"""
SYNTHESIS_INSTRUCTION = (
    "Não chame mais nenhuma ferramenta. "
    "Responda usando exclusivamente as evidências "
    "já retornadas acima. Se elas forem insuficientes, "
    "declare isso explicitamente."
)
MAX_TOOL_ROUNDS = 3
MODEL_CALL_LIMIT_PREFIX = "Model call limits exceeded"
_GEMINI_UPSTREAM_ERRORS = (
    genai_errors.APIError,
    genai_errors.ClientError,
    ModelError,
    ChatGoogleGenerativeAIError,
)


class GeminiResearchAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: VectorStore | None = None,
        graph: Any | None = None,
        synthesis_model: Any | None = None,
    ) -> None:
        active_settings = settings or get_settings()

        if not active_settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        self.settings = active_settings
        self.store = store or VectorStore(
            active_settings.chroma_path,
            active_settings.collection_name,
        )
        self.search_documents_tool = SearchDocumentsTool(
            settings=active_settings,
            store=self.store,
        )
        self.extract_section_tool = ExtractSectionTool(settings=active_settings)
        self.graph = graph
        self.synthesis_model = synthesis_model

        if self.graph is None or self.synthesis_model is None:
            chat_model = ChatGoogleGenerativeAI(
                model=active_settings.gemini_model,
                temperature=0.0,
                google_api_key=active_settings.gemini_api_key,
            )
            if self.graph is None:
                self.graph = create_agent(
                    model=chat_model,
                    tools=[
                        self.search_documents_tool.as_langchain_tool(),
                        self.extract_section_tool.as_langchain_tool(),
                    ],
                    system_prompt=SYSTEM_INSTRUCTION,
                    middleware=[
                        ModelCallLimitMiddleware(
                            run_limit=MAX_TOOL_ROUNDS,
                            exit_behavior="end",
                        )
                    ],
                )
            if self.synthesis_model is None:
                self.synthesis_model = chat_model

    def ask(self, request: AskRequest) -> AskResponse:
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=request.question)]}
            )
        except _GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("Gemini request failed: %s", error)
            raise ModelUnavailableError(f"Gemini request failed: {error}") from error

        messages = list(result["messages"])
        sources, sections, tool_errors = _collect_tool_outputs(messages)

        if _is_model_call_limit_message(_last_ai_message(messages)):
            return self._synthesize_without_tools(
                messages, sources, sections, tool_errors
            )

        answer = _ai_text(_last_ai_message(messages))
        if not answer:
            return self._no_evidence_response(
                tool_errors + ["Gemini returned an empty answer"],
                sources=sources,
                sections=sections,
            )
        return AskResponse(
            answer=answer,
            sources=_unique_sources(sources),
            sections=sections,
            evidence_found=bool(sources or sections),
            tool_errors=tool_errors,
        )

    def _synthesize_without_tools(
        self,
        messages: list[BaseMessage],
        sources: list[RetrievedDocument],
        sections: list[ExtractedSection],
        tool_errors: list[str],
    ) -> AskResponse:
        history = [
            message
            for message in messages
            if not _is_model_call_limit_message(message)
        ]
        try:
            final_response = self.synthesis_model.invoke(
                [
                    SystemMessage(content=SYSTEM_INSTRUCTION),
                    *history,
                    HumanMessage(content=SYNTHESIS_INSTRUCTION),
                ]
            )
        except _GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("Final Gemini synthesis failed: %s", error)
            raise ModelUnavailableError(
                f"Final Gemini synthesis failed: {error}"
            ) from error

        answer = _ai_text(final_response)
        if answer:
            return AskResponse(
                answer=answer,
                sources=_unique_sources(sources),
                sections=sections,
                evidence_found=bool(sources or sections),
                tool_errors=tool_errors + ["Maximum tool-calling rounds exceeded"],
            )
        return self._no_evidence_response(
            tool_errors + ["Maximum tool-calling rounds exceeded"],
            sources=sources,
            sections=sections,
        )

    @staticmethod
    def _no_evidence_response(
        tool_errors: list[str],
        sources: list[RetrievedDocument] | None = None,
        sections: list[ExtractedSection] | None = None,
    ) -> AskResponse:
        resolved_sources = _unique_sources(sources or [])
        resolved_sections = sections or []
        if resolved_sources or resolved_sections:
            return AskResponse(
                answer=(
                    "Recuperei evidências nos artigos, mas não consegui concluir "
                    "uma síntese final automaticamente. Consulte as fontes retornadas."
                ),
                sources=resolved_sources,
                sections=resolved_sections,
                evidence_found=True,
                tool_errors=tool_errors,
            )
        return AskResponse(
            answer=(
                "Não encontrei evidência suficiente nos artigos indexados "
                "para responder com segurança."
            ),
            sources=[],
            sections=[],
            evidence_found=False,
            tool_errors=tool_errors,
        )


def _collect_tool_outputs(
    messages: list[BaseMessage],
) -> tuple[list[RetrievedDocument], list[ExtractedSection], list[str]]:
    sources: list[RetrievedDocument] = []
    sections: list[ExtractedSection] = []
    tool_errors: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload: object = (
                json.loads(message.content)
                if isinstance(message.content, str)
                else message.content
            )
            result = ToolResult[object].model_validate(payload)
        except (TypeError, ValueError, ValidationError) as error:
            tool_errors.append(f"Invalid tool result: {error}")
            continue
        if not result.success:
            if result.error:
                tool_errors.append(result.error)
            continue
        if message.name == "search_documents" and isinstance(result.data, list):
            sources.extend(RetrievedDocument.model_validate(item) for item in result.data)
        elif message.name == "extract_section" and result.data is not None:
            sections.append(ExtractedSection.model_validate(result.data))
    return sources, sections, tool_errors


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _is_model_call_limit_message(message: BaseMessage | None) -> bool:
    if message is None or not isinstance(message, AIMessage):
        return False
    return _ai_text(message).startswith(MODEL_CALL_LIMIT_PREFIX)


def _ai_text(message: BaseMessage | None) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return ""


def _unique_sources(sources: list[RetrievedDocument]) -> list[RetrievedDocument]:
    unique: dict[str, RetrievedDocument] = {}
    for source in sources:
        unique[source.identifier] = source
    return list(unique.values())
