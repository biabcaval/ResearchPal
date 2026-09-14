import json
import logging
from collections.abc import Sequence
from typing import Protocol, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from researchpal.agent.errors import GEMINI_UPSTREAM_ERRORS, ModelUnavailableError
from researchpal.agent.sanity import (
    SANITY_FAIL_ERROR,
    SANITY_FALLBACK_ANSWER,
    SANITY_REJECT_ERROR,
    SANITY_REWRITE_ERROR,
    AnswerSanityChecker,
    GeminiAnswerSanityChecker,
)
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
NO_EVIDENCE_ANSWER = (
    "Não encontrei evidência suficiente nos artigos indexados "
    "para responder com segurança."
)
COULD_NOT_SYNTHESIZE_ANSWER = (
    "Recuperei evidências nos artigos, mas não consegui concluir "
    "uma síntese final automaticamente. Consulte as fontes retornadas."
)
CANNED_ANSWERS = frozenset({NO_EVIDENCE_ANSWER, COULD_NOT_SYNTHESIZE_ANSWER})
SYNTHESIS_INSTRUCTION = (
    "Não chame mais nenhuma ferramenta. "
    "Responda usando exclusivamente as evidências "
    "já retornadas acima. Se elas forem insuficientes, "
    "declare isso explicitamente."
)
MAX_TOOL_ROUNDS = 1
MODEL_CALL_LIMIT_PREFIX = "Model call limits exceeded"


class LangChainAgentState(TypedDict):
    """State passed to `AgentGraph.invoke` (LangChain messages dict)."""

    messages: list[BaseMessage]


class AgentGraph(Protocol):
    """LangChain agent runnable used by `ask` (real graph or test double)."""

    def invoke(self, payload: LangChainAgentState) -> LangChainAgentState: ...


class SynthesisModel(Protocol):
    """Chat model used for the no-further-tools synthesis pass."""

    def invoke(self, messages: Sequence[BaseMessage]) -> BaseMessage: ...


class GeminiResearchAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: VectorStore | None = None,
        graph: AgentGraph | None = None,
        synthesis_model: SynthesisModel | None = None,
        sanity_checker: AnswerSanityChecker | None = None,
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

        chat_model: ChatGoogleGenerativeAI | None = None
        if graph is None or synthesis_model is None:
            chat_model = ChatGoogleGenerativeAI(
                model=active_settings.gemini_model,
                temperature=0.0,
                google_api_key=active_settings.gemini_api_key,
            )
            if graph is None:
                graph = create_agent(
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
            if synthesis_model is None:
                synthesis_model = chat_model

        self.graph = graph
        self.synthesis_model = synthesis_model
        if sanity_checker is None:
            judge_model = chat_model or ChatGoogleGenerativeAI(
                model=active_settings.gemini_model,
                temperature=0.0,
                google_api_key=active_settings.gemini_api_key,
            )
            sanity_checker = GeminiAnswerSanityChecker(judge_model)
        self.sanity_checker = sanity_checker

    def ask(self, request: AskRequest) -> AskResponse:
        try:
            result = self.graph.invoke(
                LangChainAgentState(
                    messages=[HumanMessage(content=request.question)]
                )
            )
        except GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("Gemini request failed: %s", error)
            raise ModelUnavailableError(f"Gemini request failed: {error}") from error

        messages = list(result["messages"])
        sources, sections, tool_errors = _collect_tool_outputs(messages)

        if _is_model_call_limit_message(_last_ai_message(messages)):
            return self._synthesize_without_tools(
                request.question,
                messages,
                sources,
                sections,
                tool_errors,
            )

        answer = _ai_text(_last_ai_message(messages))
        if not answer:
            return self._no_evidence_response(
                tool_errors + ["Gemini returned an empty answer"],
                sources=sources,
                sections=sections,
            )
        return self._apply_sanity_check(
            request.question,
            messages,
            AskResponse(
                answer=answer,
                sources=_unique_sources(sources),
                sections=sections,
                evidence_found=bool(sources or sections),
                tool_errors=tool_errors,
            ),
        )

    def _synthesize_without_tools(
        self,
        question: str,
        messages: list[BaseMessage],
        sources: list[RetrievedDocument],
        sections: list[ExtractedSection],
        tool_errors: list[str],
    ) -> AskResponse:
        history = _history_without_limit_message(messages)
        try:
            final_response = self.synthesis_model.invoke(
                [
                    SystemMessage(content=SYSTEM_INSTRUCTION),
                    *history,
                    HumanMessage(content=SYNTHESIS_INSTRUCTION),
                ]
            )
        except GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("Final Gemini synthesis failed: %s", error)
            raise ModelUnavailableError(
                f"Final Gemini synthesis failed: {error}"
            ) from error

        answer = _ai_text(final_response)
        if answer:
            return self._apply_sanity_check(
                question,
                messages,
                AskResponse(
                    answer=answer,
                    sources=_unique_sources(sources),
                    sections=sections,
                    evidence_found=bool(sources or sections),
                    tool_errors=tool_errors + ["Maximum tool-calling rounds exceeded"],
                ),
            )
        return self._no_evidence_response(
            tool_errors + ["Maximum tool-calling rounds exceeded"],
            sources=sources,
            sections=sections,
        )

    def _apply_sanity_check(
        self,
        question: str,
        messages: list[BaseMessage],
        response: AskResponse,
    ) -> AskResponse:
        if response.answer in CANNED_ANSWERS:
            return response

        try:
            verdict = self.sanity_checker.check(question, response.answer)
        except GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("First sanity check Gemini request failed: %s", error)
            raise ModelUnavailableError(
                f"Sanity check Gemini request failed: {error}"
            ) from error

        if verdict.addresses_question:
            return response

        tool_errors = response.tool_errors + [SANITY_FAIL_ERROR]
        rewritten = self._rewrite_for_unanswered_parts(
            messages, verdict.unanswered_parts
        )
        if not rewritten:
            return response.model_copy(
                update={
                    "answer": SANITY_FALLBACK_ANSWER,
                    "tool_errors": tool_errors + [SANITY_REJECT_ERROR],
                }
            )

        try:
            second = self.sanity_checker.check(question, rewritten)
        except GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("Second sanity check Gemini request failed: %s", error)
            raise ModelUnavailableError(
                f"Sanity check Gemini request failed: {error}"
            ) from error

        if second.addresses_question:
            return response.model_copy(
                update={
                    "answer": rewritten,
                    "tool_errors": tool_errors + [SANITY_REWRITE_ERROR],
                }
            )
        return response.model_copy(
            update={
                "answer": SANITY_FALLBACK_ANSWER,
                "tool_errors": tool_errors
                + [SANITY_REWRITE_ERROR, SANITY_REJECT_ERROR],
            }
        )

    def _rewrite_for_unanswered_parts(
        self,
        messages: list[BaseMessage],
        unanswered_parts: list[str],
    ) -> str:
        history = _history_without_limit_message(messages)
        parts = (
            "; ".join(unanswered_parts)
            if unanswered_parts
            else "a pergunta original"
        )
        instruction = (
            "Não chame mais nenhuma ferramenta. "
            "Responda usando exclusivamente as evidências "
            "já retornadas acima. "
            "A resposta anterior não atendeu à pergunta. "
            f"Cubra explicitamente: {parts}. "
            "Não invente fatos. Se as evidências não cobrirem "
            "esses pontos, declare isso explicitamente. "
            "Responda em português."
        )
        try:
            final_response = self.synthesis_model.invoke(
                [
                    SystemMessage(content=SYSTEM_INSTRUCTION),
                    *history,
                    HumanMessage(content=instruction),
                ]
            )
        except GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("Sanity rewrite Gemini request failed: %s", error)
            raise ModelUnavailableError(
                f"Sanity rewrite Gemini request failed: {error}"
            ) from error
        return _ai_text(final_response)

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
                answer=COULD_NOT_SYNTHESIZE_ANSWER,
                sources=resolved_sources,
                sections=resolved_sections,
                evidence_found=True,
                tool_errors=tool_errors,
            )
        return AskResponse(
            answer=NO_EVIDENCE_ANSWER,
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
            if not result.success:
                if result.error:
                    tool_errors.append(result.error)
                continue
            if message.name == "search_documents" and isinstance(result.data, list):
                documents = [
                    RetrievedDocument.model_validate(item) for item in result.data
                ]
                sources.extend(documents)
            elif message.name == "extract_section" and result.data is not None:
                sections.append(ExtractedSection.model_validate(result.data))
        except (TypeError, ValueError, ValidationError) as error:
            tool_errors.append(f"Invalid tool result: {error}")
    return sources, sections, tool_errors


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _history_without_limit_message(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    return [
        message
        for message in messages
        if not _is_model_call_limit_message(message)
    ]


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
