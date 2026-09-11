import logging
from collections.abc import Iterable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from researchpal.agent.errors import ModelUnavailableError
from researchpal.agent.gemini_sdk import (
    GeminiFunctionCall,
    GeminiGenerateContentResponse,
    first_candidate,
    iter_function_calls,
)
from researchpal.config import Settings, get_settings
from researchpal.models import (
    AskRequest,
    AskResponse,
    ExtractedSection,
    ExtractSectionParams,
    RetrievedDocument,
    SearchToolParams,
    ToolResult,
)
from researchpal.tools import VectorStore, extract_section, search_documents

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
MAX_TOOL_ROUNDS = 3


class GeminiResearchAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        active_settings = settings or get_settings()

        if not active_settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        self.settings = active_settings
        self.client = genai.Client(api_key=active_settings.gemini_api_key)
        self.store = VectorStore(
            active_settings.chroma_path,
            active_settings.collection_name,
        )

    search_documents_declaration = types.FunctionDeclaration(
        name="search_documents",
        description=(
            "Busca semanticamente os chunks mais relevantes dos artigos indexados. "
            "O índice é em inglês: passe a query em inglês. Se a pergunta do usuário "
            "estiver em português, reescreva-a como uma query de busca em inglês. "
            "A resposta ao usuário continua em português. Use quando precisar "
            "encontrar evidência textual para responder à pergunta."
        ),
        parameters_json_schema=SearchToolParams.model_json_schema(),
    )

    extract_section_declaration = types.FunctionDeclaration(
        name="extract_section",
        description=(
            "Extrai uma seção específica de um artigo. "
            "Use somente para abstract, introduction ou conclusion."
        ),
        parameters_json_schema=ExtractSectionParams.model_json_schema(),
    )

    document_tools = types.Tool(
        function_declarations=[
            search_documents_declaration,
            extract_section_declaration,
        ]
    )

    def _execute_tool(
        self,
        call: GeminiFunctionCall,
    ) -> ToolResult[list[RetrievedDocument] | ExtractedSection]:
        try:
            if call.name == "search_documents":
                params = SearchToolParams.model_validate(call.arguments)
                return search_documents(params=params, store=self.store)

            if call.name == "extract_section":
                params = ExtractSectionParams.model_validate(call.arguments)
                return extract_section(params=params, settings=self.settings)
        except ValidationError as error:
            logger.warning("Invalid tool arguments for %s: %s", call.name, error)
            return ToolResult(success=False, error=f"Invalid tool arguments: {error}")
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("Tool execution failed for %s: %s", call.name, error)
            return ToolResult(success=False, error=f"Tool execution failed: {error}")

        logger.warning("Unknown tool requested: %s", call.name)
        return ToolResult(success=False, error=f"Unknown tool: {call.name}")

    def ask(self, request: AskRequest) -> AskResponse:
        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=request.question)],
            )
        ]
        sources: list[RetrievedDocument] = []
        sections: list[ExtractedSection] = []
        tool_errors: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = self._generate_with_tools(contents)
            candidate = first_candidate(response)
            if candidate is None:
                return self._no_evidence_response(
                    tool_errors + ["Gemini returned no candidate"]
                )

            function_calls = list(
                iter_function_calls(getattr(candidate.content, "parts", None) or [])
            )
            if not function_calls:
                return self._response_from_model_text(
                    response, sources, sections, tool_errors
                )

            contents.append(candidate.content)
            tool_parts = self._apply_function_calls(
                function_calls, sources, sections, tool_errors
            )
            contents.append(types.Content(role="user", parts=tool_parts))

        return self._synthesize_without_tools(contents, sources, sections, tool_errors)

    def _generate_with_tools(
        self, contents: list[types.Content]
    ) -> GeminiGenerateContentResponse:
        try:
            return self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[self.document_tools],
                    temperature=0.0,
                ),
            )
        except (genai_errors.APIError, genai_errors.ClientError) as error:
            logger.warning("Gemini request failed: %s", error)
            raise ModelUnavailableError(f"Gemini request failed: {error}") from error

    def _response_from_model_text(
        self,
        response: GeminiGenerateContentResponse,
        sources: list[RetrievedDocument],
        sections: list[ExtractedSection],
        tool_errors: list[str],
    ) -> AskResponse:
        answer = (response.text or "").strip()
        if not answer:
            return self._no_evidence_response(
                tool_errors + ["Gemini returned an empty answer"]
            )
        return AskResponse(
            answer=answer,
            sources=_unique_sources(sources),
            sections=sections,
            evidence_found=bool(sources or sections),
            tool_errors=tool_errors,
        )

    def _apply_function_calls(
        self,
        function_calls: Iterable[GeminiFunctionCall],
        sources: list[RetrievedDocument],
        sections: list[ExtractedSection],
        tool_errors: list[str],
    ) -> list[types.Part]:
        tool_response_parts: list[types.Part] = []
        for function_call in function_calls:
            result = self._execute_tool(function_call)
            if result.success and result.data is not None:
                data = result.data
                if function_call.name == "search_documents" and isinstance(data, list):
                    sources.extend(data)
                elif function_call.name == "extract_section" and isinstance(
                    data, ExtractedSection
                ):
                    sections.append(data)
            elif result.error:
                tool_errors.append(result.error)
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response=result.model_dump(mode="json"),
                )
            )
        return tool_response_parts

    def _synthesize_without_tools(
        self,
        contents: list[types.Content],
        sources: list[RetrievedDocument],
        sections: list[ExtractedSection],
        tool_errors: list[str],
    ) -> AskResponse:
        try:
            final_response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    *contents,
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "Não chame mais nenhuma ferramenta. "
                                    "Responda usando exclusivamente as evidências "
                                    "já retornadas acima. Se elas forem insuficientes, "
                                    "declare isso explicitamente."
                                )
                            )
                        ],
                    ),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                ),
            )
        except (genai_errors.APIError, genai_errors.ClientError) as error:
            logger.warning("Final Gemini synthesis failed: %s", error)
            raise ModelUnavailableError(
                f"Final Gemini synthesis failed: {error}"
            ) from error

        answer = (final_response.text or "").strip()
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


def _unique_sources(
    sources: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    unique: dict[str, RetrievedDocument] = {}
    for source in sources:
        unique[source.identifier] = source
    return list(unique.values())
