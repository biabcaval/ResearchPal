from collections.abc import Iterable
from typing import Any

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import ValidationError

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
from researchpal.tools import extract_section, search_documents
from researchpal.tools import VectorStore


SYSTEM_INSTRUCTION = """
You are a research assistant for academic papers.
Answer in Portuguese and use only evidence returned by the available tools.
Use search_documents for semantic evidence and extract_section when the user
asks about an abstract, introduction, or conclusion.
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
            "Use quando precisar encontrar evidência textual para responder à pergunta."
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
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult[Any]:
        try:
            if name == "search_documents":
                params = SearchToolParams.model_validate(arguments)
                return search_documents(params=params, store=self.store)

            if name == "extract_section":
                params = ExtractSectionParams.model_validate(arguments)
                return extract_section(params=params, settings=self.settings)
        except ValidationError as error:
            return ToolResult(success=False, error=f"Invalid tool arguments: {error}")
        except (OSError, RuntimeError, ValueError) as error:
            return ToolResult(success=False, error=f"Tool execution failed: {error}")

        return ToolResult(success=False, error=f"Unknown tool: {name}")

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
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=[self.document_tools],
                        temperature=0.0,
                    ),
                )
            except (genai_errors.APIError, genai_errors.ClientError) as error:
                return self._no_evidence_response(
                    tool_errors + [f"Gemini request failed: {error}"]
                )
            candidate = _first_candidate(response)
            if candidate is None:
                return self._no_evidence_response(
                    tool_errors + ["Gemini returned no candidate"]
                )

            function_calls = list(_function_calls(candidate.content.parts))
            if not function_calls:
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

            contents.append(candidate.content)
            tool_response_parts: list[types.Part] = []
            for function_call in function_calls:
                name = function_call.name or ""
                arguments = dict(function_call.args or {})
                result = self._execute_tool(name, arguments)
                if result.success and result.data is not None:
                    if name == "search_documents":
                        sources.extend(result.data)
                    elif name == "extract_section":
                        sections.append(result.data)
                elif result.error:
                    tool_errors.append(result.error)
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response=result.model_dump(mode="json"),
                    )
                )
            contents.append(
                types.Content(role="user", parts=tool_response_parts)
            )

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
            answer = (final_response.text or "").strip()
            if answer:
                return AskResponse(
                    answer=answer,
                    sources=_unique_sources(sources),
                    sections=sections,
                    evidence_found=bool(sources or sections),
                    tool_errors=tool_errors + ["Maximum tool-calling rounds exceeded"],
                )
        except (genai_errors.APIError, genai_errors.ClientError) as error:
            tool_errors.append(f"Final Gemini synthesis failed: {error}")

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


def _first_candidate(response: Any) -> Any | None:
    candidates = getattr(response, "candidates", None) or []
    return candidates[0] if candidates else None


def _function_calls(parts: Iterable[Any]) -> Iterable[Any]:
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            yield function_call


def _unique_sources(
    sources: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    unique: dict[str, RetrievedDocument] = {}
    for source in sources:
        unique[source.identifier] = source
    return list(unique.values())
