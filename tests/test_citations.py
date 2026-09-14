from researchpal.agent.citations import SNIPPET_MAX_CHARS, attach_citations
from researchpal.models import ChunkMetadata, RetrievedDocument


def _source(
    identifier: str,
    *,
    paper_id: str = "1706.03762",
    page: int = 1,
    text: str = "Attention improves sequence modeling.",
) -> RetrievedDocument:
    return RetrievedDocument(
        identifier=identifier,
        text=text,
        metadata=ChunkMetadata(paper_id=paper_id, page=page),
        distance=0.1,
    )


def test_known_identifier_becomes_numbered_citation() -> None:
    source = _source("1706.03762-page-1-chunk-0")

    answer, citations = attach_citations(
        "A atenção ajuda o modelo [[1706.03762-page-1-chunk-0]].",
        [source],
    )

    assert answer == "A atenção ajuda o modelo [1]."
    assert len(citations) == 1
    assert citations[0].number == 1
    assert citations[0].identifier == "1706.03762-page-1-chunk-0"
    assert citations[0].paper_id == "1706.03762"
    assert citations[0].page == 1
    assert citations[0].snippet == "Attention improves sequence modeling."


def test_repeated_identifier_reuses_the_same_number() -> None:
    source = _source("1706.03762-page-1-chunk-0")

    answer, citations = attach_citations(
        "Primeiro [[1706.03762-page-1-chunk-0]] e de novo [[1706.03762-page-1-chunk-0]].",
        [source],
    )

    assert answer == "Primeiro [1] e de novo [1]."
    assert [citation.number for citation in citations] == [1]


def test_unknown_identifier_is_stripped() -> None:
    source = _source("1706.03762-page-1-chunk-0")

    answer, citations = attach_citations(
        "Fato real [[1706.03762-page-1-chunk-0]] e inventado [[9999.99999-page-9-chunk-9]].",
        [source],
    )

    assert answer == "Fato real [1] e inventado ."
    assert [citation.identifier for citation in citations] == [
        "1706.03762-page-1-chunk-0"
    ]


def test_appearance_order_assigns_numbers_not_retrieval_order() -> None:
    first = _source("1706.03762-page-1-chunk-0")
    second = _source(
        "2005.11401-page-3-chunk-1",
        paper_id="2005.11401",
        page=3,
        text="RAG combines retrieval with generation.",
    )

    answer, citations = attach_citations(
        "RAG [[2005.11401-page-3-chunk-1]] veio depois de atenção [[1706.03762-page-1-chunk-0]].",
        [first, second],
    )

    assert answer == "RAG [1] veio depois de atenção [2]."
    assert [citation.identifier for citation in citations] == [
        "2005.11401-page-3-chunk-1",
        "1706.03762-page-1-chunk-0",
    ]


def test_no_markers_falls_back_to_retrieval_order_citations() -> None:
    sources = [
        _source("1706.03762-page-1-chunk-0"),
        _source(
            "2005.11401-page-3-chunk-1",
            paper_id="2005.11401",
            page=3,
            text="RAG combines retrieval with generation.",
        ),
    ]

    answer, citations = attach_citations("Resposta sem marcadores.", sources)

    assert answer == "Resposta sem marcadores."
    assert [citation.number for citation in citations] == [1, 2]
    assert [citation.identifier for citation in citations] == [
        "1706.03762-page-1-chunk-0",
        "2005.11401-page-3-chunk-1",
    ]


def test_empty_sources_leave_answer_and_citations_empty() -> None:
    answer, citations = attach_citations("Sem evidência.", [])

    assert answer == "Sem evidência."
    assert citations == []


def test_snippet_truncates_on_a_word_boundary() -> None:
    words = " ".join(f"word{index:03d}" for index in range(80))
    source = _source("1706.03762-page-1-chunk-0", text=words)

    _, citations = attach_citations(
        "Cite [[1706.03762-page-1-chunk-0]]",
        [source],
    )

    snippet = citations[0].snippet
    assert len(snippet) <= SNIPPET_MAX_CHARS + 1
    assert snippet.endswith("…")
    assert " " not in snippet.split("…")[0][-1:]
    assert snippet[:-1] in words
