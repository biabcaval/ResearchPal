"""Map model `[[chunk-id]]` markers onto numbered citations."""

from __future__ import annotations

import re

from researchpal.models import Citation, RetrievedDocument

SNIPPET_MAX_CHARS = 280
_MARKER = re.compile(r"\[\[([^\]]+)\]\]")


def attach_citations(
    answer: str,
    sources: list[RetrievedDocument],
) -> tuple[str, list[Citation]]:
    """Replace known `[[id]]` markers with `[n]` and build the citation list.

    Unknown markers are removed. If the answer has no markers, citations are
    numbered in `sources` order and the answer is left unchanged.

    Args:
        answer: Model prose, possibly containing `[[identifier]]` markers.
        sources: Retrieved chunks that may be cited.

    Returns:
        The rewritten answer and the ordered citation list.
    """
    by_id = {source.identifier: source for source in sources}
    if _MARKER.search(answer) is None:
        citations = [
            _citation(number, source)
            for number, source in enumerate(sources, start=1)
        ]
        return answer, citations

    assigned: dict[str, int] = {}
    citations: list[Citation] = []

    def replace(match: re.Match[str]) -> str:
        identifier = match.group(1)
        source = by_id.get(identifier)
        if source is None:
            return ""
        number = assigned.get(identifier)
        if number is None:
            number = len(assigned) + 1
            assigned[identifier] = number
            citations.append(_citation(number, source))
        return f"[{number}]"

    return _MARKER.sub(replace, answer), citations


def _citation(number: int, source: RetrievedDocument) -> Citation:
    return Citation(
        number=number,
        identifier=source.identifier,
        paper_id=source.metadata.paper_id,
        page=source.metadata.page,
        snippet=_snippet(source.text),
    )


def _snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    truncated = collapsed[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated}…"
