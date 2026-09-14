"""Turn numbered citations into hoverable HTML chips for Gradio."""

from __future__ import annotations

import html
import re

from researchpal.models import Citation

_NUMBERED_CITE = re.compile(r"\[(\d+)\]")

CITATION_CSS = """
.rp-cite {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.15em;
  height: 1.15em;
  margin: 0 0.12em;
  padding: 0 0.28em;
  border-radius: 999px;
  background: #2f6fed;
  color: #fff;
  font-size: 0.72em;
  font-weight: 700;
  line-height: 1;
  cursor: help;
  vertical-align: super;
}
.rp-cite:focus {
  outline: 2px solid #1b4fbf;
  outline-offset: 2px;
}
.rp-cite-card {
  display: none;
  position: absolute;
  left: 50%;
  bottom: calc(100% + 0.4em);
  transform: translateX(-50%);
  z-index: 20;
  width: max-content;
  max-width: 22rem;
  padding: 0.55em 0.7em;
  border-radius: 8px;
  background: #1c1c1c;
  color: #f5f5f5;
  font-size: 0.85rem;
  font-weight: 400;
  line-height: 1.35;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
  white-space: normal;
  text-align: left;
}
.rp-cite:hover .rp-cite-card,
.rp-cite:focus .rp-cite-card {
  display: block;
}
.rp-fontes {
  margin-top: 0.85em;
  font-size: 0.92em;
}
"""


def format_answer_html(answer: str, citations: list[Citation]) -> str:
    """Escape answer text and wrap `[n]` markers in hover citation chips.

    When the answer has no matching `[n]` but citations exist, append a Fontes
    line of the same chips.

    Args:
        answer: Numbered answer from the API.
        citations: Citation list from the API.

    Returns:
        Safe HTML for Gradio chat.
    """
    by_number = {citation.number: citation for citation in citations}
    escaped = html.escape(answer)

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        citation = by_number.get(number)
        if citation is None:
            return match.group(0)
        return _chip(citation)

    rendered = _NUMBERED_CITE.sub(replace, escaped)
    has_inline = any(f"[{citation.number}]" in answer for citation in citations)
    if citations and not has_inline:
        chips = " ".join(_chip(citation) for citation in citations)
        rendered = f'{rendered}<div class="rp-fontes"><strong>Fontes</strong> {chips}</div>'
    return rendered


def _chip(citation: Citation) -> str:
    snippet = html.escape(citation.snippet)
    paper_id = html.escape(citation.paper_id)
    card = f'{paper_id} · p. {citation.page} · "{snippet}"'
    return (
        f'<span class="rp-cite" tabindex="0">{citation.number}'
        f'<span class="rp-cite-card">{card}</span></span>'
    )
