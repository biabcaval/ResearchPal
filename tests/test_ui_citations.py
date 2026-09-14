from researchpal.models import Citation
from researchpal.ui.citations import format_answer_html


def _citation() -> Citation:
    return Citation(
        number=1,
        identifier="1706.03762-page-1-chunk-0",
        paper_id="1706.03762",
        page=1,
        snippet="Attention improves sequence modeling.",
    )


def test_format_answer_html_wraps_numbered_cite_in_hover_chip() -> None:
    html_output = format_answer_html("A atenção ajuda [1].", [_citation()])

    assert 'class="rp-cite"' in html_output
    assert ">1<span" in html_output
    assert "1706.03762 · p. 1" in html_output
    assert "Attention improves sequence modeling." in html_output
    assert "[1]" not in html_output


def test_format_answer_html_appends_fontes_when_answer_has_no_numbers() -> None:
    html_output = format_answer_html("Resposta sem números.", [_citation()])

    assert "Fontes" in html_output
    assert 'class="rp-cite"' in html_output
    assert "1706.03762" in html_output


def test_format_answer_html_escapes_snippet_html() -> None:
    citation = Citation(
        number=1,
        identifier="1706.03762-page-1-chunk-0",
        paper_id="1706.03762",
        page=1,
        snippet='<script>alert("x")</script>',
    )

    html_output = format_answer_html("Veja [1].", [citation])

    assert "<script>" not in html_output
    assert "&lt;script&gt;" in html_output
