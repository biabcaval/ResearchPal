from pathlib import Path

from researchpal.tools.pdf import read_pdf_pages


def _write_text_pdf(path: Path, lines: list[str]) -> None:
    """Write a one-page PDF with Type1 Helvetica so PyPDF2 can extract text."""
    content = "BT /F1 12 Tf 72 720 Td\n"
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content += f"({escaped}) Tj\n0 -16 Td\n"
    content += "ET\n"
    stream = content.encode("latin-1")
    objects = [
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n",
        (
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
            b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        ),
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        (
            f"5 0 obj<</Length {len(stream)}>>stream\n".encode("ascii")
            + stream
            + b"endstream\nendobj\n"
        ),
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)
    xref_start = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer<</Size {len(offsets)}/Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(bytes(output))


def test_read_pdf_pages_returns_extracted_text_and_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "1706.03762.pdf"
    _write_text_pdf(pdf_path, ["Attention is all you need"])

    pages = read_pdf_pages(pdf_path, "1706.03762")

    assert len(pages) == 1
    text, metadata = pages[0]
    assert "Attention is all you need" in text
    assert metadata.paper_id == "1706.03762"
    assert metadata.page == 1
