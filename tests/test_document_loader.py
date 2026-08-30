import fitz

from insurance_rag.config import AppConfig
from insurance_rag.document_loader import (
    PageExtraction,
    garbled_ratio,
    needs_ocr,
    normalize_page_text,
    parse_pdf_bytes,
)


def test_garbled_ratio_counts_replacement_characters():
    assert garbled_ratio("abc��") == 0.4


def test_needs_ocr_for_short_text():
    config = AppConfig(openai_api_key=None, min_page_text_chars=20)

    assert needs_ocr("短", config) is True


def test_needs_ocr_for_garbled_text():
    config = AppConfig(openai_api_key=None, max_garbled_ratio=0.2)

    assert needs_ocr("正常文字����", config) is True


def test_normalize_page_text_returns_quality_note_for_ocr():
    page = normalize_page_text(
        PageExtraction(page_number=3, text=" 等待期 是 90 天 ", extraction_method="ocr"),
    )

    assert page.page_number == 3
    assert page.text == "等待期 是 90 天"
    assert page.extraction_method == "ocr"
    assert "该页来自 OCR 识别，可能有误。" in page.quality_notes


def test_normalize_page_text_preserves_line_boundaries_for_heading_detection():
    page = normalize_page_text(
        PageExtraction(
            page_number=1,
            text="  第六条   等待期  \n\n 等待期为九十日。  ",
            extraction_method="text",
        ),
    )

    assert page.text == "第六条 等待期\n等待期为九十日。"


def test_parse_pdf_bytes_extracts_text_from_in_memory_pdf():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Waiting period is 90 days")
    pdf_bytes = document.tobytes()
    document.close()
    config = AppConfig(openai_api_key=None, ocr_enabled=False)

    result = parse_pdf_bytes(pdf_bytes, r"C:\uploads\policy.pdf", config)

    assert result.filename == "policy.pdf"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Waiting period is 90 days" in result.pages[0].text
    assert result.pages[0].extraction_method == "text"


def test_parse_pdf_bytes_keeps_text_pages_when_ocr_runtime_fails(mocker):
    document = fitz.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "A")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "B")
    pdf_bytes = document.tobytes()
    document.close()
    mocker.patch(
        "insurance_rag.document_loader._ocr_page",
        side_effect=ValueError("tesseract runtime failed"),
    )
    config = AppConfig(openai_api_key=None, min_page_text_chars=20, ocr_enabled=True)

    result = parse_pdf_bytes(pdf_bytes, "policy.pdf", config)

    assert len(result.pages) == 2
    assert len(result.warnings) == 1
    assert "OCR 运行失败" in result.warnings[0]
    assert "Tesseract" in result.warnings[0]
    assert "第 1 页" in result.warnings[0]
    assert result.pages[0].text == "A"
    assert result.pages[0].extraction_method == "text"
    assert result.pages[1].text == "B"
    assert result.pages[1].extraction_method == "text"


def test_parse_pdf_bytes_marks_ocr_text_with_severe_remaining_garbling(mocker):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A")
    pdf_bytes = document.tobytes()
    document.close()
    mocker.patch(
        "insurance_rag.document_loader._ocr_page",
        return_value="正常文字����",
    )
    config = AppConfig(
        openai_api_key=None,
        min_page_text_chars=20,
        max_garbled_ratio=0.2,
        ocr_enabled=True,
    )

    result = parse_pdf_bytes(pdf_bytes, "policy.pdf", config)

    assert result.pages[0].extraction_method == "ocr"
    assert "severe_ocr_uncertainty" in result.pages[0].quality_notes
