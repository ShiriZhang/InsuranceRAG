from insurance_rag.config import AppConfig
from insurance_rag.document_loader import (
    PageExtraction,
    garbled_ratio,
    needs_ocr,
    normalize_page_text,
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
