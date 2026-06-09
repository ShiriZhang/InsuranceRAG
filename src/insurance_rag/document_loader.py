from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import fitz

from insurance_rag.config import AppConfig
from insurance_rag.models import DocumentPage, ParseResult


@dataclass(frozen=True)
class PageExtraction:
    page_number: int
    text: str
    extraction_method: str


def garbled_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad_chars = sum(1 for char in text if char in {"�", "\ufffd"})
    return bad_chars / len(text)


def needs_ocr(text: str, config: AppConfig) -> bool:
    cleaned = text.strip()
    if len(cleaned) < config.min_page_text_chars:
        return True
    return garbled_ratio(cleaned) > config.max_garbled_ratio


def normalize_page_text(extraction: PageExtraction) -> DocumentPage:
    text = " ".join(extraction.text.split())
    notes: list[str] = []
    if extraction.extraction_method == "ocr":
        notes.append("该页来自 OCR 识别，可能有误。")
    if not text:
        notes.append("该页未提取到可用文本。")
    return DocumentPage(
        page_number=extraction.page_number,
        text=text,
        extraction_method=extraction.extraction_method,
        quality_notes=tuple(notes),
    )


def _ocr_page(page: fitz.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OCR 依赖不可用，请安装 pytesseract 和 Pillow。") from exc

    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    return pytesseract.image_to_string(image, lang="chi_sim+eng")


def parse_pdf_bytes(pdf_bytes: bytes, filename: str, config: AppConfig) -> ParseResult:
    warnings: list[str] = []
    pages: list[DocumentPage] = []

    with NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(pdf_bytes)
        temp_file.flush()
        try:
            document = fitz.open(temp_file.name)
        except Exception as exc:
            raise ValueError("PDF 无法打开，文件可能已加密、损坏或格式不受支持。") from exc

        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            method = "text"
            if config.ocr_enabled and needs_ocr(raw_text, config):
                try:
                    raw_text = _ocr_page(page)
                    method = "ocr"
                except RuntimeError as exc:
                    warnings.append(str(exc))
            pages.append(
                normalize_page_text(
                    PageExtraction(
                        page_number=index,
                        text=raw_text,
                        extraction_method=method,
                    )
                )
            )

    return ParseResult(filename=Path(filename).name, pages=tuple(pages), warnings=tuple(warnings))
