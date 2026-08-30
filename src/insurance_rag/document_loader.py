from dataclasses import dataclass, replace
from pathlib import Path

import fitz

from insurance_rag.config import AppConfig
from insurance_rag.models import (
    DocumentPage,
    PAGE_QUALITY_SEVERE_OCR_UNCERTAINTY,
    PAGE_QUALITY_UNREADABLE,
    ParseResult,
)


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
    lines = [
        " ".join(line.split())
        for line in extraction.text.splitlines()
        if line.strip()
    ]
    text = "\n".join(lines)
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
    warning_keys: set[str] = set()
    pages: list[DocumentPage] = []

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF 无法打开，文件可能已加密、损坏或格式不受支持。") from exc

    with document:
        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            method = "text"
            if config.ocr_enabled and needs_ocr(raw_text, config):
                try:
                    raw_text = _ocr_page(page)
                    method = "ocr"
                except Exception as exc:
                    warning_key = f"ocr_failed:{type(exc).__name__}:{exc}"
                    if warning_key not in warning_keys:
                        warnings.append(
                            f"第 {index} 页 OCR 运行失败，Tesseract 或 OCR 语言数据可能不可用；"
                            "已保留原始文本提取结果。"
                        )
                        warning_keys.add(warning_key)
            normalized_page = normalize_page_text(
                PageExtraction(
                    page_number=index,
                    text=raw_text,
                    extraction_method=method,
                )
            )
            final_text_is_severely_garbled = (
                garbled_ratio(normalized_page.text) > config.max_garbled_ratio
            )
            if method == "ocr" and final_text_is_severely_garbled:
                normalized_page = replace(
                    normalized_page,
                    quality_notes=normalized_page.quality_notes
                    + (PAGE_QUALITY_SEVERE_OCR_UNCERTAINTY,),
                )
            elif final_text_is_severely_garbled:
                normalized_page = replace(
                    normalized_page,
                    quality_notes=normalized_page.quality_notes
                    + (PAGE_QUALITY_UNREADABLE,),
                )
            pages.append(normalized_page)

    return ParseResult(filename=Path(filename).name, pages=tuple(pages), warnings=tuple(warnings))
