import re

from insurance_rag.models import DocumentChunk, DocumentPage


KNOWN_TITLES = (
    "保险责任",
    "责任免除",
    "等待期",
    "重大疾病定义",
    "保险期间",
    "保险金额",
    "犹豫期",
    "解除合同",
    "合同解除",
    "保险费",
    "豁免保险费",
)


def _looks_like_heading(normalized_line: str, title: str) -> bool:
    if normalized_line == title:
        return True
    if not normalized_line.startswith(title):
        return False

    suffix = normalized_line[len(title) :].strip()
    return bool(suffix) and len(suffix) <= 8 and re.match(r"^[：:、\-\s（(]", suffix) is not None


def infer_section_title(text: str, current_title: str) -> str:
    first_lines = [line.strip() for line in text.splitlines()[:5] if line.strip()]
    for line in first_lines:
        normalized = re.sub(r"^[第\d一二三四五六七八九十百、\.\s条款章节]+", "", line)
        for title in KNOWN_TITLES:
            if _looks_like_heading(normalized, title):
                return title
    return current_title


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    paragraphs = [part.strip() for part in re.split(r"\r\n|\n{1,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap :]
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(
    pages: tuple[DocumentPage, ...],
    source_name: str,
    source_type: str,
    chunk_size: int,
    overlap: int,
) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    current_title = "未识别条款标题"
    for page in pages:
        if not page.text:
            continue
        for part in _split_text(page.text, chunk_size=chunk_size, overlap=overlap):
            current_title = infer_section_title(part, current_title)
            chunk_id = f"{source_type}:{source_name}:p{page.page_number}:c{len(chunks) + 1}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=part,
                    page_number=page.page_number,
                    section_title=current_title,
                    source_type=source_type,
                    source_name=source_name,
                    extraction_method=page.extraction_method,
                    quality_notes=page.quality_notes,
                )
            )
    return tuple(chunks)
