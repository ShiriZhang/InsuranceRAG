import re

from insurance_rag.clause_parser import (
    KNOWN_SECTION_TITLES,
    UNKNOWN_SECTION_TITLE,
    parse_clause_metadata,
)
from insurance_rag.models import (
    CHUNKING_STRATEGIES,
    ClauseMetadata,
    DocumentChunk,
    DocumentPage,
    SourceSpan,
)


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
    candidate_lines = [line.strip() for line in text.splitlines()[:80] if line.strip()]
    for line in candidate_lines:
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


def _split_single_page_clauses(
    text: str,
    rejected_heading_offsets: set[int],
) -> list[tuple[str, int, int]]:
    page_start = len(text) - len(text.lstrip())
    page_end = len(text.rstrip())
    if page_start >= page_end:
        return []

    trusted_heading_starts: list[int] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        candidate = line.strip()
        if candidate:
            metadata = parse_clause_metadata(candidate)
            leading_space = len(line) - len(line.lstrip())
            candidate_start = cursor + leading_space
            if (
                metadata.heading_confidence in {"high", "medium"}
                and candidate_start not in rejected_heading_offsets
            ):
                trusted_heading_starts.append(candidate_start)
        cursor += len(raw_line)

    starts: list[int] = []
    if not trusted_heading_starts or page_start < trusted_heading_starts[0]:
        starts.append(page_start)
    starts.extend(trusted_heading_starts)

    unique_starts = list(dict.fromkeys(starts))
    clauses: list[tuple[str, int, int]] = []
    for index, start in enumerate(unique_starts):
        end = unique_starts[index + 1] if index + 1 < len(unique_starts) else page_end
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            clauses.append((text[start:end], start, end))
    return clauses


def _has_low_confidence_heading_candidate(text: str) -> bool:
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        metadata = parse_clause_metadata(line)
        if metadata.heading_confidence != "low":
            continue
        if any(title in line for title in KNOWN_SECTION_TITLES):
            return True
    return False


def _rejected_page_edge_medium_offsets(
    pages: tuple[DocumentPage, ...],
) -> dict[int, set[int]]:
    occurrences: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for page_index, page in enumerate(pages):
        lines: list[tuple[str, int]] = []
        cursor = 0
        for raw_line in page.text.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            candidate = line.strip()
            if candidate:
                leading_space = len(line) - len(line.lstrip())
                lines.append((candidate, cursor + leading_space))
            cursor += len(raw_line)
        if not lines:
            continue

        edge_lines = (("header", lines[0]), ("footer", lines[-1]))
        for edge, (candidate, offset) in edge_lines:
            metadata = parse_clause_metadata(candidate)
            if metadata.heading_confidence == "medium":
                occurrences.setdefault((edge, candidate), []).append(
                    (page_index, offset)
                )

    rejected: dict[int, set[int]] = {}
    for repeated in occurrences.values():
        if len(repeated) < 2:
            continue
        for page_index, offset in repeated:
            rejected.setdefault(page_index, set()).add(offset)
    return rejected


def chunk_pages(
    pages: tuple[DocumentPage, ...],
    source_name: str,
    source_type: str,
    chunk_size: int,
    overlap: int,
    strategy: str = "legacy",
) -> tuple[DocumentChunk, ...]:
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {strategy!r}")

    chunks: list[DocumentChunk] = []
    current_title = UNKNOWN_SECTION_TITLE
    rejected_offsets_by_page = (
        _rejected_page_edge_medium_offsets(pages)
        if strategy == "clause_v2"
        else {}
    )
    for page_index, page in enumerate(pages):
        if not page.text:
            continue
        rejected_heading_offsets = rejected_offsets_by_page.get(page_index, set())
        if strategy == "clause_v2":
            parts = _split_single_page_clauses(
                page.text,
                rejected_heading_offsets,
            )
        else:
            parts = [
                (part, None, None)
                for part in _split_text(
                    page.text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            ]
        for part, source_start, source_end in parts:
            metadata = parse_clause_metadata(part, current_title=current_title)
            rejected_in_part = (
                source_start is not None
                and source_end is not None
                and any(
                    source_start <= offset < source_end
                    for offset in rejected_heading_offsets
                )
            )
            if rejected_in_part and source_start in rejected_heading_offsets:
                metadata = ClauseMetadata(
                    section_title=current_title,
                    heading_confidence="low",
                    heading_source="page_header_footer",
                )
            current_title = metadata.section_title
            retrieval_context = ""
            boundary_diagnostics = ("legacy_page_line_packing",)
            if strategy == "clause_v2":
                if metadata.heading_confidence in {"high", "medium"}:
                    retrieval_context = f"Policy Clause: {metadata.heading_text}"
                    boundary_diagnostics = (
                        f"trusted_heading:{metadata.heading_confidence}:{metadata.heading_source}",
                    )
                else:
                    boundary_diagnostics = ("unknown_clause_page_fallback",)
                if _has_low_confidence_heading_candidate(part):
                    boundary_diagnostics += ("low_confidence_heading_candidate",)
                if rejected_in_part:
                    boundary_diagnostics += ("rejected_page_header_footer",)
            strategy_id = "" if strategy == "legacy" else f":{strategy}"
            chunk_id = (
                f"{source_type}:{source_name}{strategy_id}:"
                f"p{page.page_number}:c{len(chunks) + 1}"
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=part,
                    page_number=page.page_number,
                    section_title=metadata.section_title,
                    source_type=source_type,
                    source_name=source_name,
                    extraction_method=page.extraction_method,
                    quality_notes=page.quality_notes,
                    clause_id=metadata.clause_id,
                    heading_text=metadata.heading_text,
                    heading_confidence=metadata.heading_confidence,
                    heading_source=metadata.heading_source,
                    retrieval_context=retrieval_context,
                    source_spans=(
                        (
                            SourceSpan(
                                page_number=page.page_number,
                                text=part,
                                start_char=source_start,
                                end_char=source_end,
                            ),
                        )
                        if source_start is not None and source_end is not None
                        else ()
                    ),
                    boundary_diagnostics=boundary_diagnostics,
                    chunking_strategy=strategy,
                )
            )
    return tuple(chunks)
