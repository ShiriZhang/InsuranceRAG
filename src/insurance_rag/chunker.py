from dataclasses import replace
import re

from insurance_rag.clause_parser import (
    KNOWN_SECTION_TITLES,
    UNKNOWN_SECTION_TITLE,
    parse_clause_metadata,
)
from insurance_rag.models import (
    BOUNDARY_CHARACTER_WINDOW_FALLBACK,
    BOUNDARY_CROSS_PAGE_CLAUSE_CONTINUATION,
    BOUNDARY_LEGACY_PAGE_LINE_PACKING,
    BOUNDARY_LOW_CONFIDENCE_HEADING_CANDIDATE,
    BOUNDARY_PAGE_GAP_EMPTY,
    BOUNDARY_PAGE_GAP_SEVERE_OCR_UNCERTAINTY,
    BOUNDARY_PAGE_GAP_UNREADABLE,
    BOUNDARY_REJECTED_PAGE_HEADER_FOOTER,
    BOUNDARY_UNKNOWN_CLAUSE_PAGE_FALLBACK,
    CHUNKING_STRATEGIES,
    ClauseMetadata,
    DocumentChunk,
    DocumentPage,
    PAGE_QUALITY_SEVERE_OCR_UNCERTAINTY,
    PAGE_QUALITY_UNREADABLE,
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
    page_start = 0
    page_end = len(text)
    if not text.strip():
        return []

    trusted_heading_starts: list[int] = []
    rejected_heading_ends: list[int] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        candidate = line.strip()
        if candidate:
            metadata = parse_clause_metadata(candidate)
            leading_space = len(line) - len(line.lstrip())
            candidate_start = cursor + leading_space
            if candidate_start in rejected_heading_offsets:
                rejected_heading_ends.append(cursor + len(raw_line))
            if (
                metadata.heading_confidence in {"high", "medium"}
                and candidate_start not in rejected_heading_offsets
            ):
                trusted_heading_starts.append(candidate_start)
        cursor += len(raw_line)

    starts: list[int] = []
    if not trusted_heading_starts:
        starts.append(page_start)
    elif text[: trusted_heading_starts[0]].strip():
        starts.append(page_start)
    else:
        trusted_heading_starts[0] = page_start
    starts.extend(trusted_heading_starts)
    starts.extend(
        offset for offset in rejected_heading_ends if page_start < offset < page_end
    )

    unique_starts = sorted(set(starts))
    clauses: list[tuple[str, int, int]] = []
    for index, start in enumerate(unique_starts):
        end = unique_starts[index + 1] if index + 1 < len(unique_starts) else page_end
        if text[start:end].strip():
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


def _unsafe_page_gap_diagnostic(page: DocumentPage) -> str | None:
    if not page.text.strip():
        return BOUNDARY_PAGE_GAP_EMPTY
    if PAGE_QUALITY_UNREADABLE in page.quality_notes:
        return BOUNDARY_PAGE_GAP_UNREADABLE
    if PAGE_QUALITY_SEVERE_OCR_UNCERTAINTY in page.quality_notes:
        return BOUNDARY_PAGE_GAP_SEVERE_OCR_UNCERTAINTY
    return None


def _split_source_span_at_semantic_boundaries(span: SourceSpan) -> list[SourceSpan]:
    boundaries = {0, len(span.text)}
    boundaries.update(
        match.end()
        for match in re.finditer(r"[。！？!?；;][ \t]*(?:\r?\n)?", span.text)
    )
    boundaries.update(
        match.start()
        for match in re.finditer(
            r"(?m)^(?:[（(]?[零〇一二三四五六七八九十百千万两\d]+[）).、．])\s*",
            span.text,
        )
        if match.start() > 0
    )

    offsets = sorted(boundaries)
    return [
        SourceSpan(
            page_number=span.page_number,
            text=span.text[start:end],
            start_char=span.start_char + start,
            end_char=span.start_char + end,
        )
        for start, end in zip(offsets, offsets[1:])
        if end > start
    ]


def _coalesce_contiguous_spans(spans: list[SourceSpan]) -> tuple[SourceSpan, ...]:
    coalesced: list[SourceSpan] = []
    for span in spans:
        if (
            coalesced
            and coalesced[-1].page_number == span.page_number
            and coalesced[-1].end_char == span.start_char
        ):
            previous = coalesced[-1]
            coalesced[-1] = SourceSpan(
                page_number=previous.page_number,
                text=previous.text + span.text,
                start_char=previous.start_char,
                end_char=span.end_char,
            )
        else:
            coalesced.append(span)
    return tuple(coalesced)


def _spans_text(spans: tuple[SourceSpan, ...] | list[SourceSpan]) -> str:
    return "\n".join(span.text for span in _coalesce_contiguous_spans(list(spans)))


def _split_clause_v2_chunk(
    chunk: DocumentChunk,
    *,
    target_chars: int,
    hard_max_chars: int,
) -> list[DocumentChunk]:
    retrieval_prefix_chars = (
        len(chunk.retrieval_context) + 1 if chunk.retrieval_context else 0
    )
    body_hard_max_chars = hard_max_chars - retrieval_prefix_chars
    if body_hard_max_chars <= 0:
        raise ValueError(
            "hard_max_chars must exceed clause heading retrieval context length"
        )
    body_target_chars = max(1, target_chars - retrieval_prefix_chars)

    semantic_units: list[tuple[list[SourceSpan], bool]] = []
    for source_span in chunk.source_spans:
        for semantic_span in _split_source_span_at_semantic_boundaries(source_span):
            if len(semantic_span.text.strip()) <= body_hard_max_chars:
                semantic_units.append(([semantic_span], False))
                continue
            for offset in range(0, len(semantic_span.text), body_hard_max_chars):
                window_text = semantic_span.text[
                    offset : offset + body_hard_max_chars
                ]
                semantic_units.append(
                    (
                        [
                            SourceSpan(
                                page_number=semantic_span.page_number,
                                text=window_text,
                                start_char=semantic_span.start_char + offset,
                                end_char=semantic_span.start_char
                                + offset
                                + len(window_text),
                            )
                        ],
                        True,
                    )
                )

    packed: list[tuple[list[SourceSpan], bool]] = []
    current_spans: list[SourceSpan] = []
    current_uses_window = False
    for unit_spans, uses_window in semantic_units:
        candidate_spans = current_spans + unit_spans
        candidate_length = len(_spans_text(candidate_spans).strip())
        if current_spans and candidate_length > body_target_chars:
            current_length = len(_spans_text(current_spans).strip())
            current_is_bare_heading = (
                chunk.heading_text is not None
                and _spans_text(current_spans).strip() == chunk.heading_text.strip()
            )
            candidate_is_closer = (
                candidate_length <= body_hard_max_chars
                and (
                    current_is_bare_heading
                    or candidate_length - body_target_chars
                    < body_target_chars - current_length
                )
            )
            if candidate_is_closer:
                current_spans = candidate_spans
                current_uses_window = current_uses_window or uses_window
            else:
                packed.append((current_spans, current_uses_window))
                current_spans = list(unit_spans)
                current_uses_window = uses_window
        else:
            current_spans = candidate_spans
            current_uses_window = current_uses_window or uses_window
    if current_spans:
        packed.append((current_spans, current_uses_window))

    split_chunks: list[DocumentChunk] = []
    for spans, uses_window in packed:
        coalesced_spans = _coalesce_contiguous_spans(spans)
        diagnostics = chunk.boundary_diagnostics
        if uses_window:
            diagnostics = tuple(
                dict.fromkeys(diagnostics + (BOUNDARY_CHARACTER_WINDOW_FALLBACK,))
            )
        split_chunks.append(
            replace(
                chunk,
                text=_spans_text(coalesced_spans).strip(),
                page_number=coalesced_spans[0].page_number,
                source_spans=coalesced_spans,
                boundary_diagnostics=diagnostics,
            )
        )
    return split_chunks


def _chunk_legacy_pages(
    pages: tuple[DocumentPage, ...],
    *,
    source_name: str,
    source_type: str,
    chunk_size: int,
    overlap: int,
) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    current_title = UNKNOWN_SECTION_TITLE
    for page in pages:
        if not page.text:
            continue
        for part in _split_text(page.text, chunk_size=chunk_size, overlap=overlap):
            metadata = parse_clause_metadata(part, current_title=current_title)
            current_title = metadata.section_title
            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{source_type}:{source_name}:"
                        f"p{page.page_number}:c{len(chunks) + 1}"
                    ),
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
                    boundary_diagnostics=(BOUNDARY_LEGACY_PAGE_LINE_PACKING,),
                )
            )
    return tuple(chunks)


def _chunk_clause_v2_pages(
    pages: tuple[DocumentPage, ...],
    *,
    source_name: str,
    source_type: str,
    target_chars: int,
    hard_max_chars: int,
) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    current_title = UNKNOWN_SECTION_TITLE
    active_clause_chunk_index: int | None = None
    pending_gap_diagnostics: tuple[str, ...] = ()
    rejected_offsets_by_page = _rejected_page_edge_medium_offsets(pages)

    for page_index, page in enumerate(pages):
        gap_diagnostic = _unsafe_page_gap_diagnostic(page)
        if gap_diagnostic is not None:
            active_clause_chunk_index = None
            current_title = UNKNOWN_SECTION_TITLE
            pending_gap_diagnostics = tuple(
                dict.fromkeys(pending_gap_diagnostics + (gap_diagnostic,))
            )
            continue

        rejected_heading_offsets = rejected_offsets_by_page.get(page_index, set())
        parts = _split_single_page_clauses(page.text, rejected_heading_offsets)
        trusted_heading_starts = [
            source_start
            for part, source_start, _ in parts
            if parse_clause_metadata(part).heading_confidence in {"high", "medium"}
            and source_start not in rejected_heading_offsets
        ]
        first_trusted_heading_start = min(trusted_heading_starts, default=None)

        for part, source_start, source_end in parts:
            metadata = parse_clause_metadata(part, current_title=current_title)
            rejected_in_part = any(
                source_start <= offset < source_end
                for offset in rejected_heading_offsets
            )
            if rejected_in_part and source_start in rejected_heading_offsets:
                metadata = ClauseMetadata(
                    section_title=current_title,
                    heading_confidence="low",
                    heading_source="page_header_footer",
                )
            current_title = metadata.section_title

            if metadata.heading_confidence in {"high", "medium"}:
                retrieval_context = f"Policy Clause: {metadata.heading_text}"
                boundary_diagnostics = (
                    f"trusted_heading:{metadata.heading_confidence}:{metadata.heading_source}",
                )
            else:
                retrieval_context = ""
                boundary_diagnostics = (BOUNDARY_UNKNOWN_CLAUSE_PAGE_FALLBACK,)
            if _has_low_confidence_heading_candidate(part):
                boundary_diagnostics += (BOUNDARY_LOW_CONFIDENCE_HEADING_CANDIDATE,)
            if rejected_in_part:
                boundary_diagnostics += (BOUNDARY_REJECTED_PAGE_HEADER_FOOTER,)
            boundary_diagnostics += pending_gap_diagnostics

            chunk = DocumentChunk(
                chunk_id=(
                    f"{source_type}:{source_name}:clause_v2:"
                    f"p{page.page_number}:c{len(chunks) + 1}"
                ),
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
                    SourceSpan(
                        page_number=page.page_number,
                        text=part,
                        start_char=source_start,
                        end_char=source_end,
                    ),
                ),
                boundary_diagnostics=boundary_diagnostics,
                chunking_strategy="clause_v2",
            )
            continues_active_clause = (
                metadata.heading_confidence == "low"
                and not rejected_in_part
                and (
                    first_trusted_heading_start is None
                    or source_end <= first_trusted_heading_start
                )
                and active_clause_chunk_index is not None
            )
            if continues_active_clause:
                active_chunk = chunks[active_clause_chunk_index]
                chunks[active_clause_chunk_index] = replace(
                    active_chunk,
                    text=f"{active_chunk.text}\n{chunk.text}",
                    quality_notes=tuple(
                        dict.fromkeys(active_chunk.quality_notes + chunk.quality_notes)
                    ),
                    source_spans=active_chunk.source_spans + chunk.source_spans,
                    boundary_diagnostics=tuple(
                        dict.fromkeys(
                            active_chunk.boundary_diagnostics
                            + (BOUNDARY_CROSS_PAGE_CLAUSE_CONTINUATION,)
                        )
                    ),
                )
                continue

            chunks.append(chunk)
            if metadata.heading_confidence in {"high", "medium"}:
                active_clause_chunk_index = len(chunks) - 1
            pending_gap_diagnostics = ()

    if pending_gap_diagnostics and chunks:
        last_chunk = chunks[-1]
        chunks[-1] = replace(
            last_chunk,
            boundary_diagnostics=tuple(
                dict.fromkeys(
                    last_chunk.boundary_diagnostics + pending_gap_diagnostics
                )
            ),
        )

    split_chunks = [
        split_chunk
        for chunk in chunks
        for split_chunk in _split_clause_v2_chunk(
            chunk,
            target_chars=target_chars,
            hard_max_chars=hard_max_chars,
        )
    ]
    return tuple(
        replace(
            chunk,
            chunk_id=(
                f"{source_type}:{source_name}:clause_v2:"
                f"p{chunk.page_number}:c{index}"
            ),
        )
        for index, chunk in enumerate(split_chunks, start=1)
    )


def chunk_pages(
    pages: tuple[DocumentPage, ...],
    source_name: str,
    source_type: str,
    chunk_size: int,
    overlap: int,
    strategy: str = "legacy",
    target_chars: int | None = None,
    hard_max_chars: int | None = None,
) -> tuple[DocumentChunk, ...]:
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {strategy!r}")
    if strategy == "legacy":
        return _chunk_legacy_pages(
            pages,
            source_name=source_name,
            source_type=source_type,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    resolved_target_chars = chunk_size if target_chars is None else target_chars
    resolved_hard_max_chars = chunk_size if hard_max_chars is None else hard_max_chars
    if resolved_target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if resolved_hard_max_chars < resolved_target_chars:
        raise ValueError("hard_max_chars must be at least target_chars")
    return _chunk_clause_v2_pages(
        pages,
        source_name=source_name,
        source_type=source_type,
        target_chars=resolved_target_chars,
        hard_max_chars=resolved_hard_max_chars,
    )
