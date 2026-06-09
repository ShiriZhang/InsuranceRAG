from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str
    extraction_method: str
    quality_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    page_number: int | None
    section_title: str
    source_type: str
    source_name: str
    extraction_method: str
    quality_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    source_type: str
    source_name: str
    page_number: int | None
    section_title: str
    excerpt: str
    quality_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    policy_citations: tuple[Citation, ...] = ()
    builtin_citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParseResult:
    filename: str
    pages: tuple[DocumentPage, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
