from dataclasses import dataclass, field
from enum import Enum


CHUNKING_STRATEGIES = frozenset({"legacy", "clause_v2"})
PAGE_QUALITY_UNREADABLE = "unreadable_page"
PAGE_QUALITY_SEVERE_OCR_UNCERTAINTY = "severe_ocr_uncertainty"
BOUNDARY_LEGACY_PAGE_LINE_PACKING = "legacy_page_line_packing"
BOUNDARY_UNKNOWN_CLAUSE_PAGE_FALLBACK = "unknown_clause_page_fallback"
BOUNDARY_LOW_CONFIDENCE_HEADING_CANDIDATE = "low_confidence_heading_candidate"
BOUNDARY_REJECTED_PAGE_HEADER_FOOTER = "rejected_page_header_footer"
BOUNDARY_CROSS_PAGE_CLAUSE_CONTINUATION = "cross_page_clause_continuation"
BOUNDARY_CHARACTER_WINDOW_FALLBACK = "character_window_fallback"
BOUNDARY_PAGE_GAP_EMPTY = "page_gap:empty"
BOUNDARY_PAGE_GAP_UNREADABLE = "page_gap:unreadable"
BOUNDARY_PAGE_GAP_SEVERE_OCR_UNCERTAINTY = "page_gap:severe_ocr_uncertainty"


def authoritative_source_text(
    source_spans: tuple["SourceSpan", ...],
    *,
    fallback: str,
) -> str:
    if not source_spans:
        return fallback
    return "\n".join(span.text for span in source_spans)


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str
    extraction_method: str
    quality_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClauseMetadata:
    clause_id: str | None = None
    heading_text: str | None = None
    section_title: str = "未识别条款标题"
    heading_confidence: str = "low"
    heading_source: str = "fallback"


@dataclass(frozen=True)
class SourceSpan:
    page_number: int
    text: str
    start_char: int
    end_char: int


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
    clause_id: str | None = None
    heading_text: str | None = None
    heading_confidence: str = "low"
    heading_source: str = "fallback"
    retrieval_context: str = ""
    source_spans: tuple[SourceSpan, ...] = ()
    boundary_diagnostics: tuple[str, ...] = ()
    chunking_strategy: str = "legacy"

    @property
    def retrieval_text(self) -> str:
        if not self.retrieval_context:
            return self.text
        return f"{self.retrieval_context}\n{self.text}"

    @property
    def authoritative_text(self) -> str:
        return authoritative_source_text(self.source_spans, fallback=self.text)

    @property
    def authoritative_page_number(self) -> int | None:
        if not self.source_spans:
            return self.page_number
        return self.source_spans[0].page_number

    @property
    def index_compatibility_key(self) -> str:
        return f"chunking:{self.chunking_strategy}"


@dataclass(frozen=True)
class Citation:
    source_type: str
    source_name: str
    page_number: int | None
    section_title: str
    excerpt: str
    quality_notes: tuple[str, ...] = ()
    source_spans: tuple[SourceSpan, ...] = ()

    @property
    def authoritative_text(self) -> str:
        return authoritative_source_text(self.source_spans, fallback=self.excerpt)

    @property
    def page_numbers(self) -> tuple[int, ...]:
        if not self.source_spans:
            return () if self.page_number is None else (self.page_number,)
        return tuple(dict.fromkeys(span.page_number for span in self.source_spans))


class GuardStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class FactStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class FactSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    expanded_queries: tuple[str, ...]
    detected_intents: tuple[str, ...] = ()
    used_llm: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalRankDetail:
    query: str
    method: str
    rank: int
    score: float


@dataclass(frozen=True)
class RerankExplanation:
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalExplanation:
    source_type: str
    source_name: str
    page_number: int | None
    section_title: str
    final_score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    matched_terms: tuple[str, ...] = ()
    rank_details: tuple[RetrievalRankDetail, ...] = ()
    rerank_score: float | None = None
    rerank_reasons: tuple[str, ...] = ()

    @property
    def match_strength(self) -> str:
        if self.final_score >= 0.03:
            return "high"
        if self.final_score >= 0.015:
            return "medium"
        return "low"


@dataclass(frozen=True)
class VerifiedFact:
    fact_text: str
    fact_type: str
    status: FactStatus | str
    severity: FactSeverity | str
    supporting_citation_ids: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _normalize_fact_status(self.status))
        object.__setattr__(self, "severity", _normalize_fact_severity(self.severity))


def _normalize_fact_status(value: FactStatus | str) -> FactStatus:
    if isinstance(value, FactStatus):
        return value
    if isinstance(value, str):
        try:
            return FactStatus(value)
        except ValueError as exc:
            raise ValueError(f"Invalid VerifiedFact status: {value!r}") from exc
    raise ValueError(f"Invalid VerifiedFact status: {value!r}")


def _normalize_fact_severity(value: FactSeverity | str) -> FactSeverity:
    if isinstance(value, FactSeverity):
        return value
    if isinstance(value, str):
        try:
            return FactSeverity(value)
        except ValueError as exc:
            raise ValueError(f"Invalid VerifiedFact severity: {value!r}") from exc
    raise ValueError(f"Invalid VerifiedFact severity: {value!r}")


@dataclass(frozen=True)
class CitationVerificationResult:
    facts: tuple[VerifiedFact, ...] = ()
    warnings: tuple[str, ...] = ()
    block_reason: str | None = None

    @property
    def has_blocking_fact(self) -> bool:
        return any(fact.severity is FactSeverity.BLOCK for fact in self.facts)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings) or any(
            fact.severity is FactSeverity.WARN for fact in self.facts
        )


@dataclass(frozen=True)
class AnswerGuardResult:
    status: GuardStatus
    warnings: tuple[str, ...] = ()
    block_reason: str | None = None
    citation_verification: CitationVerificationResult | None = None


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    policy_citations: tuple[Citation, ...] = ()
    builtin_citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()
    retrieval_explanations: tuple[RetrievalExplanation, ...] = ()
    guard_result: AnswerGuardResult | None = None
    citation_verification: CitationVerificationResult | None = None


@dataclass(frozen=True)
class ParseResult:
    filename: str
    pages: tuple[DocumentPage, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
