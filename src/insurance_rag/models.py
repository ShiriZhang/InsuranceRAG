from dataclasses import dataclass, field
from enum import Enum


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


class GuardStatus(str, Enum):
    PASS = "pass"
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

    @property
    def match_strength(self) -> str:
        if self.final_score >= 0.03:
            return "high"
        if self.final_score >= 0.015:
            return "medium"
        return "low"


@dataclass(frozen=True)
class AnswerGuardResult:
    status: GuardStatus
    warnings: tuple[str, ...] = ()
    block_reason: str | None = None


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    policy_citations: tuple[Citation, ...] = ()
    builtin_citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()
    retrieval_explanations: tuple[RetrievalExplanation, ...] = ()
    guard_result: AnswerGuardResult | None = None


@dataclass(frozen=True)
class ParseResult:
    filename: str
    pages: tuple[DocumentPage, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
