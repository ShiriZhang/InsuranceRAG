from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from insurance_rag.chunker import chunk_pages
from insurance_rag.citation_verifier import verify_answer_facts
from insurance_rag.config import AppConfig
from insurance_rag.document_loader import parse_pdf_bytes
from insurance_rag.hybrid_retriever import HybridRetriever
from insurance_rag.models import Citation, DocumentChunk
from insurance_rag.query_rewriter import rewrite_query
from insurance_rag.retriever import InMemoryVectorIndex
from insurance_rag.rule_reranker import rerank_results


UNKNOWN_SECTION_TITLE = "未识别条款标题"
LOCAL_EVAL_QUERIES: tuple[tuple[str, str], ...] = (
    ("等待期", "等待期"),
    ("责任免除", "责任免除"),
    ("保险责任", "保险责任"),
    ("保险期间", "保险期间"),
    ("保险金额", "保险金额"),
    ("豁免保险费", "豁免保险费"),
)
LOCAL_HARD_NEGATIVE_QUERY_PAIRS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("等待期", "等待期是多久？", ("保险期间", "犹豫期", "宽限期")),
    ("责任免除", "哪些情况不赔？", ("保险责任",)),
    ("保险责任", "保障哪些内容？", ("责任免除",)),
    ("豁免保险费", "豁免保险费适用于谁？", ("保险费",)),
)


class DeterministicEvalEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_sha256_vector(text) for text in texts]


class Bm25OnlyEvalEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


@dataclass(frozen=True)
class EvalRetrievedItem:
    rank: int
    chunk_id: str
    section_title: str
    final_score: float
    vector_score: float | None
    bm25_score: float | None
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    question: str
    expected_section: str
    expected_terms: tuple[str, ...]
    retrieved_sections: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    expected_rank: int | None
    max_expected_rank: int
    top_fusion_score: float
    retrieved_items: tuple[EvalRetrievedItem, ...]
    passed: bool


@dataclass(frozen=True)
class EvalReport:
    total_cases: int
    passed_cases: int
    results: tuple[EvalCaseResult, ...]


@dataclass(frozen=True)
class HardNegativeCaseResult:
    case_id: str
    question: str
    expected_positive_chunk_id: str
    positive_rank: int | None
    max_expected_rank: int
    retrieved_chunk_ids: tuple[str, ...]
    rerank_details: tuple[str, ...]
    verifier_status: str
    passed: bool


@dataclass(frozen=True)
class HardNegativeEvalReport:
    total_cases: int
    passed_cases: int
    results: tuple[HardNegativeCaseResult, ...]


@dataclass(frozen=True)
class LocalDocumentEvalCase:
    document_name: str
    expected_term: str
    question: str
    expected_rank: int | None
    retrieved_sections: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_items: tuple[EvalRetrievedItem, ...]

    @property
    def passed_top1(self) -> bool:
        return self.expected_rank == 1

    @property
    def passed_top3(self) -> bool:
        return self.expected_rank is not None and self.expected_rank <= 3


@dataclass(frozen=True)
class LocalDocumentEvalReport:
    total_documents: int
    sampled_documents: int
    parsed_documents: int
    parse_errors: tuple[str, ...]
    total_pages: int
    total_chunks: int
    empty_pages: int
    unknown_chunks: int
    cases: tuple[LocalDocumentEvalCase, ...]

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def top1_cases(self) -> int:
        return sum(1 for case in self.cases if case.passed_top1)

    @property
    def top3_cases(self) -> int:
        return sum(1 for case in self.cases if case.passed_top3)

    @property
    def empty_page_rate(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return self.empty_pages / self.total_pages

    @property
    def unknown_chunk_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.unknown_chunks / self.total_chunks


def evaluate_synthetic_cases(
    path: Path,
    *,
    top_k: int = 3,
    max_expected_rank: int = 1,
) -> EvalReport:
    if max_expected_rank <= 0:
        raise ValueError("max_expected_rank must be at least one.")
    cases = _load_cases(path)
    embedder = DeterministicEvalEmbedder()
    results: list[EvalCaseResult] = []

    for index, case in enumerate(cases, start=1):
        _validate_case(case, index)
        chunks = _chunks_for_case(case)
        embeddings = embedder.embed_texts([chunk.retrieval_text for chunk in chunks])
        vector_index = InMemoryVectorIndex.from_embeddings(chunks, embeddings)
        retriever = HybridRetriever(
            chunks,
            vector_index,
            embedder,
            retrieval_mode="hybrid",
        )
        retrieved = retriever.search(rewrite_query(str(case["question"])), top_k=top_k)
        retrieved_sections = tuple(result.chunk.section_title for result in retrieved)
        retrieved_chunk_ids = tuple(result.chunk.chunk_id for result in retrieved)
        retrieved_items = _retrieved_items(retrieved)
        expected_section = str(case["expected_section"])
        expected_terms = tuple(str(term) for term in case.get("expected_terms", ()))
        expected_rank = _first_expected_rank(
            retrieved,
            expected_section=expected_section,
            expected_terms=expected_terms,
        )
        passed = expected_rank is not None and expected_rank <= max_expected_rank

        results.append(
            EvalCaseResult(
                case_id=_case_id(case),
                question=str(case["question"]),
                expected_section=expected_section,
                expected_terms=expected_terms,
                retrieved_sections=retrieved_sections,
                retrieved_chunk_ids=retrieved_chunk_ids,
                expected_rank=expected_rank,
                max_expected_rank=max_expected_rank,
                top_fusion_score=retrieved[0].final_score if retrieved else 0.0,
                retrieved_items=retrieved_items,
                passed=passed,
            )
        )

    return EvalReport(
        total_cases=len(results),
        passed_cases=sum(1 for result in results if result.passed),
        results=tuple(results),
    )


def evaluate_hard_negative_cases(path: Path, top_k: int = 3) -> HardNegativeEvalReport:
    if top_k <= 0:
        raise ValueError("top_k must be at least one.")

    raw_cases = _load_cases(path)
    embedder = DeterministicEvalEmbedder()
    results: list[HardNegativeCaseResult] = []

    for index, case in enumerate(raw_cases, start=1):
        _validate_hard_negative_case(case, index)
        chunks = _hard_negative_chunks_for_case(case)
        embeddings = embedder.embed_texts([chunk.retrieval_text for chunk in chunks])
        vector_index = InMemoryVectorIndex.from_embeddings(chunks, embeddings)
        retriever = HybridRetriever(
            chunks,
            vector_index,
            embedder,
            retrieval_mode="hybrid",
        )
        rewrite = rewrite_query(str(case["question"]))
        initial = retriever.search(rewrite, top_k=max(top_k, len(chunks)))
        retrieved = rerank_results(
            question=str(case["question"]),
            rewrite=rewrite,
            candidates=initial,
            top_k=top_k,
        )

        positive_id = str(case["expected_positive_chunk_id"])
        positive_rank = _first_chunk_id_rank(retrieved, positive_id)
        citations = tuple(build_eval_citation(result.chunk) for result in retrieved)
        verification = verify_answer_facts(
            answer=str(case.get("answer", "")),
            policy_citations=tuple(
                citation
                for citation in citations
                if citation.source_type != "built_in_dataset"
            ),
            builtin_citations=tuple(
                citation
                for citation in citations
                if citation.source_type == "built_in_dataset"
            ),
        )
        max_rank = int(case.get("max_expected_rank", 1))
        verifier_status = (
            "block"
            if verification.has_blocking_fact
            else "warn"
            if verification.has_warnings
            else "pass"
        )
        passed = positive_rank is not None and positive_rank <= max_rank
        if (
            case.get("answer")
            and verification.has_blocking_fact
            and "source_confusion" not in _case_id(case)
        ):
            passed = False

        results.append(
            HardNegativeCaseResult(
                case_id=_case_id(case),
                question=str(case["question"]),
                expected_positive_chunk_id=positive_id,
                positive_rank=positive_rank,
                max_expected_rank=max_rank,
                retrieved_chunk_ids=tuple(result.chunk.chunk_id for result in retrieved),
                rerank_details=tuple(
                    ",".join(result.rerank_reasons) for result in retrieved
                ),
                verifier_status=verifier_status,
                passed=passed,
            )
        )

    return HardNegativeEvalReport(
        total_cases=len(results),
        passed_cases=sum(1 for result in results if result.passed),
        results=tuple(results),
    )


def render_markdown_report(report: EvalReport) -> str:
    lines = [
        "# InsuranceRAG Evaluation Report",
        "",
        f"Passed {report.passed_cases} / {report.total_cases}",
        "",
        "| Case | Expected Section | Expected Rank | Retrieved Sections | Retrieved Chunks | Fusion Score | PASS/FAIL |",
        "| --- | --- | ---: | --- | --- | ---: | --- |",
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        details = "; ".join(
            _format_retrieved_item(item) for item in result.retrieved_items
        )
        lines.append(
            "| {case_id} | {expected_section} | {rank} | {sections} | {chunks} | {score:.6f} | {status} |".format(
                case_id=result.case_id,
                expected_section=result.expected_section,
                rank=result.expected_rank if result.expected_rank is not None else "not found",
                sections=", ".join(result.retrieved_sections),
                chunks=", ".join(result.retrieved_chunk_ids),
                score=result.top_fusion_score,
                status=status,
            )
        )
        if details:
            lines.append(f"  - Details: {details}")
    lines.append("")
    return "\n".join(lines)


def render_hard_negative_markdown_report(report: HardNegativeEvalReport) -> str:
    lines = [
        "# InsuranceRAG Hard Negative Evaluation Report",
        "",
        f"Passed {report.passed_cases} / {report.total_cases}",
        "",
        "| Case | Positive Rank | Max Rank | Retrieved Chunks | Rerank Details | Verifier | PASS/FAIL |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            "| {case_id} | {rank} | {max_rank} | {chunks} | {rerank} | {verifier} | {status} |".format(
                case_id=result.case_id,
                rank=(
                    result.positive_rank
                    if result.positive_rank is not None
                    else "not found"
                ),
                max_rank=result.max_expected_rank,
                chunks=", ".join(result.retrieved_chunk_ids),
                rerank="; ".join(result.rerank_details) or "-",
                verifier=result.verifier_status,
                status=status,
            )
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_local_documents(
    documents_dir: Path,
    *,
    sample_limit: int = 20,
    top_k: int = 3,
    config: AppConfig | None = None,
) -> LocalDocumentEvalReport:
    return _evaluate_local_documents(
        documents_dir,
        query_specs=tuple(
            (expected_term, question, ()) for expected_term, question in LOCAL_EVAL_QUERIES
        ),
        sample_limit=sample_limit,
        top_k=top_k,
        config=config,
    )


def evaluate_local_hard_negative_documents(
    documents_dir: Path,
    *,
    sample_limit: int = 20,
    top_k: int = 3,
    config: AppConfig | None = None,
) -> LocalDocumentEvalReport:
    return _evaluate_local_documents(
        documents_dir,
        query_specs=LOCAL_HARD_NEGATIVE_QUERY_PAIRS,
        sample_limit=sample_limit,
        top_k=top_k,
        config=config,
    )


def _evaluate_local_documents(
    documents_dir: Path,
    *,
    query_specs: tuple[tuple[str, str, tuple[str, ...]], ...],
    sample_limit: int,
    top_k: int,
    config: AppConfig | None,
) -> LocalDocumentEvalReport:
    if sample_limit <= 0:
        raise ValueError("sample_limit must be at least one.")
    if top_k <= 0:
        raise ValueError("top_k must be at least one.")

    pdfs = sorted(documents_dir.rglob("*.pdf"))
    sampled = _sample_evenly(pdfs, sample_limit)
    eval_config = config or AppConfig(openai_api_key=None, ocr_enabled=False)
    embedder = Bm25OnlyEvalEmbedder()

    parsed_documents = 0
    parse_errors: list[str] = []
    total_pages = 0
    total_chunks = 0
    empty_pages = 0
    unknown_chunks = 0
    cases: list[LocalDocumentEvalCase] = []

    for pdf in sampled:
        try:
            parsed = parse_pdf_bytes(pdf.read_bytes(), pdf.name, eval_config)
            chunks = chunk_pages(
                parsed.pages,
                source_name=pdf.name,
                source_type="local_eval",
                chunk_size=eval_config.chunk_size,
                overlap=eval_config.chunk_overlap,
                strategy=eval_config.chunking_strategy,
                target_chars=eval_config.chunk_target_chars,
                hard_max_chars=eval_config.chunk_hard_max_chars,
            )
        except Exception as exc:
            parse_errors.append(f"{pdf.name}: {type(exc).__name__}: {exc}")
            continue

        parsed_documents += 1
        total_pages += len(parsed.pages)
        total_chunks += len(chunks)
        empty_pages += sum(1 for page in parsed.pages if not page.text.strip())
        unknown_chunks += sum(
            1 for chunk in chunks if chunk.section_title == UNKNOWN_SECTION_TITLE
        )
        if not chunks:
            continue

        vector_index = InMemoryVectorIndex.from_embeddings(
            chunks,
            [[0.0] * 8 for _chunk in chunks],
        )
        retriever = HybridRetriever(
            chunks,
            vector_index,
            embedder,
            retrieval_mode="hybrid",
        )
        document_text = _compact_text("\n".join(chunk.text for chunk in chunks))
        for expected_term, question, negative_terms in query_specs:
            if _compact_text(expected_term) not in document_text:
                continue
            if negative_terms and not any(
                _compact_text(term) in document_text for term in negative_terms
            ):
                continue
            retrieved = retriever.search(rewrite_query(question), top_k=top_k)
            expected_rank = _first_term_rank(retrieved, expected_term)
            cases.append(
                LocalDocumentEvalCase(
                    document_name=pdf.name,
                    expected_term=expected_term,
                    question=question,
                    expected_rank=expected_rank,
                    retrieved_sections=tuple(
                        result.chunk.section_title for result in retrieved
                    ),
                    retrieved_chunk_ids=tuple(
                        result.chunk.chunk_id for result in retrieved
                    ),
                    retrieved_items=_retrieved_items(retrieved),
                )
            )

    return LocalDocumentEvalReport(
        total_documents=len(pdfs),
        sampled_documents=len(sampled),
        parsed_documents=parsed_documents,
        parse_errors=tuple(parse_errors),
        total_pages=total_pages,
        total_chunks=total_chunks,
        empty_pages=empty_pages,
        unknown_chunks=unknown_chunks,
        cases=tuple(cases),
    )


def render_local_markdown_report(report: LocalDocumentEvalReport) -> str:
    lines = [
        "# InsuranceRAG Local Document Evaluation Report",
        "",
        f"Documents: parsed {report.parsed_documents}/{report.sampled_documents} sampled from {report.total_documents}",
        f"Pages: {report.total_pages}",
        f"Chunks: {report.total_chunks}",
        f"Empty page rate: {report.empty_page_rate:.4f}",
        f"Unknown chunk title rate: {report.unknown_chunk_rate:.4f}",
        f"Retrieval Top1: {report.top1_cases}/{report.total_cases}",
        f"Retrieval Top3: {report.top3_cases}/{report.total_cases}",
        "",
        "| Document | Term | Expected Rank | Retrieved Sections | Retrieved Chunks | Scores / Matched Terms | Top3 |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for case in report.cases:
        details = "; ".join(_format_retrieved_item(item) for item in case.retrieved_items)
        lines.append(
            "| {document} | {term} | {rank} | {sections} | {chunks} | {details} | {status} |".format(
                document=case.document_name,
                term=case.expected_term,
                rank=case.expected_rank if case.expected_rank is not None else "not found",
                sections=", ".join(case.retrieved_sections),
                chunks=", ".join(case.retrieved_chunk_ids),
                details=details,
                status="PASS" if case.passed_top3 else "FAIL",
            )
        )
    if report.parse_errors:
        lines.extend(("", "## Parse Errors", ""))
        lines.extend(f"- {error}" for error in report.parse_errors)
    lines.append("")
    return "\n".join(lines)


def _sha256_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [
        int.from_bytes(digest[index : index + 4], "big") / 0xFFFFFFFF
        for index in range(0, 32, 4)
    ]


def _retrieved_items(retrieved: list[Any]) -> tuple[EvalRetrievedItem, ...]:
    return tuple(
        EvalRetrievedItem(
            rank=rank,
            chunk_id=result.chunk.chunk_id,
            section_title=result.chunk.section_title,
            final_score=result.final_score,
            vector_score=result.vector_score,
            bm25_score=result.bm25_score,
            matched_terms=result.matched_terms,
        )
        for rank, result in enumerate(retrieved, start=1)
    )


def _format_retrieved_item(item: EvalRetrievedItem) -> str:
    matched = ",".join(item.matched_terms) if item.matched_terms else "-"
    vector = "-" if item.vector_score is None else f"{item.vector_score:.4f}"
    bm25 = "-" if item.bm25_score is None else f"{item.bm25_score:.4f}"
    return (
        f"#{item.rank} {item.section_title} "
        f"fusion={item.final_score:.6f} vector={vector} bm25={bm25} matched={matched}"
    )


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Synthetic evaluation cases must be a JSON list.")
    if not data:
        raise ValueError("Synthetic evaluation requires at least one case.")
    return data


def _validate_case(case: Any, index: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"Synthetic case at index {index} must be an object.")

    label = _case_label(case, index)
    for field in ("question", "expected_section"):
        if not isinstance(case.get(field), str) or not case[field].strip():
            raise ValueError(f"Synthetic case {label} is missing invalid field: {field}.")

    if "case_id" not in case and "id" not in case:
        raise ValueError(f"Synthetic case at index {index} is missing invalid field: id.")

    chunks = case.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Synthetic case {label} is missing invalid field: chunks.")

    expected_terms = case.get("expected_terms", ())
    if not isinstance(expected_terms, (list, tuple)):
        raise ValueError(
            f"Synthetic case {label} is missing invalid field: expected_terms."
        )

    for chunk_index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Synthetic case {label} chunk {chunk_index} must be an object."
            )
        for field in ("chunk_id", "text", "section_title"):
            if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                raise ValueError(
                    f"Synthetic case {label} chunk {chunk_index} is missing invalid field: {field}."
                )


def _first_expected_rank(
    retrieved: list[Any],
    *,
    expected_section: str,
    expected_terms: tuple[str, ...],
) -> int | None:
    for rank, result in enumerate(retrieved, start=1):
        section_title = result.chunk.section_title
        if section_title == expected_section:
            return rank
        if any(term and term in section_title for term in expected_terms):
            return rank
    return None


def _first_term_rank(retrieved: list[Any], expected_term: str) -> int | None:
    compact_term = _compact_text(expected_term)
    for rank, result in enumerate(retrieved, start=1):
        candidate = _compact_text(
            f"{result.chunk.section_title}\n{result.chunk.text}"
        )
        if compact_term in candidate:
            return rank
    return None


def _first_chunk_id_rank(retrieved: list[Any], chunk_id: str) -> int | None:
    for rank, result in enumerate(retrieved, start=1):
        if result.chunk.chunk_id == chunk_id:
            return rank
    return None


def _compact_text(text: str) -> str:
    return "".join(text.split())


def _sample_evenly(items: list[Path], limit: int) -> list[Path]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def _case_id(case: dict[str, Any]) -> str:
    return str(case.get("case_id", case.get("id")))


def _case_label(case: dict[str, Any], index: int) -> str:
    if "case_id" in case or "id" in case:
        return _case_id(case)
    return f"at index {index}"


def build_eval_citation(chunk: DocumentChunk) -> Citation:
    return Citation(
        source_type=chunk.source_type,
        source_name=chunk.source_name,
        page_number=chunk.authoritative_page_number,
        section_title=chunk.section_title,
        excerpt=chunk.authoritative_text,
    )


def _validate_hard_negative_case(case: Any, index: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"Hard negative case at index {index} must be an object.")

    label = _case_label(case, index)
    for field in (
        "question",
        "expected_positive_chunk_id",
        "max_expected_rank",
        "chunks",
    ):
        if field not in case:
            raise ValueError(f"Hard negative case {label} is missing field: {field}.")

    if "case_id" not in case and "id" not in case:
        raise ValueError(f"Hard negative case at index {index} is missing field: id.")

    chunks = case.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Hard negative case {label} has invalid chunks.")

    for chunk_index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Hard negative case {label} chunk {chunk_index} must be an object."
            )
        for field in ("chunk_id", "text", "section_title"):
            if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                raise ValueError(
                    f"Hard negative case {label} chunk {chunk_index} is missing invalid field: {field}."
                )


def _hard_negative_chunks_for_case(case: dict[str, Any]) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    for index, chunk in enumerate(case.get("chunks", ()), start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=str(chunk["chunk_id"]),
                text=str(chunk["text"]),
                page_number=index,
                section_title=str(chunk["section_title"]),
                source_type=str(chunk.get("source_type", "synthetic_eval")),
                source_name=_case_id(case),
                extraction_method="synthetic",
                heading_confidence="high",
            )
        )
    return tuple(chunks)


def _chunks_for_case(case: dict[str, Any]) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    for index, chunk in enumerate(case.get("chunks", ()), start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=str(chunk["chunk_id"]),
                text=str(chunk["text"]),
                page_number=index,
                section_title=str(chunk["section_title"]),
                source_type="synthetic_eval",
                source_name=_case_id(case),
                extraction_method="synthetic",
            )
        )
    return tuple(chunks)
