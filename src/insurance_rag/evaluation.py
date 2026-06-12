from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from insurance_rag.hybrid_retriever import HybridRetriever
from insurance_rag.models import DocumentChunk
from insurance_rag.query_rewriter import rewrite_query
from insurance_rag.retriever import InMemoryVectorIndex


class DeterministicEvalEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_sha256_vector(text) for text in texts]


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
    passed: bool


@dataclass(frozen=True)
class EvalReport:
    total_cases: int
    passed_cases: int
    results: tuple[EvalCaseResult, ...]


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
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
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
                passed=passed,
            )
        )

    return EvalReport(
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
    lines.append("")
    return "\n".join(lines)


def _sha256_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [
        int.from_bytes(digest[index : index + 4], "big") / 0xFFFFFFFF
        for index in range(0, 32, 4)
    ]


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


def _case_id(case: dict[str, Any]) -> str:
    return str(case.get("case_id", case.get("id")))


def _case_label(case: dict[str, Any], index: int) -> str:
    if "case_id" in case or "id" in case:
        return _case_id(case)
    return f"at index {index}"


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
