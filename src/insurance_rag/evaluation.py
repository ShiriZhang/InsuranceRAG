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
    top_fusion_score: float
    passed: bool


@dataclass(frozen=True)
class EvalReport:
    total_cases: int
    passed_cases: int
    results: tuple[EvalCaseResult, ...]


def evaluate_synthetic_cases(path: Path, *, top_k: int = 3) -> EvalReport:
    cases = _load_cases(path)
    embedder = DeterministicEvalEmbedder()
    results: list[EvalCaseResult] = []

    for case in cases:
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
        passed = expected_section in retrieved_sections or any(
            term in section for term in expected_terms for section in retrieved_sections
        )

        results.append(
            EvalCaseResult(
                case_id=str(case["case_id"]),
                question=str(case["question"]),
                expected_section=expected_section,
                expected_terms=expected_terms,
                retrieved_sections=retrieved_sections,
                retrieved_chunk_ids=retrieved_chunk_ids,
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
        "| Case | Expected Section | Retrieved Sections | Retrieved Chunks | Fusion Score | PASS/FAIL |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            "| {case_id} | {expected_section} | {sections} | {chunks} | {score:.6f} | {status} |".format(
                case_id=result.case_id,
                expected_section=result.expected_section,
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
    return data


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
                source_name=str(case["case_id"]),
                extraction_method="synthetic",
            )
        )
    return tuple(chunks)
