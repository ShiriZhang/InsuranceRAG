from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import random
import time
from typing import Callable, Mapping, Protocol

from insurance_rag.chunker import chunk_pages
from insurance_rag.config import AppConfig
from insurance_rag.document_loader import parse_pdf_bytes
from insurance_rag.hybrid_retriever import HybridRetriever, HybridSearchResult
from insurance_rag.models import DocumentChunk, DocumentPage
from insurance_rag.models import BOUNDARY_SEMANTIC_OVERLAP_UNAVAILABLE
from insurance_rag.query_rewriter import rewrite_query
from insurance_rag.retriever import InMemoryVectorIndex
from insurance_rag.rule_reranker import rerank_results
from insurance_rag.silver_normalization import normalized_source_text


class SourceApproval(str, Enum):
    PUBLIC = "approved_public"
    PROJECT_OWNED = "approved_project_owned"
    USER_UPLOAD = "user_upload"
    USER_QUESTION = "user_question"
    GENERATED_ANSWER = "generated_answer"
    USER_TRACE = "user_trace"


APPROVED_SOURCE_TYPES = frozenset(
    {SourceApproval.PUBLIC, SourceApproval.PROJECT_OWNED}
)
JUDGE_ID = "exact-span-v1"
QUERY_REWRITE_VERSION = "production-v1"
RERANKER_VERSION = "rules-v1"


def _combine_unique_strata(
    stratum: str,
    additional_strata: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((stratum, *additional_strata)))


class Embedder(Protocol):
    model_id: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class TokenCounter(Protocol):
    tokenizer_id: str

    def __call__(self, text: str) -> int: ...


AnnotationPass = Callable[["BenchmarkSource"], tuple["AnnotationDraft", ...]]
AdjudicationPass = Callable[
    ["BenchmarkSource", "AnnotationDraft", "AnnotationDraft"],
    "AnnotationDraft",
]


@dataclass(frozen=True)
class BenchmarkSource:
    source_id: str
    source_name: str
    approval: SourceApproval
    approval_reference: str
    insurer_family: str
    product_family: str
    pages: tuple[DocumentPage, ...]
    source_content_sha256: str | None = None


@dataclass(frozen=True)
class EvidenceSpan:
    page_number: int
    start_char: int
    end_char: int
    quote: str


@dataclass(frozen=True)
class AnnotationMetadata:
    annotator_id: str
    model_id: str
    prompt_version: str
    generation_parameters: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class AnnotationResponseMetadata:
    response_id: str
    system_fingerprint: str | None
    request_timestamp: str
    retry_count: int
    prompt_tokens: int
    completion_tokens: int
    draft_order: tuple[str, ...] = ()
    returned_model: str | None = None
    response_status: str | None = None
    incomplete_reason: str | None = None


@dataclass(frozen=True)
class AnnotationDraft:
    question: str
    evidence_spans: tuple[EvidenceSpan, ...]
    stratum: str
    hard_negative_category: str | None
    metadata: AnnotationMetadata
    hard_negative_spans: tuple[EvidenceSpan, ...] = ()
    annotation_uncertain: bool = False
    additional_strata: tuple[str, ...] = ()
    response_metadata: AnnotationResponseMetadata | None = None
    slot_id: str = ""

    @property
    def strata(self) -> tuple[str, ...]:
        return _combine_unique_strata(self.stratum, self.additional_strata)


@dataclass(frozen=True)
class SilverCase:
    case_id: str
    source_id: str
    question: str
    evidence_spans: tuple[EvidenceSpan, ...]
    stratum: str
    hard_negative_category: str | None
    hard_negative_spans: tuple[EvidenceSpan, ...]
    annotation_uncertain: bool
    initial_disagreement: bool
    adjudicated: bool
    annotation_outcome: str
    annotation_metadata: tuple[AnnotationMetadata, ...]
    additional_strata: tuple[str, ...] = ()
    annotation_response_metadata: tuple[AnnotationResponseMetadata, ...] = ()
    annotation_slot_id: str = ""

    @property
    def strata(self) -> tuple[str, ...]:
        return _combine_unique_strata(self.stratum, self.additional_strata)


@dataclass(frozen=True)
class BenchmarkConfig:
    version: str
    judge_id: str
    embedding_model_id: str
    query_rewrite_version: str
    reranker_version: str
    retrieval_depth: int
    context_token_budget: int
    tokenizer_id: str

    def __post_init__(self) -> None:
        identifiers = (
            self.version,
            self.judge_id,
            self.embedding_model_id,
            self.query_rewrite_version,
            self.reranker_version,
            self.tokenizer_id,
        )
        if any(not identifier.strip() for identifier in identifiers):
            raise ValueError("Frozen benchmark identifiers cannot be empty.")
        if self.retrieval_depth < 5:
            raise ValueError("retrieval_depth must be at least five for Coverage@5.")
        if self.context_token_budget <= 0:
            raise ValueError("context_token_budget must be positive.")


@dataclass(frozen=True)
class FrozenBenchmark:
    version: str
    sources: tuple[BenchmarkSource, ...]
    cases: tuple[SilverCase, ...]
    config: BenchmarkConfig
    annotation_runs: tuple[AnnotationMetadata, ...]

    def to_manifest(self) -> dict[str, object]:
        source_records = []
        for source in self.sources:
            source_text = _source_text(source.pages)
            normalized_text = normalized_source_text(source.pages)
            source_records.append(
                {
                    "source_id": source.source_id,
                    "source_name": source.source_name,
                    "approval": source.approval.value,
                    "approval_reference": source.approval_reference,
                    "insurer_family": source.insurer_family,
                    "product_family": source.product_family,
                    "source_sha256": source.source_content_sha256
                    or _sha256(source_text),
                    "normalized_text_sha256": _sha256(normalized_text),
                    "pages": [page.page_number for page in source.pages],
                }
            )
        return {
            "version": self.version,
            "config": asdict(self.config),
            "source_records": source_records,
            "annotation_runs": [
                {
                    "annotator_id": run.annotator_id,
                    "model_id": run.model_id,
                    "prompt_version": run.prompt_version,
                    "generation_parameters": dict(run.generation_parameters),
                }
                for run in self.annotation_runs
            ],
            "cases": [
                {
                    "case_id": case.case_id,
                    "source_id": case.source_id,
                    "question": case.question,
                    "evidence_spans": [asdict(span) for span in case.evidence_spans],
                    "stratum": case.stratum,
                    "strata": list(case.strata),
                    "hard_negative_category": case.hard_negative_category,
                    "hard_negative_spans": [
                        asdict(span) for span in case.hard_negative_spans
                    ],
                    "annotation_uncertain": case.annotation_uncertain,
                    "initial_disagreement": case.initial_disagreement,
                    "adjudicated": case.adjudicated,
                    "annotation_outcome": case.annotation_outcome,
                    "annotation_slot_id": case.annotation_slot_id,
                    "annotation_responses": [
                        asdict(response)
                        for response in case.annotation_response_metadata
                    ],
                }
                for case in self.cases
            ],
        }

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.to_manifest(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return _sha256(payload)


def load_frozen_benchmark_manifest(
    payload: Mapping[str, object],
    *,
    sources: tuple[BenchmarkSource, ...],
) -> FrozenBenchmark:
    """Rehydrate a frozen manifest using separately rebuilt authoritative sources."""
    source_by_id = {source.source_id: source for source in sources}
    source_records = tuple(payload.get("source_records", ()))
    ordered_sources: list[BenchmarkSource] = []
    for record in source_records:
        if not isinstance(record, dict):
            raise ValueError("Frozen source records must be objects.")
        source_id = str(record["source_id"])
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"Frozen manifest source is unavailable: {source_id}")
        ordered_sources.append(source)

    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Frozen benchmark manifest requires a config object.")
    config = BenchmarkConfig(**config_payload)
    annotation_runs = tuple(
        _annotation_metadata_from_manifest(item)
        for item in payload.get("annotation_runs", ())
    )
    cases = tuple(
        _silver_case_from_manifest(item, annotation_runs)
        for item in payload.get("cases", ())
    )
    benchmark = FrozenBenchmark(
        version=str(payload["version"]),
        sources=tuple(ordered_sources),
        cases=cases,
        config=config,
        annotation_runs=annotation_runs,
    )
    expected = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    actual = json.dumps(
        benchmark.to_manifest(), ensure_ascii=False, sort_keys=True
    )
    if actual != expected:
        raise ValueError(
            "Rebuilt sources or labels do not match the frozen benchmark manifest."
        )
    return benchmark


def _annotation_metadata_from_manifest(value: object) -> AnnotationMetadata:
    if not isinstance(value, dict):
        raise ValueError("Frozen annotation runs must be objects.")
    parameters = value.get("generation_parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Frozen generation_parameters must be an object.")
    return AnnotationMetadata(
        annotator_id=str(value["annotator_id"]),
        model_id=str(value["model_id"]),
        prompt_version=str(value["prompt_version"]),
        generation_parameters=tuple(sorted(parameters.items())),
    )


def _silver_case_from_manifest(
    value: object,
    annotation_runs: tuple[AnnotationMetadata, ...],
) -> SilverCase:
    if not isinstance(value, dict):
        raise ValueError("Frozen Silver cases must be objects.")
    strata = tuple(str(item) for item in value.get("strata", ()))
    stratum = str(value["stratum"])
    return SilverCase(
        case_id=str(value["case_id"]),
        source_id=str(value["source_id"]),
        question=str(value["question"]),
        evidence_spans=tuple(
            EvidenceSpan(**span) for span in value.get("evidence_spans", ())
        ),
        stratum=stratum,
        additional_strata=tuple(item for item in strata if item != stratum),
        hard_negative_category=(
            str(value["hard_negative_category"])
            if value.get("hard_negative_category") is not None
            else None
        ),
        hard_negative_spans=tuple(
            EvidenceSpan(**span)
            for span in value.get("hard_negative_spans", ())
        ),
        annotation_uncertain=bool(value["annotation_uncertain"]),
        initial_disagreement=bool(value["initial_disagreement"]),
        adjudicated=bool(value["adjudicated"]),
        annotation_outcome=str(value["annotation_outcome"]),
        annotation_metadata=annotation_runs,
        annotation_response_metadata=tuple(
            AnnotationResponseMetadata(**response)
            for response in value.get("annotation_responses", ())
        ),
        annotation_slot_id=str(value.get("annotation_slot_id", "")),
    )


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    chunking_strategy: str
    target_chars: int = 900
    hard_max_chars: int = 1200
    body_overlap_mode: str = "zero_body_overlap"


@dataclass(frozen=True)
class AnnotationSummary:
    total_cases: int
    initial_disagreements: int
    adjudicated_cases: int
    uncertain_exclusions: int

    @property
    def disagreement_rate(self) -> float:
        return _rate(self.initial_disagreements, self.total_cases)

    @property
    def adjudication_success_rate(self) -> float:
        resolved = self.adjudicated_cases - self.uncertain_exclusions
        return _rate(resolved, self.adjudicated_cases)

    @property
    def exclusion_rate(self) -> float:
        return _rate(self.uncertain_exclusions, self.total_cases)


@dataclass(frozen=True)
class StrategyResult:
    strategy_name: str
    chunking_strategy: str
    target_chars: int
    hard_max_chars: int
    body_overlap_mode: str
    scored_cases: int
    coverage_at: Mapping[int, float]
    coverage_under_token_budget: float
    single_candidate_coverage: float
    irrelevant_context_proportion: float
    hard_negative_confusions: Mapping[str, int]
    embedding_tokens: int
    retrieval_unit_count: int
    chunking_latency_seconds: float
    p95_chunking_latency_seconds: float
    coverage_under_budget_by_stratum: Mapping[str, float]
    boundary_diagnostics: Mapping[str, int]
    correctness_invariants: Mapping[str, bool]
    case_coverage_at_3: tuple[float, ...]
    case_coverage_under_budget: tuple[float, ...]
    case_irrelevant_context_proportion: tuple[float, ...]
    case_hard_negative_confusion: tuple[float, ...]
    case_hard_negative_applicable: tuple[bool, ...]
    case_strata: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95


@dataclass(frozen=True)
class PairedComparison:
    baseline_strategy: str
    candidate_strategy: str
    coverage_at_3: ConfidenceInterval
    coverage_under_token_budget: ConfidenceInterval


@dataclass(frozen=True)
class BenchmarkReport:
    benchmark_version: str
    manifest_sha256: str
    annotation_summary: AnnotationSummary
    strategy_results: tuple[StrategyResult, ...]
    paired_comparisons: tuple[PairedComparison, ...]


def source_from_pdf_bytes(
    pdf_bytes: bytes,
    *,
    source_id: str,
    source_name: str,
    approval: SourceApproval,
    approval_reference: str,
    insurer_family: str,
    product_family: str,
    parse_config: AppConfig,
) -> BenchmarkSource:
    source = BenchmarkSource(
        source_id=source_id,
        source_name=source_name,
        approval=approval,
        approval_reference=approval_reference,
        insurer_family=insurer_family,
        product_family=product_family,
        pages=parse_pdf_bytes(pdf_bytes, source_name, parse_config).pages,
        source_content_sha256=_sha256_bytes(pdf_bytes),
    )
    _validate_approved_source(source)
    return source


def generate_frozen_benchmark(
    *,
    sources: tuple[BenchmarkSource, ...],
    first_pass: AnnotationPass,
    second_pass: AnnotationPass,
    adjudication_pass: AdjudicationPass,
    config: BenchmarkConfig,
    max_workers: int = 1,
) -> FrozenBenchmark:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    annotation_pairs: dict[
        str, tuple[tuple[AnnotationDraft, AnnotationDraft], ...]
    ] = {}
    adjudications: dict[tuple[str, int], AnnotationDraft] = {}

    def process_source(source: BenchmarkSource):
        _validate_approved_source(source)
        first_annotations = first_pass(source)
        second_annotations = second_pass(source)
        if len(first_annotations) != len(second_annotations):
            raise ValueError(
                "Independent annotation passes must return the same number of cases."
            )
        pairs = tuple(zip(first_annotations, second_annotations))
        source_adjudications = []
        for case_index, (first, second) in enumerate(pairs):
            if (
                first.annotation_uncertain
                or second.annotation_uncertain
                or _annotation_label(first) != _annotation_label(second)
            ):
                source_adjudications.append(
                    (
                        case_index,
                        adjudication_pass(source, first, second),
                    )
                )
        return source.source_id, pairs, tuple(source_adjudications)

    if max_workers == 1:
        results = map(process_source, sources)
        for source_id, pairs, source_adjudications in results:
            annotation_pairs[source_id] = pairs
            for case_index, adjudication in source_adjudications:
                adjudications[(source_id, case_index)] = adjudication
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_source, sources)
            for source_id, pairs, source_adjudications in results:
                annotation_pairs[source_id] = pairs
                for case_index, adjudication in source_adjudications:
                    adjudications[(source_id, case_index)] = adjudication
    return build_frozen_benchmark(
        sources=sources,
        annotation_pairs=annotation_pairs,
        adjudications=adjudications,
        config=config,
    )


def adjudicate_annotations(
    source: BenchmarkSource,
    first: AnnotationDraft,
    second: AnnotationDraft,
    *,
    adjudication: AnnotationDraft | None = None,
) -> SilverCase:
    _validate_approved_source(source)
    _validate_annotation(source, first)
    _validate_annotation(source, second)
    if first.metadata.annotator_id == second.metadata.annotator_id:
        raise ValueError("Two independent annotations require different annotator IDs.")

    agreed = (
        not first.annotation_uncertain
        and not second.annotation_uncertain
        and _annotation_label(first) == _annotation_label(second)
    )
    metadata = [first.metadata, second.metadata]
    response_metadata = [
        response
        for response in (first.response_metadata, second.response_metadata)
        if response is not None
    ]
    if agreed:
        selected = first
        outcome = "agreed"
        adjudicated = False
    else:
        if adjudication is None:
            raise ValueError("Disagreement requires a third independent adjudicator.")
        if adjudication.metadata.annotator_id in {
            first.metadata.annotator_id,
            second.metadata.annotator_id,
        }:
            raise ValueError("Disagreement requires a third independent adjudicator.")
        _validate_annotation(source, adjudication)
        selected = adjudication
        metadata.append(adjudication.metadata)
        if adjudication.response_metadata is not None:
            response_metadata.append(adjudication.response_metadata)
        outcome = "annotation_uncertain" if adjudication.annotation_uncertain else "adjudicated"
        adjudicated = True

    if selected.annotation_uncertain and selected.evidence_spans:
        raise ValueError("annotation_uncertain cases cannot freeze evidence spans.")
    if not selected.annotation_uncertain and not selected.evidence_spans:
        raise ValueError("Resolved annotations require exact evidence spans.")

    case_identity = json.dumps(
        {
            "source_id": source.source_id,
            "question": selected.question,
            "spans": [asdict(span) for span in selected.evidence_spans],
            "strata": selected.strata,
            "hard_negative_category": selected.hard_negative_category,
            "hard_negative_spans": [
                asdict(span) for span in selected.hard_negative_spans
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return SilverCase(
        case_id=f"silver-{_sha256(case_identity)[:16]}",
        source_id=source.source_id,
        question=selected.question,
        evidence_spans=selected.evidence_spans,
        stratum=selected.stratum,
        hard_negative_category=selected.hard_negative_category,
        hard_negative_spans=selected.hard_negative_spans,
        annotation_uncertain=selected.annotation_uncertain,
        initial_disagreement=not agreed,
        adjudicated=adjudicated,
        annotation_outcome=outcome,
        annotation_metadata=tuple(metadata),
        additional_strata=selected.additional_strata,
        annotation_response_metadata=tuple(response_metadata),
        annotation_slot_id=selected.slot_id,
    )


def build_frozen_benchmark(
    *,
    sources: tuple[BenchmarkSource, ...],
    annotation_pairs: Mapping[
        str, tuple[tuple[AnnotationDraft, AnnotationDraft], ...]
    ],
    adjudications: Mapping[tuple[str, int], AnnotationDraft],
    config: BenchmarkConfig,
) -> FrozenBenchmark:
    source_ids: set[str] = set()
    cases: list[SilverCase] = []
    annotation_runs: dict[str, AnnotationMetadata] = {}
    for source in sources:
        _validate_approved_source(source)
        if source.source_id in source_ids:
            raise ValueError(f"Duplicate benchmark source ID: {source.source_id}")
        source_ids.add(source.source_id)
        pairs = annotation_pairs.get(source.source_id, ())
        for case_index, (first, second) in enumerate(pairs):
            case = adjudicate_annotations(
                source,
                first,
                second,
                adjudication=adjudications.get((source.source_id, case_index)),
            )
            cases.append(case)
            for metadata in case.annotation_metadata:
                existing = annotation_runs.get(metadata.annotator_id)
                if existing is not None and existing != metadata:
                    raise ValueError(
                        f"Annotator metadata changed within one frozen version: {metadata.annotator_id}"
                    )
                annotation_runs[metadata.annotator_id] = metadata

    unknown_annotation_sources = set(annotation_pairs) - source_ids
    if unknown_annotation_sources:
        raise ValueError(
            "Annotations reference unknown sources: "
            + ", ".join(sorted(unknown_annotation_sources))
        )
    if not cases:
        raise ValueError("A frozen benchmark requires at least one annotated case.")
    return FrozenBenchmark(
        version=config.version,
        sources=sources,
        cases=tuple(cases),
        config=config,
        annotation_runs=tuple(
            annotation_runs[key] for key in sorted(annotation_runs)
        ),
    )


def run_frozen_benchmark(
    benchmark: FrozenBenchmark,
    *,
    strategies: tuple[StrategyConfig, ...],
    embedder: Embedder,
    token_counter: TokenCounter,
    bootstrap_samples: int = 2000,
) -> BenchmarkReport:
    if not strategies:
        raise ValueError("At least one strategy is required.")
    if len({strategy.name for strategy in strategies}) != len(strategies):
        raise ValueError("Strategy names must be unique.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")

    _validate_execution_config(benchmark, embedder, token_counter)
    source_by_id = {source.source_id: source for source in benchmark.sources}
    for source in benchmark.sources:
        _validate_approved_source(source)
    for case in benchmark.cases:
        source = source_by_id.get(case.source_id)
        if source is None:
            raise ValueError(
                f"Silver case references unknown source: {case.source_id}"
            )
        _validate_scored_case(source, case)

    scored_cases = tuple(
        case for case in benchmark.cases if not case.annotation_uncertain
    )
    strategy_results = tuple(
        _run_strategy(
            benchmark,
            strategy,
            scored_cases,
            source_by_id,
            embedder,
            token_counter,
        )
        for strategy in strategies
    )
    baseline = strategy_results[0]
    comparisons = tuple(
        PairedComparison(
            baseline_strategy=baseline.strategy_name,
            candidate_strategy=candidate.strategy_name,
            coverage_at_3=_paired_confidence_interval(
                baseline.case_coverage_at_3,
                candidate.case_coverage_at_3,
                bootstrap_samples=bootstrap_samples,
            ),
            coverage_under_token_budget=_paired_confidence_interval(
                baseline.case_coverage_under_budget,
                candidate.case_coverage_under_budget,
                bootstrap_samples=bootstrap_samples,
            ),
        )
        for candidate in strategy_results[1:]
    )
    annotation_summary = AnnotationSummary(
        total_cases=len(benchmark.cases),
        initial_disagreements=sum(
            case.initial_disagreement for case in benchmark.cases
        ),
        adjudicated_cases=sum(case.adjudicated for case in benchmark.cases),
        uncertain_exclusions=len(benchmark.cases) - len(scored_cases),
    )
    return BenchmarkReport(
        benchmark_version=benchmark.version,
        manifest_sha256=benchmark.manifest_sha256,
        annotation_summary=annotation_summary,
        strategy_results=strategy_results,
        paired_comparisons=comparisons,
    )


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    summary = report.annotation_summary
    lines = [
        "# Silver Supporting Evidence Benchmark",
        "",
        f"Benchmark version: `{report.benchmark_version}`",
        f"Frozen manifest SHA-256: `{report.manifest_sha256}`",
        "",
        "## Annotation quality",
        "",
        f"- Annotation disagreement: {summary.initial_disagreements}/{summary.total_cases} ({summary.disagreement_rate:.2%})",
        f"- Adjudication success: {summary.adjudicated_cases - summary.uncertain_exclusions}/{summary.adjudicated_cases} ({summary.adjudication_success_rate:.2%})",
        f"- Uncertain exclusions: {summary.uncertain_exclusions}/{summary.total_cases} ({summary.exclusion_rate:.2%})",
        "",
        "## Retrieval, context, and cost",
        "",
        "| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report.strategy_results:
        hard_negatives = ", ".join(
            f"{category}: {count}"
            for category, count in sorted(result.hard_negative_confusions.items())
        ) or "none"
        lines.append(
            "| {name} | {scored} | {at1:.2%}/{at3:.2%}/{at5:.2%} | {budget:.2%} | {single:.2%} | {irrelevant:.2%} | {tokens} | {units} | {latency:.6f} | {hard_negatives} |".format(
                name=result.strategy_name,
                scored=result.scored_cases,
                at1=result.coverage_at[1],
                at3=result.coverage_at[3],
                at5=result.coverage_at[5],
                budget=result.coverage_under_token_budget,
                single=result.single_candidate_coverage,
                irrelevant=result.irrelevant_context_proportion,
                tokens=result.embedding_tokens,
                units=result.retrieval_unit_count,
                latency=result.p95_chunking_latency_seconds,
                hard_negatives=hard_negatives,
            )
        )
    lines.extend(("", "## Boundary strata", ""))
    for result in report.strategy_results:
        strata = ", ".join(
            f"{name}: {value:.2%}"
            for name, value in result.coverage_under_budget_by_stratum.items()
        ) or "none"
        lines.append(f"- {result.strategy_name}: {strata}")
    lines.extend(("", "## Diagnostics and correctness invariants", ""))
    for result in report.strategy_results:
        diagnostics = ", ".join(
            f"{name}: {count}"
            for name, count in sorted(result.boundary_diagnostics.items())
        ) or "none"
        invariants = ", ".join(
            f"{name}: {'pass' if passed else 'FAIL'}"
            for name, passed in sorted(result.correctness_invariants.items())
        )
        lines.append(
            f"- {result.strategy_name}: diagnostics [{diagnostics}]; invariants [{invariants}]"
        )
    lines.extend(("", "## Paired 95% confidence intervals", ""))
    if not report.paired_comparisons:
        lines.append("No candidate strategy was supplied.")
    for comparison in report.paired_comparisons:
        lines.append(
            "- {candidate} minus {baseline}: Coverage@3 {coverage}; Coverage under token budget {budget}.".format(
                candidate=comparison.candidate_strategy,
                baseline=comparison.baseline_strategy,
                coverage=_format_interval(comparison.coverage_at_3),
                budget=_format_interval(comparison.coverage_under_token_budget),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _run_strategy(
    benchmark: FrozenBenchmark,
    strategy: StrategyConfig,
    cases: tuple[SilverCase, ...],
    source_by_id: Mapping[str, BenchmarkSource],
    embedder: Embedder,
    token_counter: TokenCounter,
) -> StrategyResult:
    retrievers: dict[str, HybridRetriever] = {}
    retrieval_unit_count = 0
    embedding_tokens = 0
    chunking_latency = 0.0
    source_latencies: list[float] = []
    boundary_diagnostics: dict[str, int] = {}
    correctness_invariants = {
        "hard_max_chars_respected": True,
        "non_empty_retrieval_units": True,
        "source_spans_exact": True,
        "semantic_overlap_distinct_from_source_spans": True,
        "semantic_overlap_complete": True,
        "authoritative_source_coverage_exact": True,
        "no_unintended_source_duplication": True,
        "clause_purity": True,
        "parseable_policies_indexed": True,
    }
    for source in benchmark.sources:
        if not any(page.text.strip() for page in source.pages):
            continue
        page_by_number = {page.page_number: page for page in source.pages}
        started = time.perf_counter()
        chunks = chunk_pages(
            source.pages,
            source_name=source.source_name,
            source_type="silver_benchmark",
            chunk_size=strategy.target_chars,
            overlap=0,
            strategy=strategy.chunking_strategy,
            target_chars=strategy.target_chars,
            hard_max_chars=strategy.hard_max_chars,
            body_overlap_mode=strategy.body_overlap_mode,
        )
        source_latency = time.perf_counter() - started
        chunking_latency += source_latency
        source_latencies.append(source_latency)
        if not chunks:
            raise ValueError(
                f"Strategy {strategy.name!r} produced no retrieval units for {source.source_id!r}."
            )
        texts = [chunk.retrieval_text for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        index = InMemoryVectorIndex.from_embeddings(chunks, embeddings)
        retrievers[source.source_id] = HybridRetriever(
            chunks,
            index,
            embedder,
            retrieval_mode="hybrid",
        )
        retrieval_unit_count += len(chunks)
        embedding_tokens += sum(token_counter(text) for text in texts)
        for chunk in chunks:
            for diagnostic in chunk.boundary_diagnostics:
                boundary_diagnostics[diagnostic] = (
                    boundary_diagnostics.get(diagnostic, 0) + 1
                )
            if len(chunk.retrieval_text) > strategy.hard_max_chars:
                correctness_invariants["hard_max_chars_respected"] = False
            if not chunk.retrieval_text.strip():
                correctness_invariants["non_empty_retrieval_units"] = False
            if strategy.chunking_strategy == "clause_v2":
                if any(
                    page_by_number[span.page_number].text[
                        span.start_char : span.end_char
                    ]
                    != span.text
                    for span in chunk.source_spans
                ):
                    correctness_invariants["source_spans_exact"] = False
                if (
                    BOUNDARY_SEMANTIC_OVERLAP_UNAVAILABLE
                    in chunk.boundary_diagnostics
                ):
                    correctness_invariants["semantic_overlap_complete"] = False
                if chunk.trusted_heading_count > 1:
                    correctness_invariants["clause_purity"] = False
        if strategy.chunking_strategy == "clause_v2":
            spans_by_page: dict[int, list[tuple[int, int]]] = {}
            for chunk in chunks:
                for span in chunk.source_spans:
                    spans_by_page.setdefault(span.page_number, []).append(
                        (span.start_char, span.end_char)
                    )
            for page in source.pages:
                if not page.text:
                    continue
                cursor = 0
                for start_char, end_char in sorted(
                    spans_by_page.get(page.page_number, ())
                ):
                    if start_char < cursor:
                        correctness_invariants[
                            "no_unintended_source_duplication"
                        ] = False
                    if start_char != cursor:
                        correctness_invariants[
                            "authoritative_source_coverage_exact"
                        ] = False
                    cursor = max(cursor, end_char)
                if cursor != len(page.text):
                    correctness_invariants[
                        "authoritative_source_coverage_exact"
                    ] = False

    coverage_by_k: dict[int, list[float]] = {1: [], 3: [], 5: []}
    coverage_under_budget: list[float] = []
    single_candidate: list[float] = []
    irrelevant_chars = 0
    retrieved_chars = 0
    hard_negative_confusions: dict[str, int] = {}
    case_irrelevant_context: list[float] = []
    case_hard_negative_confusion: list[float] = []
    case_hard_negative_applicable: list[bool] = []
    case_strata: list[tuple[str, ...]] = []
    for case in cases:
        source = source_by_id[case.source_id]
        rewrite = rewrite_query(case.question)
        embedding_tokens += sum(
            token_counter(query) for query in rewrite.expanded_queries
        )
        initial = retrievers[case.source_id].search(
            rewrite,
            top_k=benchmark.config.retrieval_depth,
        )
        retrieved = tuple(
            rerank_results(
                question=case.question,
                rewrite=rewrite,
                candidates=initial,
                top_k=benchmark.config.retrieval_depth,
            )
        )
        for top_k in (1, 3, 5):
            coverage_by_k[top_k].append(
                float(
                    _covers_all_spans(
                        retrieved[:top_k], case.evidence_spans, source
                    )
                )
            )
        budgeted = _within_token_budget(
            retrieved,
            benchmark.config.context_token_budget,
            token_counter,
        )
        coverage_under_budget.append(
            float(_covers_all_spans(budgeted, case.evidence_spans, source))
        )
        single_candidate.append(
            float(
                any(
                    _covers_all_spans((item,), case.evidence_spans, source)
                    for item in retrieved
                )
            )
        )
        case_irrelevant, case_retrieved = _irrelevant_context_chars(
            source,
            budgeted,
            case.evidence_spans,
        )
        irrelevant_chars += case_irrelevant
        retrieved_chars += case_retrieved
        case_irrelevant_context.append(_rate(case_irrelevant, case_retrieved))
        case_strata.append(case.strata)
        confused = False
        case_hard_negative_applicable.append(
            case.hard_negative_category is not None
        )
        if case.hard_negative_category:
            hard_negative_confusions.setdefault(case.hard_negative_category, 0)
            if _covers_any_span(
                retrieved[:1], case.hard_negative_spans, source
            ):
                hard_negative_confusions[case.hard_negative_category] += 1
                confused = True
        case_hard_negative_confusion.append(float(confused))

    coverage_under_budget_by_stratum = {
        stratum: _mean(
            [
                value
                for value, strata in zip(coverage_under_budget, case_strata)
                if stratum in strata
            ]
        )
        for stratum in sorted({item for strata in case_strata for item in strata})
    }

    return StrategyResult(
        strategy_name=strategy.name,
        chunking_strategy=strategy.chunking_strategy,
        target_chars=strategy.target_chars,
        hard_max_chars=strategy.hard_max_chars,
        body_overlap_mode=strategy.body_overlap_mode,
        scored_cases=len(cases),
        coverage_at={
            top_k: _mean(values) for top_k, values in coverage_by_k.items()
        },
        coverage_under_token_budget=_mean(coverage_under_budget),
        single_candidate_coverage=_mean(single_candidate),
        irrelevant_context_proportion=_rate(irrelevant_chars, retrieved_chars),
        hard_negative_confusions=hard_negative_confusions,
        embedding_tokens=embedding_tokens,
        retrieval_unit_count=retrieval_unit_count,
        chunking_latency_seconds=chunking_latency,
        p95_chunking_latency_seconds=_percentile(source_latencies, 95),
        coverage_under_budget_by_stratum=coverage_under_budget_by_stratum,
        boundary_diagnostics=boundary_diagnostics,
        correctness_invariants=correctness_invariants,
        case_coverage_at_3=tuple(coverage_by_k[3]),
        case_coverage_under_budget=tuple(coverage_under_budget),
        case_irrelevant_context_proportion=tuple(case_irrelevant_context),
        case_hard_negative_confusion=tuple(case_hard_negative_confusion),
        case_hard_negative_applicable=tuple(case_hard_negative_applicable),
        case_strata=tuple(case_strata),
    )


def _validate_approved_source(source: BenchmarkSource) -> None:
    if source.approval not in APPROVED_SOURCE_TYPES:
        raise ValueError(
            "Benchmark annotation sources must be approved public or project-owned text."
        )
    if not source.approval_reference.strip():
        raise ValueError("Approved sources require an approval_reference.")
    if not source.pages:
        raise ValueError("Benchmark sources require page-addressable text.")
    page_numbers = [page.page_number for page in source.pages]
    if len(set(page_numbers)) != len(page_numbers):
        raise ValueError("Benchmark source page numbers must be unique.")


def _validate_execution_config(
    benchmark: FrozenBenchmark,
    embedder: Embedder,
    token_counter: TokenCounter,
) -> None:
    config = benchmark.config
    embedding_model_id = getattr(
        embedder, "model_id", getattr(embedder, "model", None)
    )
    tokenizer_id = getattr(token_counter, "tokenizer_id", None)
    actual = {
        "judge_id": JUDGE_ID,
        "embedding_model_id": embedding_model_id,
        "query_rewrite_version": QUERY_REWRITE_VERSION,
        "reranker_version": RERANKER_VERSION,
        "tokenizer_id": tokenizer_id,
    }
    expected = {
        "judge_id": config.judge_id,
        "embedding_model_id": config.embedding_model_id,
        "query_rewrite_version": config.query_rewrite_version,
        "reranker_version": config.reranker_version,
        "tokenizer_id": config.tokenizer_id,
    }
    if actual != expected:
        differences = ", ".join(
            f"{key}: expected {expected[key]!r}, got {actual[key]!r}"
            for key in expected
            if expected[key] != actual[key]
        )
        raise ValueError(
            "Execution components do not match the frozen retrieval configuration: "
            + differences
        )


def _validate_scored_case(source: BenchmarkSource, case: SilverCase) -> None:
    pages = {page.page_number: page.text for page in source.pages}
    for span in case.evidence_spans:
        _validate_span(pages, span)
    for span in case.hard_negative_spans:
        _validate_span(pages, span)
    if case.annotation_uncertain and case.evidence_spans:
        raise ValueError("annotation_uncertain cases cannot freeze evidence spans.")
    if not case.annotation_uncertain and not case.evidence_spans:
        raise ValueError("Every scored Silver case requires exact evidence spans.")
    if bool(case.hard_negative_category) != bool(case.hard_negative_spans):
        raise ValueError(
            "Hard-negative categories require exact hard-negative spans and vice versa."
        )
    _validate_strata(case.stratum, case.additional_strata)


def _validate_span(
    pages: Mapping[int, str],
    span: EvidenceSpan,
) -> None:
    page_text = pages.get(span.page_number)
    if (
        page_text is None
        or span.start_char < 0
        or span.end_char <= span.start_char
        or span.end_char > len(page_text)
        or page_text[span.start_char : span.end_char] != span.quote
    ):
        raise ValueError(
            "Silver evidence span does not exactly map to authoritative source text."
        )


def _validate_annotation(source: BenchmarkSource, annotation: AnnotationDraft) -> None:
    if not annotation.question.strip():
        raise ValueError("Annotations require a question.")
    if not annotation.metadata.model_id.strip():
        raise ValueError("Annotations require an exact model identifier.")
    if not annotation.metadata.prompt_version.strip():
        raise ValueError("Annotations require a prompt version.")
    pages = {page.page_number: page.text for page in source.pages}
    for span in annotation.evidence_spans:
        _validate_span(pages, span)
    for span in annotation.hard_negative_spans:
        _validate_span(pages, span)
    if bool(annotation.hard_negative_category) != bool(
        annotation.hard_negative_spans
    ):
        raise ValueError(
            "Hard-negative categories require exact hard-negative spans and vice versa."
        )
    _validate_strata(annotation.stratum, annotation.additional_strata)


def _annotation_label(annotation: AnnotationDraft) -> tuple[object, ...]:
    return (
        annotation.question,
        annotation.evidence_spans,
        annotation.stratum,
        annotation.hard_negative_category,
        annotation.hard_negative_spans,
        annotation.annotation_uncertain,
        annotation.additional_strata,
    )


def _validate_strata(stratum: str, additional_strata: tuple[str, ...]) -> None:
    strata = (stratum, *additional_strata)
    if any(not item.strip() for item in strata):
        raise ValueError("Silver cases require non-empty strata identifiers.")
    if len(set(strata)) != len(strata):
        raise ValueError("Silver case strata identifiers must be unique.")


def _covers_all_spans(
    retrieved: tuple[HybridSearchResult, ...],
    evidence_spans: tuple[EvidenceSpan, ...],
    source: BenchmarkSource,
) -> bool:
    return all(
        any(
            source_span.page_number == evidence.page_number
            and source_span.start_char <= evidence.start_char
            and source_span.end_char >= evidence.end_char
            for item in retrieved
            for source_span in _authoritative_chunk_ranges(item.chunk, source)
        )
        for evidence in evidence_spans
    )


def _covers_any_span(
    retrieved: tuple[HybridSearchResult, ...],
    spans: tuple[EvidenceSpan, ...],
    source: BenchmarkSource,
) -> bool:
    return any(
        source_span.page_number == expected.page_number
        and source_span.start_char <= expected.start_char
        and source_span.end_char >= expected.end_char
        for item in retrieved
        for source_span in _authoritative_chunk_ranges(item.chunk, source)
        for expected in spans
    )


def _within_token_budget(
    retrieved: tuple[HybridSearchResult, ...],
    token_budget: int,
    token_counter: TokenCounter,
) -> tuple[HybridSearchResult, ...]:
    selected: list[HybridSearchResult] = []
    used = 0
    for item in retrieved:
        item_tokens = token_counter(item.chunk.retrieval_text)
        if used + item_tokens > token_budget:
            continue
        selected.append(item)
        used += item_tokens
    return tuple(selected)


def _irrelevant_context_chars(
    source: BenchmarkSource,
    retrieved: tuple[HybridSearchResult, ...],
    evidence_spans: tuple[EvidenceSpan, ...],
) -> tuple[int, int]:
    total = 0
    relevant = 0
    for item in retrieved:
        for source_span in _authoritative_chunk_ranges(item.chunk, source):
            span_length = source_span.end_char - source_span.start_char
            total += span_length
            for evidence in evidence_spans:
                if evidence.page_number != source_span.page_number:
                    continue
                relevant += max(
                    0,
                    min(source_span.end_char, evidence.end_char)
                    - max(source_span.start_char, evidence.start_char),
                )
    return max(0, total - relevant), total


def _authoritative_chunk_ranges(
    chunk: DocumentChunk,
    source: BenchmarkSource,
) -> tuple[EvidenceSpan, ...]:
    if chunk.source_spans:
        return tuple(
            EvidenceSpan(
                page_number=span.page_number,
                start_char=span.start_char,
                end_char=span.end_char,
                quote=span.text,
            )
            for span in chunk.source_spans
        )
    if chunk.page_number is None:
        return ()
    page = next(
        (page for page in source.pages if page.page_number == chunk.page_number),
        None,
    )
    if page is None:
        return ()
    start_char = page.text.find(chunk.text)
    if start_char < 0:
        return ()
    return (
        EvidenceSpan(
            page_number=page.page_number,
            start_char=start_char,
            end_char=start_char + len(chunk.text),
            quote=chunk.text,
        ),
    )


def _paired_confidence_interval(
    baseline: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    bootstrap_samples: int,
) -> ConfidenceInterval:
    if len(baseline) != len(candidate):
        raise ValueError("Paired comparisons require the same frozen cases.")
    differences = [right - left for left, right in zip(baseline, candidate)]
    estimate = _mean(differences)
    if not differences:
        return ConfidenceInterval(estimate=0.0, lower=0.0, upper=0.0)
    generator = random.Random(0)
    bootstrapped = sorted(
        _mean([generator.choice(differences) for _ in differences])
        for _ in range(bootstrap_samples)
    )
    lower_index = int(0.025 * (bootstrap_samples - 1))
    upper_index = int(0.975 * (bootstrap_samples - 1))
    return ConfidenceInterval(
        estimate=estimate,
        lower=bootstrapped[lower_index],
        upper=bootstrapped[upper_index],
    )


def paired_confidence_interval(
    baseline: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    bootstrap_samples: int = 2000,
) -> ConfidenceInterval:
    """Return a deterministic paired bootstrap interval for frozen cases."""
    return _paired_confidence_interval(
        baseline,
        candidate,
        bootstrap_samples=bootstrap_samples,
    )


def _source_text(pages: tuple[DocumentPage, ...]) -> str:
    return "\n\f\n".join(
        f"page:{page.page_number}\n{page.text}" for page in pages
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile / 100)]


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _format_interval(interval: ConfidenceInterval) -> str:
    return (
        f"{interval.estimate:+.2%} "
        f"[{interval.lower:+.2%}, {interval.upper:+.2%}]"
    )
