from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json

from insurance_rag.silver_benchmark import (
    APPROVED_SOURCE_TYPES,
    AdjudicationPass,
    AnnotationPass,
    BenchmarkConfig,
    BenchmarkSource,
    FrozenBenchmark,
    SilverCase,
    generate_frozen_benchmark,
)


REQUIRED_EVIDENCE_STRATA = (
    "single_sentence",
    "multi_sentence_conditions_outcomes",
    "rule_plus_exception",
    "cross_page_clause",
    "complete_short_clause",
    "internally_split_clause",
    "adjacent_or_lexically_similar_hard_negative",
)
KEY_BOUNDARY_SENSITIVE_STRATA = REQUIRED_EVIDENCE_STRATA


class DatasetSplit(str, Enum):
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


@dataclass(frozen=True)
class DocumentSplitAssignment:
    source_id: str
    split: DatasetSplit
    near_duplicate_family: str


@dataclass(frozen=True)
class FrozenDocumentSplit:
    version: str
    sources: tuple[BenchmarkSource, ...]
    assignments: tuple[DocumentSplitAssignment, ...]

    def sources_for(self, split: DatasetSplit) -> tuple[BenchmarkSource, ...]:
        selected_ids = {
            assignment.source_id
            for assignment in self.assignments
            if assignment.split is split
        }
        return tuple(
            source for source in self.sources if source.source_id in selected_ids
        )

    def split_for(self, source_id: str) -> DatasetSplit:
        for assignment in self.assignments:
            if assignment.source_id == source_id:
                return assignment.split
        raise KeyError(source_id)

    def to_manifest(self) -> dict[str, object]:
        source_by_id = {source.source_id: source for source in self.sources}
        records: list[dict[str, object]] = []
        for assignment in sorted(self.assignments, key=lambda item: item.source_id):
            source = source_by_id[assignment.source_id]
            source_text = _source_text(source)
            normalized_text = _normalized_source_text(source)
            records.append(
                {
                    "source_id": source.source_id,
                    "split": assignment.split.value,
                    "insurer_family": source.insurer_family,
                    "product_family": source.product_family,
                    "near_duplicate_family": assignment.near_duplicate_family,
                    "source_sha256": source.source_content_sha256
                    or _sha256(source_text),
                    "normalized_text_sha256": _sha256(normalized_text),
                }
            )
        return {"version": self.version, "source_records": records}

    @property
    def manifest_sha256(self) -> str:
        return _sha256(
            json.dumps(
                self.to_manifest(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


@dataclass(frozen=True)
class DatasetFreezeConfig:
    version: str
    required_development_strata: tuple[str, ...] = REQUIRED_EVIDENCE_STRATA
    key_held_out_strata: tuple[str, ...] = KEY_BOUNDARY_SENSITIVE_STRATA
    min_held_out_non_uncertain_cases: int = 200
    min_held_out_cases_per_key_stratum: int = 30
    max_held_out_policy_share: float = 0.05
    max_held_out_product_family_share: float = 0.05
    max_uncertain_overall: float = 0.10
    max_uncertain_per_key_stratum: float = 0.15
    size_grid: tuple[tuple[int, int], ...] = ((900, 1200),)
    context_token_budgets: tuple[int, ...] = (4000,)
    overlap_variants: tuple[str, ...] = (
        "zero_body_overlap",
        "preceding_semantic_unit",
    )

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("A frozen dataset release requires a version.")
        _validate_unique_identifiers(
            "required_development_strata", self.required_development_strata
        )
        _validate_unique_identifiers("key_held_out_strata", self.key_held_out_strata)
        _validate_unique_identifiers("overlap_variants", self.overlap_variants)
        if self.min_held_out_non_uncertain_cases <= 0:
            raise ValueError("min_held_out_non_uncertain_cases must be positive.")
        if self.min_held_out_cases_per_key_stratum <= 0:
            raise ValueError("min_held_out_cases_per_key_stratum must be positive.")
        for name, value in (
            ("max_held_out_policy_share", self.max_held_out_policy_share),
            (
                "max_held_out_product_family_share",
                self.max_held_out_product_family_share,
            ),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be greater than zero and at most one.")
        for name, value in (
            ("max_uncertain_overall", self.max_uncertain_overall),
            ("max_uncertain_per_key_stratum", self.max_uncertain_per_key_stratum),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one.")
        if not self.size_grid:
            raise ValueError("The development size_grid cannot be empty.")
        for target_chars, hard_max_chars in self.size_grid:
            if target_chars <= 0 or hard_max_chars < target_chars:
                raise ValueError(
                    "Every size_grid entry requires a positive target and a hard maximum at least as large."
                )
        if not self.context_token_budgets or any(
            budget <= 0 for budget in self.context_token_budgets
        ):
            raise ValueError("Development context_token_budgets must be positive.")


@dataclass(frozen=True)
class DatasetSplitReport:
    total_cases: int
    non_uncertain_cases: int
    initial_disagreements: int
    adjudicated_cases: int
    uncertain_exclusions: int

    @property
    def disagreement_rate(self) -> float:
        return _rate(self.initial_disagreements, self.total_cases)

    @property
    def adjudication_success_rate(self) -> float:
        resolved = self.adjudicated_cases - self.uncertain_exclusions
        return _rate(max(0, resolved), self.adjudicated_cases)

    @property
    def exclusion_rate(self) -> float:
        return _rate(self.uncertain_exclusions, self.total_cases)


@dataclass(frozen=True)
class DatasetFreezeReport:
    version: str
    development: DatasetSplitReport
    held_out: DatasetSplitReport
    development_stratum_counts: Mapping[str, int]
    held_out_stratum_counts: Mapping[str, int]
    key_stratum_uncertain_rates: Mapping[str, float]
    overall_uncertain_rate: float
    max_held_out_policy_share: float
    max_held_out_product_family_share: float


@dataclass(frozen=True)
class FrozenSilverDatasets:
    benchmark: FrozenBenchmark
    document_split: FrozenDocumentSplit
    config: DatasetFreezeConfig
    report: DatasetFreezeReport

    def benchmark_for(self, split: DatasetSplit) -> FrozenBenchmark:
        sources = self.document_split.sources_for(split)
        source_ids = {source.source_id for source in sources}
        cases = tuple(
            case for case in self.benchmark.cases if case.source_id in source_ids
        )
        used_annotator_ids = {
            metadata.annotator_id
            for case in cases
            for metadata in case.annotation_metadata
        }
        return FrozenBenchmark(
            version=self.benchmark.version,
            sources=sources,
            cases=cases,
            config=self.benchmark.config,
            annotation_runs=tuple(
                run
                for run in self.benchmark.annotation_runs
                if run.annotator_id in used_annotator_ids
            ),
        )

    def to_manifest(self) -> dict[str, object]:
        development_ids = {
            source.source_id
            for source in self.document_split.sources_for(DatasetSplit.DEVELOPMENT)
        }
        held_out_ids = {
            source.source_id
            for source in self.document_split.sources_for(DatasetSplit.HELD_OUT)
        }
        return {
            "version": self.config.version,
            "benchmark_manifest_sha256": self.benchmark.manifest_sha256,
            "document_split_manifest_sha256": self.document_split.manifest_sha256,
            "freeze_config": asdict(self.config),
            "document_split_manifest": self.document_split.to_manifest(),
            "benchmark_manifest": self.benchmark.to_manifest(),
            "datasets": {
                "development": [
                    case.case_id
                    for case in self.benchmark.cases
                    if case.source_id in development_ids
                ],
                "held_out": [
                    case.case_id
                    for case in self.benchmark.cases
                    if case.source_id in held_out_ids
                ],
            },
        }

    @property
    def manifest_sha256(self) -> str:
        return _sha256(
            json.dumps(
                self.to_manifest(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def freeze_document_split(
    *,
    version: str,
    sources: tuple[BenchmarkSource, ...],
    assignments: tuple[DocumentSplitAssignment, ...],
) -> FrozenDocumentSplit:
    if not version.strip():
        raise ValueError("A frozen document split requires a version.")
    if not sources:
        raise ValueError("A frozen document split requires sources.")

    source_by_id: dict[str, BenchmarkSource] = {}
    for source in sources:
        if source.source_id in source_by_id:
            raise ValueError(f"Duplicate benchmark source ID: {source.source_id}")
        if source.approval not in APPROVED_SOURCE_TYPES:
            raise ValueError(
                "Dataset sources must be approved public or project-owned text."
            )
        if not source.approval_reference.strip():
            raise ValueError("Approved dataset sources require an approval_reference.")
        if not source.pages:
            raise ValueError("Dataset sources require page-addressable text.")
        page_numbers = [page.page_number for page in source.pages]
        if len(set(page_numbers)) != len(page_numbers):
            raise ValueError("Dataset source page numbers must be unique.")
        for field_name in ("source_id", "insurer_family", "product_family"):
            if not getattr(source, field_name).strip():
                raise ValueError(f"Dataset sources require a {field_name}.")
        source_by_id[source.source_id] = source

    assignment_by_id: dict[str, DocumentSplitAssignment] = {}
    for assignment in assignments:
        if assignment.source_id in assignment_by_id:
            raise ValueError(f"Duplicate split assignment: {assignment.source_id}")
        if assignment.source_id not in source_by_id:
            raise ValueError(
                f"Split assignment references unknown source: {assignment.source_id}"
            )
        if not isinstance(assignment.split, DatasetSplit):
            raise ValueError(f"Unsupported dataset split: {assignment.split!r}")
        if not assignment.near_duplicate_family.strip():
            raise ValueError("Split assignments require a near_duplicate_family.")
        assignment_by_id[assignment.source_id] = assignment

    missing = set(source_by_id) - set(assignment_by_id)
    if missing:
        raise ValueError(
            "Every source requires a split assignment: " + ", ".join(sorted(missing))
        )
    present_splits = {assignment.split for assignment in assignments}
    if present_splits != set(DatasetSplit):
        raise ValueError("A frozen document split requires both development and held_out sources.")

    split_by_source = {
        source_id: assignment.split
        for source_id, assignment in assignment_by_id.items()
    }
    _validate_family_isolation(
        "insurer_family",
        ((source.insurer_family, split_by_source[source.source_id]) for source in sources),
    )
    _validate_family_isolation(
        "product_family",
        ((source.product_family, split_by_source[source.source_id]) for source in sources),
    )
    _validate_family_isolation(
        "near_duplicate_family",
        (
            (assignment.near_duplicate_family, assignment.split)
            for assignment in assignments
        ),
    )

    return FrozenDocumentSplit(
        version=version,
        sources=tuple(sorted(sources, key=lambda source: source.source_id)),
        assignments=tuple(sorted(assignments, key=lambda item: item.source_id)),
    )


def freeze_silver_datasets(
    *,
    benchmark: FrozenBenchmark,
    document_split: FrozenDocumentSplit,
    config: DatasetFreezeConfig,
) -> FrozenSilverDatasets:
    validated_split = freeze_document_split(
        version=document_split.version,
        sources=document_split.sources,
        assignments=document_split.assignments,
    )
    if validated_split.manifest_sha256 != document_split.manifest_sha256:
        raise ValueError("Document split does not match its frozen manifest.")

    benchmark_manifest = benchmark.to_manifest()
    split_manifest = document_split.to_manifest()
    benchmark_records = {
        record["source_id"]: record
        for record in benchmark_manifest["source_records"]
    }
    split_records = {
        record["source_id"]: record
        for record in split_manifest["source_records"]
    }
    if set(benchmark_records) != set(split_records):
        raise ValueError(
            "Frozen benchmark sources must exactly match the frozen document split."
        )
    for source_id, split_record in split_records.items():
        benchmark_record = benchmark_records[source_id]
        for field_name in (
            "insurer_family",
            "product_family",
            "source_sha256",
            "normalized_text_sha256",
        ):
            if benchmark_record[field_name] != split_record[field_name]:
                raise ValueError(
                    f"Frozen source {source_id!r} changed after the document split: {field_name}."
                )

    source_by_id = {source.source_id: source for source in benchmark.sources}
    case_ids: set[str] = set()
    for case in benchmark.cases:
        if case.case_id in case_ids:
            raise ValueError(f"Duplicate Silver case ID: {case.case_id}")
        case_ids.add(case.case_id)
        source = source_by_id.get(case.source_id)
        if source is None:
            raise ValueError(f"Silver case references unknown source: {case.source_id}")
        _validate_frozen_case(source, case)

    cases_by_split = {
        split: tuple(
            case
            for case in benchmark.cases
            if document_split.split_for(case.source_id) is split
        )
        for split in DatasetSplit
    }
    development_cases = cases_by_split[DatasetSplit.DEVELOPMENT]
    held_out_cases = cases_by_split[DatasetSplit.HELD_OUT]
    development_counts = _stratum_counts(development_cases, non_uncertain_only=True)
    held_out_counts = _stratum_counts(held_out_cases, non_uncertain_only=True)

    missing_development = [
        stratum
        for stratum in config.required_development_strata
        if development_counts.get(stratum, 0) == 0
    ]
    if missing_development:
        raise ValueError(
            "Development cases do not cover required strata: "
            + ", ".join(missing_development)
        )

    non_uncertain_held_out = tuple(
        case for case in held_out_cases if not case.annotation_uncertain
    )
    if len(non_uncertain_held_out) < config.min_held_out_non_uncertain_cases:
        raise ValueError(
            "Held-out dataset requires at least "
            f"{config.min_held_out_non_uncertain_cases} non-uncertain cases; "
            f"got {len(non_uncertain_held_out)}."
        )
    undersized_strata = [
        stratum
        for stratum in config.key_held_out_strata
        if held_out_counts.get(stratum, 0)
        < config.min_held_out_cases_per_key_stratum
    ]
    if undersized_strata:
        raise ValueError(
            "Held-out key strata do not meet the minimum case count: "
            + ", ".join(undersized_strata)
        )

    overall_uncertain_rate = _rate(
        sum(case.annotation_uncertain for case in benchmark.cases),
        len(benchmark.cases),
    )
    if overall_uncertain_rate > config.max_uncertain_overall:
        raise ValueError(
            "annotation_uncertain overall rate exceeds the frozen maximum: "
            f"{overall_uncertain_rate:.2%}."
        )
    key_uncertain_rates = {
        stratum: _uncertain_rate_for_stratum(benchmark.cases, stratum)
        for stratum in config.key_held_out_strata
    }
    unstable_strata = [
        stratum
        for stratum, rate in key_uncertain_rates.items()
        if rate > config.max_uncertain_per_key_stratum
    ]
    if unstable_strata:
        raise ValueError(
            "annotation_uncertain rate exceeds the per-stratum maximum: "
            + ", ".join(unstable_strata)
        )

    policy_counts = Counter(case.source_id for case in non_uncertain_held_out)
    product_counts = Counter(
        source_by_id[case.source_id].product_family
        for case in non_uncertain_held_out
    )
    held_out_total = len(non_uncertain_held_out)
    maximum_policy_share = max(
        (_rate(count, held_out_total) for count in policy_counts.values()),
        default=0.0,
    )
    maximum_product_share = max(
        (_rate(count, held_out_total) for count in product_counts.values()),
        default=0.0,
    )
    if maximum_policy_share > config.max_held_out_policy_share:
        raise ValueError(
            "A policy contributes more than the frozen held-out share limit: "
            f"{maximum_policy_share:.2%}."
        )
    if maximum_product_share > config.max_held_out_product_family_share:
        raise ValueError(
            "A product family contributes more than the frozen held-out share limit: "
            f"{maximum_product_share:.2%}."
        )

    report = DatasetFreezeReport(
        version=config.version,
        development=_split_report(development_cases),
        held_out=_split_report(held_out_cases),
        development_stratum_counts=development_counts,
        held_out_stratum_counts=held_out_counts,
        key_stratum_uncertain_rates=key_uncertain_rates,
        overall_uncertain_rate=overall_uncertain_rate,
        max_held_out_policy_share=maximum_policy_share,
        max_held_out_product_family_share=maximum_product_share,
    )
    return FrozenSilverDatasets(
        benchmark=benchmark,
        document_split=document_split,
        config=config,
        report=report,
    )


def generate_frozen_silver_datasets(
    *,
    document_split: FrozenDocumentSplit,
    first_pass: AnnotationPass,
    second_pass: AnnotationPass,
    adjudication_pass: AdjudicationPass,
    benchmark_config: BenchmarkConfig,
    freeze_config: DatasetFreezeConfig,
) -> FrozenSilverDatasets:
    validated_split = freeze_document_split(
        version=document_split.version,
        sources=document_split.sources,
        assignments=document_split.assignments,
    )
    benchmark = generate_frozen_benchmark(
        sources=validated_split.sources,
        first_pass=first_pass,
        second_pass=second_pass,
        adjudication_pass=adjudication_pass,
        config=benchmark_config,
    )
    return freeze_silver_datasets(
        benchmark=benchmark,
        document_split=validated_split,
        config=freeze_config,
    )


def render_dataset_freeze_markdown(report: DatasetFreezeReport) -> str:
    lines = [
        "# Silver Supporting Evidence Dataset Freeze Report",
        "",
        f"Release version: `{report.version}`",
        f"Overall annotation_uncertain rate: {report.overall_uncertain_rate:.2%}",
        "",
        "| Split | Total cases | Non-uncertain cases | Disagreement rate | Adjudication success rate | Exclusion rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in (
        ("development", report.development),
        ("held_out", report.held_out),
    ):
        lines.append(
            f"| {name} | {summary.total_cases} | {summary.non_uncertain_cases} | "
            f"{summary.disagreement_rate:.2%} | "
            f"{summary.adjudication_success_rate:.2%} | "
            f"{summary.exclusion_rate:.2%} |"
        )
    lines.extend(("", "## Development stratum coverage", ""))
    lines.extend(
        f"- `{stratum}`: {count}"
        for stratum, count in report.development_stratum_counts.items()
    )
    lines.extend(("", "## Held-out key strata", ""))
    lines.extend(
        f"- `{stratum}`: {report.held_out_stratum_counts.get(stratum, 0)} cases; "
        f"annotation_uncertain {rate:.2%}"
        for stratum, rate in report.key_stratum_uncertain_rates.items()
    )
    lines.extend(
        (
            "",
            "## Held-out diversity",
            "",
            f"- Maximum policy share: {report.max_held_out_policy_share:.2%}",
            "- Maximum product-family share: "
            f"{report.max_held_out_product_family_share:.2%}",
            "",
        )
    )
    return "\n".join(lines)


def _validate_family_isolation(
    field_name: str,
    values: Iterable[tuple[str, DatasetSplit]],
) -> None:
    split_by_family: dict[str, DatasetSplit] = {}
    for family, split in values:
        existing = split_by_family.get(family)
        if existing is not None and existing is not split:
            raise ValueError(f"{field_name} {family!r} occurs on both splits.")
        split_by_family[family] = split


def _validate_unique_identifiers(name: str, values: tuple[str, ...]) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{name} requires non-empty identifiers.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} identifiers must be unique.")


def _validate_frozen_case(source: BenchmarkSource, case: SilverCase) -> None:
    strata = case.strata
    _validate_unique_identifiers("Silver case strata", strata)
    if case.annotation_uncertain and case.evidence_spans:
        raise ValueError("annotation_uncertain cases cannot freeze evidence spans.")
    if not case.annotation_uncertain and not case.evidence_spans:
        raise ValueError("Every accepted Silver case requires exact evidence spans.")
    if case.annotation_uncertain:
        if not (
            case.initial_disagreement
            and case.adjudicated
            and case.annotation_outcome == "annotation_uncertain"
        ):
            raise ValueError(
                "annotation_uncertain cases require disagreement and third-pass adjudication metadata."
            )
    elif case.annotation_outcome == "agreed":
        if case.initial_disagreement or case.adjudicated:
            raise ValueError("Agreed Silver cases cannot record adjudication metadata.")
    elif case.annotation_outcome == "adjudicated":
        if not case.initial_disagreement or not case.adjudicated:
            raise ValueError(
                "Adjudicated Silver cases require disagreement and third-pass metadata."
            )
    else:
        raise ValueError(
            f"Unsupported Silver annotation outcome: {case.annotation_outcome!r}"
        )
    pages = {page.page_number: page.text for page in source.pages}
    for span in (*case.evidence_spans, *case.hard_negative_spans):
        page_text = pages.get(span.page_number)
        if (
            page_text is None
            or span.start_char < 0
            or span.end_char <= span.start_char
            or span.end_char > len(page_text)
            or page_text[span.start_char : span.end_char] != span.quote
        ):
            raise ValueError(
                "Silver evidence span does not exactly map to the frozen source identified by its manifest hashes."
            )
    if bool(case.hard_negative_category) != bool(case.hard_negative_spans):
        raise ValueError(
            "Hard-negative categories require exact hard-negative spans and vice versa."
        )


def _stratum_counts(
    cases: tuple[SilverCase, ...],
    *,
    non_uncertain_only: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in cases:
        if non_uncertain_only and case.annotation_uncertain:
            continue
        counts.update(case.strata)
    return dict(sorted(counts.items()))


def _uncertain_rate_for_stratum(
    cases: tuple[SilverCase, ...],
    stratum: str,
) -> float:
    matching = tuple(case for case in cases if stratum in case.strata)
    return _rate(sum(case.annotation_uncertain for case in matching), len(matching))


def _split_report(cases: tuple[SilverCase, ...]) -> DatasetSplitReport:
    return DatasetSplitReport(
        total_cases=len(cases),
        non_uncertain_cases=sum(not case.annotation_uncertain for case in cases),
        initial_disagreements=sum(case.initial_disagreement for case in cases),
        adjudicated_cases=sum(case.adjudicated for case in cases),
        uncertain_exclusions=sum(case.annotation_uncertain for case in cases),
    )


def _source_text(source: BenchmarkSource) -> str:
    return "\n\f\n".join(
        f"page:{page.page_number}\n{page.text}" for page in source.pages
    )


def _normalized_source_text(source: BenchmarkSource) -> str:
    return "\n\f\n".join(
        f"page:{page.page_number}\n{' '.join(page.text.split())}"
        for page in source.pages
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
