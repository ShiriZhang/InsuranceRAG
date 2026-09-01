from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version as package_version
from typing import Mapping

from insurance_rag.hybrid_retriever import (
    BM25_TOKENIZER_VERSION,
    DEFAULT_RRF_K,
)
from insurance_rag.silver_benchmark import (
    BenchmarkReport,
    Embedder,
    FrozenBenchmark,
    StrategyConfig,
    StrategyResult,
    TokenCounter,
    paired_confidence_interval,
    render_benchmark_markdown,
    run_frozen_benchmark,
)
from insurance_rag.silver_dataset import DatasetSplit


BOUNDARY_SENSITIVE_STRATA = frozenset(
    {
        "multi_sentence_conditions_outcomes",
        "rule_plus_exception",
        "cross_page_clause",
        "internally_split_clause",
    }
)
SELECTION_RULES_VERSION = "clause-v2-development-selection/v1.0.0"


@dataclass(frozen=True)
class DevelopmentTrial:
    target_chars: int
    hard_max_chars: int
    context_token_budget: int
    report: BenchmarkReport


@dataclass(frozen=True)
class ClauseV2SelectionManifest:
    strategy: str
    target_chars: int
    hard_max_chars: int
    body_overlap_mode: str
    context_token_budget: int
    tokenizer_id: str
    benchmark_version: str
    development_benchmark_manifest_sha256: str
    document_split_manifest_sha256: str
    retrieval_configuration: Mapping[str, object]
    selection_rules_version: str = SELECTION_RULES_VERSION

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClauseV2NoSelectionManifest:
    benchmark_version: str
    development_benchmark_manifest_sha256: str
    document_split_manifest_sha256: str
    evaluated_size_grid: tuple[tuple[int, int], ...]
    evaluated_context_token_budgets: tuple[int, ...]
    primary_context_token_budget: int
    selection_status: str = "no_eligible_candidate"
    strategy: None = None
    reason: str = "No clause_v2 development candidate satisfies every promotion rule."
    selection_rules_version: str = SELECTION_RULES_VERSION

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GuardrailEvaluation:
    name: str
    value: float | bool | None
    requirement: str
    passed: bool


@dataclass(frozen=True)
class DevelopmentSelection:
    trials: tuple[DevelopmentTrial, ...]
    manifest: ClauseV2SelectionManifest | ClauseV2NoSelectionManifest
    bootstrap_samples: int


def run_development_selection(
    benchmark: FrozenBenchmark,
    *,
    dataset_split: DatasetSplit,
    size_grid: tuple[tuple[int, int], ...],
    context_token_budgets: tuple[int, ...],
    primary_context_token_budget: int,
    document_split_manifest_sha256: str,
    embedder: Embedder,
    token_counter: TokenCounter,
    bootstrap_samples: int = 2000,
) -> DevelopmentSelection:
    if dataset_split is not DatasetSplit.DEVELOPMENT:
        raise ValueError(
            "Clause v2 selection may read development labels only; held-out scoring is separate."
        )
    if not size_grid or not context_token_budgets:
        raise ValueError("Development size and context-budget grids cannot be empty.")
    if primary_context_token_budget not in context_token_budgets:
        raise ValueError(
            "The primary context-token budget must be in the frozen budget grid."
        )

    trials: list[DevelopmentTrial] = []
    for target_chars, hard_max_chars in size_grid:
        for context_token_budget in context_token_budgets:
            configured_benchmark = replace(
                benchmark,
                config=replace(
                    benchmark.config,
                    context_token_budget=context_token_budget,
                ),
            )
            report = run_frozen_benchmark(
                configured_benchmark,
                strategies=(
                    StrategyConfig(
                        name="legacy",
                        chunking_strategy="legacy",
                        target_chars=900,
                        hard_max_chars=900,
                    ),
                    StrategyConfig(
                        name="clause_v2_zero_body_overlap",
                        chunking_strategy="clause_v2",
                        target_chars=target_chars,
                        hard_max_chars=hard_max_chars,
                    ),
                    StrategyConfig(
                        name="clause_v2_preceding_semantic_unit",
                        chunking_strategy="clause_v2",
                        target_chars=target_chars,
                        hard_max_chars=hard_max_chars,
                        body_overlap_mode="preceding_semantic_unit",
                    ),
                ),
                embedder=embedder,
                token_counter=token_counter,
                bootstrap_samples=bootstrap_samples,
            )
            trials.append(
                DevelopmentTrial(
                    target_chars=target_chars,
                    hard_max_chars=hard_max_chars,
                    context_token_budget=context_token_budget,
                    report=report,
                )
            )

    config = benchmark.config
    try:
        selected_trial, selected_result = select_development_trial(
            tuple(trials),
            primary_context_token_budget=primary_context_token_budget,
            bootstrap_samples=bootstrap_samples,
        )
    except ValueError as error:
        if str(error) != (
            "No clause_v2 development candidate satisfies every promotion rule."
        ):
            raise
        return DevelopmentSelection(
            trials=tuple(trials),
            manifest=ClauseV2NoSelectionManifest(
                benchmark_version=benchmark.version,
                development_benchmark_manifest_sha256=benchmark.manifest_sha256,
                document_split_manifest_sha256=document_split_manifest_sha256,
                evaluated_size_grid=size_grid,
                evaluated_context_token_budgets=context_token_budgets,
                primary_context_token_budget=primary_context_token_budget,
            ),
            bootstrap_samples=bootstrap_samples,
        )
    return DevelopmentSelection(
        trials=tuple(trials),
        manifest=ClauseV2SelectionManifest(
            strategy="clause_v2",
            target_chars=selected_trial.target_chars,
            hard_max_chars=selected_trial.hard_max_chars,
            body_overlap_mode=(
                selected_result.body_overlap_mode
            ),
            context_token_budget=selected_trial.context_token_budget,
            tokenizer_id=config.tokenizer_id,
            benchmark_version=benchmark.version,
            development_benchmark_manifest_sha256=(
                benchmark.manifest_sha256
            ),
            document_split_manifest_sha256=document_split_manifest_sha256,
            retrieval_configuration={
                "embedding_model_id": config.embedding_model_id,
                "retrieval_mode": "hybrid",
                "lexical_retrieval": {
                    "algorithm": "BM25Okapi",
                    "package": "rank-bm25",
                    "package_version": package_version("rank-bm25"),
                    "tokenizer_version": BM25_TOKENIZER_VERSION,
                },
                "fusion": {
                    "algorithm": "reciprocal_rank_fusion",
                    "rrf_k": DEFAULT_RRF_K,
                },
                "query_rewrite_version": config.query_rewrite_version,
                "reranker_version": config.reranker_version,
                "retrieval_depth": config.retrieval_depth,
                "context_token_budget": selected_trial.context_token_budget,
            },
        ),
        bootstrap_samples=bootstrap_samples,
    )


def select_development_trial(
    trials: tuple[DevelopmentTrial, ...],
    *,
    primary_context_token_budget: int,
    bootstrap_samples: int = 2000,
) -> tuple[DevelopmentTrial, StrategyResult]:
    eligible: list[tuple[DevelopmentTrial, StrategyResult]] = []
    primary_trial_found = False
    for trial in trials:
        results = {result.strategy_name: result for result in trial.report.strategy_results}
        required = {
            "legacy",
            "clause_v2_zero_body_overlap",
            "clause_v2_preceding_semantic_unit",
        }
        if set(results) != required:
            raise ValueError(
                "Every development trial must compare exactly the three frozen strategy families."
            )
        if trial.context_token_budget != primary_context_token_budget:
            continue
        primary_trial_found = True
        baseline = results["legacy"]
        zero_overlap = results["clause_v2_zero_body_overlap"]
        semantic_overlap = results["clause_v2_preceding_semantic_unit"]
        if _passes_guardrails(
            baseline, zero_overlap, bootstrap_samples=bootstrap_samples
        ):
            eligible.append((trial, zero_overlap))
        if _passes_guardrails(
            baseline, semantic_overlap, bootstrap_samples=bootstrap_samples
        ):
            overlap_gain = paired_confidence_interval(
                zero_overlap.case_coverage_under_budget,
                semantic_overlap.case_coverage_under_budget,
                bootstrap_samples=bootstrap_samples,
            )
            if overlap_gain.lower > 0:
                eligible.append((trial, semantic_overlap))
    if not primary_trial_found:
        raise ValueError("No development trial uses the primary context-token budget.")
    if not eligible:
        raise ValueError(
            "No clause_v2 development candidate satisfies every promotion rule."
        )
    return max(eligible, key=_candidate_selection_key)


def render_development_selection_markdown(
    selection: DevelopmentSelection,
) -> str:
    sections = ["# Clause v2 development selection", ""]
    for trial in selection.trials:
        results = {
            result.strategy_name: result
            for result in trial.report.strategy_results
        }
        sections.extend(
            (
                f"## target={trial.target_chars}, hard_max={trial.hard_max_chars}, budget={trial.context_token_budget}",
                "",
                "### Promotion guardrails",
                "",
                "| Candidate | Guardrail | Value | Requirement | Result |",
                "| --- | --- | ---: | --- | --- |",
            )
        )
        baseline = results["legacy"]
        for candidate_name in (
            "clause_v2_zero_body_overlap",
            "clause_v2_preceding_semantic_unit",
        ):
            for evaluation in evaluate_guardrails(
                baseline,
                results[candidate_name],
                bootstrap_samples=selection.bootstrap_samples,
            ):
                value = (
                    "n/a"
                    if evaluation.value is None
                    else str(evaluation.value).lower()
                    if isinstance(evaluation.value, bool)
                    else f"{evaluation.value:.6f}"
                )
                sections.append(
                    f"| `{candidate_name}` | `{evaluation.name}` | {value} | "
                    f"{evaluation.requirement} | "
                    f"{'PASS' if evaluation.passed else 'FAIL'} |"
                )
        sections.extend(
            (
                "",
                render_benchmark_markdown(trial.report),
            )
        )
    if isinstance(selection.manifest, ClauseV2NoSelectionManifest):
        sections.extend(_render_failure_analysis(selection))
    sections.extend(("## Frozen selection", ""))
    if isinstance(selection.manifest, ClauseV2SelectionManifest):
        sections.extend(
            (
                f"- Status: `selected`",
                f"- Strategy: `{selection.manifest.strategy}`",
                f"- Body overlap: `{selection.manifest.body_overlap_mode}`",
                f"- Size: `{selection.manifest.target_chars}/{selection.manifest.hard_max_chars}` characters",
                f"- Context budget: `{selection.manifest.context_token_budget}` tokens using `{selection.manifest.tokenizer_id}`",
                "",
            )
        )
    else:
        sections.extend(
            (
                f"- Status: `{selection.manifest.selection_status}`",
                f"- Decision: no configuration was selected or promoted.",
                f"- Reason: {selection.manifest.reason}",
                "",
            )
        )
    return "\n".join(sections)


def _render_failure_analysis(
    selection: DevelopmentSelection,
) -> tuple[str, ...]:
    candidate_evaluations: list[tuple[GuardrailEvaluation, ...]] = []
    semantic_overlap_incomplete = False
    for trial in selection.trials:
        results = {
            result.strategy_name: result
            for result in trial.report.strategy_results
        }
        baseline = results["legacy"]
        for candidate_name in (
            "clause_v2_zero_body_overlap",
            "clause_v2_preceding_semantic_unit",
        ):
            candidate = results[candidate_name]
            if candidate_name == "clause_v2_preceding_semantic_unit":
                semantic_overlap_incomplete = semantic_overlap_incomplete or not (
                    candidate.correctness_invariants.get(
                        "semantic_overlap_complete", True
                    )
                )
            candidate_evaluations.append(
                evaluate_guardrails(
                    baseline,
                    candidate,
                    bootstrap_samples=selection.bootstrap_samples,
                )
            )
    candidate_count = len(candidate_evaluations)
    passing_count = sum(
        all(evaluation.passed for evaluation in evaluations)
        for evaluations in candidate_evaluations
    )
    failures = Counter(
        evaluation.name
        for evaluations in candidate_evaluations
        for evaluation in evaluations
        if not evaluation.passed
    )
    lines = ["## Failure analysis", ""]
    if not failures:
        lines.extend(("All evaluated candidates passed every guardrail.", ""))
        return tuple(lines)
    lines.extend(
        (
            (
                f"No candidate passed every guardrail (0/{candidate_count})."
                if passing_count == 0
                else f"{passing_count}/{candidate_count} candidates passed every guardrail."
            ),
            "",
            "Failed guardrails across the full sensitivity grid:",
            "",
        )
    )
    for name, count in sorted(
        failures.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"- `{name}`: {count}/{candidate_count} candidates failed.")
    universal = sorted(
        name for name, count in failures.items() if count == candidate_count
    )
    if universal:
        lines.extend(
            (
                "",
                "Universal blockers: "
                + ", ".join(f"`{name}`" for name in universal)
                + ".",
            )
        )
    lines.append("")
    if semantic_overlap_incomplete:
        lines.append(
            "The semantic-overlap candidates also report "
            "`semantic_overlap_complete: FAIL` when a complete preceding semantic "
            "unit cannot fit without exceeding `hard_max_chars`. No threshold was "
            "relaxed and no candidate was selected."
        )
    else:
        lines.append("No threshold was relaxed and no candidate was selected.")
    lines.append("")
    return tuple(lines)


def _passes_guardrails(
    baseline: StrategyResult,
    candidate: StrategyResult,
    *,
    bootstrap_samples: int,
) -> bool:
    return all(
        evaluation.passed
        for evaluation in evaluate_guardrails(
            baseline,
            candidate,
            bootstrap_samples=bootstrap_samples,
        )
    )


def evaluate_guardrails(
    baseline: StrategyResult,
    candidate: StrategyResult,
    *,
    bootstrap_samples: int,
) -> tuple[GuardrailEvaluation, ...]:
    coverage_at_3 = paired_confidence_interval(
        baseline.case_coverage_at_3,
        candidate.case_coverage_at_3,
        bootstrap_samples=bootstrap_samples,
    )
    coverage_under_budget = paired_confidence_interval(
        baseline.case_coverage_under_budget,
        candidate.case_coverage_under_budget,
        bootstrap_samples=bootstrap_samples,
    )
    boundary_indexes = tuple(
        index
        for index, strata in enumerate(candidate.case_strata)
        if BOUNDARY_SENSITIVE_STRATA.intersection(strata)
    )
    boundary_coverage = (
        paired_confidence_interval(
            tuple(
                baseline.case_coverage_under_budget[index]
                for index in boundary_indexes
            ),
            tuple(
                candidate.case_coverage_under_budget[index]
                for index in boundary_indexes
            ),
            bootstrap_samples=bootstrap_samples,
        )
        if boundary_indexes
        else None
    )
    hard_negative_indexes = tuple(
        index
        for index, applicable in enumerate(
            baseline.case_hard_negative_applicable
        )
        if applicable
    )
    hard_negative = (
        paired_confidence_interval(
            tuple(
                baseline.case_hard_negative_confusion[index]
                for index in hard_negative_indexes
            ),
            tuple(
                candidate.case_hard_negative_confusion[index]
                for index in hard_negative_indexes
            ),
            bootstrap_samples=bootstrap_samples,
        )
        if hard_negative_indexes
        else None
    )
    irrelevant_context = paired_confidence_interval(
        baseline.case_irrelevant_context_proportion,
        candidate.case_irrelevant_context_proportion,
        bootstrap_samples=bootstrap_samples,
    )
    embedding_ratio = _ratio(candidate.embedding_tokens, baseline.embedding_tokens)
    retrieval_unit_ratio = _ratio(
        candidate.retrieval_unit_count,
        baseline.retrieval_unit_count,
    )
    latency_ratio = _ratio(
        candidate.p95_chunking_latency_seconds,
        baseline.p95_chunking_latency_seconds,
    )
    return (
        GuardrailEvaluation(
            "coverage_at_3_ci_lower",
            coverage_at_3.lower,
            ">= -0.01",
            coverage_at_3.lower >= -0.01,
        ),
        GuardrailEvaluation(
            "coverage_budget_ci_lower",
            coverage_under_budget.lower,
            ">= -0.01",
            coverage_under_budget.lower >= -0.01,
        ),
        GuardrailEvaluation(
            "boundary_coverage_gain",
            None if boundary_coverage is None else boundary_coverage.estimate,
            ">= 0.05",
            boundary_coverage is not None and boundary_coverage.estimate >= 0.05,
        ),
        GuardrailEvaluation(
            "boundary_coverage_ci_lower",
            None if boundary_coverage is None else boundary_coverage.lower,
            "> 0",
            boundary_coverage is not None and boundary_coverage.lower > 0,
        ),
        GuardrailEvaluation(
            "hard_negative_ci_upper",
            None if hard_negative is None else hard_negative.upper,
            "<= 0.01",
            hard_negative is not None and hard_negative.upper <= 0.01,
        ),
        GuardrailEvaluation(
            "irrelevant_context_ci_upper",
            irrelevant_context.upper,
            "<= 0.02",
            irrelevant_context.upper <= 0.02,
        ),
        GuardrailEvaluation(
            "embedding_token_ratio",
            embedding_ratio,
            "<= 1.15",
            embedding_ratio <= 1.15,
        ),
        GuardrailEvaluation(
            "retrieval_unit_ratio",
            retrieval_unit_ratio,
            "<= 1.25",
            retrieval_unit_ratio <= 1.25,
        ),
        GuardrailEvaluation(
            "p95_chunking_latency_ratio",
            latency_ratio,
            "<= 2.0",
            latency_ratio <= 2,
        ),
        GuardrailEvaluation(
            "correctness_invariants",
            all(candidate.correctness_invariants.values()),
            "all true",
            all(candidate.correctness_invariants.values()),
        ),
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def _candidate_selection_key(
    candidate: tuple[DevelopmentTrial, StrategyResult],
) -> tuple[float, ...]:
    trial, result = candidate
    boundary_values = [
        value
        for name, value in result.coverage_under_budget_by_stratum.items()
        if name in BOUNDARY_SENSITIVE_STRATA
    ]
    boundary_coverage = (
        sum(boundary_values) / len(boundary_values)
        if boundary_values
        else 0.0
    )
    return (
        result.coverage_under_token_budget,
        result.coverage_at[3],
        boundary_coverage,
        float(result.body_overlap_mode == "zero_body_overlap"),
        -float(result.embedding_tokens),
        -float(result.retrieval_unit_count),
        -result.p95_chunking_latency_seconds,
        -float(trial.context_token_budget),
        -float(trial.hard_max_chars),
        -float(trial.target_chars),
    )
