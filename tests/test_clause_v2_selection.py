from dataclasses import replace

import pytest

import insurance_rag.clause_v2_selection as selection_module

from insurance_rag.clause_v2_selection import (
    ClauseV2NoSelectionManifest,
    ClauseV2SelectionManifest,
    DevelopmentSelection,
    DevelopmentTrial,
    render_development_selection_markdown,
    run_development_selection,
    select_development_trial,
)
from insurance_rag.silver_benchmark import (
    AnnotationSummary,
    BenchmarkConfig,
    BenchmarkReport,
    FrozenBenchmark,
    StrategyResult,
)
from insurance_rag.silver_dataset import DatasetSplit


def _result(name: str, coverage: tuple[float, ...]) -> StrategyResult:
    count = len(coverage)
    return StrategyResult(
        strategy_name=name,
        chunking_strategy="legacy" if name == "legacy" else "clause_v2",
        target_chars=900,
        hard_max_chars=1200,
        body_overlap_mode=(
            "preceding_semantic_unit"
            if name.endswith("preceding_semantic_unit")
            else "zero_body_overlap"
        ),
        scored_cases=count,
        coverage_at={1: 0.0, 3: 0.0, 5: 0.0},
        coverage_under_token_budget=sum(coverage) / count,
        single_candidate_coverage=0.0,
        irrelevant_context_proportion=0.1,
        hard_negative_confusions={"similar_clause": 0},
        embedding_tokens=1100 if name != "legacy" else 1000,
        retrieval_unit_count=110 if name != "legacy" else 100,
        chunking_latency_seconds=1.0,
        p95_chunking_latency_seconds=1.0,
        coverage_under_budget_by_stratum={
            "multi_sentence_conditions_outcomes": sum(coverage) / count
        },
        boundary_diagnostics={},
        correctness_invariants={"hard_max_chars_respected": True},
        case_coverage_at_3=tuple(0.0 for _ in coverage),
        case_coverage_under_budget=coverage,
        case_irrelevant_context_proportion=tuple(0.1 for _ in coverage),
        case_hard_negative_confusion=tuple(0.0 for _ in coverage),
        case_hard_negative_applicable=tuple(True for _ in coverage),
        case_strata=tuple(
            ("multi_sentence_conditions_outcomes",) for _ in coverage
        ),
    )


def _trial(
    *,
    zero_improvements: int = 4,
    semantic_improvements: int = 4,
) -> DevelopmentTrial:
    baseline_coverage = tuple(0.0 for _ in range(40))
    zero_coverage = tuple(
        1.0 if index < zero_improvements else 0.0 for index in range(40)
    )
    semantic_coverage = tuple(
        1.0 if index < semantic_improvements else 0.0
        for index in range(40)
    )
    report = BenchmarkReport(
        benchmark_version="development-v1",
        manifest_sha256="development-manifest",
        annotation_summary=AnnotationSummary(40, 0, 0, 0),
        strategy_results=(
            _result("legacy", baseline_coverage),
            _result("clause_v2_zero_body_overlap", zero_coverage),
            _result(
                "clause_v2_preceding_semantic_unit", semantic_coverage
            ),
        ),
        paired_comparisons=(),
    )
    return DevelopmentTrial(900, 1200, 4000, report)


def test_statistically_unclear_overlap_selects_zero_body_overlap():
    trial = _trial()

    selected_trial, selected_result = select_development_trial(
        (trial,), primary_context_token_budget=4000, bootstrap_samples=1000
    )

    assert selected_trial is trial
    assert selected_result.strategy_name == "clause_v2_zero_body_overlap"


def test_semantic_overlap_requires_clear_paired_gain_over_zero_overlap():
    trial = _trial(zero_improvements=4, semantic_improvements=8)

    _, selected_result = select_development_trial(
        (trial,), primary_context_token_budget=4000, bootstrap_samples=1000
    )

    assert (
        selected_result.strategy_name
        == "clause_v2_preceding_semantic_unit"
    )


def test_selection_compares_all_trials_instead_of_taking_input_order():
    weaker = _trial(zero_improvements=4, semantic_improvements=4)
    stronger = replace(
        _trial(zero_improvements=8, semantic_improvements=8),
        target_chars=1200,
        hard_max_chars=1600,
    )

    selected_trial, _ = select_development_trial(
        (weaker, stronger),
        primary_context_token_budget=4000,
        bootstrap_samples=1000,
    )
    reversed_trial, _ = select_development_trial(
        (stronger, weaker),
        primary_context_token_budget=4000,
        bootstrap_samples=1000,
    )

    assert selected_trial is stronger
    assert reversed_trial is stronger


def test_semantic_overlap_can_win_when_zero_overlap_fails_guardrails():
    trial = _trial(zero_improvements=0, semantic_improvements=8)

    _, selected_result = select_development_trial(
        (trial,), primary_context_token_budget=4000, bootstrap_samples=1000
    )

    assert selected_result.body_overlap_mode == "preceding_semantic_unit"


def test_development_selection_rejects_held_out_before_retrieval_runs():
    config = BenchmarkConfig(
        version="fixture-v1",
        judge_id="exact-span-v1",
        embedding_model_id="fixture-embedding-v1",
        query_rewrite_version="production-v1",
        reranker_version="rules-v1",
        retrieval_depth=5,
        context_token_budget=100,
        tokenizer_id="fixture-tokenizer-v1",
    )
    benchmark = FrozenBenchmark(
        version=config.version,
        sources=(),
        cases=(),
        config=config,
        annotation_runs=(),
    )

    with pytest.raises(ValueError, match="development labels only"):
        run_development_selection(
            benchmark,
            dataset_split=DatasetSplit.HELD_OUT,
            size_grid=((900, 1200),),
            context_token_budgets=(100,),
            primary_context_token_budget=100,
            document_split_manifest_sha256="split-manifest",
            embedder=None,  # type: ignore[arg-type]
            token_counter=None,  # type: ignore[arg-type]
        )


def test_every_trial_requires_exactly_the_three_preregistered_families():
    trial = _trial()
    incomplete = replace(
        trial,
        report=replace(
            trial.report,
            strategy_results=trial.report.strategy_results[:2],
        ),
    )

    with pytest.raises(ValueError, match="exactly the three"):
        select_development_trial(
            (incomplete,), primary_context_token_budget=4000
        )


def test_no_eligible_candidate_is_recorded_without_forcing_a_selection():
    trial = _trial(zero_improvements=0, semantic_improvements=0)
    manifest = ClauseV2NoSelectionManifest(
        benchmark_version="development-v1",
        development_benchmark_manifest_sha256="development-manifest",
        document_split_manifest_sha256="split-manifest",
        evaluated_size_grid=((900, 1200),),
        evaluated_context_token_budgets=(4000,),
        primary_context_token_budget=4000,
    )
    selection = DevelopmentSelection(
        trials=(trial,),
        manifest=manifest,
        bootstrap_samples=100,
    )

    markdown = render_development_selection_markdown(selection)

    assert manifest.to_manifest()["strategy"] is None
    assert manifest.to_manifest()["selection_status"] == "no_eligible_candidate"
    assert "no configuration was selected or promoted" in markdown
    assert "FAIL" in markdown
    assert "semantic_overlap_complete: FAIL" not in markdown


def test_selection_uses_only_the_preregistered_primary_budget():
    primary = _trial(zero_improvements=4, semantic_improvements=4)
    non_primary = replace(
        _trial(zero_improvements=8, semantic_improvements=8),
        context_token_budget=8000,
    )

    selected_trial, _ = select_development_trial(
        (non_primary, primary),
        primary_context_token_budget=4000,
        bootstrap_samples=1000,
    )

    assert selected_trial is primary


def test_selected_manifest_references_the_independent_benchmark(monkeypatch):
    config = BenchmarkConfig(
        version="fixture-v1",
        judge_id="exact-span-v1",
        embedding_model_id="fixture-embedding-v1",
        query_rewrite_version="production-v1",
        reranker_version="rules-v1",
        retrieval_depth=5,
        context_token_budget=4000,
        tokenizer_id="fixture-tokenizer-v1",
    )
    benchmark = FrozenBenchmark(
        version=config.version,
        sources=(),
        cases=(),
        config=config,
        annotation_runs=(),
    )
    trial_report = replace(
        _trial().report,
        manifest_sha256="temporary-trial-manifest",
    )
    monkeypatch.setattr(
        selection_module,
        "run_frozen_benchmark",
        lambda *_args, **_kwargs: trial_report,
    )

    selection = run_development_selection(
        benchmark,
        dataset_split=DatasetSplit.DEVELOPMENT,
        size_grid=((900, 1200),),
        context_token_budgets=(4000,),
        primary_context_token_budget=4000,
        document_split_manifest_sha256="split-manifest",
        embedder=None,  # type: ignore[arg-type]
        token_counter=None,  # type: ignore[arg-type]
        bootstrap_samples=100,
    )

    assert (
        selection.manifest.development_benchmark_manifest_sha256
        == benchmark.manifest_sha256
    )


def test_successful_selection_report_does_not_claim_no_candidate_passed():
    manifest = ClauseV2SelectionManifest(
        strategy="clause_v2",
        target_chars=900,
        hard_max_chars=1200,
        body_overlap_mode="zero_body_overlap",
        context_token_budget=4000,
        tokenizer_id="fixture-tokenizer-v1",
        benchmark_version="development-v1",
        development_benchmark_manifest_sha256="development-manifest",
        document_split_manifest_sha256="split-manifest",
        retrieval_configuration={},
    )
    selection = DevelopmentSelection(
        trials=(_trial(),),
        manifest=manifest,
        bootstrap_samples=100,
    )

    markdown = render_development_selection_markdown(selection)

    assert "## Failure analysis" not in markdown
    assert "No candidate passed" not in markdown
    assert "Status: `selected`" in markdown
