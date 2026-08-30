from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from insurance_rag.models import DocumentPage
from insurance_rag.silver_benchmark import (
    AnnotationDraft,
    AnnotationMetadata,
    BenchmarkConfig,
    BenchmarkSource,
    EvidenceSpan,
    FrozenBenchmark,
    SilverCase,
    SourceApproval,
    adjudicate_annotations,
    build_frozen_benchmark,
)
from insurance_rag.silver_dataset import (
    REQUIRED_EVIDENCE_STRATA,
    DatasetFreezeConfig,
    DatasetSplit,
    DocumentSplitAssignment,
    FrozenDocumentSplit,
    freeze_silver_datasets,
    freeze_document_split,
    generate_frozen_silver_datasets,
    render_dataset_freeze_markdown,
)


def _source(
    source_id: str,
    *,
    insurer_family: str,
    product_family: str,
) -> BenchmarkSource:
    return BenchmarkSource(
        source_id=source_id,
        source_name=f"{source_id}.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family=insurer_family,
        product_family=product_family,
        pages=(
            DocumentPage(
                page_number=1,
                text=f"{source_id} authoritative policy text",
                extraction_method="text",
            ),
        ),
    )


def _case(
    case_id: str,
    source: BenchmarkSource,
    stratum: str,
    *,
    additional_strata: tuple[str, ...] = (),
    uncertain: bool = False,
    disagreement: bool = False,
    adjudicated: bool = False,
) -> SilverCase:
    quote = source.pages[0].text
    return SilverCase(
        case_id=case_id,
        source_id=source.source_id,
        question=f"SECRET QUESTION {case_id}",
        evidence_spans=()
        if uncertain
        else (EvidenceSpan(1, 0, len(quote), quote),),
        stratum=stratum,
        hard_negative_category=None,
        hard_negative_spans=(),
        annotation_uncertain=uncertain,
        initial_disagreement=disagreement,
        adjudicated=adjudicated,
        annotation_outcome=(
            "annotation_uncertain"
            if uncertain
            else "adjudicated"
            if adjudicated
            else "agreed"
        ),
        annotation_metadata=(
            AnnotationMetadata(
                "fixture-annotator",
                "fixture-model-v1",
                "fixture-prompt-v1",
                (("temperature", 0),),
            ),
        ),
        additional_strata=additional_strata,
    )


def _benchmark(
    sources: tuple[BenchmarkSource, ...],
    cases: tuple[SilverCase, ...],
) -> FrozenBenchmark:
    metadata = cases[0].annotation_metadata if cases else ()
    return FrozenBenchmark(
        version="fixture-benchmark-v1",
        sources=sources,
        cases=cases,
        config=BenchmarkConfig(
            version="fixture-benchmark-v1",
            judge_id="exact-span-v1",
            embedding_model_id="fixture-embedding-v1",
            query_rewrite_version="production-v1",
            reranker_version="rules-v1",
            retrieval_depth=5,
            context_token_budget=100,
            tokenizer_id="fixture-tokenizer-v1",
        ),
        annotation_runs=metadata,
    )


def _valid_freeze_inputs():
    development = _source(
        "valid-development-policy",
        insurer_family="valid-development-insurer",
        product_family="valid-development-product",
    )
    held_out_a = _source(
        "valid-held-out-policy-a",
        insurer_family="valid-held-out-insurer-a",
        product_family="valid-held-out-product-a",
    )
    held_out_b = _source(
        "valid-held-out-policy-b",
        insurer_family="valid-held-out-insurer-b",
        product_family="valid-held-out-product-b",
    )
    sources = (development, held_out_a, held_out_b)
    split = freeze_document_split(
        version="valid-split-v1",
        sources=sources,
        assignments=(
            DocumentSplitAssignment(
                development.source_id,
                DatasetSplit.DEVELOPMENT,
                "valid-development-versions",
            ),
            DocumentSplitAssignment(
                held_out_a.source_id,
                DatasetSplit.HELD_OUT,
                "valid-held-out-a-versions",
            ),
            DocumentSplitAssignment(
                held_out_b.source_id,
                DatasetSplit.HELD_OUT,
                "valid-held-out-b-versions",
            ),
        ),
    )
    development_cases = tuple(
        _case(f"valid-development-{index}", development, stratum)
        for index, stratum in enumerate(REQUIRED_EVIDENCE_STRATA)
    )
    held_out_cases = (
        _case(
            "valid-held-out-a-1",
            held_out_a,
            "cross_page_clause",
            additional_strata=("rule_plus_exception",),
        ),
        _case("valid-held-out-a-2", held_out_a, "internally_split_clause"),
        _case("valid-held-out-b-1", held_out_b, "cross_page_clause"),
        _case(
            "valid-held-out-b-2",
            held_out_b,
            "rule_plus_exception",
            additional_strata=("internally_split_clause",),
        ),
    )
    config = DatasetFreezeConfig(
        version="valid-release-v1",
        key_held_out_strata=(
            "cross_page_clause",
            "rule_plus_exception",
            "internally_split_clause",
        ),
        min_held_out_non_uncertain_cases=4,
        min_held_out_cases_per_key_stratum=1,
        max_held_out_policy_share=0.5,
        max_held_out_product_family_share=0.5,
        size_grid=((900, 1200),),
        context_token_budgets=(100,),
        overlap_variants=("zero_body_overlap", "preceding_semantic_unit"),
    )
    return (
        _benchmark(sources, development_cases + held_out_cases),
        split,
        config,
    )


def test_document_split_is_frozen_before_generation_and_keeps_families_on_one_side():
    development = _source(
        "development-policy-v1",
        insurer_family="development-insurer",
        product_family="development-product",
    )
    held_out = _source(
        "held-out-policy-v1",
        insurer_family="held-out-insurer",
        product_family="held-out-product",
    )
    assignments = (
        DocumentSplitAssignment(
            source_id=development.source_id,
            split=DatasetSplit.DEVELOPMENT,
            near_duplicate_family="development-policy-versions",
        ),
        DocumentSplitAssignment(
            source_id=held_out.source_id,
            split=DatasetSplit.HELD_OUT,
            near_duplicate_family="held-out-policy-versions",
        ),
    )

    frozen = freeze_document_split(
        version="fixture-split-v1",
        sources=(development, held_out),
        assignments=assignments,
    )

    assert frozen.sources_for(DatasetSplit.DEVELOPMENT) == (development,)
    assert frozen.sources_for(DatasetSplit.HELD_OUT) == (held_out,)
    assert frozen.manifest_sha256
    records = frozen.to_manifest()["source_records"]
    assert [record["source_id"] for record in records] == [
        "development-policy-v1",
        "held-out-policy-v1",
    ]
    assert [record["split"] for record in records] == [
        "development",
        "held_out",
    ]
    assert records[0]["near_duplicate_family"] == "development-policy-versions"
    assert all(record["source_sha256"] for record in records)
    assert all(record["normalized_text_sha256"] for record in records)

    crossed = replace(
        assignments[1],
        near_duplicate_family="development-policy-versions",
    )
    with pytest.raises(ValueError, match="near_duplicate_family.*both splits"):
        freeze_document_split(
            version="fixture-split-v1",
            sources=(development, held_out),
            assignments=(assignments[0], crossed),
        )


@pytest.mark.parametrize("family_field", ["insurer_family", "product_family"])
def test_document_split_rejects_insurer_or_product_family_leakage(family_field):
    development = _source(
        "development-policy-v1",
        insurer_family="insurer-a",
        product_family="product-a",
    )
    held_out = _source(
        "held-out-policy-v1",
        insurer_family="insurer-b",
        product_family="product-b",
    )
    held_out = replace(
        held_out,
        **{family_field: getattr(development, family_field)},
    )

    with pytest.raises(ValueError, match=f"{family_field}.*both splits"):
        freeze_document_split(
            version="fixture-split-v1",
            sources=(development, held_out),
            assignments=(
                DocumentSplitAssignment(
                    development.source_id,
                    DatasetSplit.DEVELOPMENT,
                    "development-policy-versions",
                ),
                DocumentSplitAssignment(
                    held_out.source_id,
                    DatasetSplit.HELD_OUT,
                    "held-out-policy-versions",
                ),
            ),
        )


def test_document_split_rejects_sources_without_auditable_approval_or_pages():
    development = _source(
        "development-policy",
        insurer_family="development-insurer",
        product_family="development-product",
    )
    held_out = _source(
        "held-out-policy",
        insurer_family="held-out-insurer",
        product_family="held-out-product",
    )
    assignments = (
        DocumentSplitAssignment(
            development.source_id,
            DatasetSplit.DEVELOPMENT,
            "development-versions",
        ),
        DocumentSplitAssignment(
            held_out.source_id,
            DatasetSplit.HELD_OUT,
            "held-out-versions",
        ),
    )

    with pytest.raises(ValueError, match="approval_reference"):
        freeze_document_split(
            version="fixture-split-v1",
            sources=(replace(development, approval_reference=""), held_out),
            assignments=assignments,
        )
    with pytest.raises(ValueError, match="page-addressable"):
        freeze_document_split(
            version="fixture-split-v1",
            sources=(replace(development, pages=()), held_out),
            assignments=assignments,
        )


def test_silver_case_can_belong_to_overlapping_strata_and_freezes_them():
    source = _source(
        "overlapping-strata-policy",
        insurer_family="insurer-a",
        product_family="product-a",
    )
    quote = source.pages[0].text
    first = AnnotationDraft(
        question="Which evidence crosses boundaries?",
        evidence_spans=(EvidenceSpan(1, 0, len(quote), quote),),
        stratum="multi_sentence_conditions_outcomes",
        hard_negative_category=None,
        metadata=AnnotationMetadata("first", "model-a", "prompt-v1", (("temperature", 0),)),
        additional_strata=("cross_page_clause", "internally_split_clause"),
    )
    second = replace(
        first,
        metadata=AnnotationMetadata("second", "model-b", "prompt-v1", (("temperature", 0),)),
    )

    case = adjudicate_annotations(source, first, second)
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((first, second),)},
        adjudications={},
        config=BenchmarkConfig(
            version="fixture-v1",
            judge_id="exact-span-v1",
            embedding_model_id="fixture-embedding-v1",
            query_rewrite_version="production-v1",
            reranker_version="rules-v1",
            retrieval_depth=5,
            context_token_budget=100,
            tokenizer_id="fixture-tokenizer-v1",
        ),
    )

    assert case.strata == (
        "multi_sentence_conditions_outcomes",
        "cross_page_clause",
        "internally_split_clause",
    )
    assert benchmark.to_manifest()["cases"][0]["strata"] == list(case.strata)


def test_freeze_silver_datasets_accepts_valid_release_and_report_has_no_source_content():
    development = _source(
        "development-policy",
        insurer_family="development-insurer",
        product_family="development-product",
    )
    held_out_a = _source(
        "held-out-policy-a",
        insurer_family="held-out-insurer-a",
        product_family="held-out-product-a",
    )
    held_out_b = _source(
        "held-out-policy-b",
        insurer_family="held-out-insurer-b",
        product_family="held-out-product-b",
    )
    sources = (development, held_out_a, held_out_b)
    split = freeze_document_split(
        version="fixture-split-v1",
        sources=sources,
        assignments=(
            DocumentSplitAssignment(
                development.source_id,
                DatasetSplit.DEVELOPMENT,
                "development-versions",
            ),
            DocumentSplitAssignment(
                held_out_a.source_id,
                DatasetSplit.HELD_OUT,
                "held-out-a-versions",
            ),
            DocumentSplitAssignment(
                held_out_b.source_id,
                DatasetSplit.HELD_OUT,
                "held-out-b-versions",
            ),
        ),
    )
    development_cases = tuple(
        _case(f"development-{index}", development, stratum)
        for index, stratum in enumerate(REQUIRED_EVIDENCE_STRATA)
    )
    held_out_cases = (
        _case(
            "held-out-a-1",
            held_out_a,
            "cross_page_clause",
            additional_strata=("rule_plus_exception",),
            disagreement=True,
            adjudicated=True,
        ),
        _case("held-out-a-2", held_out_a, "internally_split_clause"),
        _case("held-out-b-1", held_out_b, "cross_page_clause"),
        _case(
            "held-out-b-2",
            held_out_b,
            "rule_plus_exception",
            additional_strata=("internally_split_clause",),
        ),
    )
    benchmark = _benchmark(sources, development_cases + held_out_cases)
    config = DatasetFreezeConfig(
        version="fixture-release-v1",
        key_held_out_strata=(
            "cross_page_clause",
            "rule_plus_exception",
            "internally_split_clause",
        ),
        min_held_out_non_uncertain_cases=4,
        min_held_out_cases_per_key_stratum=1,
        max_held_out_policy_share=0.5,
        max_held_out_product_family_share=0.5,
        size_grid=((900, 1200),),
        context_token_budgets=(100,),
        overlap_variants=("zero_body_overlap", "preceding_semantic_unit"),
    )

    release = freeze_silver_datasets(
        benchmark=benchmark,
        document_split=split,
        config=config,
    )

    manifest = release.to_manifest()
    assert manifest["benchmark_manifest_sha256"] == benchmark.manifest_sha256
    assert manifest["document_split_manifest_sha256"] == split.manifest_sha256
    assert manifest["datasets"]["development"] == [
        case.case_id for case in development_cases
    ]
    assert manifest["datasets"]["held_out"] == [
        case.case_id for case in held_out_cases
    ]
    assert manifest["benchmark_manifest"]["annotation_runs"][0]["model_id"] == (
        "fixture-model-v1"
    )
    assert release.manifest_sha256
    development_benchmark = release.benchmark_for(DatasetSplit.DEVELOPMENT)
    assert development_benchmark.sources == (development,)
    assert development_benchmark.cases == development_cases
    assert not {
        case.case_id for case in development_benchmark.cases
    }.intersection(manifest["datasets"]["held_out"])

    markdown = render_dataset_freeze_markdown(release.report)
    assert "Disagreement rate" in markdown
    assert "Adjudication success rate" in markdown
    assert "Exclusion rate" in markdown
    assert "SECRET QUESTION" not in markdown
    assert development.pages[0].text not in markdown


def test_generation_revalidates_document_split_before_annotation_callbacks():
    development = _source(
        "development-policy",
        insurer_family="shared-insurer",
        product_family="development-product",
    )
    held_out = _source(
        "held-out-policy",
        insurer_family="shared-insurer",
        product_family="held-out-product",
    )
    invalid_split = FrozenDocumentSplit(
        version="invalid-split-v1",
        sources=(development, held_out),
        assignments=(
            DocumentSplitAssignment(
                development.source_id,
                DatasetSplit.DEVELOPMENT,
                "development-versions",
            ),
            DocumentSplitAssignment(
                held_out.source_id,
                DatasetSplit.HELD_OUT,
                "held-out-versions",
            ),
        ),
    )
    annotation_calls: list[str] = []

    def annotation_pass(source):
        annotation_calls.append(source.source_id)
        return ()

    with pytest.raises(ValueError, match="insurer_family.*both splits"):
        generate_frozen_silver_datasets(
            document_split=invalid_split,
            first_pass=annotation_pass,
            second_pass=annotation_pass,
            adjudication_pass=lambda *_args: pytest.fail(
                "Invalid split must fail before adjudication"
            ),
            benchmark_config=BenchmarkConfig(
                version="fixture-benchmark-v1",
                judge_id="exact-span-v1",
                embedding_model_id="fixture-embedding-v1",
                query_rewrite_version="production-v1",
                reranker_version="rules-v1",
                retrieval_depth=5,
                context_token_budget=100,
                tokenizer_id="fixture-tokenizer-v1",
            ),
            freeze_config=DatasetFreezeConfig(version="fixture-release-v1"),
        )

    assert annotation_calls == []


def test_production_dataset_threshold_defaults_match_issue_16():
    config = DatasetFreezeConfig(version="production-defaults-v1")

    assert config.min_held_out_non_uncertain_cases == 200
    assert config.min_held_out_cases_per_key_stratum == 30
    assert config.max_held_out_policy_share == 0.05
    assert config.max_held_out_product_family_share == 0.05
    assert config.max_uncertain_overall == 0.10
    assert config.max_uncertain_per_key_stratum == 0.15
    assert config.required_development_strata == REQUIRED_EVIDENCE_STRATA


@pytest.mark.parametrize(
    ("config_change", "message"),
    [
        ({"min_held_out_non_uncertain_cases": 5}, "at least 5 non-uncertain"),
        (
            {
                "key_held_out_strata": ("single_sentence",),
                "min_held_out_cases_per_key_stratum": 1,
            },
            "key strata.*minimum",
        ),
        ({"max_held_out_policy_share": 0.49}, "policy contributes more"),
        (
            {"max_held_out_product_family_share": 0.49},
            "product family contributes more",
        ),
    ],
)
def test_freeze_rejects_held_out_size_stratum_and_diversity_failures(
    config_change,
    message,
):
    benchmark, split, config = _valid_freeze_inputs()

    with pytest.raises(ValueError, match=message):
        freeze_silver_datasets(
            benchmark=benchmark,
            document_split=split,
            config=replace(config, **config_change),
        )


def test_freeze_rejects_missing_development_stratum():
    benchmark, split, config = _valid_freeze_inputs()
    without_single_sentence = replace(
        benchmark,
        cases=tuple(
            case
            for case in benchmark.cases
            if not (
                split.split_for(case.source_id) is DatasetSplit.DEVELOPMENT
                and case.stratum == "single_sentence"
            )
        ),
    )

    with pytest.raises(ValueError, match="Development.*single_sentence"):
        freeze_silver_datasets(
            benchmark=without_single_sentence,
            document_split=split,
            config=config,
        )


def test_freeze_rejects_overall_and_per_stratum_uncertainty_limits():
    benchmark, split, config = _valid_freeze_inputs()
    held_out_source = split.sources_for(DatasetSplit.HELD_OUT)[0]
    uncertain = _case(
        "uncertain-held-out",
        held_out_source,
        "cross_page_clause",
        uncertain=True,
        disagreement=True,
        adjudicated=True,
    )
    unstable = replace(benchmark, cases=(*benchmark.cases, uncertain))

    with pytest.raises(ValueError, match="overall rate exceeds"):
        freeze_silver_datasets(
            benchmark=unstable,
            document_split=split,
            config=replace(config, max_uncertain_overall=0.05),
        )
    with pytest.raises(ValueError, match="per-stratum maximum.*cross_page_clause"):
        freeze_silver_datasets(
            benchmark=unstable,
            document_split=split,
            config=replace(
                config,
                max_uncertain_overall=0.10,
                max_uncertain_per_key_stratum=0.20,
            ),
        )


def test_freeze_rejects_inconsistent_uncertain_adjudication_metadata():
    benchmark, split, config = _valid_freeze_inputs()
    held_out_source = split.sources_for(DatasetSplit.HELD_OUT)[0]
    bypassed = _case(
        "bypassed-uncertain-held-out",
        held_out_source,
        "cross_page_clause",
        uncertain=True,
        disagreement=False,
        adjudicated=False,
    )

    with pytest.raises(ValueError, match="annotation_uncertain.*adjudication"):
        freeze_silver_datasets(
            benchmark=replace(benchmark, cases=(*benchmark.cases, bypassed)),
            document_split=split,
            config=replace(
                config,
                max_uncertain_overall=1.0,
                max_uncertain_per_key_stratum=1.0,
            ),
        )


def test_freeze_revalidates_exact_spans_against_manifest_hashed_source():
    benchmark, split, config = _valid_freeze_inputs()
    original = benchmark.cases[-1]
    invalid = replace(
        original,
        evidence_spans=(
            replace(original.evidence_spans[0], quote="tampered policy content"),
        ),
    )

    with pytest.raises(ValueError, match="exactly map.*manifest hashes"):
        freeze_silver_datasets(
            benchmark=replace(benchmark, cases=(*benchmark.cases[:-1], invalid)),
            document_split=split,
            config=config,
        )


def test_release_hash_changes_when_frozen_selection_inputs_or_labels_change():
    benchmark, split, config = _valid_freeze_inputs()
    original = freeze_silver_datasets(
        benchmark=benchmark,
        document_split=split,
        config=config,
    )
    changed_selection = freeze_silver_datasets(
        benchmark=benchmark,
        document_split=split,
        config=replace(config, size_grid=((800, 1000),)),
    )
    changed_case = replace(
        benchmark.cases[-1],
        additional_strata=(
            *benchmark.cases[-1].additional_strata,
            "complete_short_clause",
        ),
    )
    changed_labels = freeze_silver_datasets(
        benchmark=replace(
            benchmark,
            cases=(*benchmark.cases[:-1], changed_case),
        ),
        document_split=split,
        config=config,
    )

    assert original.manifest_sha256 != changed_selection.manifest_sha256
    assert original.manifest_sha256 != changed_labels.manifest_sha256


def test_source_bearing_dataset_manifests_are_ignored_by_git():
    ignored = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")

    assert "silver_dataset_manifests/" in ignored.splitlines()
