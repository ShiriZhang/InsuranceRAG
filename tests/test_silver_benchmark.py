from __future__ import annotations

from dataclasses import replace

import fitz
import pytest

from insurance_rag.config import AppConfig
from insurance_rag.models import DocumentPage
from insurance_rag.silver_benchmark import (
    AnnotationDraft,
    AnnotationMetadata,
    BenchmarkConfig,
    BenchmarkSource,
    EvidenceSpan,
    FrozenBenchmark,
    SourceApproval,
    StrategyConfig,
    adjudicate_annotations,
    build_frozen_benchmark,
    generate_frozen_benchmark,
    load_frozen_benchmark_manifest,
    render_benchmark_markdown,
    run_frozen_benchmark,
    source_from_pdf_bytes,
)


class KeywordEmbedder:
    model_id = "fixture-embedding-v1"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float("等待期" in text),
                float("责任免除" in text),
                float("保险责任" in text),
            ]
            for text in texts
        ]


class CharacterTokenCounter:
    tokenizer_id = "fixture-char-count-v1"

    def __call__(self, text: str) -> int:
        return len(text)


def _source(*, approval: SourceApproval = SourceApproval.PROJECT_OWNED) -> BenchmarkSource:
    return BenchmarkSource(
        source_id="fixture-policy-v1",
        source_name="fixture-policy.pdf",
        approval=approval,
        approval_reference="repository test fixture",
        insurer_family="fixture-insurer",
        product_family="fixture-product",
        pages=(
            DocumentPage(
                page_number=1,
                text="第六条 等待期\n等待期为九十日。",
                extraction_method="text",
            ),
            DocumentPage(
                page_number=2,
                text="第七条 责任免除\n酒后驾驶导致的事故不承担保险责任。",
                extraction_method="text",
            ),
        ),
    )


def _metadata(annotator_id: str, model: str) -> AnnotationMetadata:
    return AnnotationMetadata(
        annotator_id=annotator_id,
        model_id=model,
        prompt_version="silver-v1",
        generation_parameters=(("temperature", 0),),
    )


def _draft(
    annotator_id: str,
    model: str,
    *,
    quote: str = "等待期为九十日。",
    start_char: int = 8,
) -> AnnotationDraft:
    return AnnotationDraft(
        question="等待期是多久？",
        evidence_spans=(
            EvidenceSpan(
                page_number=1,
                start_char=start_char,
                end_char=start_char + len(quote),
                quote=quote,
            ),
        ),
        stratum="single_sentence",
        hard_negative_category="similar_clause",
        metadata=_metadata(annotator_id, model),
        hard_negative_spans=(
            EvidenceSpan(
                page_number=2,
                start_char=0,
                end_char=len("第七条 责任免除"),
                quote="第七条 责任免除",
            ),
        ),
    )


def test_benchmark_rejects_unapproved_and_private_runtime_sources():
    private_source = _source(approval=SourceApproval.USER_UPLOAD)

    with pytest.raises(ValueError, match="approved public or project-owned"):
        build_frozen_benchmark(
            sources=(private_source,),
            annotation_pairs={private_source.source_id: ((_draft("a", "m1"), _draft("b", "m2")),)},
            adjudications={},
            config=_config(),
        )


def test_exact_span_must_map_to_authoritative_page_text():
    source = _source()
    invalid = _draft("a", "m1", start_char=0)

    with pytest.raises(ValueError, match="does not exactly map"):
        adjudicate_annotations(source, invalid, _draft("b", "m2"))


def test_disagreement_requires_independent_third_adjudicator_and_can_be_uncertain():
    source = _source()
    first = _draft("first", "model-a")
    second = _draft("second", "model-b", quote="第六条 等待期", start_char=0)

    with pytest.raises(ValueError, match="third independent"):
        adjudicate_annotations(source, first, second)

    uncertain = adjudicate_annotations(
        source,
        first,
        second,
        adjudication=AnnotationDraft(
            question=first.question,
            evidence_spans=(),
                stratum=first.stratum,
                hard_negative_category=first.hard_negative_category,
                metadata=_metadata("third", "model-c"),
                hard_negative_spans=first.hard_negative_spans,
                annotation_uncertain=True,
        ),
    )

    assert uncertain.annotation_uncertain is True
    assert uncertain.adjudicated is True


def test_frozen_manifest_records_hashes_models_prompts_parameters_and_outcomes():
    source = _source()
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((_draft("a", "model-a"), _draft("b", "model-b")),)},
        adjudications={},
        config=_config(),
    )

    manifest = benchmark.to_manifest()

    assert manifest["source_records"][0]["source_sha256"]
    assert manifest["source_records"][0]["normalized_text_sha256"]
    assert manifest["cases"][0]["annotation_outcome"] == "agreed"
    assert {item["model_id"] for item in manifest["annotation_runs"]} == {
        "model-a",
        "model-b",
    }
    assert manifest["annotation_runs"][0]["prompt_version"] == "silver-v1"
    assert manifest["annotation_runs"][0]["generation_parameters"] == {
        "temperature": 0
    }


def test_frozen_manifest_can_be_rehydrated_only_with_matching_sources():
    source = _source()
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={
            source.source_id: ((_draft("a", "model-a"), _draft("b", "model-b")),)
        },
        adjudications={},
        config=_config(),
    )

    loaded = load_frozen_benchmark_manifest(
        benchmark.to_manifest(), sources=(source,)
    )

    assert loaded.manifest_sha256 == benchmark.manifest_sha256
    with pytest.raises(ValueError, match="source is unavailable"):
        load_frozen_benchmark_manifest(benchmark.to_manifest(), sources=())


def test_small_frozen_fixture_uses_same_judge_and_retrieval_configuration():
    source = _source()
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((_draft("a", "model-a"), _draft("b", "model-b")),)},
        adjudications={},
        config=_config(),
    )

    report = run_frozen_benchmark(
        benchmark,
        strategies=(
            StrategyConfig(name="legacy", chunking_strategy="legacy"),
            StrategyConfig(name="clause_v2", chunking_strategy="clause_v2"),
        ),
        embedder=KeywordEmbedder(),
        token_counter=CharacterTokenCounter(),
        bootstrap_samples=200,
    )

    assert report.manifest_sha256 == benchmark.manifest_sha256
    assert [result.strategy_name for result in report.strategy_results] == [
        "legacy",
        "clause_v2",
    ]
    for result in report.strategy_results:
        assert result.scored_cases == 1
        assert result.coverage_at[1] == 1.0
        assert result.coverage_at[3] == 1.0
        assert result.coverage_at[5] == 1.0
        assert result.coverage_under_token_budget == 1.0
        assert result.single_candidate_coverage == 1.0
        assert 0.0 <= result.irrelevant_context_proportion <= 1.0
        assert result.embedding_tokens > 0
        assert result.retrieval_unit_count > 0
        assert result.chunking_latency_seconds >= 0
        assert result.hard_negative_confusions == {"similar_clause": 0}
        assert all(result.correctness_invariants.values())
    assert report.paired_comparisons[0].coverage_at_3.confidence_level == 0.95

    markdown = render_benchmark_markdown(report)
    assert "Coverage@1/@3/@5" in markdown
    assert "Coverage under token budget" in markdown
    assert "Annotation disagreement" in markdown
    assert "Paired 95% confidence intervals" in markdown


def test_irrelevant_context_is_measured_within_the_fixed_token_budget():
    source = _source()
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={
            source.source_id: ((_draft("a", "model-a"), _draft("b", "model-b")),)
        },
        adjudications={},
        config=replace(_config(), context_token_budget=len(source.pages[0].text)),
    )

    report = run_frozen_benchmark(
        benchmark,
        strategies=(StrategyConfig(name="legacy", chunking_strategy="legacy"),),
        embedder=KeywordEmbedder(),
        token_counter=CharacterTokenCounter(),
        bootstrap_samples=100,
    )

    expected = (
        len(source.pages[0].text) - len("等待期为九十日。")
    ) / len(source.pages[0].text)
    assert report.strategy_results[0].irrelevant_context_proportion == pytest.approx(
        expected
    )


def test_uncertain_cases_are_excluded_from_primary_scores_but_reported():
    source = _source()
    first = _draft("first", "model-a")
    second = _draft("second", "model-b", quote="第六条 等待期", start_char=0)
    third = replace(
        first,
        evidence_spans=(),
        metadata=_metadata("third", "model-c"),
        annotation_uncertain=True,
    )
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((first, second),)},
        adjudications={(source.source_id, 0): third},
        config=_config(),
    )

    report = run_frozen_benchmark(
        benchmark,
        strategies=(StrategyConfig(name="legacy", chunking_strategy="legacy"),),
        embedder=KeywordEmbedder(),
        token_counter=CharacterTokenCounter(),
    )

    assert report.annotation_summary.total_cases == 1
    assert report.annotation_summary.initial_disagreements == 1
    assert report.annotation_summary.adjudicated_cases == 1
    assert report.annotation_summary.uncertain_exclusions == 1
    assert report.strategy_results[0].scored_cases == 0


def test_unparseable_source_with_only_uncertain_cases_is_not_indexed():
    source = _source()
    first = _draft("first", "model-a")
    second = _draft("second", "model-b", quote="第六条 等待期", start_char=0)
    third = replace(
        first,
        evidence_spans=(),
        hard_negative_category=None,
        hard_negative_spans=(),
        metadata=_metadata("third", "model-c"),
        annotation_uncertain=True,
    )
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((first, second),)},
        adjudications={(source.source_id, 0): third},
        config=_config(),
    )
    empty_source = replace(
        source,
        pages=tuple(replace(page, text="") for page in source.pages),
    )

    report = run_frozen_benchmark(
        replace(benchmark, sources=(empty_source,)),
        strategies=(StrategyConfig(name="legacy", chunking_strategy="legacy"),),
        embedder=KeywordEmbedder(),
        token_counter=CharacterTokenCounter(),
    )

    result = report.strategy_results[0]
    assert result.scored_cases == 0
    assert result.retrieval_unit_count == 0
    assert result.embedding_tokens == 0
    assert all(result.correctness_invariants.values())


def test_changed_judge_or_retrieval_configuration_requires_new_frozen_version():
    source = _source()
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((_draft("a", "model-a"), _draft("b", "model-b")),)},
        adjudications={},
        config=_config(),
    )

    changed = FrozenBenchmark(
        version=benchmark.version,
        sources=benchmark.sources,
        cases=benchmark.cases,
        config=replace(benchmark.config, retrieval_depth=99),
        annotation_runs=benchmark.annotation_runs,
    )

    assert changed.manifest_sha256 != benchmark.manifest_sha256

    mismatched_execution = replace(
        changed,
        config=replace(changed.config, embedding_model_id="different-model"),
    )
    with pytest.raises(ValueError, match="retrieval configuration"):
        run_frozen_benchmark(
            mismatched_execution,
            strategies=(StrategyConfig(name="legacy", chunking_strategy="legacy"),),
            embedder=KeywordEmbedder(),
            token_counter=CharacterTokenCounter(),
        )


def test_run_revalidates_scored_spans_even_for_directly_constructed_benchmark():
    source = _source()
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((_draft("a", "model-a"), _draft("b", "model-b")),)},
        adjudications={},
        config=_config(),
    )
    invalid_case = replace(
        benchmark.cases[0],
        evidence_spans=(
            EvidenceSpan(
                page_number=1,
                start_char=0,
                end_char=len("等待期为九十日。"),
                quote="等待期为九十日。",
            ),
        ),
    )
    bypassed = replace(benchmark, cases=(invalid_case,))

    with pytest.raises(ValueError, match="does not exactly map"):
        run_frozen_benchmark(
            bypassed,
            strategies=(StrategyConfig(name="legacy", chunking_strategy="legacy"),),
            embedder=KeywordEmbedder(),
            token_counter=CharacterTokenCounter(),
        )


def test_generation_is_evidence_first_and_annotators_never_receive_chunks():
    source = _source()
    observations: list[tuple[str, bool]] = []

    def first_pass(received_source):
        observations.append((received_source.source_id, hasattr(received_source, "chunks")))
        return (_draft("first", "model-a"),)

    def second_pass(received_source):
        observations.append((received_source.source_id, hasattr(received_source, "chunks")))
        return (_draft("second", "model-b"),)

    benchmark = generate_frozen_benchmark(
        sources=(source,),
        first_pass=first_pass,
        second_pass=second_pass,
        adjudication_pass=lambda *_args: pytest.fail(
            "agreeing annotations must not be adjudicated"
        ),
        config=_config(),
    )

    assert observations == [
        (source.source_id, False),
        (source.source_id, False),
    ]
    assert benchmark.cases[0].annotation_outcome == "agreed"


def test_generation_supports_bounded_source_concurrency_and_preserves_order():
    first_source = _source()
    second_source = replace(
        first_source,
        source_id="fixture-policy-v2",
        source_name="fixture-policy-v2.pdf",
    )

    benchmark = generate_frozen_benchmark(
        sources=(first_source, second_source),
        first_pass=lambda _source: (_draft("first", "model-a"),),
        second_pass=lambda _source: (_draft("second", "model-b"),),
        adjudication_pass=lambda *_args: pytest.fail(
            "agreeing annotations must not be adjudicated"
        ),
        config=_config(),
        max_workers=2,
    )

    assert tuple(source.source_id for source in benchmark.sources) == (
        "fixture-policy-v1",
        "fixture-policy-v2",
    )


def test_generation_rejects_non_positive_worker_count():
    with pytest.raises(ValueError, match="max_workers"):
        generate_frozen_benchmark(
            sources=(_source(),),
            first_pass=lambda _source: (_draft("first", "model-a"),),
            second_pass=lambda _source: (_draft("second", "model-b"),),
            adjudication_pass=lambda *_args: pytest.fail("must not run"),
            config=_config(),
            max_workers=0,
        )


def test_generation_adjudicates_even_when_both_passes_return_same_uncertain_label():
    source = _source()
    uncertain_first = AnnotationDraft(
        question="等待期是多久？",
        evidence_spans=(),
        stratum="single_sentence",
        hard_negative_category=None,
        metadata=_metadata("first", "model-a"),
        annotation_uncertain=True,
    )
    uncertain_second = replace(
        uncertain_first,
        metadata=_metadata("second", "model-b"),
    )
    adjudication_calls: list[str] = []

    def adjudication_pass(received_source, _first, _second):
        adjudication_calls.append(received_source.source_id)
        return replace(
            _draft("third", "model-c"),
            hard_negative_category=None,
            hard_negative_spans=(),
        )

    benchmark = generate_frozen_benchmark(
        sources=(source,),
        first_pass=lambda _source: (uncertain_first,),
        second_pass=lambda _source: (uncertain_second,),
        adjudication_pass=adjudication_pass,
        config=_config(),
    )

    assert adjudication_calls == [source.source_id]
    assert benchmark.cases[0].adjudicated is True
    assert benchmark.cases[0].annotation_outcome == "adjudicated"


def test_source_fixture_runs_through_production_pdf_page_parser():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Waiting period is 90 days.")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    source = source_from_pdf_bytes(
        pdf_bytes,
        source_id="parsed-fixture-v1",
        source_name="parsed-fixture.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="fixture-insurer",
        product_family="fixture-product",
        parse_config=AppConfig(openai_api_key=None, ocr_enabled=False),
    )

    assert source.pages[0].text.strip() == "Waiting period is 90 days."

    quote = "Waiting period is 90 days."
    first = AnnotationDraft(
        question="How long is the waiting period?",
        evidence_spans=(EvidenceSpan(1, 0, len(quote), quote),),
        stratum="single_sentence",
        hard_negative_category=None,
        metadata=_metadata("pdf-first", "model-a"),
    )
    second = replace(first, metadata=_metadata("pdf-second", "model-b"))
    benchmark = build_frozen_benchmark(
        sources=(source,),
        annotation_pairs={source.source_id: ((first, second),)},
        adjudications={},
        config=_config(),
    )
    report = run_frozen_benchmark(
        benchmark,
        strategies=(StrategyConfig(name="legacy", chunking_strategy="legacy"),),
        embedder=KeywordEmbedder(),
        token_counter=CharacterTokenCounter(),
    )

    assert report.strategy_results[0].coverage_at[1] == 1.0
    assert "Coverage@1/@3/@5" in render_benchmark_markdown(report)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        version="fixture-v1",
        judge_id="exact-span-v1",
        embedding_model_id="fixture-embedding-v1",
        query_rewrite_version="production-v1",
        reranker_version="rules-v1",
        retrieval_depth=5,
        context_token_budget=100,
        tokenizer_id="fixture-char-count-v1",
    )
