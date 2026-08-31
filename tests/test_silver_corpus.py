from __future__ import annotations

from pathlib import Path
from collections import Counter
from dataclasses import replace
from uuid import uuid4

import fitz

from insurance_rag.config import AppConfig
from insurance_rag.silver_benchmark import SourceApproval
from insurance_rag.silver_corpus import (
    build_adjudication_source_window,
    build_annotation_source_window,
    build_approved_corpus_inventory,
    build_evidence_case_plan,
    freeze_insurer_document_split,
)
from insurance_rag.silver_dataset import DatasetSplit
from insurance_rag.silver_benchmark import BenchmarkSource, EvidenceSpan
from insurance_rag.models import DocumentPage
from insurance_rag.silver_dataset import (
    DocumentSplitAssignment,
    freeze_document_split,
)


def _write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    path.write_bytes(document.tobytes())
    document.close()


ROOT = Path(__file__).parents[1]


def test_inventory_supports_both_directory_shapes_and_freezes_insurer_split():
    documents = ROOT / "tmp" / "silver-corpus-tests" / uuid4().hex / "documents"
    _write_pdf(documents / "insurer-a" / "direct-product.pdf", "Policy A")
    _write_pdf(
        documents / "insurer-b" / "nested-product-1" / "policy.pdf",
        "Policy B1",
    )
    _write_pdf(
        documents / "insurer-b" / "nested-product-2" / "policy.pdf",
        "Policy B2",
    )
    _write_pdf(documents / "insurer-c" / "direct-product.pdf", "Policy C")
    _write_pdf(documents / "insurer-d" / "product" / "policy.pdf", "Policy D")

    inventory = build_approved_corpus_inventory(
        documents,
        approval_reference=(
            "project-owner-approved-documents-for-llm-annotation/2026-08-30"
        ),
        parse_config=AppConfig(openai_api_key=None, ocr_enabled=False),
    )
    repeated = build_approved_corpus_inventory(
        documents,
        approval_reference=(
            "project-owner-approved-documents-for-llm-annotation/2026-08-30"
        ),
        parse_config=AppConfig(openai_api_key=None, ocr_enabled=False),
    )

    assert len(inventory.entries) == 5
    assert inventory.sources == repeated.sources
    assert {entry.relative_path for entry in inventory.entries} == {
        "insurer-a/direct-product.pdf",
        "insurer-b/nested-product-1/policy.pdf",
        "insurer-b/nested-product-2/policy.pdf",
        "insurer-c/direct-product.pdf",
        "insurer-d/product/policy.pdf",
    }
    assert all(source.approval is SourceApproval.PROJECT_OWNED for source in inventory.sources)
    assert len({source.product_family for source in inventory.sources}) == 5
    assert all(source.source_id.startswith("benchmark-source-v1-") for source in inventory.sources)

    frozen = freeze_insurer_document_split(
        inventory,
        version="silver-document-split/v1.0.0",
        held_out_fraction=0.40,
        seed="silver-document-split/v1.0.0",
    )

    split_by_insurer: dict[str, set[DatasetSplit]] = {}
    source_by_id = {source.source_id: source for source in frozen.sources}
    for assignment in frozen.assignments:
        source = source_by_id[assignment.source_id]
        split_by_insurer.setdefault(source.insurer_family, set()).add(assignment.split)
        assert assignment.near_duplicate_family == source.product_family
    assert all(len(splits) == 1 for splits in split_by_insurer.values())
    assert len(frozen.sources_for(DatasetSplit.HELD_OUT)) == 2


def test_case_plan_balances_required_strata_and_uses_more_held_out_cases():
    sources = tuple(
        BenchmarkSource(
            source_id=f"source-{index}",
            source_name=f"source-{index}.pdf",
            approval=SourceApproval.PROJECT_OWNED,
            approval_reference="repository test fixture",
            insurer_family=f"insurer-{index}",
            product_family=f"product-{index}",
            pages=(DocumentPage(1, "Policy text", "text"),),
        )
        for index in range(14)
    )
    frozen = freeze_document_split(
        version="fixture-split-v1",
        sources=sources,
        assignments=tuple(
            DocumentSplitAssignment(
                source_id=source.source_id,
                split=(
                    DatasetSplit.DEVELOPMENT
                    if index < 7
                    else DatasetSplit.HELD_OUT
                ),
                near_duplicate_family=source.product_family,
            )
            for index, source in enumerate(sources)
        ),
    )

    plan = build_evidence_case_plan(
        frozen,
        development_cases_per_source=3,
        held_out_cases_per_source=5,
    )

    assert all(
        len(plan(source)) == 3
        for source in frozen.sources_for(DatasetSplit.DEVELOPMENT)
    )
    assert all(
        len(plan(source)) == 5
        for source in frozen.sources_for(DatasetSplit.HELD_OUT)
    )
    held_out_strata = Counter(
        request.stratum
        for source in frozen.sources_for(DatasetSplit.HELD_OUT)
        for request in plan(source)
    )
    assert set(held_out_strata.values()) == {5}
    hard_negative_requests = [
        request
        for source in frozen.sources
        for request in plan(source)
        if request.stratum == "adjacent_or_lexically_similar_hard_negative"
    ]
    assert all(
        request.hard_negative_category == "similar_clause"
        for request in hard_negative_requests
    )


def test_case_plan_assigns_held_out_cross_page_cases_to_best_boundaries():
    sources = tuple(
        BenchmarkSource(
            source_id=f"ranked-source-{index}",
            source_name=f"ranked-source-{index}.pdf",
            approval=SourceApproval.PROJECT_OWNED,
            approval_reference="repository test fixture",
            insurer_family=f"ranked-insurer-{index}",
            product_family=f"ranked-product-{index}",
            pages=(
                (
                    DocumentPage(1, "本公司承担保险责任，", "text"),
                    DocumentPage(2, "并按约定给付保险金。", "text"),
                )
                if index < 5
                else (DocumentPage(1, "完整条款。", "text"),)
            ),
        )
        for index in range(8)
    )
    frozen = freeze_document_split(
        version="ranked-split-v1",
        sources=sources,
        assignments=tuple(
            DocumentSplitAssignment(
                source_id=source.source_id,
                split=(
                    DatasetSplit.DEVELOPMENT
                    if source.source_id == "ranked-source-7"
                    else DatasetSplit.HELD_OUT
                ),
                near_duplicate_family=source.product_family,
            )
            for source in sources
        ),
    )

    plan = build_evidence_case_plan(
        frozen,
        development_cases_per_source=3,
        held_out_cases_per_source=5,
    )

    cross_page_source_ids = {
        source.source_id
        for source in frozen.sources_for(DatasetSplit.HELD_OUT)
        if any(request.stratum == "cross_page_clause" for request in plan(source))
    }
    assert cross_page_source_ids == {
        f"ranked-source-{index}" for index in range(5)
    }


def test_annotation_windows_are_deterministic_complete_pages():
    source = BenchmarkSource(
        source_id="source-a",
        source_name="source-a.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(DocumentPage(1, "fixture", "text"),),
    )
    source = replace(
        source,
        pages=tuple(
            DocumentPage(index, character * 20, "text")
            for index, character in enumerate("甲乙丙丁戊己", start=1)
        ),
    )

    windows = tuple(
        build_annotation_source_window(
            source,
            slot_id=f"slot-{index + 1}",
            slot_index=index,
            slot_count=3,
            max_chars=45,
        )
        for index in range(3)
    )

    assert [[page.page_number for page in window.pages] for window in windows] == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    assert windows[0].source_id.endswith(":slot-1")
    assert windows[0].pages[0].text == source.pages[0].text


def test_annotation_window_segments_long_pages_without_losing_source_text():
    long_text = "甲" * 50
    source = BenchmarkSource(
        source_id="source-long",
        source_name="source-long.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(DocumentPage(7, long_text, "text"),),
    )

    windows = tuple(
        build_annotation_source_window(
            source,
            slot_id=f"slot-{index}",
            slot_index=index,
            slot_count=3,
            max_chars=20,
        )
        for index in range(3)
    )

    assert "".join(window.pages[0].text for window in windows) == long_text
    assert all(window.pages[0].page_number == 7 for window in windows)
    assert all(window.authoritative_pages == source.pages for window in windows)


def test_cross_page_window_uses_adjacent_page_boundary_within_budget():
    source = BenchmarkSource(
        source_id="source-cross-page",
        source_name="source-cross-page.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(
            DocumentPage(1, "甲" * 60 + "第一页结尾", "text"),
            DocumentPage(2, "第二页开头" + "乙" * 60, "text"),
        ),
    )

    window = build_annotation_source_window(
        source,
        slot_id="cross-page-slot",
        slot_index=0,
        slot_count=1,
        max_chars=40,
        stratum="cross_page_clause",
    )

    assert [page.page_number for page in window.pages] == [1, 2]
    assert window.pages[0].text.endswith("第一页结尾")
    assert window.pages[1].text.startswith("第二页开头")
    assert sum(len(page.text) for page in window.pages) <= 40
    assert window.authoritative_pages == source.pages


def test_cross_page_window_prefers_boundary_with_clause_continuation():
    source = BenchmarkSource(
        source_id="source-cross-page-ranked",
        source_name="source-cross-page-ranked.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(
            DocumentPage(1, "第一条 本公司承担保险责任，", "text"),
            DocumentPage(2, "并按约定给付保险金。本页结束。", "text"),
            DocumentPage(3, "第三条 责任免除。", "text"),
        ),
    )

    window = build_annotation_source_window(
        source,
        slot_id="cross-page-ranked-slot",
        slot_index=0,
        slot_count=1,
        max_chars=80,
        stratum="cross_page_clause",
    )

    assert [page.page_number for page in window.pages] == [1, 2]


def test_cross_page_window_prefers_substantive_continuation_over_short_header():
    source = BenchmarkSource(
        source_id="source-cross-page-substantive",
        source_name="source-cross-page-substantive.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(
            DocumentPage(1, "责任，", "text"),
            DocumentPage(2, "并给付。本公司在被保险人确诊约定疾病且满足等待期要求后，", "text"),
            DocumentPage(3, "按照基本保险金额给付重大疾病保险金。", "text"),
        ),
    )

    window = build_annotation_source_window(
        source,
        slot_id="cross-page-substantive-slot",
        slot_index=0,
        slot_count=1,
        max_chars=120,
        stratum="cross_page_clause",
    )

    assert [page.page_number for page in window.pages] == [2, 3]


def test_multi_sentence_window_prefers_conditions_and_outcomes():
    source = BenchmarkSource(
        source_id="source-multi-sentence-ranked",
        source_name="source-multi-sentence-ranked.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(
            DocumentPage(1, "产品名称。公司名称。", "text"),
            DocumentPage(
                2,
                "若被保险人在等待期后确诊重大疾病，本公司承担保险责任。满足约定条件时，本公司给付重大疾病保险金。",
                "text",
            ),
            DocumentPage(3, "释义。附表。", "text"),
        ),
    )

    window = build_annotation_source_window(
        source,
        slot_id="multi-sentence-slot",
        slot_index=0,
        slot_count=3,
        max_chars=55,
        stratum="multi_sentence_conditions_outcomes",
    )

    assert [page.page_number for page in window.pages] == [2]


def test_adjudication_window_keeps_disputed_spans_when_full_pages_are_too_large():
    source = BenchmarkSource(
        source_id="source-dispute",
        source_name="source-dispute.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(
            DocumentPage(1, "甲" * 200 + "第一证据" + "乙" * 200, "text"),
            DocumentPage(2, "丙" * 200 + "第二证据" + "丁" * 200, "text"),
        ),
    )
    first = EvidenceSpan(1, 200, 204, "第一证据")
    second = EvidenceSpan(2, 200, 204, "第二证据")

    window = build_adjudication_source_window(
        source,
        (first,),
        (second,),
        work_item_id="case-1",
        max_chars=100,
    )

    combined = "".join(page.text for page in window.pages)
    assert "第一证据" in combined
    assert "第二证据" in combined


def test_adjudication_window_reuses_the_annotation_window_when_available():
    source = BenchmarkSource(
        source_id="source-adjudication-fallback",
        source_name="source-adjudication-fallback.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="insurer-a",
        product_family="product-a",
        pages=(
            DocumentPage(1, "第一页末尾的条件，", "text"),
            DocumentPage(2, "并在第二页给付保险金。", "text"),
        ),
    )
    fallback = build_annotation_source_window(
        source,
        slot_id="slot-1",
        slot_index=0,
        slot_count=1,
        max_chars=80,
        stratum="cross_page_clause",
    )

    window = build_adjudication_source_window(
        source,
        (EvidenceSpan(1, 0, 9, "第一页末尾的条件，"),),
        (),
        work_item_id="adjudication",
        max_chars=80,
        fallback_window=fallback,
    )

    assert window.pages == fallback.pages
    assert [page.page_number for page in window.pages] == [1, 2]
