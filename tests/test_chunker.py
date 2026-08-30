import pytest

from insurance_rag.chunker import infer_section_title, chunk_pages
from insurance_rag.models import DocumentPage, SourceSpan


def test_chunk_pages_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        chunk_pages(
            (),
            source_name="policy.pdf",
            source_type="user_policy",
            chunk_size=900,
            overlap=0,
            strategy="typo",
        )


def test_clause_v2_keeps_complete_single_page_clauses_independent():
    pages = (
        DocumentPage(
            page_number=1,
            text=(
                "第六条 等待期\n"
                "等待期为九十日。\n"
                "责任免除\n"
                "酒后驾驶导致的事故不承担保险责任。"
            ),
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert [chunk.text for chunk in chunks] == [
        "第六条 等待期\n等待期为九十日。",
        "责任免除\n酒后驾驶导致的事故不承担保险责任。",
    ]
    assert [chunk.section_title for chunk in chunks] == ["等待期", "责任免除"]
    assert [chunk.heading_confidence for chunk in chunks] == ["high", "medium"]


def test_clause_v2_exposes_retrieval_context_source_span_and_boundary_diagnostic():
    source_text = "第六条 等待期\n等待期为九十日。"
    pages = (
        DocumentPage(
            page_number=7,
            text=source_text,
            extraction_method="text",
        ),
    )

    chunk = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )[0]

    assert chunk.text == source_text
    assert chunk.source_spans == (
        SourceSpan(page_number=7, text=source_text, start_char=0, end_char=len(source_text)),
    )
    assert chunk.retrieval_context == "Policy Clause: 第六条 等待期"
    assert chunk.retrieval_text == f"Policy Clause: 第六条 等待期\n{source_text}"
    assert "trusted_heading:high:line_pattern" in chunk.boundary_diagnostics
    assert chunk.chunking_strategy == "clause_v2"


def test_clause_v2_continues_established_clause_across_readable_pages():
    first_page_text = "第六条 等待期\n等待期自合同生效日起计算。"
    second_page_text = "等待期为九十日。"
    pages = (
        DocumentPage(
            page_number=7,
            text=first_page_text,
            extraction_method="text",
        ),
        DocumentPage(
            page_number=8,
            text=second_page_text,
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert len(chunks) == 1
    assert chunks[0].text == "第六条 等待期\n等待期自合同生效日起计算。\n等待期为九十日。"
    assert chunks[0].source_spans == (
        SourceSpan(
            page_number=7,
            text=first_page_text,
            start_char=0,
            end_char=len(first_page_text),
        ),
        SourceSpan(
            page_number=8,
            text=second_page_text,
            start_char=0,
            end_char=len(second_page_text),
        ),
    )


def test_clause_v2_continues_until_next_trusted_heading_on_later_page():
    pages = (
        DocumentPage(
            page_number=1,
            text="第六条 等待期\n等待期自合同生效日起计算。",
            extraction_method="text",
        ),
        DocumentPage(
            page_number=2,
            text="等待期为九十日。\n第七条 责任免除\n酒后驾驶导致的事故不承担保险责任。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert [chunk.text for chunk in chunks] == [
        "第六条 等待期\n等待期自合同生效日起计算。\n等待期为九十日。",
        "第七条 责任免除\n酒后驾驶导致的事故不承担保险责任。",
    ]
    assert [chunk.section_title for chunk in chunks] == ["等待期", "责任免除"]
    assert [span.page_number for span in chunks[0].source_spans] == [1, 2]


def test_clause_v2_empty_page_stops_cross_page_continuation():
    pages = (
        DocumentPage(
            page_number=1,
            text="第六条 等待期\n等待期自合同生效日起计算。",
            extraction_method="text",
        ),
        DocumentPage(page_number=2, text="", extraction_method="text"),
        DocumentPage(
            page_number=3,
            text="等待期为九十日。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert [chunk.text for chunk in chunks] == [
        "第六条 等待期\n等待期自合同生效日起计算。",
        "等待期为九十日。",
    ]
    assert chunks[1].section_title == "未识别条款标题"
    assert "page_gap:empty" in chunks[1].boundary_diagnostics
    assert "unknown_clause_page_fallback" in chunks[1].boundary_diagnostics


def test_clause_v2_trailing_empty_page_leaves_observable_diagnostic():
    pages = (
        DocumentPage(
            page_number=1,
            text="第六条 等待期\n等待期为九十日。",
            extraction_method="text",
        ),
        DocumentPage(page_number=2, text="", extraction_method="text"),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert "page_gap:empty" in chunks[-1].boundary_diagnostics


@pytest.mark.parametrize(
    ("quality_note", "expected_diagnostic"),
    [
        ("unreadable_page", "page_gap:unreadable"),
        ("severe_ocr_uncertainty", "page_gap:severe_ocr_uncertainty"),
    ],
)
def test_clause_v2_unsafe_page_quality_stops_cross_page_continuation(
    quality_note,
    expected_diagnostic,
):
    pages = (
        DocumentPage(
            page_number=1,
            text="第六条 等待期\n等待期自合同生效日起计算。",
            extraction_method="text",
        ),
        DocumentPage(
            page_number=2,
            text="无法可靠读取的页面文本",
            extraction_method="ocr",
            quality_notes=(quality_note,),
        ),
        DocumentPage(
            page_number=3,
            text="等待期为九十日。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert [chunk.page_number for chunk in chunks] == [1, 3]
    assert chunks[1].section_title == "未识别条款标题"
    assert expected_diagnostic in chunks[1].boundary_diagnostics


def test_clause_v2_splits_unknown_content_within_its_page_at_sentences():
    page = DocumentPage(
        page_number=4,
        text="投保人应如实告知。保险费应按时交纳。合同变更应书面申请。",
        extraction_method="text",
    )

    chunks = chunk_pages(
        (page,),
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
        target_chars=10,
        hard_max_chars=16,
    )

    assert [chunk.text for chunk in chunks] == [
        "投保人应如实告知。",
        "保险费应按时交纳。",
        "合同变更应书面申请。",
    ]
    assert all(chunk.page_number == 4 for chunk in chunks)
    assert all(
        "unknown_clause_page_fallback" in chunk.boundary_diagnostics
        for chunk in chunks
    )


def test_clause_v2_packs_complete_numbered_items_toward_soft_target():
    page = DocumentPage(
        page_number=5,
        text=(
            "第六条 保险责任\n"
            "一、给付身故保险金。\n"
            "二、给付全残保险金。\n"
            "三、给付疾病保险金。"
        ),
        extraction_method="text",
    )

    chunks = chunk_pages(
        (page,),
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
        target_chars=26,
        hard_max_chars=32,
    )

    assert [chunk.text for chunk in chunks] == [
        "第六条 保险责任\n一、给付身故保险金。\n二、给付全残保险金。\n",
        "三、给付疾病保险金。",
    ]
    assert all(
        chunk.retrieval_context == "Policy Clause: 第六条 保险责任"
        for chunk in chunks
    )


def test_clause_v2_does_not_emit_heading_without_governed_body():
    page = DocumentPage(
        page_number=5,
        text="第六条 保险责任\n一、给付身故保险金。\n二、给付全残保险金。",
        extraction_method="text",
    )

    chunks = chunk_pages(
        (page,),
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
        target_chars=9,
        hard_max_chars=24,
    )

    assert chunks[0].text == "第六条 保险责任\n一、给付身故保险金。\n"
    assert all(chunk.text.strip() != "第六条 保险责任" for chunk in chunks)


def test_clause_v2_uses_observable_windows_for_indivisible_overlong_sentence():
    body = "本合同对一次事故造成的全部损失按照约定比例承担保险责任且不超过保险金额。"
    page = DocumentPage(
        page_number=6,
        text=f"第六条 保险责任\n{body}",
        extraction_method="text",
    )

    chunks = chunk_pages(
        (page,),
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
        target_chars=16,
        hard_max_chars=20,
    )

    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert any(
        "character_window_fallback" in chunk.boundary_diagnostics
        for chunk in chunks
    )
    assert all(chunk.text.strip() != "第六条 保险责任" for chunk in chunks)
    reconstructed = "".join(
        span.text for chunk in chunks for span in chunk.source_spans
    )
    assert reconstructed == page.text


def test_clause_v2_reconstruction_and_clause_purity_hold_across_boundary_modes():
    pages = (
        DocumentPage(
            page_number=1,
            text="投保人应如实告知。保险费应按时交纳。",
            extraction_method="text",
        ),
        DocumentPage(
            page_number=2,
            text="第六条 等待期\n等待期自合同生效日起计算。",
            extraction_method="text",
        ),
        DocumentPage(
            page_number=3,
            text="等待期为九十日。\n第七条 责任免除\n酒后驾驶导致的事故不承担保险责任。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
        target_chars=16,
        hard_max_chars=28,
    )

    for page in pages:
        coverage = [0] * len(page.text)
        for chunk in chunks:
            for span in chunk.source_spans:
                if span.page_number != page.page_number:
                    continue
                assert page.text[span.start_char : span.end_char] == span.text
                for offset in range(span.start_char, span.end_char):
                    coverage[offset] += 1
        assert all(
            count == 1
            for character, count in zip(page.text, coverage)
            if not character.isspace()
        )

    assert all(
        not ("第六条 等待期" in chunk.text and "第七条 责任免除" in chunk.text)
        for chunk in chunks
    )


def test_clause_v2_source_spans_are_exact_slices_of_original_page_text():
    page_text = (
        "  第六条 等待期\r\n\r\n"
        "  等待期为九十日。\r\n"
        "  责任免除\r\n"
        "  酒后驾驶导致的事故不承担保险责任。  "
    )
    page = DocumentPage(page_number=3, text=page_text, extraction_method="text")

    chunks = chunk_pages(
        (page,),
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert len(chunks) == 2
    for chunk in chunks:
        span = chunk.source_spans[0]
        assert page_text[span.start_char : span.end_char] == span.text
        assert span.text == chunk.text


def test_clause_v2_keeps_low_confidence_heading_mentions_in_diagnostics():
    page = DocumentPage(
        page_number=1,
        text="第六条 等待期\n本公司在等待期内不承担保险责任。",
        extraction_method="text",
    )

    chunks = chunk_pages(
        (page,),
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    assert len(chunks) == 1
    assert "low_confidence_heading_candidate" in chunks[0].boundary_diagnostics


def test_clause_v2_rejects_repeated_medium_heading_page_headers():
    pages = (
        DocumentPage(
            page_number=1,
            text="保险责任\n第六条 等待期\n等待期为九十日。",
            extraction_method="text",
        ),
        DocumentPage(
            page_number=2,
            text="保险责任\n第七条 责任免除\n酒后驾驶导致的事故不承担保险责任。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )

    header_chunks = [chunk for chunk in chunks if chunk.text == "保险责任"]
    assert len(header_chunks) == 2
    assert all(chunk.heading_confidence == "low" for chunk in header_chunks)
    assert all(
        "rejected_page_header_footer" in chunk.boundary_diagnostics
        for chunk in header_chunks
    )
    assert [
        chunk.section_title
        for chunk in chunks
        if chunk.heading_confidence in {"high", "medium"}
    ] == ["等待期", "责任免除"]


def test_chunk_pages_attaches_high_confidence_clause_metadata():
    pages = (
        DocumentPage(
            page_number=6,
            text="第六条 等待期\n等待期为90天。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=200,
        overlap=0,
    )

    assert chunks[0].section_title == "等待期"
    assert chunks[0].clause_id == "第六条"
    assert chunks[0].heading_text == "第六条 等待期"
    assert chunks[0].heading_confidence == "high"


def test_chunk_pages_preserves_unnumbered_known_heading_suffix_compatibility():
    pages = (
        DocumentPage(
            page_number=6,
            text="等待期：90天",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=200,
        overlap=0,
    )

    assert chunks[0].section_title == "等待期"
    assert chunks[0].clause_id is None
    assert chunks[0].heading_text == "等待期：90天"
    assert chunks[0].heading_confidence == "medium"
    assert chunks[0].heading_source == "known_title"


def test_chunk_pages_preserves_fallback_title_for_following_chunks():
    pages = (
        DocumentPage(
            page_number=1,
            text="第二条 保险期间\n本合同保险期间为一年。\n后续内容继续说明保险期间。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=20,
        overlap=0,
    )

    assert chunks[0].section_title == "保险期间"
    assert all(chunk.section_title == "保险期间" for chunk in chunks)


def test_chunk_pages_uses_real_heading_after_directory_line():
    pages = (
        DocumentPage(
            page_number=3,
            text="2.3 保险责任 ........ 5\n第六条 等待期\n等待期为90天。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=200,
        overlap=0,
    )

    assert chunks[0].section_title == "等待期"
    assert chunks[0].clause_id == "第六条"
    assert chunks[0].heading_text == "第六条 等待期"
    assert chunks[0].heading_confidence == "high"


def test_chunk_pages_low_confidence_chunks_preserve_current_title():
    pages = (
        DocumentPage(
            page_number=1,
            text="第二条 保险期间\n本合同保险期间为一年。\n本页后续说明不包含新标题。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=18,
        overlap=0,
    )

    assert len(chunks) > 1
    assert chunks[0].section_title == "保险期间"
    assert chunks[1].section_title == "保险期间"
    assert chunks[1].heading_confidence == "low"


def test_infer_section_title_matches_known_clause_heading():
    text = "责任免除\n因下列情形之一导致被保险人发生疾病的，本公司不承担保险责任。"

    assert infer_section_title(text, current_title="未识别条款标题") == "责任免除"


def test_infer_section_title_matches_numbered_clause_heading():
    text = "第七条 保险责任\n本合同的保险责任包括重大疾病保险金。"

    assert infer_section_title(text, current_title="未识别条款标题") == "保险责任"


def test_infer_section_title_matches_heading_after_page_header_lines():
    text = (
        "某保险条款，第4页，共20页\n"
        "被保险人因意外伤害事故导致身故的，无等待期。\n"
        "等待期内发生以下情形之一时，我们不承担给付保险金的责任。\n"
        "一、被保险人在等待期内确诊相关疾病。\n"
        "二、被保险人在等待期内因疾病导致身故。\n"
        "本项下累计已交保险费按照合同约定计算。\n"
        "第七条 保险责任\n"
        "在本合同保险期间内且本合同有效，我们承担相应保险责任。"
    )

    assert infer_section_title(text, current_title="未识别条款标题") == "保险责任"


def test_chunk_pages_preserves_page_and_source_metadata():
    pages = (
        DocumentPage(
            page_number=2,
            text="保险责任\n本合同保障重大疾病。\n等待期\n等待期为九十日。",
            extraction_method="text",
            quality_notes=("OCR text may contain errors",),
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="user.pdf",
        source_type="user_policy",
        chunk_size=20,
        overlap=5,
    )

    assert chunks
    assert chunks[0].page_number == 2
    assert chunks[0].section_title == "保险责任"
    assert chunks[0].source_type == "user_policy"
    assert chunks[0].source_name == "user.pdf"
    assert chunks[0].extraction_method == "text"
    assert chunks[0].quality_notes == ("OCR text may contain errors",)


def test_infer_section_title_ignores_body_mentions_of_known_titles():
    text = "因下列情形之一导致被保险人发生疾病的，本公司不承担保险责任。"

    assert infer_section_title(text, current_title="责任免除") == "责任免除"


def test_chunk_pages_carries_forward_current_title_for_continuation_chunks():
    pages = (
        DocumentPage(
            page_number=1,
            text="责任免除\n因下列情形之一导致被保险人发生疾病的，本公司不承担保险责任。\n其他说明继续描述除外责任。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="user.pdf",
        source_type="user_policy",
        chunk_size=24,
        overlap=0,
    )

    assert len(chunks) > 1
    assert chunks[0].section_title == "责任免除"
    assert chunks[1].section_title == "责任免除"
