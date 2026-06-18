from insurance_rag.chunker import infer_section_title, chunk_pages
from insurance_rag.models import DocumentPage


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
