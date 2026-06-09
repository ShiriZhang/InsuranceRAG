from insurance_rag.chunker import infer_section_title, chunk_pages
from insurance_rag.models import DocumentPage


def test_infer_section_title_matches_known_clause_heading():
    text = "责任免除\n因下列情形之一导致被保险人发生疾病的，本公司不承担保险责任。"

    assert infer_section_title(text, current_title="未识别条款标题") == "责任免除"


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
