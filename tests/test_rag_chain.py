from insurance_rag.models import DocumentChunk
from insurance_rag.rag_chain import (
    REFUSAL_ANSWER,
    build_citation,
    build_messages,
    should_use_builtin_context,
)


def make_chunk(source_type: str = "user_policy") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="c1",
        text="等待期为九十日。",
        page_number=4,
        section_title="等待期",
        source_type=source_type,
        source_name="user.pdf" if source_type == "user_policy" else "内置条款.pdf",
        extraction_method="text",
    )


def test_should_use_builtin_context_for_term_question():
    assert should_use_builtin_context("什么是等待期？", policy_result_count=2) is True


def test_should_not_use_builtin_context_for_specific_policy_question():
    assert should_use_builtin_context("这份保单等待期是多少？", policy_result_count=3) is False


def test_build_citation_uses_chunk_metadata():
    citation = build_citation(make_chunk())

    assert citation.page_number == 4
    assert citation.section_title == "等待期"
    assert citation.excerpt == "等待期为九十日。"


def test_build_messages_include_no_claim_decision_rule():
    messages = build_messages("等待期是多少？", [make_chunk()], [])
    joined = "\n".join(message["content"] for message in messages)

    assert "不得做最终理赔判断" in joined
    assert "用户保单资料" in joined


def test_refusal_answer_is_evidence_limited():
    assert "没有找到足够明确的依据" in REFUSAL_ANSWER
