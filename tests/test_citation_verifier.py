from insurance_rag.citation_verifier import verify_answer_facts
from insurance_rag.models import Citation


def citation(title: str, excerpt: str, source_type: str = "user_policy") -> Citation:
    return Citation(
        source_type=source_type,
        source_name="policy.pdf",
        page_number=6,
        section_title=title,
        excerpt=excerpt,
    )


def test_verifier_supports_numeric_fact_in_same_citation():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(citation("等待期", "本合同等待期为九十日。"),),
        builtin_citations=(),
    )

    assert result.facts[0].status == "supported"
    assert result.facts[0].severity == "info"


def test_verifier_blocks_numeric_fact_when_number_belongs_to_other_clause():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(
            citation("等待期", "本合同等待期为30天。"),
            citation("保险期间", "保险期间为90天。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason


def test_verifier_blocks_builtin_content_as_user_policy_fact():
    result = verify_answer_facts(
        answer="你的保单写明等待期是保险合同生效后的一段观察时间。",
        policy_citations=(),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is True
    assert "内置资料" in result.block_reason


def test_verifier_warns_for_partial_support():
    result = verify_answer_facts(
        answer="等待期条款需要结合原文核对。",
        policy_citations=(citation("等待期", "等待期内不承担保险责任。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    assert result.has_warnings is True


def test_verifier_normalizes_chinese_number_days():
    result = verify_answer_facts(
        answer="等待期是九十日。",
        policy_citations=(citation("等待期", "本合同等待期为90天。"),),
        builtin_citations=(),
    )

    assert result.facts[0].status == "supported"
    assert result.facts[0].severity == "info"


def test_verifier_blocks_unsupported_numeric_fact_without_policy_citations():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason
