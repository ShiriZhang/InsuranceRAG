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


def test_verifier_supports_title_term_with_excerpt_number():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(citation("等待期", "为90天。"),),
        builtin_citations=(),
    )

    assert result.facts[0].status == "supported"
    assert result.facts[0].severity == "info"
    assert result.has_blocking_fact is False


def test_verifier_blocks_compound_title_with_bare_excerpt_number():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(citation("等待期、保险期间", "为90天。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason


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


def test_verifier_blocks_numeric_fact_when_number_belongs_to_same_citation_wrong_clause():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(citation("等待期", "等待期为30天。保险期间为90天。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason


def test_verifier_supports_multi_clause_answer_without_cross_clause_facts():
    result = verify_answer_facts(
        answer="等待期是90天，保险期间是1年。",
        policy_citations=(
            citation("等待期", "等待期为90天。"),
            citation("保险期间", "保险期间为1年。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [fact for fact in result.facts if fact.status == "supported"]
    assert len(supported_facts) >= 2


def test_verifier_does_not_cross_pair_terms_and_numbers_joined_by_he():
    result = verify_answer_facts(
        answer="等待期是90天和保险期间是1年。",
        policy_citations=(
            citation("等待期", "等待期为90天。"),
            citation("保险期间", "保险期间为1年。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [
        fact.fact_text for fact in result.facts if fact.status == "supported"
    ]
    assert supported_facts == ["等待期90天", "保险期间1年"]


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


def test_verifier_allows_source_confusing_phrase_when_policy_numeric_fact_supported():
    result = verify_answer_facts(
        answer="你的保单写明等待期为30天。",
        policy_citations=(citation("等待期", "等待期为30天。"),),
        builtin_citations=(
            citation(
                "等待期",
                "等待期是保险合同生效后的一段观察时间。",
                "built_in_dataset",
            ),
        ),
    )

    assert result.has_blocking_fact is False


def test_verifier_blocks_unsupported_source_confusing_sentence_when_numeric_fact_supported():
    result = verify_answer_facts(
        answer="你的保单写明等待期为30天。你的保单写明等待期是保险合同生效后的一段观察时间。",
        policy_citations=(citation("等待期", "等待期为30天。"),),
        builtin_citations=(
            citation(
                "等待期",
                "等待期是保险合同生效后的一段观察时间。",
                "built_in_dataset",
            ),
        ),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "内置资料" in result.block_reason


def test_verifier_allows_source_confusing_phrase_with_equivalent_copula():
    result = verify_answer_facts(
        answer="你的保单写明等待期是30天。",
        policy_citations=(citation("等待期", "等待期为30天。"),),
        builtin_citations=(
            citation(
                "等待期",
                "等待期是保险合同生效后的一段观察时间。",
                "built_in_dataset",
            ),
        ),
    )

    assert result.has_blocking_fact is False


def test_verifier_blocks_builtin_content_as_user_policy_fact_with_unrelated_policy_citation():
    result = verify_answer_facts(
        answer="你的保单写明等待期是保险合同生效后的一段观察时间。",
        policy_citations=(citation("保险金额", "保险金额为10万元。"),),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "内置资料" in result.block_reason


def test_verifier_blocks_source_confusion_when_policy_term_citation_is_unrelated():
    result = verify_answer_facts(
        answer="你的保单写明等待期是保险合同生效后的一段观察时间。",
        policy_citations=(citation("等待期", "等待期内不承担保险责任。"),),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "内置资料" in result.block_reason


def test_verifier_warns_for_partial_support():
    result = verify_answer_facts(
        answer="等待期条款需要结合原文核对。",
        policy_citations=(citation("等待期", "等待期内不承担保险责任。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    assert result.has_warnings is True


def test_verifier_blocks_unsupported_unlisted_numeric_policy_fact():
    result = verify_answer_facts(
        answer="免赔额是1万元。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "免赔额1万元" in result.block_reason


def test_verifier_blocks_source_confusing_unlisted_numeric_policy_fact():
    result = verify_answer_facts(
        answer="你的保单写明观察期是30天。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "观察期30天" in result.block_reason


def test_verifier_supports_common_alias_numeric_policy_facts():
    result = verify_answer_facts(
        answer="观察期是30天，保额是10万元。",
        policy_citations=(
            citation("等待期", "等待期为30天。"),
            citation("保险金额", "保险金额为10万元。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [
        fact.fact_text for fact in result.facts if fact.status == "supported"
    ]
    assert supported_facts == ["观察期30天", "保额10万元"]


def test_verifier_warns_with_term_matching_citation_when_available():
    result = verify_answer_facts(
        answer="等待期条款需要结合原文核对。",
        policy_citations=(
            citation("保险金额", "保险金额为10万元。"),
            citation("等待期", "等待期内不承担保险责任。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    assert result.has_warnings is True
    assert result.facts[0].supporting_citation_ids == ("policy.pdf:6:等待期",)


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
