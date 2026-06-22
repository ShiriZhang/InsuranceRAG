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


def test_verifier_blocks_unsupported_policy_age_fact():
    result = verify_answer_facts(
        answer="投保年龄是60周岁。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "投保年龄60周岁" in result.block_reason


def test_verifier_supports_policy_age_fact_in_policy_citation():
    result = verify_answer_facts(
        answer="投保年龄是60周岁。",
        policy_citations=(citation("投保年龄", "投保年龄为60周岁。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [
        fact.fact_text for fact in result.facts if fact.status == "supported"
    ]
    assert supported_facts == ["投保年龄60周岁"]


def test_verifier_blocks_unsupported_payment_period_fact():
    result = verify_answer_facts(
        answer="交费期间是20年。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "交费期间20年" in result.block_reason


def test_verifier_blocks_unsupported_benefit_limit_fact():
    result = verify_answer_facts(
        answer="给付限额为100元。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "给付限额100元" in result.block_reason


def test_verifier_blocks_comma_formatted_amount_without_policy_citation():
    result = verify_answer_facts(
        answer="免赔额是1,000元。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "免赔额" in result.block_reason


def test_verifier_supports_comma_formatted_amount_with_plain_policy_citation():
    result = verify_answer_facts(
        answer="免赔额是1,000元。",
        policy_citations=(citation("免赔额", "免赔额为1000元。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [
        fact.fact_text for fact in result.facts if fact.status == "supported"
    ]
    assert supported_facts == ["免赔额1000元"]


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


def test_verifier_blocks_unsupported_responsibility_exemption_fact():
    result = verify_answer_facts(
        answer="酒后驾驶属于责任免除。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "酒后驾驶" in result.block_reason or "责任免除" in result.block_reason


def test_verifier_supports_responsibility_exemption_fact_in_policy_citation():
    result = verify_answer_facts(
        answer="酒后驾驶属于责任免除。",
        policy_citations=(citation("责任免除", "酒后驾驶属于责任免除。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False


def test_verifier_blocks_waiver_subject_mismatch():
    result = verify_answer_facts(
        answer="投保人可豁免保险费。",
        policy_citations=(citation("豁免保险费", "被保险人可豁免保险费。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "投保人" in result.block_reason


def test_verifier_supports_waiver_subject_in_policy_citation():
    result = verify_answer_facts(
        answer="投保人可豁免保险费。",
        policy_citations=(citation("豁免保险费", "投保人可豁免保险费。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False


def test_verifier_blocks_waiver_subject_mismatch_without_modal():
    result = verify_answer_facts(
        answer="投保人豁免保险费。",
        policy_citations=(citation("豁免保险费", "被保险人豁免保险费。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "投保人豁免保险费" in result.block_reason


def test_verifier_supports_waiver_subject_without_modal_in_policy_citation():
    result = verify_answer_facts(
        answer="投保人豁免保险费。",
        policy_citations=(citation("豁免保险费", "投保人豁免保险费。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [
        fact.fact_text for fact in result.facts if fact.status == "supported"
    ]
    assert supported_facts == ["投保人豁免保险费"]


def test_verifier_blocks_unsupported_shared_numeric_fact():
    result = verify_answer_facts(
        answer="等待期、保险期间均为90天。",
        policy_citations=(
            citation("保险期间", "保险期间为90天。"),
            citation("等待期", "等待期为30天。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason


def test_verifier_supports_shared_numeric_facts():
    result = verify_answer_facts(
        answer="等待期、保险期间均为90天。",
        policy_citations=(
            citation("等待期", "等待期为90天。"),
            citation("保险期间", "保险期间为90天。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [
        fact.fact_text for fact in result.facts if fact.status == "supported"
    ]
    assert supported_facts == ["等待期90天", "保险期间90天"]


def test_verifier_supports_shared_numeric_facts_in_same_policy_citation():
    result = verify_answer_facts(
        answer="等待期、保险期间均为90天。",
        policy_citations=(citation("等待期", "等待期、保险期间均为90天。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    supported_facts = [
        fact.fact_text for fact in result.facts if fact.status == "supported"
    ]
    assert supported_facts == ["等待期90天", "保险期间90天"]


def test_verifier_blocks_source_confusion_with_supported_number_and_builtin_definition():
    result = verify_answer_facts(
        answer="你的保单写明等待期为30天且等待期是保险合同生效后的一段观察时间。",
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


def test_verifier_allows_safe_policy_upload_instruction_with_builtin_citation():
    result = verify_answer_facts(
        answer="没有找到你的保单原文，请上传保单后再核对。",
        policy_citations=(),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is False
    assert result.facts == ()


def test_verifier_allows_safe_policy_fallback_that_mentions_policy_term_with_builtin_citation():
    result = verify_answer_facts(
        answer="没有找到你的保单原文，无法确认等待期，请上传保单后再核对。",
        policy_citations=(),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is False
    assert not any(fact.fact_type == "source_confusion" for fact in result.facts)


def test_verifier_allows_supported_policy_number_with_generic_caveat_and_builtin_citation():
    result = verify_answer_facts(
        answer="你的保单写明等待期为30天，请以保单条款为准。",
        policy_citations=(citation("等待期", "等待期为30天。"),),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is False
    assert not any(fact.fact_type == "source_confusion" for fact in result.facts)


def test_verifier_blocks_unsupported_number_before_caveat_without_separator():
    result = verify_answer_facts(
        answer="等待期为90天请以保单条款为准。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason


def test_verifier_allows_supported_policy_number_before_caveat_without_separator():
    result = verify_answer_facts(
        answer="你的保单写明等待期为30天请以保单条款为准。",
        policy_citations=(citation("等待期", "等待期为30天。"),),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is False
    assert not any(fact.fact_type == "source_confusion" for fact in result.facts)


def test_verifier_allows_temporarily_missing_policy_upload_instruction_with_builtin_citation():
    result = verify_answer_facts(
        answer="你的保单暂未上传，请上传后核对。",
        policy_citations=(),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is False
    assert result.facts == ()


def test_verifier_does_not_extract_uncertain_user_proposed_number_as_policy_fact():
    result = verify_answer_facts(
        answer="没有找到你的保单原文，无法确认等待期是否为90天，请上传保单后再核对。",
        policy_citations=(),
        builtin_citations=(
            citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),
        ),
    )

    assert result.has_blocking_fact is False
    assert not any(
        fact.fact_type == "number"
        and fact.status == "unsupported"
        and fact.fact_text == "等待期90天"
        for fact in result.facts
    )
    assert not any(fact.fact_type == "source_confusion" for fact in result.facts)


def test_verifier_blocks_asserted_number_after_fallback_phrase():
    result = verify_answer_facts(
        answer="没有找到你的保单原文但等待期为90天。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason


def test_verifier_does_not_extract_uncertain_shared_number_as_policy_fact():
    result = verify_answer_facts(
        answer="没有找到你的保单原文，无法确认等待期、保险期间均为90天，请上传保单后再核对。",
        policy_citations=(),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    assert result.facts == ()
