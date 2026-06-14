from insurance_rag.answer_guard import BLOCKED_ANSWER, check_answer
from insurance_rag.models import Citation, GuardStatus, RetrievalExplanation


def citation(
    *,
    source_type: str = "policy",
    quality_notes: tuple[str, ...] = (),
    section_title: str = "保障责任",
    excerpt: str = "住院医疗费用按合同约定给付。",
) -> Citation:
    return Citation(
        source_type=source_type,
        source_name="sample.pdf",
        page_number=1,
        section_title=section_title,
        excerpt=excerpt,
        quality_notes=quality_notes,
    )


def explanation(final_score: float = 0.03) -> RetrievalExplanation:
    return RetrievalExplanation(
        source_type="policy",
        source_name="sample.pdf",
        page_number=1,
        section_title="保障责任",
        final_score=final_score,
    )


def check(
    answer: str,
    *,
    policy_citations: tuple[Citation, ...] = (),
    builtin_citations: tuple[Citation, ...] = (),
    retrieval_explanations: tuple[RetrievalExplanation, ...] = (),
):
    return check_answer(
        question="住院医疗能赔吗？",
        answer=answer,
        policy_citations=policy_citations,
        builtin_citations=builtin_citations,
        retrieval_explanations=retrieval_explanations,
    )


def test_blocks_specific_policy_answer_without_user_policy_citation():
    result = check("这份保单写明住院医疗费用可以报销。")

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason == "回答包含具体保单事实，但没有用户保单引用。"
    assert "不能直接给出该结论" in BLOCKED_ANSWER


def test_blocks_policy_fact_not_supported_by_cited_evidence():
    result = check(
        "这份保单的等待期为九十日。",
        policy_citations=(
            citation(section_title="保障责任", excerpt="住院医疗费用按合同约定给付。"),
            citation(section_title="责任免除", excerpt="酒后驾驶属于责任免除。"),
        ),
        retrieval_explanations=(explanation(final_score=0.03),),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "未被用户保单引用支持" in result.block_reason


def test_blocks_unsupported_policy_fact_without_trigger_phrase():
    result = check(
        "等待期是90天。",
        policy_citations=(
            citation(section_title="保险责任", excerpt="本合同保险责任包括重大疾病保险金。"),
            citation(section_title="责任免除", excerpt="酒后驾驶属于责任免除。"),
        ),
        retrieval_explanations=(explanation(final_score=0.03),),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "未被用户保单引用支持" in result.block_reason


def test_allows_supported_policy_fact_with_equivalent_chinese_number_unit():
    result = check(
        "等待期是90天。",
        policy_citations=(
            citation(section_title="等待期", excerpt="本合同等待期为九十日。"),
            citation(section_title="保险责任", excerpt="等待期后按合同约定承担保险责任。"),
        ),
        retrieval_explanations=(explanation(final_score=0.03),),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()


def test_blocks_numeric_policy_fact_when_number_belongs_to_different_clause():
    result = check(
        "等待期是90天。",
        policy_citations=(
            citation(section_title="等待期", excerpt="本合同等待期为30天。"),
            citation(section_title="保险期间", excerpt="保险期间可至90天。"),
        ),
        retrieval_explanations=(explanation(final_score=0.03),),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "未被用户保单引用支持" in result.block_reason


def test_allows_policy_fact_supported_by_cited_evidence():
    result = check(
        "这份保单写明住院医疗费用按合同约定给付。",
        policy_citations=(
            citation(excerpt="住院医疗费用按合同约定给付。"),
            citation(excerpt="住院医疗费用按合同约定给付。"),
        ),
        retrieval_explanations=(explanation(final_score=0.03),),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()


def test_blocks_final_claim_decision():
    result = check(
        "保险公司必须赔付这次住院费用。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "最终理赔判断" in result.block_reason


def test_blocks_direct_non_payment_decision():
    result = check(
        "根据条款，这种情况不赔。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "最终理赔判断" in result.block_reason


def test_blocks_direct_claim_payment_decision():
    result = check(
        "根据条款，这次可以理赔。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "最终理赔判断" in result.block_reason


def test_blocks_direct_no_reimbursement_decision():
    result = check(
        "根据条款，该费用不予赔付。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "最终理赔判断" in result.block_reason


def test_blocks_direct_reimbursement_decision():
    result = check(
        "该费用可以报销。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "最终理赔判断" in result.block_reason


def test_allows_cautionary_phrase_around_certain_non_payment():
    result = check(
        "不能直接判断一定不赔，应核对免责条款。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()


def test_allows_cautionary_phrase_around_insurer_must_pay():
    result = check(
        "不应写保险公司必须赔，应以保险公司和合同原文为准。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()


def test_allows_cautionary_phrase_around_certain_payment():
    result = check(
        "不能说肯定赔，需要结合事故事实和条款。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()


def test_allows_cautionary_phrase_around_direct_non_payment():
    result = check(
        "不能直接判断不赔，应核对免责条款。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()


def test_allows_cautionary_phrase_around_claim_payment():
    result = check(
        "不能说可以理赔，需要结合事故事实和条款。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()


def test_still_blocks_direct_final_claim_decision():
    result = check(
        "这种情况一定赔。",
        policy_citations=(citation(), citation()),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "最终理赔判断" in result.block_reason


def test_blocks_when_builtin_context_is_treated_as_user_policy():
    result = check(
        "你的保单写明等待期是保险合同生效后的一段观察时间。",
        policy_citations=(citation(), citation()),
        builtin_citations=(
            citation(
                source_type="builtin",
                section_title="等待期",
                excerpt="等待期是保险合同生效后的一段观察时间。",
            ),
        ),
    )

    assert result.status is GuardStatus.BLOCK
    assert result.block_reason is not None
    assert "内置资料库" in result.block_reason


def test_warns_for_low_score_evidence():
    result = check(
        "根据检索到的条款，住院医疗需要继续核对免赔额和等待期。",
        policy_citations=(citation(), citation()),
        retrieval_explanations=(explanation(final_score=0.009),),
    )

    assert result.status is GuardStatus.WARN
    assert any("检索分数较低" in warning for warning in result.warnings)


def test_passes_grounded_explanation():
    result = check(
        "根据用户保单引用，住院医疗责任需要核对等待期、免赔额和责任免除条款。",
        policy_citations=(citation(), citation()),
        retrieval_explanations=(explanation(final_score=0.03),),
    )

    assert result.status is GuardStatus.PASS
    assert result.warnings == ()
    assert result.block_reason is None


def test_warns_for_ocr_or_text_quality_notes():
    result = check(
        "根据用户保单引用，住院医疗责任需要结合原文核对。",
        policy_citations=(citation(quality_notes=("OCR confidence is low",)), citation()),
    )

    assert result.status is GuardStatus.WARN
    assert any("OCR" in warning or "文本提取" in warning for warning in result.warnings)
