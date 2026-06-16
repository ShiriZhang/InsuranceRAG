from insurance_rag.models import (
    AnswerGuardResult,
    AnswerPayload,
    CitationVerificationResult,
    ClauseMetadata,
    DocumentChunk,
    GuardStatus,
    QueryRewriteResult,
    RerankExplanation,
    RetrievalExplanation,
    RetrievalRankDetail,
    VerifiedFact,
)


def test_guard_status_values():
    assert GuardStatus.PASS == "pass"
    assert GuardStatus.WARN == "warn"
    assert GuardStatus.BLOCK == "block"


def test_query_rewrite_result_defaults():
    result = QueryRewriteResult(
        original_query="住院医疗怎么赔？",
        expanded_queries=("住院医疗保险责任",),
    )

    assert result.original_query == "住院医疗怎么赔？"
    assert result.expanded_queries == ("住院医疗保险责任",)
    assert result.detected_intents == ()
    assert result.used_llm is False
    assert result.warnings == ()


def test_retrieval_rank_detail_carries_rank_data():
    detail = RetrievalRankDetail(
        query="住院医疗",
        method="bm25",
        rank=2,
        score=0.42,
    )

    assert detail.query == "住院医疗"
    assert detail.method == "bm25"
    assert detail.rank == 2
    assert detail.score == 0.42


def test_retrieval_explanation_match_strength_thresholds():
    high = RetrievalExplanation(
        source_type="builtin",
        source_name="sample.pdf",
        page_number=1,
        section_title="保障责任",
        final_score=0.03,
    )
    medium = RetrievalExplanation(
        source_type="policy",
        source_name="policy.pdf",
        page_number=3,
        section_title="等待期",
        final_score=0.015,
    )
    low = RetrievalExplanation(
        source_type="builtin",
        source_name="sample.pdf",
        page_number=None,
        section_title="除外责任",
        final_score=0.0149,
    )

    assert high.match_strength == "high"
    assert medium.match_strength == "medium"
    assert low.match_strength == "low"


def test_retrieval_explanation_optional_scores_and_details():
    rank_detail = RetrievalRankDetail(
        query="等待期",
        method="vector",
        rank=1,
        score=0.88,
    )
    explanation = RetrievalExplanation(
        source_type="policy",
        source_name="policy.pdf",
        page_number=5,
        section_title="等待期",
        final_score=0.02,
        vector_score=0.88,
        bm25_score=0.12,
        matched_terms=("等待期", "90天"),
        rank_details=(rank_detail,),
    )

    assert explanation.vector_score == 0.88
    assert explanation.bm25_score == 0.12
    assert explanation.matched_terms == ("等待期", "90天")
    assert explanation.rank_details == (rank_detail,)


def test_answer_guard_result_defaults_and_block_reason():
    warning = AnswerGuardResult(
        status=GuardStatus.WARN,
        warnings=("缺少引用",),
    )
    blocked = AnswerGuardResult(
        status=GuardStatus.BLOCK,
        block_reason="答案没有足够依据",
    )

    assert warning.status is GuardStatus.WARN
    assert warning.warnings == ("缺少引用",)
    assert warning.block_reason is None
    assert blocked.status is GuardStatus.BLOCK
    assert blocked.warnings == ()
    assert blocked.block_reason == "答案没有足够依据"


def test_answer_payload_carries_guard_result_and_retrieval_explanations():
    explanation = RetrievalExplanation(
        source_type="builtin",
        source_name="sample.pdf",
        page_number=1,
        section_title="保障责任",
        final_score=0.04,
    )
    guard_result = AnswerGuardResult(status=GuardStatus.PASS)

    payload = AnswerPayload(
        answer="可以赔付住院医疗费用。",
        retrieval_explanations=(explanation,),
        guard_result=guard_result,
    )

    assert payload.retrieval_explanations == (explanation,)
    assert payload.guard_result is guard_result


def test_clause_metadata_defaults_are_low_confidence_unknown():
    metadata = ClauseMetadata()

    assert metadata.clause_id is None
    assert metadata.heading_text is None
    assert metadata.section_title == "未识别条款标题"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"


def test_document_chunk_carries_clause_metadata_fields():
    chunk = DocumentChunk(
        chunk_id="c1",
        text="第六条 等待期\n等待期为90天。",
        page_number=6,
        section_title="等待期",
        source_type="user_policy",
        source_name="policy.pdf",
        extraction_method="text",
        clause_id="第六条",
        heading_text="第六条 等待期",
        heading_confidence="high",
        heading_source="line_pattern",
    )

    assert chunk.clause_id == "第六条"
    assert chunk.heading_text == "第六条 等待期"
    assert chunk.heading_confidence == "high"
    assert chunk.heading_source == "line_pattern"


def test_verification_result_carries_facts():
    fact = VerifiedFact(
        fact_text="等待期是90天",
        fact_type="number",
        status="supported",
        severity="info",
        supporting_citation_ids=("policy.pdf:6:等待期",),
        reason="同一引用中找到等待期和90天。",
    )
    result = CitationVerificationResult(facts=(fact,))

    assert result.facts == (fact,)
    assert result.has_blocking_fact is False
    assert result.has_warnings is False


def test_rerank_explanation_defaults():
    explanation = RerankExplanation(score=1.5, reasons=("title_intent_match",))

    assert explanation.score == 1.5
    assert explanation.reasons == ("title_intent_match",)
