from insurance_rag.hybrid_retriever import HybridSearchResult
from insurance_rag.models import DocumentChunk
from insurance_rag.query_rewriter import rewrite_query
from insurance_rag.rule_reranker import rerank_results


def chunk(
    chunk_id: str, title: str, text: str, *, confidence: str = "high"
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        page_number=1,
        section_title=title,
        source_type="user_policy",
        source_name="policy.pdf",
        extraction_method="text",
        heading_confidence=confidence,
    )


def result(
    chunk_id: str, title: str, text: str, score: float = 0.01
) -> HybridSearchResult:
    return HybridSearchResult(chunk=chunk(chunk_id, title, text), final_score=score)


def test_waiting_period_question_ranks_waiting_period_before_insurance_period():
    candidates = [
        result("period", "保险期间", "保险期间为90天。", score=0.05),
        result("waiting", "等待期", "等待期为90天。", score=0.01),
    ]

    reranked = rerank_results(
        question="等待期是多久？",
        rewrite=rewrite_query("等待期是多久？"),
        candidates=candidates,
        top_k=2,
    )

    assert reranked[0].chunk.chunk_id == "waiting"
    assert "title_intent_match" in reranked[0].rerank_reasons


def test_exclusion_question_ranks_exclusion_before_coverage():
    candidates = [
        result("coverage", "保险责任", "本合同承担重大疾病保险责任。", score=0.05),
        result("exclusion", "责任免除", "酒后驾驶属于责任免除。", score=0.01),
    ]

    reranked = rerank_results(
        question="哪些情况不赔？",
        rewrite=rewrite_query("哪些情况不赔？"),
        candidates=candidates,
        top_k=2,
    )

    assert reranked[0].chunk.chunk_id == "exclusion"
    assert "exclusion_fact_type_match" in reranked[0].rerank_reasons


def test_low_heading_confidence_adds_negative_reason():
    candidates = [
        HybridSearchResult(
            chunk=chunk("low", "等待期", "等待期为90天。", confidence="low"),
            final_score=0.01,
        )
    ]

    reranked = rerank_results(
        question="等待期是多久？",
        rewrite=rewrite_query("等待期是多久？"),
        candidates=candidates,
        top_k=1,
    )

    assert "low_heading_confidence" in reranked[0].rerank_reasons


def test_strong_hybrid_result_is_not_swamped_by_small_rule_hit():
    candidates = [
        result("strong", "保险期间", "保险期间为90天。", score=0.50),
        result("weak", "等待期", "等待期为90天。", score=0.01),
    ]

    reranked = rerank_results(
        question="等待期是多久？",
        rewrite=rewrite_query("等待期是多久？"),
        candidates=candidates,
        top_k=2,
    )

    assert reranked[0].chunk.chunk_id == "strong"
    assert "title_intent_match" in reranked[1].rerank_reasons


def test_definition_question_with_concrete_term_prefers_concrete_clause():
    candidates = [
        result("definition", "释义", "等待期是指合同生效后的一段期间。", score=0.05),
        result("waiting", "等待期", "等待期为90天。", score=0.01),
    ]

    reranked = rerank_results(
        question="什么是等待期？",
        rewrite=rewrite_query("什么是等待期？"),
        candidates=candidates,
        top_k=2,
    )

    assert reranked[0].chunk.chunk_id == "waiting"
    assert "title_intent_match" in reranked[0].rerank_reasons


def test_exclusion_synonym_heading_is_promoted():
    candidates = [
        result("coverage", "保险责任", "本合同承担重大疾病保险责任。", score=0.05),
        result("synonym", "免除责任", "酒后驾驶属于免除责任。", score=0.01),
    ]

    reranked = rerank_results(
        question="哪些情况不赔？",
        rewrite=rewrite_query("哪些情况不赔？"),
        candidates=candidates,
        top_k=2,
    )

    assert reranked[0].chunk.chunk_id == "synonym"
    assert "title_intent_match" in reranked[0].rerank_reasons


def test_no_intent_no_match_preserves_hybrid_order():
    candidates = [
        result("first", "其他说明", "甲内容。", score=0.05),
        result("second", "其他说明", "乙内容。", score=0.01),
    ]

    reranked = rerank_results(
        question="随便问问",
        rewrite=rewrite_query("随便问问"),
        candidates=candidates,
        top_k=2,
    )

    assert [item.chunk.chunk_id for item in reranked] == ["first", "second"]
    assert reranked[0].rerank_reasons == ()
    assert reranked[1].rerank_reasons == ()


def test_directory_like_chunk_is_demoted_but_explained():
    directory_chunk = HybridSearchResult(
        chunk=chunk(
            "directory",
            "等待期",
            "1 等待期 ........ 2\n2 保险期间 ........ 3\n3 责任免除 ........ 4",
        ),
        final_score=0.06,
    )
    strong_chunk = result("strong", "其他说明", "正文内容。", score=0.50)

    reranked = rerank_results(
        question="等待期是多久？",
        rewrite=rewrite_query("等待期是多久？"),
        candidates=[directory_chunk, strong_chunk],
        top_k=2,
    )

    assert reranked[0].chunk.chunk_id == "strong"
    assert "directory_like_chunk" in reranked[1].rerank_reasons
