import pytest

import insurance_rag.hybrid_retriever as hybrid_module
from insurance_rag.chunker import chunk_pages
from insurance_rag.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    tokenize_for_bm25,
)
from insurance_rag.models import DocumentChunk, DocumentPage, QueryRewriteResult
from insurance_rag.retriever import InMemoryVectorIndex


WAITING_PERIOD = "\u7b49\u5f85\u671f"
EXCLUSION = "\u8d23\u4efb\u514d\u9664"
COVERAGE = "\u4fdd\u9669\u8d23\u4efb"
CRITICAL_ILLNESS = "\u91cd\u5927\u75be\u75c5"
WAIVER = "\u6295\u4fdd\u4eba\u8c41\u514d"
EFFECTIVE_DATE = "\u751f\u6548\u65e5"
INSURED_AMOUNT = "\u4fdd\u9669\u91d1\u989d"
PROTECTION = "\u4fdd\u969c"


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        page_number=1,
        section_title=COVERAGE,
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
    )


class FakeEmbedder:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self.embeddings[: len(texts)]


class RuntimeErrorVectorIndex:
    index_compatibility_key = "chunking:legacy"

    def search(self, query_embedding: list[float], top_k: int):
        raise RuntimeError("unexpected vector failure")


def make_rewrite(*expanded_queries: str) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=expanded_queries[0] if expanded_queries else "",
        expanded_queries=expanded_queries,
    )


def test_tokenize_for_bm25_keeps_insurance_terms_and_cjk_bigrams():
    tokens = tokenize_for_bm25(f"{WAITING_PERIOD}90\u5929 ABC123 {EXCLUSION}")

    assert WAITING_PERIOD in tokens
    assert EXCLUSION in tokens
    assert "ABC123" in tokens
    assert "\u7b49" in tokens
    assert "\u5f85" in tokens
    assert "\u7b49\u5f85" in tokens
    assert "\u5f85\u671f" in tokens


def test_hybrid_search_uses_bm25_to_recover_exact_term():
    chunks = (
        make_chunk("semantic", "\u666e\u901a\u4fdd\u969c\u8bf4\u660e"),
        make_chunk("exact", f"\u672c\u4ea7\u54c1\u660e\u786e\u5217\u51fa{WAIVER}\u7684\u9002\u7528\u6761\u4ef6"),
        make_chunk("other-1", f"{WAITING_PERIOD}\u548c{EFFECTIVE_DATE}\u53e6\u6709\u7ea6\u5b9a"),
        make_chunk("other-2", f"{INSURED_AMOUNT}\u4ee5\u4fdd\u9669\u5355\u8f7d\u660e\u4e3a\u51c6"),
    )
    index = InMemoryVectorIndex.from_embeddings(
        chunks,
        [
            [1.0, 0.0],
            [0.05, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder, rrf_k=1)

    results = retriever.search(make_rewrite(WAIVER), top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["exact", "semantic"]
    assert results[0].bm25_score is not None
    assert results[0].vector_score is not None
    assert WAIVER in results[0].matched_terms
    assert {detail.method for detail in results[0].rank_details} == {"vector", "bm25"}
    explanation = results[0].to_explanation()
    assert explanation.source_name == "user.pdf"
    assert explanation.final_score == results[0].final_score
    assert explanation.matched_terms == results[0].matched_terms


def test_hybrid_search_uses_retrieval_only_clause_context_for_lexical_matching():
    chunks = chunk_pages(
        (
            DocumentPage(
                1,
                "第六条 等待期\n等待期为九十日。\n第七条 保险责任\n保障重大疾病。",
                "text",
            ),
        ),
        source_name="user.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[0.0, 0.0], [0.0, 0.0]])
    retriever = HybridRetriever(chunks, index, FakeEmbedder([[0.0, 0.0]]))

    results = retriever.search(make_rewrite("Policy Clause"), top_k=1)

    assert results
    assert {"Policy", "Clause"}.issubset(results[0].matched_terms)


def test_hybrid_retriever_rejects_index_without_compatibility_identity():
    class UnversionedIndex:
        def search(self, query_embedding: list[float], top_k: int):
            return []

    chunk = make_chunk("legacy", "等待期为九十日。")

    with pytest.raises(ValueError, match="compatibility identity"):
        HybridRetriever((chunk,), UnversionedIndex(), FakeEmbedder([[1.0, 0.0]]))


def test_hybrid_search_result_explanation_carries_rerank_details():
    chunk = make_chunk("c1", "等待期为90天。")
    result = HybridSearchResult(
        chunk=chunk,
        final_score=0.02,
        rerank_score=2.0,
        rerank_reasons=("title_intent_match",),
    )

    explanation = result.to_explanation()

    assert explanation.rerank_score == 2.0
    assert explanation.rerank_reasons == ("title_intent_match",)


def test_exact_insurance_term_match_outranks_partial_token_matches():
    chunks = (
        make_chunk("partial-1", "保险 责任 保障 范围 " * 20),
        make_chunk("partial-2", "保单主要保障责任范围，保险费用责任说明 " * 20),
        make_chunk("exact", "第七条 保险责任 本合同承担重大疾病保险责任。"),
        make_chunk("partial-3", "保险金额和责任免除另有约定 " * 20),
    )
    index = InMemoryVectorIndex.from_embeddings(
        chunks,
        [[0.0, 0.0] for _chunk in chunks],
    )
    embedder = FakeEmbedder([[0.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite(COVERAGE), top_k=3)

    assert results[0].chunk.chunk_id == "exact"
    assert COVERAGE in results[0].matched_terms


def test_hybrid_search_deduplicates_chunks_across_queries():
    chunks = (
        make_chunk("a", f"{COVERAGE}\u5305\u62ec{CRITICAL_ILLNESS}\u4fdd\u969c"),
        make_chunk("b", f"{WAITING_PERIOD}\u540e\u6309\u6761\u6b3e\u8d54\u4ed8"),
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = FakeEmbedder([[1.0, 0.0], [1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite(COVERAGE, CRITICAL_ILLNESS), top_k=2)

    assert [result.chunk.chunk_id for result in results].count("a") == 1
    assert len(results) == 2
    assert len([detail for detail in results[0].rank_details if detail.method == "vector"]) == 2


def test_vector_mode_skips_bm25_scores():
    chunks = (
        make_chunk("a", f"{COVERAGE}\u5305\u62ec{CRITICAL_ILLNESS}\u4fdd\u969c"),
        make_chunk("b", f"{WAITING_PERIOD}\u540e\u6309\u6761\u6b3e\u8d54\u4ed8"),
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder, retrieval_mode="vector")

    results = retriever.search(make_rewrite(WAITING_PERIOD), top_k=2)

    assert all(result.bm25_score is None for result in results)
    assert all(result.matched_terms == () for result in results)
    assert {detail.method for result in results for detail in result.rank_details} == {"vector"}


def test_vector_mode_embedding_count_mismatch_raises_value_error():
    chunks = (
        make_chunk("a", "alpha coverage"),
        make_chunk("b", "beta waiting period"),
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder, retrieval_mode="vector")

    with pytest.raises(ValueError, match="Embedding count"):
        retriever.search(make_rewrite("alpha", "beta"), top_k=2)


def test_hybrid_retriever_rejects_index_from_incompatible_chunking_strategy():
    clause_v2_chunk = DocumentChunk(
        chunk_id="clause-v2",
        text="等待期为九十日。",
        page_number=1,
        section_title=WAITING_PERIOD,
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
        chunking_strategy="clause_v2",
    )
    legacy_chunk = make_chunk("legacy", "等待期为九十日。")
    clause_v2_index = InMemoryVectorIndex.from_embeddings(
        (clause_v2_chunk,),
        [[1.0, 0.0]],
    )

    with pytest.raises(ValueError, match="chunking strategy"):
        HybridRetriever(
            (legacy_chunk,),
            clause_v2_index,
            FakeEmbedder([[1.0, 0.0]]),
        )


def test_top_k_less_than_one_returns_empty():
    chunks = (make_chunk("a", f"{COVERAGE}\u5305\u62ec{CRITICAL_ILLNESS}\u4fdd\u969c"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    assert retriever.search(make_rewrite(COVERAGE), top_k=0) == []
    assert embedder.calls == []


def test_no_expanded_queries_returns_empty_without_calling_embedder():
    chunks = (make_chunk("a", f"{COVERAGE}\u5305\u62ec{CRITICAL_ILLNESS}\u4fdd\u969c"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    assert retriever.search(make_rewrite(), top_k=1) == []
    assert embedder.calls == []


def test_blank_chunk_does_not_crash_bm25_construction():
    chunks = (make_chunk("blank", "   \n\t"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite(COVERAGE), top_k=1)

    assert [result.chunk.chunk_id for result in results] == ["blank"]
    assert results[0].bm25_score is None


def test_hybrid_search_falls_back_to_vector_when_bm25_construction_fails(monkeypatch):
    chunks = (
        make_chunk("semantic", f"{COVERAGE}包括{CRITICAL_ILLNESS}保障"),
        make_chunk("other", f"{WAITING_PERIOD}后按条款赔付"),
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])

    def raise_bm25_error(_tokens):
        raise RuntimeError("bm25 unavailable")

    monkeypatch.setattr(hybrid_module, "BM25Okapi", raise_bm25_error)
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite(COVERAGE), top_k=1)

    assert [result.chunk.chunk_id for result in results] == ["semantic"]
    assert results[0].vector_score is not None
    assert results[0].bm25_score is None


def test_hybrid_search_falls_back_to_vector_when_bm25_search_fails():
    chunks = (
        make_chunk("semantic", f"{COVERAGE}包括{CRITICAL_ILLNESS}保障"),
        make_chunk("other", f"{WAITING_PERIOD}后按条款赔付"),
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    class BrokenBM25:
        def get_scores(self, _tokens):
            raise RuntimeError("bm25 search failed")

    retriever._bm25 = BrokenBM25()

    results = retriever.search(make_rewrite(COVERAGE), top_k=1)

    assert [result.chunk.chunk_id for result in results] == ["semantic"]
    assert results[0].vector_score is not None
    assert results[0].bm25_score is None


def test_hybrid_search_keeps_bm25_when_embedder_returns_fewer_embeddings():
    chunks = (
        make_chunk("semantic", "\u666e\u901a\u4fdd\u969c\u8bf4\u660e"),
        make_chunk("exact", f"\u672c\u4ea7\u54c1\u660e\u786e\u5217\u51fa{WAIVER}\u7684\u9002\u7528\u6761\u4ef6"),
        make_chunk("other-1", f"{WAITING_PERIOD}\u548c{EFFECTIVE_DATE}\u53e6\u6709\u7ea6\u5b9a"),
        make_chunk("other-2", f"{INSURED_AMOUNT}\u4ee5\u4fdd\u9669\u5355\u8f7d\u660e\u4e3a\u51c6"),
    )
    index = InMemoryVectorIndex.from_embeddings(
        chunks,
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite("\u666e\u901a\u4fdd\u969c", WAIVER), top_k=3)

    exact = next(result for result in results if result.chunk.chunk_id == "exact")
    assert exact.bm25_score is not None
    assert any(
        detail.method == "bm25" and detail.query == WAIVER
        for detail in exact.rank_details
    )


def test_hybrid_search_continues_bm25_when_vector_search_fails():
    chunks = (
        make_chunk("exact", f"\u672c\u4ea7\u54c1\u660e\u786e\u5217\u51fa{WAIVER}\u7684\u9002\u7528\u6761\u4ef6"),
        make_chunk("other-1", f"{WAITING_PERIOD}\u548c{EFFECTIVE_DATE}\u53e6\u6709\u7ea6\u5b9a"),
        make_chunk("other-2", f"{INSURED_AMOUNT}\u4ee5\u4fdd\u9669\u5355\u8f7d\u660e\u4e3a\u51c6"),
        make_chunk("other-3", f"{EXCLUSION}\u6761\u6b3e\u53e6\u89c1\u5408\u540c"),
    )
    index = InMemoryVectorIndex.from_embeddings(
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )
    embedder = FakeEmbedder([[1.0, 0.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite(WAIVER), top_k=1)

    assert [result.chunk.chunk_id for result in results] == ["exact"]
    assert results[0].vector_score is None
    assert results[0].bm25_score is not None


def test_hybrid_search_propagates_unexpected_vector_errors():
    chunks = (
        make_chunk("exact", "exactterm coverage applies"),
        make_chunk("other-1", "waiting period details"),
        make_chunk("other-2", "coverage amount details"),
        make_chunk("other-3", "exclusion details"),
    )
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, RuntimeErrorVectorIndex(), embedder)

    with pytest.raises(RuntimeError, match="unexpected vector failure"):
        retriever.search(make_rewrite("exactterm"), top_k=1)


def test_matched_terms_excludes_single_character_cjk_display_terms():
    chunks = (
        make_chunk("exact", f"{WAIVER} ABC {PROTECTION}\u9002\u7528"),
        make_chunk("other-1", f"{WAITING_PERIOD}\u548c{EFFECTIVE_DATE}\u53e6\u6709\u7ea6\u5b9a"),
        make_chunk("other-2", f"{INSURED_AMOUNT}\u4ee5\u4fdd\u9669\u5355\u8f7d\u660e\u4e3a\u51c6"),
        make_chunk("other-3", f"{EXCLUSION}\u6761\u6b3e\u53e6\u89c1\u5408\u540c"),
    )
    index = InMemoryVectorIndex.from_embeddings(
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite(f"{WAIVER} ABC {PROTECTION}"), top_k=1)

    assert results[0].matched_terms == (WAIVER, "ABC", PROTECTION)
