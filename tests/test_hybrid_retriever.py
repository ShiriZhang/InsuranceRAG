from insurance_rag.hybrid_retriever import HybridRetriever, tokenize_for_bm25
from insurance_rag.models import DocumentChunk, QueryRewriteResult
from insurance_rag.retriever import InMemoryVectorIndex


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        page_number=1,
        section_title="保险责任",
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


def make_rewrite(*expanded_queries: str) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=expanded_queries[0] if expanded_queries else "",
        expanded_queries=expanded_queries,
    )


def test_tokenize_for_bm25_keeps_insurance_terms_and_cjk_bigrams():
    tokens = tokenize_for_bm25("等待期90天 ABC123 责任免除")

    assert "等待期" in tokens
    assert "责任免除" in tokens
    assert "ABC123" in tokens
    assert "等" in tokens
    assert "待" in tokens
    assert "等待" in tokens
    assert "待期" in tokens


def test_hybrid_search_uses_bm25_to_recover_exact_term():
    chunks = (
        make_chunk("semantic", "普通保障说明，包含常见理赔流程。"),
        make_chunk("exact", "本产品明确列出投保人豁免的适用条件。"),
        make_chunk("other-1", "等待期和生效日另有约定。"),
        make_chunk("other-2", "保险金额以保险单载明为准。"),
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

    results = retriever.search(make_rewrite("投保人豁免"), top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["exact", "semantic"]
    assert results[0].bm25_score is not None
    assert results[0].vector_score is not None
    assert "投保人豁免" in results[0].matched_terms
    assert {detail.method for detail in results[0].rank_details} == {"vector", "bm25"}
    explanation = results[0].to_explanation()
    assert explanation.source_name == "user.pdf"
    assert explanation.final_score == results[0].final_score
    assert explanation.matched_terms == results[0].matched_terms


def test_hybrid_search_deduplicates_chunks_across_queries():
    chunks = (
        make_chunk("a", "保险责任包括重大疾病保障。"),
        make_chunk("b", "等待期后按条款赔付。"),
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = FakeEmbedder([[1.0, 0.0], [1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    results = retriever.search(make_rewrite("保险责任", "重大疾病"), top_k=2)

    assert [result.chunk.chunk_id for result in results].count("a") == 1
    assert len(results) == 2
    assert len([detail for detail in results[0].rank_details if detail.method == "vector"]) == 2


def test_vector_mode_skips_bm25_scores():
    chunks = (
        make_chunk("a", "保险责任包括重大疾病保障。"),
        make_chunk("b", "等待期后按条款赔付。"),
    )
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder, retrieval_mode="vector")

    results = retriever.search(make_rewrite("等待期"), top_k=2)

    assert all(result.bm25_score is None for result in results)
    assert all(result.matched_terms == () for result in results)
    assert {detail.method for result in results for detail in result.rank_details} == {"vector"}


def test_top_k_less_than_one_returns_empty():
    chunks = (make_chunk("a", "保险责任包括重大疾病保障。"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    assert retriever.search(make_rewrite("保险责任"), top_k=0) == []
    assert embedder.calls == []


def test_no_expanded_queries_returns_empty_without_calling_embedder():
    chunks = (make_chunk("a", "保险责任包括重大疾病保障。"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder([[1.0, 0.0]])
    retriever = HybridRetriever(chunks, index, embedder)

    assert retriever.search(make_rewrite(), top_k=1) == []
    assert embedder.calls == []
