import pytest

from insurance_rag.chunker import chunk_pages
from insurance_rag.models import DocumentChunk, DocumentPage
from insurance_rag.retriever import InMemoryVectorIndex, build_index


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


def test_vector_index_returns_most_similar_chunk():
    chunks = (make_chunk("a", "等待期为九十日"), make_chunk("b", "责任免除条款"))
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = index.search([0.9, 0.1], top_k=1)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].score > 0.9


def test_vector_index_rejects_mismatched_chunk_and_embedding_counts():
    chunks = (make_chunk("a", "等待期为九十日"), make_chunk("b", "责任免除条款"))

    with pytest.raises(ValueError, match="chunks and embeddings"):
        InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])


def test_vector_index_rejects_empty_embeddings():
    with pytest.raises(ValueError, match="Cannot build vector index without embeddings"):
        InMemoryVectorIndex.from_embeddings((), [])


def test_vector_index_rejects_query_dimension_mismatch():
    chunks = (make_chunk("a", "等待期为九十日"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])

    with pytest.raises(ValueError, match="Query embedding dimension"):
        index.search([1.0, 0.0, 0.0], top_k=1)


def test_vector_index_returns_empty_for_non_positive_top_k():
    chunks = (make_chunk("a", "等待期为九十日"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])

    assert index.search([1.0, 0.0], top_k=0) == []
    assert index.search([1.0, 0.0], top_k=-1) == []


def test_vector_index_returns_empty_for_zero_query():
    chunks = (make_chunk("a", "等待期为九十日"),)
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])

    assert index.search([0.0, 0.0], top_k=1) == []


def test_build_index_rejects_empty_chunks_without_calling_embedder():
    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return []

    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="Cannot build vector index without embeddings"):
        build_index((), embedder)

    assert embedder.calls == []


def test_build_index_embeds_retrieval_context_with_authoritative_source_text():
    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return [[1.0, 0.0]]

    chunk = chunk_pages(
        (DocumentPage(1, "第六条 等待期\n等待期为九十日。", "text"),),
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
    )[0]
    embedder = FakeEmbedder()

    build_index((chunk,), embedder)

    assert embedder.calls == [[chunk.retrieval_text]]


def test_clause_v2_representative_policy_builds_production_equivalent_index():
    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    pages = (
        DocumentPage(1, "投保前请仔细阅读。", "text"),
        DocumentPage(2, "第六条 等待期\n等待期自合同生效日起计算。", "text"),
        DocumentPage(3, "等待期为九十日。", "text"),
        DocumentPage(
            4,
            "第七条 保险责任\n"
            "本合同对一次事故造成的全部损失按照约定比例承担保险责任且不超过保险金额。",
            "text",
        ),
    )
    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=900,
        overlap=0,
        strategy="clause_v2",
        target_chars=40,
        hard_max_chars=48,
    )
    embedder = FakeEmbedder()

    index = build_index(chunks, embedder)

    assert index.chunks == chunks
    assert embedder.calls == [[chunk.retrieval_text for chunk in chunks]]
    assert all(chunk.index_compatibility_key == "chunking:clause_v2" for chunk in chunks)
    assert all(len(chunk.retrieval_text) <= 48 for chunk in chunks)
