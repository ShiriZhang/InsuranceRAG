import pytest

from insurance_rag.models import DocumentChunk
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
