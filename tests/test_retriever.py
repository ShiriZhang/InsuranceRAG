from insurance_rag.models import DocumentChunk
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


def test_vector_index_returns_most_similar_chunk():
    chunks = (make_chunk("a", "等待期为九十日"), make_chunk("b", "责任免除条款"))
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = index.search([0.9, 0.1], top_k=1)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].score > 0.9
