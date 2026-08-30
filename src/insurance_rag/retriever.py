from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from insurance_rag.models import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class InMemoryVectorIndex:
    def __init__(self, chunks: tuple[DocumentChunk, ...], matrix: np.ndarray) -> None:
        self.chunks = chunks
        compatibility_keys = {chunk.index_compatibility_key for chunk in chunks}
        if len(compatibility_keys) != 1:
            raise ValueError("Index chunks must use one chunking strategy.")
        self.index_compatibility_key = next(iter(compatibility_keys))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    @classmethod
    def from_embeddings(
        cls,
        chunks: tuple[DocumentChunk, ...],
        embeddings: list[list[float]],
    ) -> "InMemoryVectorIndex":
        if not embeddings:
            raise ValueError("Cannot build vector index without embeddings.")
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")
        matrix = np.array(embeddings, dtype=np.float32)
        return cls(chunks=chunks, matrix=matrix)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query = np.array(query_embedding, dtype=np.float32)
        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                "Query embedding dimension must match index embedding dimension."
            )
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm
        scores = self.matrix @ query
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked
        ]


def build_index(
    chunks: tuple[DocumentChunk, ...],
    embedder: OpenAIEmbedder,
) -> InMemoryVectorIndex:
    if not chunks:
        raise ValueError("Cannot build vector index without embeddings.")
    embeddings = embedder.embed_texts([chunk.retrieval_text for chunk in chunks])
    return InMemoryVectorIndex.from_embeddings(chunks, embeddings)
