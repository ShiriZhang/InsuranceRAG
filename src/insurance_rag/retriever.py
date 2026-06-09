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
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    @classmethod
    def from_embeddings(
        cls,
        chunks: tuple[DocumentChunk, ...],
        embeddings: list[list[float]],
    ) -> "InMemoryVectorIndex":
        matrix = np.array(embeddings, dtype=np.float32)
        return cls(chunks=chunks, matrix=matrix)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        query = np.array(query_embedding, dtype=np.float32)
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
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    return InMemoryVectorIndex.from_embeddings(chunks, embeddings)
