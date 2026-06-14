import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from insurance_rag.models import (
    DocumentChunk,
    QueryRewriteResult,
    RetrievalExplanation,
    RetrievalRankDetail,
)


_INSURANCE_TERMS: tuple[str, ...] = (
    "等待期",
    "责任免除",
    "除外责任",
    "免责条款",
    "保险责任",
    "赔付条件",
    "保险金额",
    "保险期间",
    "生效日",
    "豁免保险费",
    "投保人豁免",
    "被保险人豁免",
    "重大疾病",
    "既往症",
)

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_TOKEN_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []

    for term in _INSURANCE_TERMS:
        if term in text:
            tokens.append(term)

    tokens.extend(match.group(0) for match in _ASCII_TOKEN_RE.finditer(text))

    cjk_chars = _CJK_CHAR_RE.findall(text)
    tokens.extend(cjk_chars)
    tokens.extend(
        cjk_chars[index] + cjk_chars[index + 1]
        for index in range(len(cjk_chars) - 1)
    )

    return tokens


@dataclass(frozen=True)
class HybridSearchResult:
    chunk: DocumentChunk
    final_score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    matched_terms: tuple[str, ...] = ()
    rank_details: tuple[RetrievalRankDetail, ...] = ()

    def to_explanation(self) -> RetrievalExplanation:
        return RetrievalExplanation(
            source_type=self.chunk.source_type,
            source_name=self.chunk.source_name,
            page_number=self.chunk.page_number,
            section_title=self.chunk.section_title,
            final_score=self.final_score,
            vector_score=self.vector_score,
            bm25_score=self.bm25_score,
            matched_terms=self.matched_terms,
            rank_details=self.rank_details,
        )


@dataclass
class _AccumulatedResult:
    chunk: DocumentChunk
    final_score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    matched_terms: tuple[str, ...] = ()
    rank_details: tuple[RetrievalRankDetail, ...] = ()

    def add_rank_detail(
        self,
        *,
        query: str,
        method: str,
        rank: int,
        score: float,
        rrf_score: float,
    ) -> None:
        self.final_score += rrf_score
        self.rank_details = self.rank_details + (
            RetrievalRankDetail(
                query=query,
                method=method,
                rank=rank,
                score=score,
            ),
        )

    def merge_matched_terms(self, terms: tuple[str, ...]) -> None:
        merged: list[str] = list(self.matched_terms)
        seen = set(merged)
        for term in terms:
            if term not in seen:
                seen.add(term)
                merged.append(term)
        self.matched_terms = tuple(merged)

    def to_result(self) -> HybridSearchResult:
        return HybridSearchResult(
            chunk=self.chunk,
            final_score=self.final_score,
            vector_score=self.vector_score,
            bm25_score=self.bm25_score,
            matched_terms=self.matched_terms,
            rank_details=self.rank_details,
        )


class HybridRetriever:
    def __init__(
        self,
        chunks,
        vector_index,
        embedder,
        *,
        rrf_k: int = 60,
        retrieval_mode: str = "hybrid",
    ) -> None:
        self.chunks: tuple[DocumentChunk, ...] = tuple(chunks)
        self.vector_index = vector_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.retrieval_mode = retrieval_mode
        self._tokenized_chunks = [
            tokenize_for_bm25(chunk.text) for chunk in self.chunks
        ]
        self._chunk_token_sets = [set(tokens) for tokens in self._tokenized_chunks]
        self._bm25 = None
        if any(self._tokenized_chunks):
            try:
                self._bm25 = BM25Okapi(self._tokenized_chunks)
            except Exception:
                self._bm25 = None

    def search(
        self,
        rewrite: QueryRewriteResult,
        top_k: int,
    ) -> list[HybridSearchResult]:
        if top_k <= 0 or not rewrite.expanded_queries:
            return []

        expanded_queries = list(rewrite.expanded_queries)
        query_embeddings = self.embedder.embed_texts(expanded_queries)
        if (
            self.retrieval_mode == "vector"
            and len(query_embeddings) != len(expanded_queries)
        ):
            raise ValueError(
                "Embedding count must match expanded query count in vector mode."
            )
        accumulated: dict[str, _AccumulatedResult] = {}

        for index, query in enumerate(expanded_queries):
            if index < len(query_embeddings):
                try:
                    self._add_vector_results(
                        accumulated,
                        query=query,
                        query_embedding=query_embeddings[index],
                        top_k=top_k,
                    )
                except ValueError:
                    if self.retrieval_mode != "hybrid":
                        raise
            if self.retrieval_mode == "hybrid":
                self._add_bm25_results(accumulated, query)

        return [
            result.to_result()
            for result in sorted(
                accumulated.values(),
                key=lambda result: result.final_score,
                reverse=True,
            )[:top_k]
        ]

    def _add_vector_results(
        self,
        accumulated: dict[str, _AccumulatedResult],
        *,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> None:
        vector_results = self.vector_index.search(query_embedding, top_k)
        for rank, result in enumerate(vector_results, start=1):
            merged = self._get_or_create(accumulated, result.chunk)
            merged.vector_score = _max_optional(merged.vector_score, result.score)
            merged.add_rank_detail(
                query=query,
                method="vector",
                rank=rank,
                score=result.score,
                rrf_score=self._rrf_score(rank),
            )

    def _add_bm25_results(
        self,
        accumulated: dict[str, _AccumulatedResult],
        query: str,
    ) -> None:
        if self._bm25 is None:
            return

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return

        try:
            scores = self._bm25.get_scores(query_tokens)
        except Exception:
            return
        ranked_indexes = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: scores[index],
            reverse=True,
        )
        query_token_set = set(query_tokens)
        if not ranked_indexes:
            overlap_scores = [
                len(query_token_set & chunk_token_set)
                for chunk_token_set in self._chunk_token_sets
            ]
            ranked_indexes = sorted(
                (index for index, score in enumerate(overlap_scores) if score > 0),
                key=lambda index: overlap_scores[index],
                reverse=True,
            )
            scores = overlap_scores

        for rank, index in enumerate(ranked_indexes, start=1):
            chunk = self.chunks[index]
            score = float(scores[index])
            merged = self._get_or_create(accumulated, chunk)
            merged.bm25_score = _max_optional(merged.bm25_score, score)
            merged.merge_matched_terms(
                _matched_terms(
                    query_tokens,
                    query_token_set,
                    self._chunk_token_sets[index],
                    query,
                    chunk.text,
                )
            )
            merged.add_rank_detail(
                query=query,
                method="bm25",
                rank=rank,
                score=score,
                rrf_score=self._rrf_score(rank),
            )

    def _get_or_create(
        self,
        accumulated: dict[str, _AccumulatedResult],
        chunk: DocumentChunk,
    ) -> _AccumulatedResult:
        if chunk.chunk_id not in accumulated:
            accumulated[chunk.chunk_id] = _AccumulatedResult(chunk=chunk)
        return accumulated[chunk.chunk_id]

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)


def _max_optional(current: float | None, candidate: float) -> float:
    if current is None:
        return candidate
    return max(current, candidate)


def _matched_terms(
    query_tokens: list[str],
    query_token_set: set[str],
    chunk_token_set: set[str],
    query_text: str,
    chunk_text: str,
) -> tuple[str, ...]:
    seen: set[str] = set()
    matched: list[str] = []
    for term in query_tokens:
        if (
            term in query_token_set
            and term in chunk_token_set
            and _is_display_matched_term(term, query_text, chunk_text)
            and term not in seen
        ):
            seen.add(term)
            matched.append(term)
    return tuple(_remove_subsumed_cjk_terms(matched))


def _is_display_matched_term(term: str, query_text: str, chunk_text: str) -> bool:
    if term in _INSURANCE_TERMS:
        return True
    if _ASCII_TOKEN_RE.fullmatch(term):
        return True
    return bool(
        len(term) >= 2
        and _CJK_TOKEN_RE.fullmatch(term)
        and term in query_text
        and term in chunk_text
    )


def _remove_subsumed_cjk_terms(terms: list[str]) -> list[str]:
    filtered: list[str] = []
    for term in terms:
        if not _CJK_TOKEN_RE.fullmatch(term):
            filtered.append(term)
            continue
        if any(
            term != other
            and len(term) < len(other)
            and _CJK_TOKEN_RE.fullmatch(other)
            and term in other
            for other in terms
        ):
            continue
        filtered.append(term)
    return filtered
