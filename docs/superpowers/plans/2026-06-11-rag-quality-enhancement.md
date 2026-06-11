# RAG Quality Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hybrid retrieval, rule-based query rewriting, answer self-checking, retrieval explanations, and offline evaluation to the InsuranceRAG Streamlit MVP.

**Architecture:** Keep `rag_chain.py` as the orchestration layer and move new behavior into focused modules: `query_rewriter.py`, `hybrid_retriever.py`, `answer_guard.py`, and `evaluation.py`. Use `rank-bm25` for keyword retrieval, reciprocal rank fusion for result merging, and deterministic fake embeddings for reproducible synthetic evaluation.

**Tech Stack:** Python, Streamlit, OpenAI API, NumPy, PyMuPDF, pytest, rank-bm25.

---

## File Structure

- Modify `requirements.txt`: add `rank-bm25`.
- Modify `src/insurance_rag/config.py`: add retrieval mode, RRF, optional LLM rewrite/guard flags, and eval report directory.
- Modify `src/insurance_rag/models.py`: add query rewrite, retrieval explanation, and answer guard dataclasses; extend `AnswerPayload`.
- Create `src/insurance_rag/query_rewriter.py`: rule-based query expansion.
- Create `src/insurance_rag/hybrid_retriever.py`: BM25 index, tokenization, vector/BM25 search, RRF fusion, score explanations.
- Create `src/insurance_rag/answer_guard.py`: programmatic pass/warn/block checks.
- Modify `src/insurance_rag/rag_chain.py`: orchestrate query rewriting, hybrid retrieval, built-in background retrieval, and answer guard.
- Modify `app.py`: build/store hybrid retrievers and render collapsed retrieval details.
- Create `src/insurance_rag/evaluation.py`: synthetic/local evaluation helpers and report rendering.
- Create `scripts/evaluate_rag.py`: CLI entrypoint for offline evaluation.
- Create `evals/synthetic_cases.json`: reproducible synthetic evaluation cases.
- Modify `.gitignore`: ignore `eval_reports/` and `.rag_eval_cache/`.
- Modify `README.md`: document hybrid retrieval, answer guard, and offline evaluation commands.
- Add tests:
  - `tests/test_query_rewriter.py`
  - `tests/test_hybrid_retriever.py`
  - `tests/test_answer_guard.py`
  - `tests/test_evaluation.py`
  - update `tests/test_config.py`
  - update `tests/test_rag_chain.py`

---

## Task 1: Configuration, Dependencies, and Shared Models

**Files:**
- Modify: `requirements.txt`
- Modify: `src/insurance_rag/config.py`
- Modify: `src/insurance_rag/models.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing config tests**

Add these tests to `tests/test_config.py`:

```python
from insurance_rag.config import AppConfig


def test_retrieval_quality_config_defaults(monkeypatch):
    monkeypatch.delenv("INSURANCE_RAG_RETRIEVAL_MODE", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_RRF_K", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_QUERY_REWRITE_LLM", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_ANSWER_GUARD_LLM", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_EVAL_REPORT_DIR", raising=False)

    config = AppConfig.from_env()

    assert config.retrieval_mode == "hybrid"
    assert config.rrf_k == 60
    assert config.query_rewrite_llm is False
    assert config.answer_guard_llm is False
    assert config.eval_report_dir == "eval_reports"


def test_retrieval_quality_config_from_env(monkeypatch):
    monkeypatch.setenv("INSURANCE_RAG_RETRIEVAL_MODE", "vector")
    monkeypatch.setenv("INSURANCE_RAG_RRF_K", "25")
    monkeypatch.setenv("INSURANCE_RAG_QUERY_REWRITE_LLM", "true")
    monkeypatch.setenv("INSURANCE_RAG_ANSWER_GUARD_LLM", "yes")
    monkeypatch.setenv("INSURANCE_RAG_EVAL_REPORT_DIR", "custom_reports")

    config = AppConfig.from_env()

    assert config.retrieval_mode == "vector"
    assert config.rrf_k == 25
    assert config.query_rewrite_llm is True
    assert config.answer_guard_llm is True
    assert config.eval_report_dir == "custom_reports"
```

- [ ] **Step 2: Write failing model tests**

Create `tests/test_models.py`:

```python
from insurance_rag.models import (
    AnswerGuardResult,
    AnswerPayload,
    GuardStatus,
    QueryRewriteResult,
    RetrievalExplanation,
    RetrievalRankDetail,
)


def test_query_rewrite_result_preserves_original_and_expansions():
    result = QueryRewriteResult(
        original_query="这个赔不赔？",
        expanded_queries=("这个赔不赔？", "保险责任", "责任免除"),
        detected_intents=("claim_condition",),
    )

    assert result.original_query == "这个赔不赔？"
    assert result.expanded_queries == ("这个赔不赔？", "保险责任", "责任免除")
    assert result.used_llm is False
    assert result.warnings == ()


def test_retrieval_explanation_carries_scores_and_rank_details():
    detail = RetrievalRankDetail(query="等待期", method="bm25", rank=1, score=3.2)
    explanation = RetrievalExplanation(
        source_type="user_policy",
        source_name="policy.pdf",
        page_number=4,
        section_title="等待期",
        final_score=0.032,
        vector_score=0.81,
        bm25_score=3.2,
        matched_terms=("等待期",),
        rank_details=(detail,),
    )

    assert explanation.match_strength == "high"
    assert explanation.rank_details[0].method == "bm25"


def test_answer_payload_can_carry_guard_and_retrieval_explanations():
    guard = AnswerGuardResult(status=GuardStatus.WARN, warnings=("引用较少",))
    explanation = RetrievalExplanation(
        source_type="user_policy",
        source_name="policy.pdf",
        page_number=1,
        section_title="保险责任",
        final_score=0.02,
    )

    payload = AnswerPayload(
        answer="这份保单提到了保险责任。",
        guard_result=guard,
        retrieval_explanations=(explanation,),
    )

    assert payload.guard_result.status == GuardStatus.WARN
    assert payload.retrieval_explanations[0].section_title == "保险责任"
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```powershell
pytest tests/test_config.py tests/test_models.py -q -p no:cacheprovider
```

Expected: FAIL because `QueryRewriteResult`, retrieval explanation models, guard models, and new config fields do not exist.

- [ ] **Step 4: Add dependency**

Modify `requirements.txt`:

```text
rank-bm25>=0.2.2
```

Keep existing dependencies unchanged.

- [ ] **Step 5: Implement config fields**

Update `AppConfig` in `src/insurance_rag/config.py`:

```python
@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 900
    chunk_overlap: int = 150
    policy_top_k: int = 6
    builtin_top_k: int = 3
    min_page_text_chars: int = 80
    max_garbled_ratio: float = 0.25
    ocr_enabled: bool = True
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    query_rewrite_llm: bool = False
    answer_guard_llm: bool = False
    eval_report_dir: str = "eval_reports"
```

Add these fields to `from_env()`:

```python
retrieval_mode=os.getenv("INSURANCE_RAG_RETRIEVAL_MODE", "hybrid"),
rrf_k=int(os.getenv("INSURANCE_RAG_RRF_K", "60")),
query_rewrite_llm=_env_bool("INSURANCE_RAG_QUERY_REWRITE_LLM", False),
answer_guard_llm=_env_bool("INSURANCE_RAG_ANSWER_GUARD_LLM", False),
eval_report_dir=os.getenv("INSURANCE_RAG_EVAL_REPORT_DIR", "eval_reports"),
```

- [ ] **Step 6: Implement shared dataclasses**

Add to `src/insurance_rag/models.py`:

```python
from enum import StrEnum


class GuardStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    expanded_queries: tuple[str, ...]
    detected_intents: tuple[str, ...] = ()
    used_llm: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalRankDetail:
    query: str
    method: str
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalExplanation:
    source_type: str
    source_name: str
    page_number: int | None
    section_title: str
    final_score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    matched_terms: tuple[str, ...] = ()
    rank_details: tuple[RetrievalRankDetail, ...] = ()

    @property
    def match_strength(self) -> str:
        if self.final_score >= 0.03:
            return "high"
        if self.final_score >= 0.015:
            return "medium"
        return "low"


@dataclass(frozen=True)
class AnswerGuardResult:
    status: GuardStatus
    warnings: tuple[str, ...] = ()
    block_reason: str | None = None
```

Extend `AnswerPayload`:

```python
@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    policy_citations: tuple[Citation, ...] = ()
    builtin_citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()
    retrieval_explanations: tuple[RetrievalExplanation, ...] = ()
    guard_result: AnswerGuardResult | None = None
```

- [ ] **Step 7: Run focused tests**

Run:

```powershell
pytest tests/test_config.py tests/test_models.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add requirements.txt src\insurance_rag\config.py src\insurance_rag\models.py tests\test_config.py tests\test_models.py
git commit -m "feat: add RAG quality config and models"
```

---

## Task 2: Rule-Based Query Rewriter

**Files:**
- Create: `src/insurance_rag/query_rewriter.py`
- Create: `tests/test_query_rewriter.py`

- [ ] **Step 1: Write failing query rewriter tests**

Create `tests/test_query_rewriter.py`:

```python
from insurance_rag.query_rewriter import rewrite_query


def test_rewrites_claim_question_to_coverage_and_exclusion_terms():
    result = rewrite_query("这个病赔不赔？")

    assert result.original_query == "这个病赔不赔？"
    assert result.expanded_queries[0] == "这个病赔不赔？"
    assert "保险责任" in result.expanded_queries
    assert "责任免除" in result.expanded_queries
    assert "赔付条件" in result.expanded_queries
    assert "除外责任" in result.expanded_queries
    assert "claim_condition" in result.detected_intents


def test_rewrites_waiting_period_question():
    result = rewrite_query("这份保单等多久才生效？")

    assert "等待期" in result.expanded_queries
    assert "生效日" in result.expanded_queries
    assert "保险期间" in result.expanded_queries
    assert "waiting_period" in result.detected_intents


def test_rewrites_exclusion_question():
    result = rewrite_query("哪些情况不赔？")

    assert "责任免除" in result.expanded_queries
    assert "除外责任" in result.expanded_queries
    assert "免责条款" in result.expanded_queries
    assert "exclusion" in result.detected_intents


def test_rewrites_waiver_question():
    result = rewrite_query("有没有豁免保险费？")

    assert "豁免保险费" in result.expanded_queries
    assert "投保人豁免" in result.expanded_queries
    assert "被保险人豁免" in result.expanded_queries
    assert "waiver" in result.detected_intents


def test_plain_question_keeps_original_without_duplicates():
    result = rewrite_query("保险金额在哪里？")

    assert result.expanded_queries.count("保险金额在哪里？") == 1
    assert len(result.expanded_queries) == len(set(result.expanded_queries))
    assert result.used_llm is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_query_rewriter.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for `insurance_rag.query_rewriter`.

- [ ] **Step 3: Implement query rewriter**

Create `src/insurance_rag/query_rewriter.py`:

```python
from collections.abc import Iterable

from insurance_rag.models import QueryRewriteResult


INTENT_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "claim_condition",
        ("赔不赔", "能不能赔", "会不会赔", "是否赔", "理赔吗"),
        ("保险责任", "责任免除", "赔付条件", "除外责任"),
    ),
    (
        "exclusion",
        ("不赔", "哪些情况不赔", "什么不赔", "免责", "除外"),
        ("责任免除", "除外责任", "免责条款"),
    ),
    (
        "waiting_period",
        ("等待期", "等多久", "多久生效", "什么时候生效"),
        ("等待期", "生效日", "保险期间"),
    ),
    (
        "coverage",
        ("保什么", "保障什么", "保障哪些", "保险责任"),
        ("保险责任", "保障范围", "保险金额"),
    ),
    (
        "waiver",
        ("豁免", "豁免保险费", "免交保费"),
        ("豁免保险费", "投保人豁免", "被保险人豁免"),
    ),
    (
        "definition",
        ("什么是", "是什么意思", "定义", "如何理解"),
        ("释义", "定义", "术语解释"),
    ),
)


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return tuple(deduped)


def rewrite_query(question: str, *, use_llm: bool = False) -> QueryRewriteResult:
    normalized_question = question.strip()
    if not normalized_question:
        return QueryRewriteResult(
            original_query=question,
            expanded_queries=(),
            warnings=("问题为空，无法生成检索查询。",),
        )

    expanded: list[str] = [normalized_question]
    intents: list[str] = []
    for intent, triggers, additions in INTENT_RULES:
        if any(trigger in normalized_question for trigger in triggers):
            intents.append(intent)
            expanded.extend(additions)

    if use_llm:
        return QueryRewriteResult(
            original_query=question,
            expanded_queries=_dedupe(expanded),
            detected_intents=tuple(intents),
            used_llm=False,
            warnings=("LLM 查询改写尚未启用，已使用规则改写。",),
        )

    return QueryRewriteResult(
        original_query=question,
        expanded_queries=_dedupe(expanded),
        detected_intents=tuple(intents),
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
pytest tests/test_query_rewriter.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src\insurance_rag\query_rewriter.py tests\test_query_rewriter.py
git commit -m "feat: add rule-based query rewriting"
```

---

## Task 3: Hybrid Retriever with BM25 and RRF

**Files:**
- Create: `src/insurance_rag/hybrid_retriever.py`
- Modify: `src/insurance_rag/retriever.py`
- Create: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Write failing hybrid retriever tests**

Create `tests/test_hybrid_retriever.py`:

```python
from insurance_rag.hybrid_retriever import HybridRetriever, tokenize_for_bm25
from insurance_rag.models import DocumentChunk, QueryRewriteResult
from insurance_rag.retriever import InMemoryVectorIndex


class FakeEmbedder:
    def __init__(self, embeddings_by_text):
        self.embeddings_by_text = embeddings_by_text
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(texts)
        return [self.embeddings_by_text[text] for text in texts]


def make_chunk(chunk_id, text, section_title):
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        page_number=1,
        section_title=section_title,
        source_type="user_policy",
        source_name="policy.pdf",
        extraction_method="text",
    )


def test_tokenize_for_bm25_keeps_insurance_terms_and_cjk_bigrams():
    tokens = tokenize_for_bm25("等待期为90日，责任免除包括既往症。")

    assert "等待期" in tokens
    assert "责任免除" in tokens
    assert "既往症" in tokens
    assert "90" in tokens


def test_hybrid_search_uses_bm25_to_recover_exact_term():
    chunks = (
        make_chunk("c1", "本合同的保险责任包括重大疾病保险金。", "保险责任"),
        make_chunk("c2", "等待期为九十日。", "等待期"),
        make_chunk("c3", "责任免除包括投保前已患疾病。", "责任免除"),
    )
    vector_index = InMemoryVectorIndex.from_embeddings(
        chunks,
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]],
    )
    embedder = FakeEmbedder({"责任免除": [0.9, 0.1]})
    retriever = HybridRetriever(
        chunks=chunks,
        vector_index=vector_index,
        embedder=embedder,
        rrf_k=60,
        retrieval_mode="hybrid",
    )
    rewrite = QueryRewriteResult(
        original_query="哪些情况不赔？",
        expanded_queries=("责任免除",),
        detected_intents=("exclusion",),
    )

    results = retriever.search(rewrite, top_k=2)

    assert results[0].chunk.chunk_id == "c3"
    assert results[0].bm25_score is not None
    assert "责任免除" in results[0].matched_terms
    assert results[0].to_explanation().section_title == "责任免除"


def test_hybrid_search_deduplicates_chunks_across_queries():
    chunks = (make_chunk("c1", "等待期为九十日。", "等待期"),)
    vector_index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder({"等待期": [1.0, 0.0], "生效日": [1.0, 0.0]})
    retriever = HybridRetriever(chunks, vector_index, embedder, rrf_k=60)
    rewrite = QueryRewriteResult(
        original_query="等多久？",
        expanded_queries=("等待期", "生效日"),
    )

    results = retriever.search(rewrite, top_k=5)

    assert [result.chunk.chunk_id for result in results] == ["c1"]
    assert len(results[0].rank_details) >= 2


def test_vector_mode_skips_bm25_scores():
    chunks = (make_chunk("c1", "等待期为九十日。", "等待期"),)
    vector_index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder({"等待期": [1.0, 0.0]})
    retriever = HybridRetriever(chunks, vector_index, embedder, retrieval_mode="vector")
    rewrite = QueryRewriteResult(original_query="等待期", expanded_queries=("等待期",))

    results = retriever.search(rewrite, top_k=1)

    assert results[0].vector_score == 1.0
    assert results[0].bm25_score is None


def test_top_k_less_than_one_returns_empty():
    chunks = (make_chunk("c1", "等待期为九十日。", "等待期"),)
    vector_index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0]])
    embedder = FakeEmbedder({"等待期": [1.0, 0.0]})
    retriever = HybridRetriever(chunks, vector_index, embedder)
    rewrite = QueryRewriteResult(original_query="等待期", expanded_queries=("等待期",))

    assert retriever.search(rewrite, top_k=0) == []
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_hybrid_retriever.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for `insurance_rag.hybrid_retriever`.

- [ ] **Step 3: Implement hybrid retriever**

Create `src/insurance_rag/hybrid_retriever.py` with these public objects:

```python
from dataclasses import dataclass, field
import re

import numpy as np
from rank_bm25 import BM25Okapi

from insurance_rag.models import (
    DocumentChunk,
    QueryRewriteResult,
    RetrievalExplanation,
    RetrievalRankDetail,
)
from insurance_rag.retriever import InMemoryVectorIndex, OpenAIEmbedder


INSURANCE_TERMS = (
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


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for term in INSURANCE_TERMS:
        if term in text:
            tokens.append(term)
    tokens.extend(re.findall(r"[A-Za-z0-9]+", text))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(cjk_chars)
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:]))
    return tokens


@dataclass
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
```

Add `HybridRetriever`:

```python
@dataclass
class _Accumulator:
    chunk: DocumentChunk
    final_score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    matched_terms: set[str] = field(default_factory=set)
    rank_details: list[RetrievalRankDetail] = field(default_factory=list)


class HybridRetriever:
    def __init__(
        self,
        chunks: tuple[DocumentChunk, ...],
        vector_index: InMemoryVectorIndex,
        embedder: OpenAIEmbedder,
        *,
        rrf_k: int = 60,
        retrieval_mode: str = "hybrid",
    ) -> None:
        self.chunks = chunks
        self.vector_index = vector_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.retrieval_mode = retrieval_mode
        self._tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized_chunks) if chunks else None

    def search(self, rewrite: QueryRewriteResult, top_k: int) -> list[HybridSearchResult]:
        if top_k <= 0 or not rewrite.expanded_queries:
            return []

        accumulators: dict[str, _Accumulator] = {}
        query_embeddings = self.embedder.embed_texts(list(rewrite.expanded_queries))
        for query, query_embedding in zip(rewrite.expanded_queries, query_embeddings):
            self._add_vector_results(query, query_embedding, top_k, accumulators)
            if self.retrieval_mode == "hybrid":
                self._add_bm25_results(query, top_k, accumulators)

        ranked = sorted(
            accumulators.values(),
            key=lambda item: item.final_score,
            reverse=True,
        )
        return [
            HybridSearchResult(
                chunk=item.chunk,
                final_score=item.final_score,
                vector_score=item.vector_score,
                bm25_score=item.bm25_score,
                matched_terms=tuple(sorted(item.matched_terms)),
                rank_details=tuple(item.rank_details),
            )
            for item in ranked[:top_k]
        ]

    def _get_accumulator(
        self,
        chunk: DocumentChunk,
        accumulators: dict[str, _Accumulator],
    ) -> _Accumulator:
        if chunk.chunk_id not in accumulators:
            accumulators[chunk.chunk_id] = _Accumulator(chunk=chunk)
        return accumulators[chunk.chunk_id]

    def _add_vector_results(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        accumulators: dict[str, _Accumulator],
    ) -> None:
        for rank, result in enumerate(self.vector_index.search(query_embedding, top_k), start=1):
            item = self._get_accumulator(result.chunk, accumulators)
            item.final_score += 1.0 / (self.rrf_k + rank)
            item.vector_score = max(item.vector_score or result.score, result.score)
            item.rank_details.append(
                RetrievalRankDetail(query=query, method="vector", rank=rank, score=result.score)
            )

    def _add_bm25_results(
        self,
        query: str,
        top_k: int,
        accumulators: dict[str, _Accumulator],
    ) -> None:
        if self._bm25 is None:
            return
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return
        scores = self._bm25.get_scores(query_tokens)
        ranked_indexes = np.argsort(scores)[::-1][:top_k]
        query_terms = set(query_tokens)
        for rank, chunk_index in enumerate(ranked_indexes, start=1):
            score = float(scores[chunk_index])
            if score <= 0:
                continue
            chunk = self.chunks[int(chunk_index)]
            item = self._get_accumulator(chunk, accumulators)
            item.final_score += 1.0 / (self.rrf_k + rank)
            item.bm25_score = max(item.bm25_score or score, score)
            item.matched_terms.update(query_terms.intersection(self._tokenized_chunks[chunk_index]))
            item.rank_details.append(
                RetrievalRankDetail(query=query, method="bm25", rank=rank, score=score)
            )
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
pytest tests/test_hybrid_retriever.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Run retriever regression tests**

Run:

```powershell
pytest tests/test_retriever.py tests/test_hybrid_retriever.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\insurance_rag\hybrid_retriever.py tests\test_hybrid_retriever.py
git commit -m "feat: add hybrid BM25 vector retriever"
```

---

## Task 4: Answer Guard

**Files:**
- Create: `src/insurance_rag/answer_guard.py`
- Create: `tests/test_answer_guard.py`

- [ ] **Step 1: Write failing guard tests**

Create `tests/test_answer_guard.py`:

```python
from insurance_rag.answer_guard import BLOCKED_ANSWER, check_answer
from insurance_rag.models import Citation, GuardStatus, RetrievalExplanation


def citation(source_type="user_policy"):
    return Citation(
        source_type=source_type,
        source_name="policy.pdf" if source_type == "user_policy" else "builtin.pdf",
        page_number=2,
        section_title="保险责任",
        excerpt="本合同保险责任包括重大疾病保险金。",
    )


def explanation(final_score=0.04):
    return RetrievalExplanation(
        source_type="user_policy",
        source_name="policy.pdf",
        page_number=2,
        section_title="保险责任",
        final_score=final_score,
    )


def test_blocks_specific_policy_answer_without_user_policy_citation():
    result = check_answer(
        question="这份保单保什么？",
        answer="这份保单写明保障重大疾病保险金。",
        policy_citations=(),
        builtin_citations=(),
        retrieval_explanations=(),
    )

    assert result.status == GuardStatus.BLOCK
    assert result.block_reason


def test_blocks_final_claim_decision():
    result = check_answer(
        question="这个情况赔不赔？",
        answer="根据条款，这种情况一定赔。",
        policy_citations=(citation(),),
        builtin_citations=(),
        retrieval_explanations=(explanation(),),
    )

    assert result.status == GuardStatus.BLOCK
    assert "最终理赔判断" in result.block_reason


def test_warns_when_builtin_context_may_be_treated_as_policy():
    result = check_answer(
        question="什么是等待期？",
        answer="你的保单写明等待期是保险合同生效后的一段时间。",
        policy_citations=(citation(),),
        builtin_citations=(citation("built_in_dataset"),),
        retrieval_explanations=(explanation(),),
    )

    assert result.status in {GuardStatus.WARN, GuardStatus.BLOCK}
    assert result.warnings or result.block_reason


def test_warns_for_low_score_evidence():
    result = check_answer(
        question="等待期是多少？",
        answer="保单条款显示等待期为九十日。",
        policy_citations=(citation(),),
        builtin_citations=(),
        retrieval_explanations=(explanation(final_score=0.005),),
    )

    assert result.status == GuardStatus.WARN
    assert any("检索分数较低" in warning for warning in result.warnings)


def test_passes_grounded_explanation():
    result = check_answer(
        question="等待期是多少？",
        answer="根据用户保单引用，等待期为九十日。这里只是解释条款含义，不构成最终理赔结论。",
        policy_citations=(citation(),),
        builtin_citations=(),
        retrieval_explanations=(explanation(),),
    )

    assert result.status == GuardStatus.PASS
    assert result.warnings == ()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_answer_guard.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for `insurance_rag.answer_guard`.

- [ ] **Step 3: Implement answer guard**

Create `src/insurance_rag/answer_guard.py`:

```python
from insurance_rag.models import (
    AnswerGuardResult,
    Citation,
    GuardStatus,
    RetrievalExplanation,
)


BLOCKED_ANSWER = (
    "这份回答可能超出了保单解释助手的边界，因此我不能直接给出该结论。"
    "请以保单原文、保险公司解释和专业人士意见为准。"
)

FINAL_CLAIM_PATTERNS = (
    "一定赔",
    "肯定赔",
    "必须赔",
    "一定不赔",
    "肯定不赔",
    "不会赔",
    "保险公司必须",
)

POLICY_FACT_PATTERNS = ("这份保单", "你的保单", "保单写明", "条款显示", "合同约定")


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def check_answer(
    *,
    question: str,
    answer: str,
    policy_citations: tuple[Citation, ...],
    builtin_citations: tuple[Citation, ...],
    retrieval_explanations: tuple[RetrievalExplanation, ...],
) -> AnswerGuardResult:
    warnings: list[str] = []

    if not policy_citations and _contains_any(answer, POLICY_FACT_PATTERNS):
        return AnswerGuardResult(
            status=GuardStatus.BLOCK,
            block_reason="回答包含具体保单事实，但没有用户保单引用。",
        )

    if _contains_any(answer, FINAL_CLAIM_PATTERNS):
        return AnswerGuardResult(
            status=GuardStatus.BLOCK,
            block_reason="回答包含最终理赔判断。",
        )

    if builtin_citations and ("你的保单" in answer or "这份保单写明" in answer):
        warnings.append("回答可能把内置资料库背景解释表述成用户保单内容。")

    if len(policy_citations) == 1:
        warnings.append("用户保单引用较少，请结合原文继续核对。")

    if retrieval_explanations:
        top_score = max(item.final_score for item in retrieval_explanations)
        if top_score < 0.01:
            warnings.append("检索分数较低，回答依据可能较弱。")

    if any(citation.quality_notes for citation in policy_citations):
        warnings.append("部分引用来自质量提示页面，OCR 或文本抽取可能不完整。")

    if builtin_citations:
        warnings.append("内置资料库内容仅用于术语或背景解释，不能替代用户保单。")

    if warnings:
        return AnswerGuardResult(status=GuardStatus.WARN, warnings=tuple(warnings))
    return AnswerGuardResult(status=GuardStatus.PASS)
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
pytest tests/test_answer_guard.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src\insurance_rag\answer_guard.py tests\test_answer_guard.py
git commit -m "feat: add answer guard checks"
```

---

## Task 5: Integrate Query Rewriting, Hybrid Retrieval, and Guard into RAG Chain

**Files:**
- Modify: `src/insurance_rag/rag_chain.py`
- Modify: `tests/test_rag_chain.py`

- [ ] **Step 1: Update RAG chain tests for hybrid retriever flow**

In `tests/test_rag_chain.py`, replace the fake index with a fake hybrid retriever:

```python
class FakeHybridRetriever:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.calls = []

    def search(self, rewrite, top_k):
        self.calls.append((rewrite, top_k))
        if self.error:
            raise self.error
        from insurance_rag.hybrid_retriever import HybridSearchResult
        return [
            HybridSearchResult(
                chunk=chunk,
                final_score=0.04,
                vector_score=0.9,
                bm25_score=2.0,
                matched_terms=("等待期",),
            )
            for chunk in self.chunks
        ]
```

Add tests:

```python
def test_answer_calls_query_rewriter_and_hybrid_retriever(monkeypatch):
    policy_chunk = make_chunk()
    policy_retriever = FakeHybridRetriever([policy_chunk])
    chain, client = make_chain(monkeypatch, policy_retriever=policy_retriever)

    payload = chain.answer("这个赔不赔？")

    rewrite = policy_retriever.calls[0][0]
    assert "保险责任" in rewrite.expanded_queries
    assert payload.retrieval_explanations[0].matched_terms == ("等待期",)
    assert len(client.calls) == 1


def test_answer_guard_block_replaces_model_answer(monkeypatch):
    policy_chunk = make_chunk()
    chain, client = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([policy_chunk]),
        chat_client=FakeChatClient("这种情况一定赔。"),
    )

    payload = chain.answer("这个赔不赔？")

    assert "不能直接给出该结论" in payload.answer
    assert payload.guard_result.status == "block"


def test_answer_guard_warn_preserves_model_answer(monkeypatch):
    policy_chunk = make_chunk()
    chain, client = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([policy_chunk]),
        chat_client=FakeChatClient("根据用户保单引用，等待期为九十日。"),
    )

    payload = chain.answer("等待期是多少？")

    assert payload.answer == "根据用户保单引用，等待期为九十日。"
    assert payload.guard_result.status in {"pass", "warn"}
```

Update `make_chain()` to pass `policy_retriever` and `builtin_retriever` into `RagChain`. Keep existing tests for refusal and built-in degradation, but make them use `FakeHybridRetriever`.

- [ ] **Step 2: Run RAG tests to verify failure**

Run:

```powershell
pytest tests/test_rag_chain.py -q -p no:cacheprovider
```

Expected: FAIL because `RagChain` does not accept hybrid retrievers and does not call the query rewriter or guard.

- [ ] **Step 3: Update `RagChain.__init__`**

Modify `RagChain` constructor in `src/insurance_rag/rag_chain.py`:

```python
from insurance_rag.answer_guard import BLOCKED_ANSWER, check_answer
from insurance_rag.hybrid_retriever import HybridRetriever
from insurance_rag.models import GuardStatus
from insurance_rag.query_rewriter import rewrite_query


class RagChain:
    def __init__(
        self,
        config: AppConfig,
        policy_retriever: HybridRetriever,
        builtin_retriever: HybridRetriever | None = None,
    ) -> None:
        if not config.openai_api_key:
            raise ValueError("缺少 OPENAI_API_KEY。")
        self.config = config
        self.policy_retriever = policy_retriever
        self.builtin_retriever = builtin_retriever
        self.client = OpenAI(api_key=config.openai_api_key)
```

- [ ] **Step 4: Update `answer()` orchestration**

Modify `answer()`:

```python
def answer(self, question: str) -> AnswerPayload:
    warnings: list[str] = []
    rewrite = rewrite_query(question, use_llm=self.config.query_rewrite_llm)
    warnings.extend(rewrite.warnings)

    try:
        policy_results = self.policy_retriever.search(
            rewrite,
            top_k=self.config.policy_top_k,
        )
    except Exception as error:
        return AnswerPayload(
            answer=REFUSAL_ANSWER,
            warnings=(f"保单检索失败：{error}",),
        )

    policy_chunks = [result.chunk for result in policy_results]
    if not policy_chunks:
        return AnswerPayload(answer=REFUSAL_ANSWER, warnings=tuple(warnings))

    builtin_results = []
    if self.builtin_retriever and should_use_builtin_context(question, len(policy_chunks)):
        try:
            builtin_results = self.builtin_retriever.search(
                rewrite,
                top_k=self.config.builtin_top_k,
            )
        except Exception as error:
            warnings.append(f"内置资料库检索失败，已仅使用用户保单资料回答：{error}")

    builtin_chunks = [result.chunk for result in builtin_results]
    retrieval_explanations = tuple(
        result.to_explanation() for result in [*policy_results, *builtin_results]
    )

    messages = build_messages(question, policy_chunks, builtin_chunks)
    response = self.client.chat.completions.create(
        model=self.config.chat_model,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content or REFUSAL_ANSWER
    policy_citations = tuple(build_citation(chunk) for chunk in policy_chunks)
    builtin_citations = tuple(build_citation(chunk) for chunk in builtin_chunks)
    guard_result = check_answer(
        question=question,
        answer=answer,
        policy_citations=policy_citations,
        builtin_citations=builtin_citations,
        retrieval_explanations=retrieval_explanations,
    )
    warnings.extend(guard_result.warnings)
    if guard_result.status == GuardStatus.BLOCK:
        answer = BLOCKED_ANSWER
        if guard_result.block_reason:
            warnings.append(guard_result.block_reason)

    return AnswerPayload(
        answer=answer,
        policy_citations=policy_citations,
        builtin_citations=builtin_citations,
        warnings=tuple(warnings),
        retrieval_explanations=retrieval_explanations,
        guard_result=guard_result,
    )
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
pytest tests/test_rag_chain.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Run related unit tests**

Run:

```powershell
pytest tests/test_query_rewriter.py tests/test_hybrid_retriever.py tests/test_answer_guard.py tests/test_rag_chain.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src\insurance_rag\rag_chain.py tests\test_rag_chain.py
git commit -m "feat: integrate hybrid retrieval into RAG chain"
```

---

## Task 6: Streamlit UI and Session State Integration

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update imports**

Modify imports in `app.py`:

```python
from insurance_rag.hybrid_retriever import HybridRetriever
from insurance_rag.models import AnswerPayload, RetrievalExplanation
from insurance_rag.retriever import OpenAIEmbedder, build_index
```

Remove unused `InMemoryVectorIndex` import if it is no longer used.

- [ ] **Step 2: Update session state keys**

In `init_state()` add:

```python
st.session_state.setdefault("policy_retriever", None)
st.session_state.setdefault("builtin_retriever", None)
```

In `clear_policy_state()` clear:

```python
st.session_state.policy_retriever = None
st.session_state.builtin_retriever = None
```

Keep `policy_index` and `builtin_index` only if still needed during migration. Prefer storing retrievers as the objects passed into `RagChain`.

- [ ] **Step 3: Build hybrid retriever for uploaded policy**

After `policy_index = build_index(chunks, embedder)` in `process_upload()` add:

```python
policy_retriever = HybridRetriever(
    chunks=chunks,
    vector_index=policy_index,
    embedder=embedder,
    rrf_k=config.rrf_k,
    retrieval_mode=config.retrieval_mode,
)
```

Then store:

```python
st.session_state.policy_retriever = policy_retriever
```

Keep storing `policy_chunks` for UI statistics and built-in trigger logic.

- [ ] **Step 4: Build hybrid retriever for built-in background**

Change `build_builtin_background_index()` to return `HybridRetriever | None`. After building the vector index:

```python
vector_index = build_index(tuple(chunks), embedder)
return HybridRetriever(
    chunks=tuple(chunks),
    vector_index=vector_index,
    embedder=embedder,
    rrf_k=config.rrf_k,
    retrieval_mode=config.retrieval_mode,
)
```

Store the result in `st.session_state.builtin_retriever` instead of `builtin_index`.

- [ ] **Step 5: Update question flow**

Replace the `RagChain` creation with:

```python
chain = RagChain(
    config=config,
    policy_retriever=st.session_state.policy_retriever,
    builtin_retriever=st.session_state.builtin_retriever,
)
```

Change the readiness check from policy index to policy retriever:

```python
if not st.session_state.policy_retriever:
    st.info("请先上传并解析一份保险 PDF。用户上传内容只在当前会话中使用。")
    st.warning("使用 OpenAI API 时，问题和被检索到的保单片段会发送给 OpenAI 用于生成回答。")
    return
```

- [ ] **Step 6: Render retrieval details**

Add this helper in `app.py`:

```python
def render_retrieval_details(explanations: tuple[RetrievalExplanation, ...]) -> None:
    if not explanations:
        return
    with st.expander("检索依据详情", expanded=False):
        for index, item in enumerate(explanations, start=1):
            page = f"第 {item.page_number} 页" if item.page_number else "页码未知"
            st.markdown(
                f"**{index}. {item.source_name}｜{page}｜{item.section_title}｜匹配度：{item.match_strength}**"
            )
            score_parts = [f"融合分数：{item.final_score:.4f}"]
            if item.vector_score is not None:
                score_parts.append(f"向量分数：{item.vector_score:.4f}")
            if item.bm25_score is not None:
                score_parts.append(f"BM25 分数：{item.bm25_score:.4f}")
            st.caption("；".join(score_parts))
            if item.matched_terms:
                st.caption("命中关键词：" + "、".join(item.matched_terms))
            for detail in item.rank_details:
                st.caption(
                    f"{detail.method}｜query={detail.query}｜rank={detail.rank}｜score={detail.score:.4f}"
                )
```

Call it inside `render_citations(payload)` after citation sections:

```python
render_retrieval_details(payload.retrieval_explanations)
```

- [ ] **Step 7: Run import and existing tests**

Run:

```powershell
python -c "import app"
pytest tests -q -p no:cacheprovider
```

Expected: import succeeds and tests pass.

- [ ] **Step 8: Commit**

Run:

```powershell
git add app.py
git commit -m "feat: show hybrid retrieval details in UI"
```

---

## Task 7: Offline Synthetic and Optional Local Evaluation

**Files:**
- Create: `src/insurance_rag/evaluation.py`
- Create: `scripts/evaluate_rag.py`
- Create: `evals/synthetic_cases.json`
- Create: `tests/test_evaluation.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write synthetic cases**

Create `evals/synthetic_cases.json`:

```json
[
  {
    "id": "waiting-period",
    "question": "这份保单等多久才生效？",
    "expected_terms": ["等待期"],
    "expected_section": "等待期",
    "chunks": [
      {
        "chunk_id": "synthetic-waiting",
        "text": "等待期为九十日。等待期内因疾病导致保险事故的，本公司不承担保险责任。",
        "page_number": 3,
        "section_title": "等待期"
      },
      {
        "chunk_id": "synthetic-coverage",
        "text": "本合同的保险责任包括重大疾病保险金和身故保险金。",
        "page_number": 5,
        "section_title": "保险责任"
      },
      {
        "chunk_id": "synthetic-exclusion",
        "text": "责任免除包括投保前已患疾病、故意自伤以及酒后驾驶。",
        "page_number": 8,
        "section_title": "责任免除"
      }
    ]
  },
  {
    "id": "exclusion",
    "question": "哪些情况不赔？",
    "expected_terms": ["责任免除", "除外责任"],
    "expected_section": "责任免除",
    "chunks": [
      {
        "chunk_id": "synthetic-exclusion",
        "text": "责任免除包括投保前已患疾病、故意自伤以及酒后驾驶。",
        "page_number": 8,
        "section_title": "责任免除"
      },
      {
        "chunk_id": "synthetic-waiver",
        "text": "投保人豁免保险费责任适用于合同约定的特定情形。",
        "page_number": 11,
        "section_title": "豁免保险费"
      },
      {
        "chunk_id": "synthetic-term",
        "text": "保险期间为终身，自合同生效日零时开始。",
        "page_number": 2,
        "section_title": "保险期间"
      }
    ]
  }
]
```

- [ ] **Step 2: Write failing evaluation tests**

Create `tests/test_evaluation.py`:

```python
from pathlib import Path

from insurance_rag.evaluation import (
    DeterministicEvalEmbedder,
    evaluate_synthetic_cases,
    render_markdown_report,
)


def test_deterministic_eval_embedder_returns_stable_vectors():
    embedder = DeterministicEvalEmbedder()

    first = embedder.embed_texts(["等待期", "责任免除"])
    second = embedder.embed_texts(["等待期", "责任免除"])

    assert first == second
    assert len(first[0]) == 8


def test_synthetic_evaluation_reports_expected_section():
    report = evaluate_synthetic_cases(
        Path("evals/synthetic_cases.json"),
        top_k=2,
    )

    assert report.total_cases >= 2
    assert report.passed_cases >= 1
    assert report.results[0].retrieved_sections
    assert report.results[0].expected_section


def test_markdown_report_contains_scores_and_pass_fail():
    report = evaluate_synthetic_cases(
        Path("evals/synthetic_cases.json"),
        top_k=2,
    )

    markdown = render_markdown_report(report)

    assert "# InsuranceRAG Evaluation Report" in markdown
    assert "Fusion Score" in markdown
    assert "PASS" in markdown or "FAIL" in markdown
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```powershell
pytest tests/test_evaluation.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for `insurance_rag.evaluation`.

- [ ] **Step 4: Implement evaluation helpers**

Create `src/insurance_rag/evaluation.py`:

```python
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from insurance_rag.hybrid_retriever import HybridRetriever
from insurance_rag.models import DocumentChunk
from insurance_rag.query_rewriter import rewrite_query
from insurance_rag.retriever import InMemoryVectorIndex


class DeterministicEvalEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(8)]


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    question: str
    expected_section: str
    expected_terms: tuple[str, ...]
    retrieved_sections: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    top_fusion_score: float | None
    passed: bool


@dataclass(frozen=True)
class EvalReport:
    total_cases: int
    passed_cases: int
    results: tuple[EvalCaseResult, ...]


def _chunk_from_case(case_id: str, raw: dict) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=raw["chunk_id"],
        text=raw["text"],
        page_number=raw.get("page_number"),
        section_title=raw.get("section_title", "未识别条款标题"),
        source_type="synthetic_eval",
        source_name=f"{case_id}.json",
        extraction_method="synthetic",
    )


def evaluate_synthetic_cases(path: Path, *, top_k: int = 3) -> EvalReport:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    embedder = DeterministicEvalEmbedder()
    results: list[EvalCaseResult] = []

    for raw_case in raw_cases:
        chunks = tuple(_chunk_from_case(raw_case["id"], raw) for raw in raw_case["chunks"])
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_index = InMemoryVectorIndex.from_embeddings(chunks, embeddings)
        retriever = HybridRetriever(
            chunks=chunks,
            vector_index=vector_index,
            embedder=embedder,
            retrieval_mode="hybrid",
        )
        rewrite = rewrite_query(raw_case["question"])
        search_results = retriever.search(rewrite, top_k=top_k)
        sections = tuple(result.chunk.section_title for result in search_results)
        chunk_ids = tuple(result.chunk.chunk_id for result in search_results)
        expected_section = raw_case["expected_section"]
        expected_terms = tuple(raw_case.get("expected_terms", ()))
        passed = expected_section in sections or any(
            term in " ".join(sections) for term in expected_terms
        )
        results.append(
            EvalCaseResult(
                case_id=raw_case["id"],
                question=raw_case["question"],
                expected_section=expected_section,
                expected_terms=expected_terms,
                retrieved_sections=sections,
                retrieved_chunk_ids=chunk_ids,
                top_fusion_score=search_results[0].final_score if search_results else None,
                passed=passed,
            )
        )

    return EvalReport(
        total_cases=len(results),
        passed_cases=sum(1 for result in results if result.passed),
        results=tuple(results),
    )


def render_markdown_report(report: EvalReport) -> str:
    lines = [
        "# InsuranceRAG Evaluation Report",
        "",
        f"Passed: {report.passed_cases}/{report.total_cases}",
        "",
        "| Case | Status | Expected Section | Retrieved Sections | Fusion Score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        score = "" if result.top_fusion_score is None else f"{result.top_fusion_score:.4f}"
        lines.append(
            f"| {result.case_id} | {status} | {result.expected_section} | "
            f"{', '.join(result.retrieved_sections)} | {score} |"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 5: Implement evaluation CLI**

Create `scripts/evaluate_rag.py`:

```python
from argparse import ArgumentParser
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_rag.config import AppConfig
from insurance_rag.evaluation import evaluate_synthetic_cases, render_markdown_report


def main() -> int:
    parser = ArgumentParser(description="Run InsuranceRAG offline evaluation.")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic eval cases.")
    parser.add_argument("--cases", default="evals/synthetic_cases.json")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--local-documents", default=None)
    args = parser.parse_args()

    config = AppConfig.from_env()
    report_dir = Path(args.report_dir or config.eval_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.local_documents:
        local_path = Path(args.local_documents)
        if not local_path.exists():
            print(f"Local documents path not found, skipping: {local_path}")
        else:
            print("Local document evaluation is available as an optional path; synthetic evaluation remains the reproducible CI target.")

    if not args.synthetic:
        print("No evaluation selected. Use --synthetic.")
        return 2

    report = evaluate_synthetic_cases(Path(args.cases))
    markdown = render_markdown_report(report)
    output_path = report_dir / "synthetic_eval_report.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote report to {output_path}")
    return 0 if report.passed_cases == report.total_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Update `.gitignore`**

Add:

```text
eval_reports/
.rag_eval_cache/
```

Keep the existing `documents/` ignore rule.

- [ ] **Step 7: Run focused tests and CLI**

Run:

```powershell
pytest tests/test_evaluation.py -q -p no:cacheprovider
python scripts\evaluate_rag.py --synthetic
```

Expected: tests pass; CLI writes `eval_reports/synthetic_eval_report.md`. If one synthetic case fails because deterministic embeddings make vector ranking noisy, adjust only the synthetic text or expected top-k so BM25 exact terms can pass consistently.

- [ ] **Step 8: Confirm ignored report output**

Run:

```powershell
git status --short
git check-ignore -v eval_reports\synthetic_eval_report.md
```

Expected: `eval_reports/synthetic_eval_report.md` is ignored by `.gitignore`; `git status --short` does not show the generated report.

- [ ] **Step 9: Commit**

Run:

```powershell
git add .gitignore evals\synthetic_cases.json scripts\evaluate_rag.py src\insurance_rag\evaluation.py tests\test_evaluation.py
git commit -m "feat: add offline RAG evaluation"
```

---

## Task 8: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add retrieval quality section**

Add a section to `README.md`:

````markdown
## RAG 检索增强

系统默认使用 hybrid retrieval：

- OpenAI embeddings 做语义检索。
- `rank-bm25` 做关键词检索。
- 使用 RRF 融合排序。
- 查询会先做规则扩展，例如“赔不赔”会同时检索“保险责任”“责任免除”“赔付条件”“除外责任”。

可以通过环境变量切换：

```powershell
$env:INSURANCE_RAG_RETRIEVAL_MODE="vector"
$env:INSURANCE_RAG_RRF_K="60"
```
````

- [ ] **Step 2: Add answer guard section**

Add:

```markdown
## 回答自检

回答生成后会做程序化自检：

- 如果回答没有用户保单依据却陈述具体保单事实，会阻断。
- 如果回答包含“肯定赔”“一定不赔”等最终理赔判断，会阻断。
- 如果引用较少、检索分数较低、OCR 质量可能影响依据，会显示 warning。

本项目仍只做条款解释，不构成法律、医疗、财务、保险理赔或核保建议。
```

- [ ] **Step 3: Add offline evaluation section**

Add:

````markdown
## 离线评测

repo 内置合成评测集，可复现运行：

```powershell
python scripts\evaluate_rag.py --synthetic
```

评测报告默认写入 `eval_reports/`，该目录被 Git 忽略。

如果本地存在 `documents/`，可以传入本地资料目录做可选实验：

```powershell
python scripts\evaluate_rag.py --synthetic --local-documents documents
```

本地真实文档和评测报告不应提交到 GitHub。
````

- [ ] **Step 4: Run markdown and test sanity checks**

Run:

```powershell
git diff --check
pytest tests -q -p no:cacheprovider
```

Expected: no whitespace errors; tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add README.md
git commit -m "docs: document RAG quality enhancements"
```

---

## Task 9: Final Verification and Review

**Files:**
- No new files unless verification reveals a bug.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
pytest tests -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run evaluation CLI**

Run:

```powershell
python scripts\evaluate_rag.py --synthetic
```

Expected: exits `0`, prints markdown report, and writes `eval_reports/synthetic_eval_report.md`.

- [ ] **Step 3: Confirm sensitive outputs are ignored**

Run:

```powershell
git check-ignore -v documents eval_reports\synthetic_eval_report.md .rag_eval_cache
git ls-files documents eval_reports .rag_eval_cache
```

Expected: first command shows ignore rules; second command prints nothing.

- [ ] **Step 4: Inspect git diff summary**

Run:

```powershell
git status --short
git log --oneline -8
```

Expected: clean working tree after task commits; recent commits correspond to the planned tasks.

- [ ] **Step 5: Request code review**

Use `superpowers:requesting-code-review`. Ask the reviewer to focus on:

- Whether built-in dataset content can still only support terminology/background.
- Whether hybrid retrieval preserves citations and source separation.
- Whether answer guard blocks final claim judgments without excessive false positives.
- Whether evaluation reports avoid committing real policy content.
- Whether tests cover the major failure modes in the spec.

- [ ] **Step 6: Address review findings**

If review returns actionable findings, use `superpowers:receiving-code-review`, verify each finding technically, implement accepted fixes with focused tests, and commit each fix.

- [ ] **Step 7: Final verification before completion**

Run:

```powershell
pytest tests -q -p no:cacheprovider
python scripts\evaluate_rag.py --synthetic
git status --short --branch
```

Expected: tests pass, synthetic evaluation passes, and the branch is ready for merge or push.

---

## Spec Coverage Checklist

- Hybrid retrieval with BM25 and vector search: Task 3, Task 5, Task 6.
- Query rewriting: Task 2, Task 5.
- Answer self-checking: Task 4, Task 5.
- Retrieval explanation in UI and reports: Task 1, Task 3, Task 6, Task 7.
- Synthetic evaluation for CI-style reproducibility: Task 7.
- Optional local document evaluation path: Task 7.
- `.gitignore` protection for generated reports and caches: Task 7, Task 9.
- README updates: Task 8.
- Full verification and code review: Task 9.
