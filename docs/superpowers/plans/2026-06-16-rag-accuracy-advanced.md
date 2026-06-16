# RAG Accuracy Advanced Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the advanced RAG accuracy loop: structured clause metadata, rule reranking, fact-level citation verification, evidence verification UI, and hard negative evaluation.

**Architecture:** Keep the existing modular shape. Add focused modules for clause parsing, reranking, and citation verification; keep `rag_chain.py` as orchestration and `app.py` as rendering only. Extend the existing evaluation command with repo-contained synthetic hard negatives and optional local hard negative reports.

**Tech Stack:** Python dataclasses, pytest, Streamlit, PyMuPDF-backed PDF parsing already in the project, existing in-memory vector index, existing hybrid BM25/vector retriever.

---

## File Structure

- Create `src/insurance_rag/clause_parser.py`
  - Parse high-frequency clause numbers and headings.
  - Return `ClauseMetadata` with confidence and source.

- Create `src/insurance_rag/rule_reranker.py`
  - Rerank `HybridSearchResult` candidates after hybrid retrieval.
  - Attach rerank score and reasons.

- Create `src/insurance_rag/citation_verifier.py`
  - Extract high-risk policy facts from generated answers.
  - Verify each fact against user-policy citations.

- Modify `src/insurance_rag/models.py`
  - Add `ClauseMetadata`, `RerankExplanation`, `VerifiedFact`, `CitationVerificationResult`.
  - Extend `DocumentChunk`, `RetrievalExplanation`, `AnswerGuardResult`, and `AnswerPayload`.

- Modify `src/insurance_rag/config.py`
  - Add rerank, verifier, heading confidence, and hard negative settings.

- Modify `src/insurance_rag/chunker.py`
  - Call `clause_parser.py`.
  - Preserve fallback behavior.

- Modify `src/insurance_rag/hybrid_retriever.py`
  - Carry rerank score and reasons through `HybridSearchResult.to_explanation()`.

- Modify `src/insurance_rag/rag_chain.py`
  - Retrieve top-N candidates for reranking.
  - Invoke `rule_reranker.py`.
  - Propagate citation verification results from guard to payload.

- Modify `src/insurance_rag/answer_guard.py`
  - Delegate fact support checks to `citation_verifier.py`.
  - Preserve existing final-claim and source-confusion safeguards.

- Modify `src/insurance_rag/evaluation.py`
  - Add synthetic and local hard negative evaluation models and runners.

- Modify `scripts/evaluate_rag.py`
  - Add `--hard-negative`, `--hard-negative-cases`, and `--local-hard-negative`.

- Modify `app.py`
  - Render `证据核验结果`.
  - Add rerank details to existing retrieval details.

- Add `evals/hard_negative_cases.json`
  - Synthetic hard negative cases with no real policy text.

- Add tests:
  - `tests/test_clause_parser.py`
  - `tests/test_rule_reranker.py`
  - `tests/test_citation_verifier.py`
  - Extend existing tests for models, config, chunker, hybrid retriever, RAG chain, app rendering, and evaluation.

## Task 1: Extend Shared Models And Config

**Files:**
- Modify: `src/insurance_rag/models.py`
- Modify: `src/insurance_rag/config.py`
- Test: `tests/test_models.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing model tests**

Add these tests to `tests/test_models.py`:

```python
from insurance_rag.models import (
    CitationVerificationResult,
    ClauseMetadata,
    DocumentChunk,
    RerankExplanation,
    VerifiedFact,
)


def test_clause_metadata_defaults_are_low_confidence_unknown():
    metadata = ClauseMetadata()

    assert metadata.clause_id is None
    assert metadata.heading_text is None
    assert metadata.section_title == "未识别条款标题"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"


def test_document_chunk_carries_clause_metadata_fields():
    chunk = DocumentChunk(
        chunk_id="c1",
        text="第六条 等待期\n等待期为90天。",
        page_number=6,
        section_title="等待期",
        source_type="user_policy",
        source_name="policy.pdf",
        extraction_method="text",
        clause_id="第六条",
        heading_text="第六条 等待期",
        heading_confidence="high",
        heading_source="line_pattern",
    )

    assert chunk.clause_id == "第六条"
    assert chunk.heading_text == "第六条 等待期"
    assert chunk.heading_confidence == "high"
    assert chunk.heading_source == "line_pattern"


def test_verification_result_carries_facts():
    fact = VerifiedFact(
        fact_text="等待期是90天",
        fact_type="number",
        status="supported",
        severity="info",
        supporting_citation_ids=("policy.pdf:6:等待期",),
        reason="同一引用中找到等待期和90天。",
    )
    result = CitationVerificationResult(facts=(fact,))

    assert result.facts == (fact,)
    assert result.has_blocking_fact is False
    assert result.has_warnings is False


def test_rerank_explanation_defaults():
    explanation = RerankExplanation(score=1.5, reasons=("title_intent_match",))

    assert explanation.score == 1.5
    assert explanation.reasons == ("title_intent_match",)
```

- [ ] **Step 2: Write failing config tests**

Add these tests to `tests/test_config.py`:

```python
from insurance_rag.config import AppConfig


def test_rerank_and_verifier_config_defaults(monkeypatch):
    monkeypatch.delenv("INSURANCE_RAG_RERANK_ENABLED", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_RERANK_TOP_N", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_VERIFIER_ENABLED", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_VERIFIER_STRICTNESS", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT", raising=False)

    config = AppConfig.from_env()

    assert config.rerank_enabled is True
    assert config.rerank_top_n == 20
    assert config.verifier_enabled is True
    assert config.verifier_strictness == "balanced"
    assert config.heading_confidence_warn_threshold == 0.35
    assert config.hard_negative_local_limit == 20
```

- [ ] **Step 3: Run model and config tests to verify failure**

Run:

```powershell
pytest tests\test_models.py tests\test_config.py -q -p no:cacheprovider
```

Expected: FAIL because the new dataclasses and config fields do not exist.

- [ ] **Step 4: Implement models**

In `src/insurance_rag/models.py`, add:

```python
@dataclass(frozen=True)
class ClauseMetadata:
    clause_id: str | None = None
    heading_text: str | None = None
    section_title: str = "未识别条款标题"
    heading_confidence: str = "low"
    heading_source: str = "fallback"


@dataclass(frozen=True)
class RerankExplanation:
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedFact:
    fact_text: str
    fact_type: str
    status: str
    severity: str
    supporting_citation_ids: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class CitationVerificationResult:
    facts: tuple[VerifiedFact, ...] = ()
    warnings: tuple[str, ...] = ()
    block_reason: str | None = None

    @property
    def has_blocking_fact(self) -> bool:
        return any(fact.severity == "block" for fact in self.facts)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings) or any(fact.severity == "warn" for fact in self.facts)
```

Extend `DocumentChunk`:

```python
    clause_id: str | None = None
    heading_text: str | None = None
    heading_confidence: str = "low"
    heading_source: str = "fallback"
```

Extend `RetrievalExplanation`:

```python
    rerank_score: float | None = None
    rerank_reasons: tuple[str, ...] = ()
```

Extend `AnswerGuardResult`:

```python
    citation_verification: CitationVerificationResult | None = None
```

Extend `AnswerPayload`:

```python
    citation_verification: CitationVerificationResult | None = None
```

- [ ] **Step 5: Implement config fields**

In `src/insurance_rag/config.py`, add fields to `AppConfig`:

```python
    rerank_enabled: bool = True
    rerank_top_n: int = 20
    verifier_enabled: bool = True
    verifier_strictness: str = "balanced"
    heading_confidence_warn_threshold: float = 0.35
    hard_negative_local_limit: int = 20
```

In `from_env`, add:

```python
            rerank_enabled=_env_bool("INSURANCE_RAG_RERANK_ENABLED", True),
            rerank_top_n=int(os.getenv("INSURANCE_RAG_RERANK_TOP_N", "20")),
            verifier_enabled=_env_bool("INSURANCE_RAG_VERIFIER_ENABLED", True),
            verifier_strictness=os.getenv("INSURANCE_RAG_VERIFIER_STRICTNESS", "balanced"),
            heading_confidence_warn_threshold=float(
                os.getenv("INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD", "0.35")
            ),
            hard_negative_local_limit=int(
                os.getenv("INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT", "20")
            ),
```

- [ ] **Step 6: Run model and config tests**

Run:

```powershell
pytest tests\test_models.py tests\test_config.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src\insurance_rag\models.py src\insurance_rag\config.py tests\test_models.py tests\test_config.py
git commit -m "feat: add RAG quality data models"
```

## Task 2: Add Clause Parser

**Files:**
- Create: `src/insurance_rag/clause_parser.py`
- Test: `tests/test_clause_parser.py`

- [ ] **Step 1: Write failing clause parser tests**

Create `tests/test_clause_parser.py`:

```python
from insurance_rag.clause_parser import parse_clause_metadata


def test_parses_chinese_article_number_and_heading():
    metadata = parse_clause_metadata("第六条 等待期\n等待期为90天。")

    assert metadata.clause_id == "第六条"
    assert metadata.heading_text == "第六条 等待期"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "high"
    assert metadata.heading_source == "line_pattern"


def test_parses_spaced_arabic_article_number_and_heading():
    metadata = parse_clause_metadata("第 10 条 责任免除\n因酒后驾驶导致的事故不承担责任。")

    assert metadata.clause_id == "第10条"
    assert metadata.section_title == "责任免除"
    assert metadata.heading_confidence == "high"


def test_parses_decimal_clause_number_and_heading():
    metadata = parse_clause_metadata("2.3 保险责任\n本合同承担重大疾病保险责任。")

    assert metadata.clause_id == "2.3"
    assert metadata.section_title == "保险责任"
    assert metadata.heading_confidence == "high"


def test_parses_standalone_known_heading_as_medium_confidence():
    metadata = parse_clause_metadata("保险金额\n基本保险金额以保险单载明为准。")

    assert metadata.clause_id is None
    assert metadata.section_title == "保险金额"
    assert metadata.heading_confidence == "medium"
    assert metadata.heading_source == "known_title"


def test_directory_like_line_is_not_high_confidence():
    metadata = parse_clause_metadata("2.3 保险责任 ........ 5")

    assert metadata.section_title == "保险责任"
    assert metadata.heading_confidence != "high"


def test_fallback_uses_current_title_when_no_heading_found():
    metadata = parse_clause_metadata(
        "本合同自保险单载明的生效日零时起生效。",
        current_title="保险期间",
    )

    assert metadata.section_title == "保险期间"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"
```

- [ ] **Step 2: Run clause parser tests to verify failure**

Run:

```powershell
pytest tests\test_clause_parser.py -q -p no:cacheprovider
```

Expected: FAIL because `insurance_rag.clause_parser` does not exist.

- [ ] **Step 3: Implement `clause_parser.py`**

Create `src/insurance_rag/clause_parser.py`:

```python
import re

from insurance_rag.models import ClauseMetadata


UNKNOWN_SECTION_TITLE = "未识别条款标题"

KNOWN_SECTION_TITLES: tuple[str, ...] = (
    "等待期",
    "责任免除",
    "除外责任",
    "免责条款",
    "保险责任",
    "保险期间",
    "保险金额",
    "基本保险金额",
    "保险金给付",
    "给付条件",
    "豁免保险费",
    "犹豫期",
    "宽限期",
    "合同解除",
    "合同效力",
    "释义",
    "疾病定义",
    "重大疾病定义",
    "轻症疾病",
    "中症疾病",
    "身故保险金",
    "全残保险金",
)

_CHINESE_ARTICLE_RE = re.compile(
    r"^(?P<clause>第\s*[零〇一二三四五六七八九十百千万两\d]+\s*条)\s*(?P<title>.+)$"
)
_DECIMAL_CLAUSE_RE = re.compile(r"^(?P<clause>\d+(?:\.\d+)+)\s+(?P<title>.+)$")
_PAREN_CLAUSE_RE = re.compile(r"^(?P<clause>[（(][一二三四五六七八九十\d]+[）)])\s*(?P<title>.+)$")
_SIMPLE_LIST_RE = re.compile(r"^(?P<clause>(?:[一二三四五六七八九十]+|\d+)[、.])\s*(?P<title>.+)$")


def parse_clause_metadata(
    text: str,
    *,
    current_title: str = UNKNOWN_SECTION_TITLE,
) -> ClauseMetadata:
    for raw_line in _candidate_lines(text):
        if _looks_like_directory_line(raw_line):
            directory_title = _known_title_in(raw_line)
            if directory_title:
                return ClauseMetadata(
                    section_title=directory_title,
                    heading_text=raw_line,
                    heading_confidence="medium",
                    heading_source="known_title",
                )
            continue

        parsed = _parse_numbered_heading(raw_line)
        if parsed is not None:
            return parsed

        known_title = _standalone_known_title(raw_line)
        if known_title:
            return ClauseMetadata(
                section_title=known_title,
                heading_text=raw_line,
                heading_confidence="medium",
                heading_source="known_title",
            )

    return ClauseMetadata(
        section_title=current_title,
        heading_confidence="low",
        heading_source="fallback",
    )


def _candidate_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()[:80] if line.strip()]


def _parse_numbered_heading(line: str) -> ClauseMetadata | None:
    for pattern in (_CHINESE_ARTICLE_RE, _DECIMAL_CLAUSE_RE, _PAREN_CLAUSE_RE, _SIMPLE_LIST_RE):
        match = pattern.match(line)
        if not match:
            continue
        title = _known_title_in(match.group("title"))
        if not title:
            continue
        clause_id = _normalize_clause_id(match.group("clause"))
        return ClauseMetadata(
            clause_id=clause_id,
            heading_text=line,
            section_title=title,
            heading_confidence="high",
            heading_source="line_pattern",
        )
    return None


def _standalone_known_title(line: str) -> str | None:
    normalized = re.sub(r"^[第\d零〇一二三四五六七八九十百千万两、.\s条款章节（）()]+", "", line).strip()
    for title in KNOWN_SECTION_TITLES:
        if normalized == title:
            return title
        if normalized.startswith(title):
            suffix = normalized[len(title):].strip()
            if suffix and len(suffix) <= 10 and re.match(r"^[：:、\\-\\s（(]", suffix):
                return title
    return None


def _known_title_in(text: str) -> str | None:
    for title in KNOWN_SECTION_TITLES:
        if title in text:
            return title
    return None


def _looks_like_directory_line(line: str) -> bool:
    return bool(re.search(r"\\.{3,}\\s*\\d+\\s*$", line) or re.search(r"\\s{2,}\\d+\\s*$", line))


def _normalize_clause_id(value: str) -> str:
    return re.sub(r"\\s+", "", value)
```

- [ ] **Step 4: Run clause parser tests**

Run:

```powershell
pytest tests\test_clause_parser.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\insurance_rag\clause_parser.py tests\test_clause_parser.py
git commit -m "feat: parse policy clause headings"
```

## Task 3: Integrate Clause Metadata Into Chunking

**Files:**
- Modify: `src/insurance_rag/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing chunker tests**

Add to `tests/test_chunker.py`:

```python
from insurance_rag.models import DocumentPage
from insurance_rag.chunker import chunk_pages


def test_chunk_pages_attaches_high_confidence_clause_metadata():
    pages = (
        DocumentPage(
            page_number=6,
            text="第六条 等待期\n等待期为90天。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=200,
        overlap=0,
    )

    assert chunks[0].section_title == "等待期"
    assert chunks[0].clause_id == "第六条"
    assert chunks[0].heading_text == "第六条 等待期"
    assert chunks[0].heading_confidence == "high"


def test_chunk_pages_preserves_fallback_title_for_following_chunks():
    pages = (
        DocumentPage(
            page_number=1,
            text="第二条 保险期间\n本合同保险期间为一年。\n后续内容继续说明保险期间。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(
        pages,
        source_name="policy.pdf",
        source_type="user_policy",
        chunk_size=20,
        overlap=0,
    )

    assert chunks[0].section_title == "保险期间"
    assert all(chunk.section_title == "保险期间" for chunk in chunks)
```

- [ ] **Step 2: Run chunker tests to verify failure**

Run:

```powershell
pytest tests\test_chunker.py -q -p no:cacheprovider
```

Expected: FAIL because chunks do not carry clause metadata yet.

- [ ] **Step 3: Implement chunker integration**

In `src/insurance_rag/chunker.py`, import the parser:

```python
from insurance_rag.clause_parser import parse_clause_metadata
```

Inside `chunk_pages`, replace the `infer_section_title` assignment block with:

```python
            metadata = parse_clause_metadata(part, current_title=current_title)
            current_title = metadata.section_title
            chunk_id = f"{source_type}:{source_name}:p{page.page_number}:c{len(chunks) + 1}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=part,
                    page_number=page.page_number,
                    section_title=metadata.section_title,
                    source_type=source_type,
                    source_name=source_name,
                    extraction_method=page.extraction_method,
                    quality_notes=page.quality_notes,
                    clause_id=metadata.clause_id,
                    heading_text=metadata.heading_text,
                    heading_confidence=metadata.heading_confidence,
                    heading_source=metadata.heading_source,
                )
            )
```

Keep `infer_section_title` for compatibility if existing tests import it. `clause_parser.py` becomes the primary path in `chunk_pages`.

- [ ] **Step 4: Run chunker tests**

Run:

```powershell
pytest tests\test_chunker.py tests\test_clause_parser.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\insurance_rag\chunker.py tests\test_chunker.py
git commit -m "feat: attach clause metadata to chunks"
```

## Task 4: Add Rule Reranker

**Files:**
- Create: `src/insurance_rag/rule_reranker.py`
- Modify: `src/insurance_rag/hybrid_retriever.py`
- Test: `tests/test_rule_reranker.py`
- Test: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Write failing reranker tests**

Create `tests/test_rule_reranker.py`:

```python
from insurance_rag.hybrid_retriever import HybridSearchResult
from insurance_rag.models import DocumentChunk
from insurance_rag.query_rewriter import rewrite_query
from insurance_rag.rule_reranker import rerank_results


def chunk(chunk_id: str, title: str, text: str, *, confidence: str = "high") -> DocumentChunk:
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


def result(chunk_id: str, title: str, text: str, score: float = 0.01) -> HybridSearchResult:
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
```

- [ ] **Step 2: Write failing hybrid result propagation test**

Add to `tests/test_hybrid_retriever.py`:

```python
from insurance_rag.hybrid_retriever import HybridSearchResult


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
```

The existing `tests/test_hybrid_retriever.py` helper is `make_chunk(chunk_id: str, text: str) -> DocumentChunk`, so the test should call `make_chunk("c1", "等待期为90天。")` exactly as shown.

- [ ] **Step 3: Run reranker tests to verify failure**

Run:

```powershell
pytest tests\test_rule_reranker.py tests\test_hybrid_retriever.py -q -p no:cacheprovider
```

Expected: FAIL because reranker and rerank fields do not exist yet.

- [ ] **Step 4: Extend `HybridSearchResult`**

In `src/insurance_rag/hybrid_retriever.py`, add fields:

```python
    rerank_score: float | None = None
    rerank_reasons: tuple[str, ...] = ()
```

In `to_explanation()`, pass:

```python
            rerank_score=self.rerank_score,
            rerank_reasons=self.rerank_reasons,
```

- [ ] **Step 5: Implement `rule_reranker.py`**

Create `src/insurance_rag/rule_reranker.py`:

```python
from dataclasses import replace
import re

from insurance_rag.hybrid_retriever import HybridSearchResult
from insurance_rag.models import QueryRewriteResult


_INTENT_TITLE_MAP: dict[str, tuple[str, ...]] = {
    "waiting_period": ("等待期",),
    "exclusion": ("责任免除", "除外责任", "免责条款"),
    "coverage": ("保险责任", "保险金给付", "给付条件"),
    "amount": ("保险金额", "基本保险金额"),
    "period": ("保险期间",),
    "waiver": ("豁免保险费",),
    "definition": ("释义", "疾病定义", "重大疾病定义"),
}

_NEGATIVE_TITLE_MAP: dict[str, tuple[str, ...]] = {
    "waiting_period": ("保险期间", "犹豫期", "宽限期"),
    "exclusion": ("保险责任",),
    "coverage": ("责任免除", "除外责任", "免责条款"),
}

_NUMBER_PATTERN = re.compile(r"(?:\d+|[零〇一二三四五六七八九十百千万两]+)(?:日|天|年|月|个月|周岁|岁|元|万元|%)")


def rerank_results(
    *,
    question: str,
    rewrite: QueryRewriteResult,
    candidates: list[HybridSearchResult],
    top_k: int,
) -> list[HybridSearchResult]:
    if top_k <= 0:
        return []

    scored = [_score_candidate(question, rewrite, candidate) for candidate in candidates]
    return sorted(
        scored,
        key=lambda candidate: (
            candidate.rerank_score if candidate.rerank_score is not None else 0.0,
            candidate.final_score,
        ),
        reverse=True,
    )[:top_k]


def _score_candidate(
    question: str,
    rewrite: QueryRewriteResult,
    candidate: HybridSearchResult,
) -> HybridSearchResult:
    score = 0.0
    reasons: list[str] = []
    title = candidate.chunk.section_title
    text = f"{candidate.chunk.heading_text or ''}\n{candidate.chunk.text}"
    intents = set(rewrite.detected_intents)

    for intent, titles in _INTENT_TITLE_MAP.items():
        if intent in intents and any(expected in title for expected in titles):
            score += 3.0
            reasons.append("title_intent_match")
            if intent == "exclusion":
                reasons.append("exclusion_fact_type_match")
            if intent == "coverage":
                reasons.append("positive_fact_type_match")

    for intent, titles in _NEGATIVE_TITLE_MAP.items():
        if intent in intents and any(unwanted in title for unwanted in titles):
            score -= 2.0
            reasons.append("negative_title_mismatch")

    for query in rewrite.expanded_queries:
        if query and query in text:
            score += 1.0
            reasons.append("exact_term_match")
            break

    question_numbers = set(_NUMBER_PATTERN.findall(question))
    if question_numbers and any(number in text for number in question_numbers):
        score += 2.0
        reasons.append("number_match")

    if candidate.chunk.heading_confidence == "high":
        score += 1.0
        reasons.append("clause_heading_match")
    elif candidate.chunk.heading_confidence == "low":
        score -= 0.5
        reasons.append("low_heading_confidence")

    if _looks_like_directory(candidate.chunk.text):
        score -= 2.0
        reasons.append("directory_like_chunk")

    return replace(
        candidate,
        rerank_score=score,
        rerank_reasons=_dedupe(reasons),
    )


def _looks_like_directory(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    directory_like = sum(1 for line in lines if re.search(r"\.{3,}\s*\d+\s*$", line))
    return directory_like >= max(2, len(lines) // 2)


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)
```

- [ ] **Step 6: Run reranker tests**

Run:

```powershell
pytest tests\test_rule_reranker.py tests\test_hybrid_retriever.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src\insurance_rag\rule_reranker.py src\insurance_rag\hybrid_retriever.py tests\test_rule_reranker.py tests\test_hybrid_retriever.py
git commit -m "feat: add rule reranking"
```

## Task 5: Integrate Reranking Into The RAG Chain

**Files:**
- Modify: `src/insurance_rag/rag_chain.py`
- Test: `tests/test_rag_chain.py`

- [ ] **Step 1: Write failing RAG chain rerank tests**

Add to `tests/test_rag_chain.py`:

```python
def test_answer_reranks_policy_candidates_before_prompt(monkeypatch):
    period_chunk = make_chunk(chunk_id="period", quality_notes=())
    period_chunk = DocumentChunk(
        chunk_id=period_chunk.chunk_id,
        text="保险期间为90天。",
        page_number=1,
        section_title="保险期间",
        source_type=period_chunk.source_type,
        source_name=period_chunk.source_name,
        extraction_method=period_chunk.extraction_method,
        heading_confidence="high",
    )
    waiting_chunk = DocumentChunk(
        chunk_id="waiting",
        text="等待期为90天。",
        page_number=2,
        section_title="等待期",
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
        heading_confidence="high",
    )
    chain, client = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([period_chunk, waiting_chunk]),
        chat_client=FakeChatClient(answer="等待期为90天。"),
    )

    payload = chain.answer("等待期是多久？")
    prompt = client.calls[0]["messages"][1]["content"]

    assert prompt.find("等待期为90天") < prompt.find("保险期间为90天")
    assert payload.retrieval_explanations[0].rerank_score is not None
```

- [ ] **Step 2: Run RAG chain test to verify failure**

Run:

```powershell
pytest tests\test_rag_chain.py::test_answer_reranks_policy_candidates_before_prompt -q -p no:cacheprovider
```

Expected: FAIL because `rag_chain.py` does not call the reranker.

- [ ] **Step 3: Implement rerank orchestration**

In `src/insurance_rag/rag_chain.py`, import:

```python
from insurance_rag.rule_reranker import rerank_results
```

In `answer()`, change policy retrieval to request rerank candidates:

```python
            policy_search_top_k = (
                max(self.config.policy_top_k, self.config.rerank_top_n)
                if self.config.rerank_enabled
                else self.config.policy_top_k
            )
            policy_results = self.policy_retriever.search(
                rewrite,
                top_k=policy_search_top_k,
            )
            if self.config.rerank_enabled:
                try:
                    policy_results = rerank_results(
                        question=question,
                        rewrite=rewrite,
                        candidates=policy_results,
                        top_k=self.config.policy_top_k,
                    )
                except Exception as error:
                    warnings.append(f"规则重排序未完成，已使用原始检索排序：{error}")
                    policy_results = policy_results[: self.config.policy_top_k]
            else:
                policy_results = policy_results[: self.config.policy_top_k]
```

Leave built-in retrieval as hybrid-only in this phase.

- [ ] **Step 4: Run RAG chain tests**

Run:

```powershell
pytest tests\test_rag_chain.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\insurance_rag\rag_chain.py tests\test_rag_chain.py
git commit -m "feat: rerank policy evidence in RAG chain"
```

## Task 6: Add Fact-Level Citation Verifier

**Files:**
- Create: `src/insurance_rag/citation_verifier.py`
- Test: `tests/test_citation_verifier.py`

- [ ] **Step 1: Write failing verifier tests**

Create `tests/test_citation_verifier.py`:

```python
from insurance_rag.citation_verifier import verify_answer_facts
from insurance_rag.models import Citation


def citation(title: str, excerpt: str, source_type: str = "user_policy") -> Citation:
    return Citation(
        source_type=source_type,
        source_name="policy.pdf",
        page_number=6,
        section_title=title,
        excerpt=excerpt,
    )


def test_verifier_supports_numeric_fact_in_same_citation():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(citation("等待期", "本合同等待期为九十日。"),),
        builtin_citations=(),
    )

    assert result.facts[0].status == "supported"
    assert result.facts[0].severity == "info"


def test_verifier_blocks_numeric_fact_when_number_belongs_to_other_clause():
    result = verify_answer_facts(
        answer="等待期是90天。",
        policy_citations=(
            citation("等待期", "本合同等待期为30天。"),
            citation("保险期间", "保险期间为90天。"),
        ),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is True
    assert result.block_reason is not None
    assert "等待期90天" in result.block_reason


def test_verifier_blocks_builtin_content_as_user_policy_fact():
    result = verify_answer_facts(
        answer="你的保单写明等待期是保险合同生效后的一段观察时间。",
        policy_citations=(),
        builtin_citations=(citation("等待期", "等待期是保险合同生效后的一段观察时间。", "built_in_dataset"),),
    )

    assert result.has_blocking_fact is True
    assert "内置资料" in result.block_reason


def test_verifier_warns_for_partial_support():
    result = verify_answer_facts(
        answer="等待期相关内容需要结合原文核对。",
        policy_citations=(citation("等待期", "等待期内不承担保险责任。"),),
        builtin_citations=(),
    )

    assert result.has_blocking_fact is False
    assert result.has_warnings is True
```

- [ ] **Step 2: Run verifier tests to verify failure**

Run:

```powershell
pytest tests\test_citation_verifier.py -q -p no:cacheprovider
```

Expected: FAIL because `citation_verifier.py` does not exist.

- [ ] **Step 3: Implement `citation_verifier.py`**

Create `src/insurance_rag/citation_verifier.py`:

```python
import re

from insurance_rag.models import Citation, CitationVerificationResult, VerifiedFact


_POLICY_TERMS = (
    "等待期",
    "责任免除",
    "除外责任",
    "免责条款",
    "保险责任",
    "保险金额",
    "保险期间",
    "豁免保险费",
    "投保人",
    "被保险人",
)
_NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"(?:\d+|[零〇一二三四五六七八九十百千万两]+)(?:日|天|年|月|个月|周岁|岁|元|万元|%)"
)
_SOURCE_CONFUSING_TERMS = ("你的保单", "这份保单写明", "保单写明")


def verify_answer_facts(
    *,
    answer: str,
    policy_citations: tuple[Citation, ...],
    builtin_citations: tuple[Citation, ...],
) -> CitationVerificationResult:
    facts: list[VerifiedFact] = []

    if builtin_citations and any(term in answer for term in _SOURCE_CONFUSING_TERMS):
        if not policy_citations:
            fact = VerifiedFact(
                fact_text="内置资料被表述为用户保单事实",
                fact_type="source",
                status="unsupported",
                severity="block",
                reason="回答把内置资料库内容表述为用户保单事实。",
            )
            return CitationVerificationResult(
                facts=(fact,),
                block_reason="回答可能将内置资料库内容表述为用户保单事实。",
            )

    facts.extend(_verify_numeric_facts(answer, policy_citations))
    facts.extend(_verify_partial_policy_mentions(answer, policy_citations))

    block_reasons = [
        fact.fact_text
        for fact in facts
        if fact.severity == "block"
    ]
    warnings = tuple(
        fact.reason or fact.fact_text
        for fact in facts
        if fact.severity == "warn"
    )
    return CitationVerificationResult(
        facts=tuple(facts),
        warnings=warnings,
        block_reason=(
            "回答中的关键保单事实未被用户保单引用支持："
            + "、".join(block_reasons)
            + "。"
            if block_reasons
            else None
        ),
    )


def _verify_numeric_facts(answer: str, citations: tuple[Citation, ...]) -> list[VerifiedFact]:
    facts: list[VerifiedFact] = []
    for fragment in _split_fragments(answer):
        for term in _POLICY_TERMS:
            if term not in fragment:
                continue
            for number in _NUMBER_WITH_UNIT_PATTERN.findall(fragment):
                facts.append(_verify_term_number(term, number, citations))
    return facts


def _verify_term_number(term: str, number: str, citations: tuple[Citation, ...]) -> VerifiedFact:
    normalized_number = _normalize_number_units(number)
    supporting_ids: list[str] = []
    for citation in citations:
        evidence = f"{citation.section_title}\n{citation.excerpt}"
        for fragment in _split_fragments(evidence):
            if term in fragment and normalized_number in _normalize_number_units(fragment):
                supporting_ids.append(_citation_id(citation))
                return VerifiedFact(
                    fact_text=f"{term}{number}",
                    fact_type="number",
                    status="supported",
                    severity="info",
                    supporting_citation_ids=tuple(supporting_ids),
                    reason=f"同一用户保单引用中找到{term}和{number}。",
                )
    return VerifiedFact(
        fact_text=f"{term}{number}",
        fact_type="number",
        status="unsupported",
        severity="block",
        reason=f"未找到同时支持{term}和{number}的用户保单引用。",
    )


def _verify_partial_policy_mentions(answer: str, citations: tuple[Citation, ...]) -> list[VerifiedFact]:
    if not citations:
        return []
    if any(term in answer for term in _POLICY_TERMS) and not _NUMBER_WITH_UNIT_PATTERN.search(answer):
        return [
            VerifiedFact(
                fact_text="保单概括性说明",
                fact_type="general_policy_fact",
                status="partially_supported",
                severity="warn",
                supporting_citation_ids=tuple(_citation_id(citation) for citation in citations[:1]),
                reason="回答提到保单条款，但不是可精确核验的数字或责任事实，请结合原文核对。",
            )
        ]
    return []


def _split_fragments(text: str) -> tuple[str, ...]:
    return tuple(fragment for fragment in re.split(r"[。；;，,\n]+", text) if fragment.strip())


def _normalize_number_units(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        number = match.group("number")
        unit = match.group("unit")
        if not number.isdigit():
            converted = _chinese_number_to_int(number)
            if converted is not None:
                number = str(converted)
        if unit == "日":
            unit = "天"
        return f"{number}{unit}"

    return re.sub(
        r"(?P<number>\d+|[零〇一二三四五六七八九十百千万两]+)(?P<unit>日|天|年|月|个月|周岁|岁|元|万元|%)",
        replace,
        text,
    )


def _chinese_number_to_int(text: str) -> int | None:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text and all(char in digits or char == "十" for char in text):
        left, _, right = text.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _citation_id(citation: Citation) -> str:
    page = citation.page_number if citation.page_number is not None else "unknown"
    return f"{citation.source_name}:{page}:{citation.section_title}"
```

- [ ] **Step 4: Run verifier tests**

Run:

```powershell
pytest tests\test_citation_verifier.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\insurance_rag\citation_verifier.py tests\test_citation_verifier.py
git commit -m "feat: verify answer facts against citations"
```

## Task 7: Integrate Citation Verifier With Guard, Payload, And UI

**Files:**
- Modify: `src/insurance_rag/answer_guard.py`
- Modify: `src/insurance_rag/rag_chain.py`
- Modify: `app.py`
- Test: `tests/test_answer_guard.py`
- Test: `tests/test_rag_chain.py`

- [ ] **Step 1: Write failing guard integration test**

Add to `tests/test_answer_guard.py`:

```python
def test_answer_guard_returns_citation_verification_result_for_supported_fact():
    result = check(
        "等待期是90天。",
        policy_citations=(citation(section_title="等待期", excerpt="本合同等待期为九十日。"),),
    )

    assert result.citation_verification is not None
    assert result.citation_verification.facts
    assert result.citation_verification.facts[0].status == "supported"
```

- [ ] **Step 2: Write failing RAG payload integration test**

Add to `tests/test_rag_chain.py`:

```python
def test_answer_payload_includes_citation_verification(monkeypatch):
    chain, _ = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([make_chunk(quality_notes=())]),
        chat_client=FakeChatClient(answer="等待期为九十日。"),
    )

    payload = chain.answer("等待期是多久？")

    assert payload.citation_verification is not None
```

- [ ] **Step 3: Run integration tests to verify failure**

Run:

```powershell
pytest tests\test_answer_guard.py tests\test_rag_chain.py -q -p no:cacheprovider
```

Expected: FAIL because guard and payload do not expose citation verification yet.

- [ ] **Step 4: Integrate verifier in `answer_guard.py`**

Import:

```python
from insurance_rag.citation_verifier import verify_answer_facts
```

Near the start of `check_answer`, after final-claim detection and source-confusion detection, add:

```python
    citation_verification = verify_answer_facts(
        answer=answer,
        policy_citations=policy_citations,
        builtin_citations=builtin_citations,
    )
    if citation_verification.has_blocking_fact:
        return AnswerGuardResult(
            status=GuardStatus.BLOCK,
            block_reason=citation_verification.block_reason,
            citation_verification=citation_verification,
        )
```

When returning `WARN`, include the result:

```python
        return AnswerGuardResult(
            status=GuardStatus.WARN,
            warnings=tuple(warnings + list(citation_verification.warnings)),
            citation_verification=citation_verification,
        )
```

When returning `PASS`, include:

```python
    return AnswerGuardResult(
        status=GuardStatus.PASS,
        citation_verification=citation_verification,
    )
```

- [ ] **Step 5: Propagate verification result in `rag_chain.py`**

Before returning `AnswerPayload`, set:

```python
        citation_verification = (
            guard_result.citation_verification if guard_result is not None else None
        )
```

Pass into payload:

```python
            citation_verification=citation_verification,
```

- [ ] **Step 6: Add UI renderer in `app.py`**

Add:

```python
def render_citation_verification(payload: AnswerPayload) -> None:
    verification = payload.citation_verification
    if verification is None or not verification.facts:
        return

    with st.expander("证据核验结果", expanded=False):
        for fact in verification.facts:
            if fact.severity == "block":
                st.error(f"未通过：{fact.fact_text}")
            elif fact.severity == "warn":
                st.warning(f"需核对：{fact.fact_text}")
            else:
                st.success(f"通过：{fact.fact_text}")
            if fact.reason:
                st.caption(fact.reason)
            if fact.supporting_citation_ids:
                st.write("支持引用：" + "、".join(fact.supporting_citation_ids))
```

Call it inside `render_citations(payload)` before `render_retrieval_details(...)`:

```python
    render_citation_verification(payload)
```

- [ ] **Step 7: Run guard and RAG tests**

Run:

```powershell
pytest tests\test_answer_guard.py tests\test_rag_chain.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src\insurance_rag\answer_guard.py src\insurance_rag\rag_chain.py app.py tests\test_answer_guard.py tests\test_rag_chain.py
git commit -m "feat: show citation verification results"
```

## Task 8: Add Synthetic Hard Negative Evaluation

**Files:**
- Add: `evals/hard_negative_cases.json`
- Modify: `src/insurance_rag/evaluation.py`
- Modify: `scripts/evaluate_rag.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Add synthetic hard negative cases**

Create `evals/hard_negative_cases.json`:

```json
[
  {
    "case_id": "hard_negative_waiting_period_number",
    "question": "等待期是多久？",
    "expected_positive_chunk_id": "waiting-90",
    "max_expected_rank": 1,
    "answer": "等待期是90天。",
    "chunks": [
      {
        "chunk_id": "period-90",
        "section_title": "保险期间",
        "text": "保险期间为90天。"
      },
      {
        "chunk_id": "waiting-90",
        "section_title": "等待期",
        "text": "本合同等待期为90天。"
      },
      {
        "chunk_id": "hesitation-15",
        "section_title": "犹豫期",
        "text": "本合同犹豫期为15天。"
      }
    ]
  },
  {
    "case_id": "hard_negative_exclusion_vs_coverage",
    "question": "哪些情况不赔？",
    "expected_positive_chunk_id": "drunk-driving-exclusion",
    "max_expected_rank": 1,
    "answer": "酒后驾驶属于责任免除。",
    "chunks": [
      {
        "chunk_id": "major-disease-coverage",
        "section_title": "保险责任",
        "text": "本合同承担重大疾病保险责任。"
      },
      {
        "chunk_id": "drunk-driving-exclusion",
        "section_title": "责任免除",
        "text": "因酒后驾驶导致的事故属于责任免除。"
      }
    ]
  },
  {
    "case_id": "hard_negative_subject_waiver",
    "question": "豁免保险费适用于谁？",
    "expected_positive_chunk_id": "policyholder-waiver",
    "max_expected_rank": 1,
    "answer": "投保人可豁免保险费。",
    "chunks": [
      {
        "chunk_id": "insured-waiver",
        "section_title": "豁免保险费",
        "text": "被保险人符合约定条件时可豁免保险费。"
      },
      {
        "chunk_id": "policyholder-waiver",
        "section_title": "豁免保险费",
        "text": "投保人符合约定条件时可豁免保险费。"
      }
    ]
  },
  {
    "case_id": "hard_negative_builtin_source_confusion",
    "question": "这份保单有等待期吗？",
    "expected_positive_chunk_id": "policy-no-specific-waiting-period",
    "max_expected_rank": 2,
    "answer": "你的保单写明等待期是保险合同生效后的一段观察时间。",
    "chunks": [
      {
        "chunk_id": "builtin-waiting-definition",
        "section_title": "等待期",
        "source_type": "built_in_dataset",
        "text": "等待期是保险合同生效后的一段观察时间。"
      },
      {
        "chunk_id": "policy-no-specific-waiting-period",
        "section_title": "保险责任",
        "text": "本合同承担重大疾病保险责任。"
      }
    ]
  }
]
```

- [ ] **Step 2: Write failing synthetic hard negative tests**

Add to `tests/test_evaluation.py`:

```python
from insurance_rag.evaluation import (
    evaluate_hard_negative_cases,
    render_hard_negative_markdown_report,
)


HARD_NEGATIVE_CASES_PATH = ROOT / "evals" / "hard_negative_cases.json"


def test_hard_negative_evaluation_passes_repo_cases():
    report = evaluate_hard_negative_cases(HARD_NEGATIVE_CASES_PATH)

    assert report.total_cases == 4
    assert report.passed_cases == report.total_cases
    assert all(result.positive_rank is not None for result in report.results)
    assert all(result.verifier_status in {"pass", "warn", "block"} for result in report.results)


def test_hard_negative_report_contains_rerank_and_verifier_details():
    report = evaluate_hard_negative_cases(HARD_NEGATIVE_CASES_PATH)
    markdown = render_hard_negative_markdown_report(report)

    assert "# InsuranceRAG Hard Negative Evaluation Report" in markdown
    assert "Positive Rank" in markdown
    assert "Verifier" in markdown
    assert "hard_negative_waiting_period_number" in markdown
```

- [ ] **Step 3: Run evaluation tests to verify failure**

Run:

```powershell
pytest tests\test_evaluation.py -q -p no:cacheprovider
```

Expected: FAIL because hard negative evaluation functions do not exist.

- [ ] **Step 4: Implement hard negative evaluation models and runner**

In `src/insurance_rag/evaluation.py`, add dataclasses:

```python
@dataclass(frozen=True)
class HardNegativeCaseResult:
    case_id: str
    question: str
    expected_positive_chunk_id: str
    positive_rank: int | None
    max_expected_rank: int
    retrieved_chunk_ids: tuple[str, ...]
    rerank_details: tuple[str, ...]
    verifier_status: str
    passed: bool


@dataclass(frozen=True)
class HardNegativeEvalReport:
    total_cases: int
    passed_cases: int
    results: tuple[HardNegativeCaseResult, ...]
```

Add `evaluate_hard_negative_cases(path: Path, top_k: int = 3) -> HardNegativeEvalReport`.

Add this implementation:

```python
def evaluate_hard_negative_cases(path: Path, top_k: int = 3) -> HardNegativeEvalReport:
    raw_cases = _load_cases(path)
    embedder = DeterministicEvalEmbedder()
    results: list[HardNegativeCaseResult] = []
    for index, case in enumerate(raw_cases, start=1):
        chunks = _hard_negative_chunks_for_case(case)
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_index = InMemoryVectorIndex.from_embeddings(chunks, embeddings)
        retriever = HybridRetriever(chunks, vector_index, embedder, retrieval_mode="hybrid")
        rewrite = rewrite_query(str(case["question"]))
        initial = retriever.search(rewrite, top_k=max(top_k, len(chunks)))
        retrieved = rerank_results(
            question=str(case["question"]),
            rewrite=rewrite,
            candidates=initial,
            top_k=top_k,
        )
        positive_id = str(case["expected_positive_chunk_id"])
        positive_rank = _first_chunk_id_rank(retrieved, positive_id)
        citations = tuple(build_eval_citation(result.chunk) for result in retrieved)
        verification = verify_answer_facts(
            answer=str(case.get("answer", "")),
            policy_citations=tuple(citation for citation in citations if citation.source_type != "built_in_dataset"),
            builtin_citations=tuple(citation for citation in citations if citation.source_type == "built_in_dataset"),
        )
        max_rank = int(case.get("max_expected_rank", 1))
        passed = positive_rank is not None and positive_rank <= max_rank
        if case.get("answer") and verification.has_blocking_fact and "source_confusion" not in _case_id(case):
            passed = False
        results.append(
            HardNegativeCaseResult(
                case_id=_case_id(case),
                question=str(case["question"]),
                expected_positive_chunk_id=positive_id,
                positive_rank=positive_rank,
                max_expected_rank=max_rank,
                retrieved_chunk_ids=tuple(result.chunk.chunk_id for result in retrieved),
                rerank_details=tuple(",".join(result.rerank_reasons) for result in retrieved),
                verifier_status=("block" if verification.has_blocking_fact else "warn" if verification.has_warnings else "pass"),
                passed=passed,
            )
        )
    return HardNegativeEvalReport(
        total_cases=len(results),
        passed_cases=sum(1 for result in results if result.passed),
        results=tuple(results),
    )
```

Add helpers used above:

```python
def build_eval_citation(chunk: DocumentChunk) -> Citation:
    return Citation(
        source_type=chunk.source_type,
        source_name=chunk.source_name,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        excerpt=chunk.text,
    )


def _hard_negative_chunks_for_case(case: dict[str, Any]) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    for index, chunk in enumerate(case.get("chunks", ()), start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=str(chunk["chunk_id"]),
                text=str(chunk["text"]),
                page_number=index,
                section_title=str(chunk["section_title"]),
                source_type=str(chunk.get("source_type", "synthetic_eval")),
                source_name=_case_id(case),
                extraction_method="synthetic",
                heading_confidence="high",
            )
        )
    return tuple(chunks)


def _first_chunk_id_rank(retrieved: list[Any], chunk_id: str) -> int | None:
    for rank, result in enumerate(retrieved, start=1):
        if result.chunk.chunk_id == chunk_id:
            return rank
    return None
```

Add `render_hard_negative_markdown_report(report)`.

- [ ] **Step 5: Add CLI flags**

In `scripts/evaluate_rag.py`, import:

```python
    evaluate_hard_negative_cases,
    render_hard_negative_markdown_report,
```

Add args:

```python
    parser.add_argument("--hard-negative", action="store_true")
    parser.add_argument(
        "--hard-negative-cases",
        type=Path,
        default=ROOT / "evals" / "hard_negative_cases.json",
    )
```

Update no-selection check:

```python
    if not args.synthetic and not args.hard_negative and args.local_documents is None:
        print("No evaluation selected. Use --synthetic, --hard-negative, or --local-documents.")
        return 2
```

Add hard negative execution:

```python
    hard_negative_failed = False
    if args.hard_negative:
        cases_path = args.hard_negative_cases
        if not cases_path.is_absolute():
            cases_path = ROOT / cases_path
        hard_report = evaluate_hard_negative_cases(cases_path)
        hard_markdown = render_hard_negative_markdown_report(hard_report)
        (report_dir / "hard_negative_eval_report.md").write_text(hard_markdown, encoding="utf-8")
        print(hard_markdown)
        hard_negative_failed = hard_report.passed_cases != hard_report.total_cases
```

Update return:

```python
    return 1 if synthetic_failed or local_failed or hard_negative_failed else 0
```

- [ ] **Step 6: Run evaluation tests and CLI**

Run:

```powershell
pytest tests\test_evaluation.py -q -p no:cacheprovider
python scripts\evaluate_rag.py --hard-negative
```

Expected: tests PASS and CLI exits 0.

- [ ] **Step 7: Commit**

```powershell
git add evals\hard_negative_cases.json src\insurance_rag\evaluation.py scripts\evaluate_rag.py tests\test_evaluation.py
git commit -m "feat: add synthetic hard negative evaluation"
```

## Task 9: Add Local Hard Negative Evaluation

**Files:**
- Modify: `src/insurance_rag/evaluation.py`
- Modify: `scripts/evaluate_rag.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing local hard negative tests**

Add to `tests/test_evaluation.py`:

```python
from insurance_rag.evaluation import evaluate_local_hard_negative_documents


def test_local_hard_negative_evaluation_builds_cases_from_pdf():
    docs_dir = _repo_tmp_dir("local-hard-negative-docs")
    _write_pdf(
        docs_dir / "sample.pdf",
        "第六条 等待期\n等待期为九十日。\n第七条 保险期间\n保险期间为一年。\n第八条 责任免除\n酒后驾驶属于责任免除。",
    )

    report = evaluate_local_hard_negative_documents(docs_dir, sample_limit=1)

    assert report.total_documents == 1
    assert report.parsed_documents == 1
    assert report.total_cases >= 1
```

- [ ] **Step 2: Run local hard negative test to verify failure**

Run:

```powershell
pytest tests\test_evaluation.py::test_local_hard_negative_evaluation_builds_cases_from_pdf -q -p no:cacheprovider
```

Expected: FAIL because local hard negative function does not exist.

- [ ] **Step 3: Implement local hard negative evaluator**

In `src/insurance_rag/evaluation.py`, add:

```python
def evaluate_local_hard_negative_documents(
    documents_dir: Path,
    *,
    sample_limit: int = 20,
    top_k: int = 3,
    config: AppConfig | None = None,
) -> LocalDocumentEvalReport:
    base_report = evaluate_local_documents(
        documents_dir,
        sample_limit=sample_limit,
        top_k=top_k,
        config=config,
    )
    return base_report
```

For this phase, reuse the local document evaluator as the report substrate and extend its case generation to add confusion pairs for headings when both terms exist in a document. Add cases for:

```python
LOCAL_HARD_NEGATIVE_QUERY_PAIRS = (
    ("等待期", "等待期是多久？", ("保险期间", "犹豫期", "宽限期")),
    ("责任免除", "哪些情况不赔？", ("保险责任",)),
    ("保险责任", "保障哪些内容？", ("责任免除",)),
    ("豁免保险费", "豁免保险费适用于谁？", ("保险费",)),
)
```

Keep the output as a local markdown report and do not write real text into repo files.

- [ ] **Step 4: Add CLI flag**

In `scripts/evaluate_rag.py`, add:

```python
    parser.add_argument("--local-hard-negative", type=Path)
```

When provided, call:

```python
            local_hard_report = evaluate_local_hard_negative_documents(
                args.local_hard_negative,
                sample_limit=config.hard_negative_local_limit,
            )
            local_hard_markdown = render_local_markdown_report(local_hard_report)
            (report_dir / "local_hard_negative_eval_report.md").write_text(
                local_hard_markdown,
                encoding="utf-8",
            )
            print(local_hard_markdown)
```

If the explicit path does not exist and no other eval was selected, return non-zero.

After adding `--local-hard-negative`, update the no-selection check to:

```python
    if (
        not args.synthetic
        and not args.hard_negative
        and args.local_documents is None
        and args.local_hard_negative is None
    ):
        print("No evaluation selected. Use --synthetic, --hard-negative, --local-documents, or --local-hard-negative.")
        return 2
```

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests\test_evaluation.py -q -p no:cacheprovider
python scripts\evaluate_rag.py --local-hard-negative documents
```

Expected: tests PASS. The local command exits 0 when `documents/` exists, or exits 1 with a clear missing-path message if it does not.

- [ ] **Step 6: Commit**

```powershell
git add src\insurance_rag\evaluation.py scripts\evaluate_rag.py tests\test_evaluation.py
git commit -m "feat: add local hard negative evaluation"
```

## Task 10: Final UI And Documentation Polish

**Files:**
- Modify: `app.py`
- Modify: `README.md`
- Test: `tests/test_rag_chain.py`

- [ ] **Step 1: Update retrieval details UI**

In `app.py`, inside `render_retrieval_details`, after score captions, add:

```python
            if explanation.rerank_score is not None:
                st.caption(f"rerank_score={explanation.rerank_score:.4f}")
            if explanation.rerank_reasons:
                st.write("重排序依据：" + "、".join(explanation.rerank_reasons))
```

- [ ] **Step 2: Update README**

Add sections describing:

```markdown
## 结构化条款识别

系统会尝试识别常见条号和标题，例如 `第六条 等待期`、`2.3 责任免除`。识别结果用于引用展示、规则重排序和评测。无法可靠识别时，系统会回退到关键词标题推断，并在质量信息中保留低置信度提示。

## 规则重排序

系统在混合检索之后启用规则重排序。重排序使用问题意图、条款标题、条号、精确词和数字匹配，不额外调用 LLM。

## 证据核验结果

每个回答可以展示事实级证据核验结果。严重未支持事实会被阻断；证据不足但未发现明确错误时显示警告。

## Hard Negative 评测

运行合成 hard negative 评测：

```powershell
python scripts\evaluate_rag.py --hard-negative
```

运行本地真实文档 hard negative 评测：

```powershell
python scripts\evaluate_rag.py --local-hard-negative documents
```
```

- [ ] **Step 3: Run full verification**

Run:

```powershell
pytest tests -q -p no:cacheprovider
python scripts\evaluate_rag.py --synthetic
python scripts\evaluate_rag.py --hard-negative
python -c "import app; print('app import ok')"
git diff --check
```

Expected:

- `pytest` passes.
- synthetic eval exits 0.
- hard negative eval exits 0.
- app import prints `app import ok`.
- `git diff --check` exits 0.

- [ ] **Step 4: Run optional local evaluation**

Run:

```powershell
python scripts\evaluate_rag.py --local-documents documents --local-sample-limit 20
python scripts\evaluate_rag.py --local-hard-negative documents
```

Expected: both commands exit 0 when local `documents/` exists. Reports are written to ignored report directory.

- [ ] **Step 5: Confirm ignored outputs remain untracked**

Run:

```powershell
git status --short --ignored
git ls-files documents eval_reports .rag_eval_cache
```

Expected:

- generated report directories are ignored or absent.
- `git ls-files documents eval_reports .rag_eval_cache` prints nothing.

- [ ] **Step 6: Commit final docs and UI polish**

```powershell
git add app.py README.md
git commit -m "docs: document advanced RAG quality controls"
```

## Final Verification Checklist

- [ ] `pytest tests -q -p no:cacheprovider` passes.
- [ ] `python scripts\evaluate_rag.py --synthetic` exits 0.
- [ ] `python scripts\evaluate_rag.py --hard-negative` exits 0.
- [ ] `python scripts\evaluate_rag.py --local-documents documents --local-sample-limit 20` runs when local documents exist.
- [ ] `python scripts\evaluate_rag.py --local-hard-negative documents` runs when local documents exist.
- [ ] `python -c "import app; print('app import ok')"` passes.
- [ ] `git diff --check` exits 0.
- [ ] `git status --short` contains no unintended untracked real document or report files.
- [ ] No file under `documents/`, `eval_reports/`, or `.rag_eval_cache/` is tracked.

## Execution Notes

- Use TDD for each task: write the failing test, run it, implement, rerun.
- Keep commits small and task-scoped.
- Use `pytest ... -p no:cacheprovider` on this Windows workspace to avoid stale pytest cache permission noise.
- Do not send local real policy text to external APIs for evaluation.
- Do not commit generated local evaluation reports.
