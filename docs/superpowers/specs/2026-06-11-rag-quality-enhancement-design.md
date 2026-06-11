# RAG Quality Enhancement Design

Date: 2026-06-11

## Overview

This design extends the current InsuranceRAG MVP with product-oriented RAG quality improvements. The goal is to make retrieval more accurate, more explainable, and easier to regression-test without changing the core product boundary.

The system remains a Chinese policy explanation assistant. It explains insurance policy clauses in plain Chinese, cites original policy evidence, and must not make final claim approval, denial, legal, medical, financial, or underwriting conclusions.

The uploaded user policy remains the primary evidence source. The built-in `documents/` dataset remains secondary and may only be used for terminology or background explanation after user policy evidence has been found.

## Goals

- Improve retrieval recall for exact insurance terms such as `等待期`, `责任免除`, `豁免保险费`, `保险责任`, and `除外责任`.
- Add query rewriting so colloquial user questions retrieve policy clauses more reliably.
- Add answer self-checking to catch unsupported answers, source confusion, and final claim judgments.
- Add retrieval explanation data for UI debugging and demo transparency.
- Add offline evaluation for both reproducible synthetic cases and optional local real-document cases.

## Non-Goals

- Do not turn the assistant into a final claim decision system.
- Do not use built-in documents to answer specific policy questions when the uploaded user policy has no evidence.
- Do not require persistent storage of uploaded user policies.
- Do not require every question to make extra LLM calls for rewriting or self-checking.
- Do not commit local `documents/` content, evaluation reports, or real policy text to GitHub.
- Do not introduce a full workflow or agent framework for this enhancement.

## Product Behavior

Normal users continue to see Chinese answers and source citations. The app adds a collapsed retrieval-details section for each answer. That section explains why each chunk was selected, including simplified match strength, matched terms, and score summaries.

When the answer is safe but evidence is weak, the app shows warnings and keeps the answer. When the generated answer violates a serious boundary, the app blocks the answer and replaces it with a safe evidence-limited response.

Offline evaluation produces a full report with retrieval ranks, vector scores, BM25 scores, fusion scores, matched terms, pass/fail status, and answer-guard warnings or blocks.

## Architecture

The enhancement uses a modular productization path. `rag_chain.py` remains the orchestration layer, while retrieval, rewriting, answer checking, and evaluation live in separate modules.

### `query_rewriter.py`

Transforms the original user question into one or more retrieval queries.

Responsibilities:

- Preserve the original user question.
- Generate rule-based expanded queries for common insurance intents.
- Detect coarse intents such as coverage, exclusions, waiting period, claim condition, waiver, definition, and policy term.
- Avoid duplicate expanded queries.
- Return warnings if rewriting fails and allow retrieval to continue with the original question.
- Reserve an optional LLM rewrite interface for complex questions, controlled by configuration.

First-version examples:

- `赔不赔`, `能不能赔` -> `保险责任`, `责任免除`, `赔付条件`, `除外责任`
- `什么不赔`, `哪些情况不赔` -> `责任免除`, `除外责任`, `免责条款`
- `等多久`, `多久生效` -> `等待期`, `生效日`, `保险期间`
- `保什么`, `保障哪些` -> `保险责任`, `保障范围`, `保险金额`
- `豁免` -> `豁免保险费`, `投保人豁免`, `被保险人豁免`

Suggested return model:

- `original_query`
- `expanded_queries`
- `detected_intents`
- `used_llm`
- `warnings`

### `hybrid_retriever.py`

Combines vector retrieval with BM25 keyword retrieval.

Responsibilities:

- Build a BM25 index from the same `DocumentChunk` objects used by the vector index.
- Execute vector search and BM25 search for each expanded query.
- Fuse rankings with reciprocal rank fusion by default.
- Deduplicate chunks across multiple rewritten queries.
- Preserve score details for UI and evaluation reports.
- Degrade to vector-only mode when BM25 construction or search fails.

The first implementation should use `rank-bm25`. This dependency is light, keeps code small, and makes the retrieval strategy easier to explain.

Suggested result model:

- `chunk`
- `final_score`
- `vector_score`
- `bm25_score`
- `matched_terms`
- `rank_details`

Default fusion:

- Use reciprocal rank fusion rather than manually normalizing vector and BM25 scores.
- Use configurable `rrf_k`, default `60`.
- Return `top_k` final chunks after fusion.

### `answer_guard.py`

Checks the generated answer after the chat model responds.

Responsibilities:

- Detect unsupported policy-specific answers.
- Detect final claim decisions.
- Detect confusion between user policy evidence and built-in background evidence.
- Return a graded result: `pass`, `warn`, or `block`.
- Reserve an optional LLM guard interface for ambiguous failures, controlled by configuration.

Suggested return model:

- `status`: `pass`, `warn`, or `block`
- `warnings`
- `block_reason`

Blocking rules:

- No user policy citation exists, but the answer states a specific fact about the uploaded policy.
- The answer makes a final claim judgment such as `一定赔`, `肯定不赔`, `必须赔`, or equivalent wording.
- The answer presents built-in dataset content as if it were written in the uploaded user policy.

Warning rules:

- User policy citation count is low.
- The top fused retrieval score is weak.
- Cited chunks contain OCR quality warnings.
- Built-in context was heavily used and should be treated only as background.
- The answer appears to need clearer uncertainty language.

### `rag_chain.py`

Keeps responsibility for orchestration.

Flow:

1. Call `query_rewriter`.
2. Retrieve user policy chunks through `hybrid_retriever`.
3. Refuse early if no user policy evidence is available.
4. Decide whether built-in background is allowed.
5. Retrieve built-in background through `hybrid_retriever` only when allowed.
6. Build prompt with separate user-policy and built-in-background sections.
7. Generate Chinese answer with OpenAI chat model.
8. Call `answer_guard`.
9. Return the final answer, citations, warnings, and retrieval explanations.

The chat prompt should continue to emphasize:

- User policy evidence is primary.
- Built-in dataset content is background only.
- The assistant explains clauses and does not decide claims.
- Evidence gaps must be stated explicitly.

### `app.py`

Keeps the existing Streamlit product flow.

UI additions:

- Collapsed `检索依据详情` section under each assistant answer.
- Per-citation match label such as high, medium, or low.
- Matched keyword display.
- Score summary for vector, BM25, and fused ranking.
- Warnings for weak evidence, OCR quality, built-in background use, and answer-guard findings.
- Safe replacement answer display when the guard returns `block`.

### `scripts/evaluate_rag.py` and `evals/`

Adds offline evaluation.

Synthetic evaluation:

- Store reproducible synthetic cases in `evals/synthetic_cases.json`.
- Use artificial insurance clause snippets rather than real policy text.
- Avoid OpenAI dependency where possible by using fake or deterministic embeddings.
- Be suitable for CI and normal `pytest`-style verification.

Optional local evaluation:

- Provide a command that can read local `documents/` when present.
- Do not require local document evaluation in CI.
- Do not commit local reports containing real policy text.
- Write reports to an ignored report directory.

Evaluation report contents:

- Query.
- Expected section, title, or keyword.
- Retrieved chunks.
- Rank of expected evidence.
- Vector score.
- BM25 score.
- Fusion score.
- Matched terms.
- Pass/fail.
- Answer-guard warnings or block status.

## Data Flow

### Upload and Indexing

1. Parse uploaded PDF into pages.
2. Chunk pages into `DocumentChunk` objects.
3. Generate embeddings and build the existing in-memory vector index.
4. Build a BM25 index from the same chunks.
5. Store a policy hybrid retriever in the current Streamlit session.
6. Lazily build the built-in hybrid retriever only when a question needs terminology or background support.

### Question Answering

1. Receive original user question.
2. Rewrite the query into expanded retrieval queries.
3. Search the user policy with vector and BM25 retrieval.
4. Fuse results with reciprocal rank fusion.
5. If no user policy evidence is found, refuse without calling the chat model.
6. If the question needs background explanation, search the built-in dataset.
7. Build the prompt with separate source sections.
8. Generate the answer.
9. Run the answer guard.
10. Return one of:
    - normal answer
    - answer with warnings
    - safe replacement answer after blocking

## Configuration

Suggested environment variables:

- `INSURANCE_RAG_RETRIEVAL_MODE`: `hybrid` or `vector`, default `hybrid`
- `INSURANCE_RAG_RRF_K`: default `60`
- `INSURANCE_RAG_QUERY_REWRITE_LLM`: default `false`
- `INSURANCE_RAG_ANSWER_GUARD_LLM`: default `false`
- `INSURANCE_RAG_EVAL_REPORT_DIR`: default `eval_reports/`

The first implementation should keep LLM rewriting and LLM guard disabled by default. Rule-based rewriting and programmatic guard checks should be the default path.

## Error Handling

- Query rewriting failure: fall back to the original question and show a warning.
- BM25 index construction failure: fall back to vector retrieval and show a warning.
- BM25 search failure: fall back to vector-only results and show a warning.
- Vector retrieval failure for user policy: return an evidence-limited refusal.
- Built-in retrieval failure: continue with user policy evidence only.
- Answer guard runtime failure: keep the model answer but show an `自检未完成` warning.
- Answer guard block: replace the model answer with a safe evidence-limited response.
- Offline evaluation case failure: record the failure and continue the batch.

## Testing Strategy

### Unit Tests: Query Rewriting

- `赔不赔` expands to coverage, exclusion, and claim-condition terms.
- `等多久生效` expands to waiting-period and effective-date terms.
- Ordinary questions preserve the original query.
- Duplicate expanded queries are removed.
- Rewriting failures fall back to the original query.

### Unit Tests: Hybrid Retrieval

- BM25 finds exact insurance terms.
- Vector and BM25 results are fused.
- The same chunk is not returned multiple times.
- `top_k <= 0` returns an empty result.
- BM25 failure degrades to vector-only retrieval.
- Score explanations are preserved.

### Unit Tests: Answer Guard

- No user policy citation plus specific policy fact returns `block`.
- Final claim wording returns `block`.
- Built-in context phrased as user policy evidence returns `block` or `warn`.
- Low citation count returns `warn`.
- Normal grounded explanation returns `pass`.

### RAG Chain Tests

- The chain calls query rewriting.
- The chain uses hybrid retrieval.
- Built-in retrieval still only triggers for terminology or background questions.
- Guard `block` suppresses the original model answer.
- Guard `warn` preserves the model answer and returns warnings.
- Retrieval explanations are returned in `AnswerPayload`.

### Offline Evaluation Tests

- Synthetic cases run without access to local `documents/`.
- Synthetic evaluation reports expected evidence rank and pass/fail.
- Optional local document evaluation skips gracefully when `documents/` is absent.
- Real-document reports are written only to ignored output directories.

## Git Hygiene

Implementation should update `.gitignore` to ignore:

- `eval_reports/`
- `.rag_eval_cache/`

The existing `documents/` ignore rule must remain in place.

## Acceptance Criteria

- Hybrid retrieval is the default retrieval mode, with vector-only fallback.
- Query rewriting improves recall for common insurance questions without changing the user-facing question.
- Answers are checked after generation and serious boundary violations are blocked.
- UI exposes retrieval details in a collapsed section.
- Synthetic evaluation is reproducible from the repository alone.
- Optional local evaluation can run against `documents/` when available.
- Tests cover rewriting, hybrid retrieval, answer guard, RAG orchestration, and evaluation behavior.
- No user-uploaded document content, local `documents/` content, or real-document evaluation report is committed.
