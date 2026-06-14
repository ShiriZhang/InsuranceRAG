# RAG Accuracy Advanced Design

Date: 2026-06-14

## Overview

This phase upgrades InsuranceRAG from a working RAG demo into a more verifiable policy-question answering system. The target is not to add many surface features, but to make the system more reliably find the correct policy clauses and prove that key answer facts are supported by user-policy citations.

The product boundary remains unchanged:

- The assistant explains insurance policy clauses in Chinese.
- The uploaded user policy is the primary evidence source.
- Built-in documents are background material only.
- The system does not make final claim approval, denial, legal, medical, financial, or underwriting conclusions.
- Uploaded user documents and generated user-policy indexes remain session-only.

This phase adds one product-quality loop:

1. Structured clause parsing.
2. More accurate retrieval through rule reranking.
3. Fact-level citation verification.
4. Hard negative evaluation.

## Goals

- Improve clause metadata by identifying common clause numbers and headings.
- Use clause structure, question intent, exact terms, and numbers to rerank retrieved chunks.
- Verify that important policy facts in an answer are directly supported by user-policy citations.
- Show users a concise evidence-verification result below each answer.
- Add synthetic and local hard negative evaluations to catch similar-but-wrong evidence.

## Non-Goals

- Do not implement full chapter/article/paragraph tree parsing.
- Do not implement complex PDF table structure extraction.
- Do not introduce LLM reranking.
- Do not introduce a local cross-encoder reranker.
- Do not support multi-document upload in this phase.
- Do not persist user-uploaded documents, parsed user text, embeddings, or chat history.
- Do not commit local `documents/` content, local hard negative reports, or real policy text.

## User Experience Changes

Normal users continue to see:

- Chinese answer.
- User policy citations.
- Built-in background citations when applicable.
- Collapsed retrieval details.

This phase adds a new collapsed section under each assistant answer:

- `证据核验结果`

For each extracted key fact, this section shows:

- Fact text.
- Verification status.
- Supporting citation when found.
- Risk reason when evidence is incomplete or missing.
- System handling: pass, warning, or blocked.

Examples:

- Fact: `等待期是90天`
- Status: supported
- Supporting citation: page 6, `等待期`, original excerpt
- Reason: same user-policy citation contains both `等待期` and `90天`

If a severe unsupported fact is found, the original model answer is blocked and replaced with a safe evidence-limited answer.

## Architecture

The implementation should stay modular. `rag_chain.py` remains the orchestration layer. New parsing, reranking, verification, and evaluation behavior should live in focused modules.

### `clause_parser.py`

New module for identifying clause numbers and headings from page or chunk text.

The parser does not claim to cover all insurance PDF formats. It supports a configurable and extensible set of high-frequency patterns, with confidence levels and fallback behavior.

Clause number examples:

- `第六条`, `第十条`, `第二十三条`
- `第6条`, `第 10 条`
- `1.1`, `2.3`, `2.3.1`
- `（一）`, `（二）`
- `（1）`, `(1)`, `1）`
- `一、`, `二、`
- `1、`, `1.`
- Combined number and heading lines such as `2.1 保险责任`

Heading examples:

- `等待期`
- `保险责任`
- `责任免除`
- `除外责任`
- `免责条款`
- `保险期间`
- `保险金额`
- `基本保险金额`
- `保险金给付`
- `给付条件`
- `豁免保险费`
- `犹豫期`
- `宽限期`
- `合同解除`
- `合同效力`
- `释义`
- `疾病定义`
- `重大疾病定义`
- `轻症疾病`
- `中症疾病`
- `身故保险金`
- `全残保险金`

Recognition levels:

- High confidence: the same line contains a clear clause number and a known heading, such as `第六条 等待期` or `2.3 责任免除`.
- Medium confidence: a line contains a known heading, or a nearby number and known heading can be associated.
- Low confidence: fallback from the existing keyword-based section inference, or `未识别条款标题`.

Suggested output:

- `clause_id`: for example `第六条`, `2.3`, `（一）`
- `heading_text`: original heading line, for example `第六条 等待期`
- `section_title`: normalized title, for example `等待期`
- `heading_confidence`: `high`, `medium`, or `low`
- `heading_source`: `line_pattern`, `known_title`, or `fallback`

The parser should avoid treating directory lines, page headers, and footer-like lines as high-confidence clause headings.

### `chunker.py`

`chunker.py` continues to generate chunks, but calls `clause_parser.py` to attach clause metadata.

`DocumentChunk` should be extended with optional fields:

- `clause_id`
- `heading_text`
- `heading_confidence`
- `heading_source`

`section_title` remains the simplified title used by UI, retrieval explanations, and evaluation reports.

If clause parsing succeeds with high or medium confidence, chunk metadata should use the parsed result. If parsing fails, the chunker falls back to the current keyword-based behavior so existing functionality keeps working.

### `rule_reranker.py`

New module that reranks candidates after hybrid retrieval.

Inputs:

- Original user question.
- Query rewrite result.
- Hybrid retrieval candidates.
- Candidate chunk clause metadata.

Outputs:

- Reranked candidates.
- Rerank score per candidate.
- Rerank reasons per candidate.

Positive rerank reasons:

- `title_intent_match`: question intent matches section title, such as a waiting-period question matching `等待期`.
- `exact_term_match`: exact policy term match.
- `number_match`: matching number, period, age, or amount.
- `clause_heading_match`: strong clause heading match.
- `positive_fact_type_match`: coverage question matches insurance responsibility clauses.
- `exclusion_fact_type_match`: exclusion or non-payment question matches exemption clauses.

Negative rerank reasons:

- `negative_title_mismatch`: question asks one clause type but chunk mainly belongs to another.
- `directory_like_chunk`: chunk looks like a table of contents or page navigation text.
- `low_heading_confidence`: heading metadata has low confidence.

The reranker does not replace hybrid retrieval. It only reorders the top candidates. If it fails, the system falls back to the original hybrid order and adds a warning.

### `citation_verifier.py`

New module for fact-level citation verification.

It extracts important policy facts from an answer and checks whether those facts are directly supported by user-policy citations.

Fact types:

- Numeric, period, age, or amount facts, such as `等待期是90天` or `保险金额为10万元`.
- Coverage and exclusion facts, such as `酒后驾驶属于责任免除`.
- Subject facts, such as `投保人可豁免保险费`.
- Source facts, such as `你的保单写明...`.

Suggested output per fact:

- `fact_text`
- `fact_type`
- `status`: `supported`, `partially_supported`, or `unsupported`
- `severity`: `info`, `warn`, or `block`
- `supporting_citation_ids`
- `reason`

`answer_guard.py` should call `citation_verifier.py`. Severe unsupported facts block the model answer. Minor uncertainty produces warnings.

### `evaluation.py` and `scripts/evaluate_rag.py`

The existing evaluation path should be extended with hard negative evaluation.

New repo-contained file:

- `evals/hard_negative_cases.json`

Synthetic hard negative evaluation should be stable and suitable for CI. It should use artificial insurance text, not real policy text.

Local hard negative evaluation should be optional and run against local `documents/`. It should write reports to an ignored evaluation directory and must not commit real document content.

Hard negative reports should include:

- Question.
- Expected positive chunk.
- Hard negative chunks.
- Positive rank.
- Whether a hard negative outranked the positive chunk.
- Rerank reasons.
- Verifier result.
- Pass/fail status.

### `app.py`

`app.py` should only render verification results. It should not implement verification logic.

Under each assistant answer, add a `证据核验结果` expander showing:

- Extracted facts.
- Verification status.
- Supporting citation.
- Risk reason.
- Handling result: pass, warning, or blocked.

The existing `检索依据详情` expander remains and should include rerank score and rerank reasons.

## Data Model Changes

Add models in `models.py`:

- `ClauseMetadata`
- `RerankExplanation`
- `VerifiedFact`
- `CitationVerificationResult`

Extend `DocumentChunk`:

- `clause_id`
- `heading_text`
- `heading_confidence`
- `heading_source`

Extend `RetrievalExplanation`:

- `rerank_score`
- `rerank_reasons`

Extend `AnswerPayload`:

- `citation_verification`

These models allow UI, evaluation, and RAG orchestration to share structured data instead of parsing strings.

## RAG Data Flow

### Upload and Parsing

1. `document_loader.py` parses PDF pages.
2. `chunker.py` generates chunks.
3. `chunker.py` calls `clause_parser.py` to attach clause metadata.
4. The app records parsing quality:
   - total pages
   - chunk count
   - high, medium, and low confidence heading counts
   - unknown section title rate
   - OCR page count and warnings
5. If low-confidence heading rate is high, the UI shows a clear Chinese warning.

The session-only privacy model remains unchanged.

### Retrieval and Reranking

1. `query_rewriter.py` expands the user question and detects coarse intent.
2. `HybridRetriever` performs embedding + BM25 + RRF retrieval.
3. `rule_reranker.py` reranks the top candidates.
4. Final top-k user-policy chunks are used in the prompt.
5. `RetrievalExplanation` records hybrid scores, rerank scores, matched terms, and rerank reasons.

The reranker should not discard all low-rule-score chunks. It should reorder candidates while preserving recall. If reranking fails, the system uses the original hybrid order.

### Answer Generation and Verification

1. `rag_chain.py` builds user-policy context from reranked chunks.
2. Built-in background is retrieved only when user-policy evidence exists and the question needs terminology or background context.
3. The prompt continues to require:
   - Chinese answer.
   - Clause explanation only.
   - No final claim decision.
   - User policy as primary evidence.
   - Built-in documents as background only.
   - Explicit uncertainty when evidence is insufficient.
4. The chat model generates an answer.
5. Citations are generated.
6. `answer_guard.py` invokes `citation_verifier.py`.
7. The answer is returned as pass, warning, or blocked.

## Verification Severity Rules

### Block

The original model answer is blocked when:

- It contains a numeric, period, age, or amount fact that is not supported by citations.
- It claims an action, condition, disease, or fee belongs to insurance responsibility or responsibility exemption without citation support.
- It states a subject-specific fact, such as policyholder waiver versus insured-person waiver, without support.
- It presents built-in background content as user-policy fact.
- It makes a final claim decision such as `一定赔`, `肯定不赔`, or `保险公司必须赔`.

Example:

- Answer says `等待期是90天`.
- Citations contain `等待期为30天` and `保险期间为90天`.
- Result: block.

### Warn

The answer is preserved with warnings when:

- Citations partially support a fact but the excerpt is incomplete.
- The answer is a broad explanation rather than a specific policy fact.
- User-policy citation count is low.
- Top retrieval score is weak.
- OCR quality warnings exist.
- Heading confidence is low.
- The verifier is uncertain but does not find a clear contradiction.

### Pass

The answer passes when all extracted key facts are supported by user-policy citations and no final claim decision or source confusion is detected.

## Hard Negative Evaluation

### Synthetic Hard Negative Evaluation

Synthetic evaluation reads `evals/hard_negative_cases.json`.

Each case should contain:

- Question.
- Positive chunk.
- Hard negative chunks.
- Expected positive rank threshold.
- Optional answer text for verifier testing.

The runner executes:

1. Query rewrite.
2. Hybrid retrieval.
3. Rule rerank.
4. Positive-rank check.
5. Citation verifier check when answer text is provided.

Synthetic hard negative failures should make the command return non-zero.

Required synthetic categories:

- Numeric confusion.
- Clause type confusion.
- Subject confusion.
- Source confusion.

### Local Hard Negative Evaluation

Local evaluation reads local `documents/` when present.

It should automatically build hard negatives from real documents where possible:

- `等待期` versus `犹豫期` versus `保险期间`.
- `保险责任` versus `责任免除`.
- `投保人豁免` versus `被保险人豁免`.
- Similar but different numbers, periods, ages, or amounts.

The report should include:

- sampled documents
- parsed documents
- hard negative case count
- positive top1/top3
- hard negative outrank rate
- unknown and low-confidence heading rate
- verifier pass/warn/block counts

First version does not need 100% local pass rate. It must expose useful failure cases.

## Error Handling

`clause_parser.py` failure:

- Do not block upload or answering.
- Fallback to existing section inference.
- Mark heading confidence as low.

`rule_reranker.py` failure:

- Do not block answering.
- Use original hybrid ranking.
- Add a user-facing or debug warning.
- Record failure in evaluation reports.

`citation_verifier.py` runtime exception:

- Fail closed and block the original answer.
- Show `证据核验未完成，请结合原文核对`.

Verifier uncertainty:

- Do not block.
- Add a warning.

Hard negative evaluation failure:

- Case-level failures should be recorded.
- Synthetic hard negative failures return non-zero.
- Explicit local-only evaluation with missing `documents/` returns non-zero.

OpenAI API failure:

- Embedding failure prevents index construction and shows a Chinese setup/network/quota hint.
- Chat failure preserves parsed document state and asks the user to retry.
- Developer stack traces should not be shown in normal UI.

## Configuration

Add or extend environment variables:

- `INSURANCE_RAG_RERANK_ENABLED`
  - default: `true`
- `INSURANCE_RAG_RERANK_TOP_N`
  - default: `20`
- `INSURANCE_RAG_VERIFIER_ENABLED`
  - default: `true`
- `INSURANCE_RAG_VERIFIER_STRICTNESS`
  - default: `balanced`
  - choices: `strict`, `balanced`, `warn_only`
- `INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD`
  - default: `0.35`
- `INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT`
  - default: `20`

Existing variables remain:

- `INSURANCE_RAG_RETRIEVAL_MODE`
- `INSURANCE_RAG_RRF_K`
- `INSURANCE_RAG_EVAL_REPORT_DIR`

## Testing Strategy

### Unit Tests

`clause_parser.py`:

- Recognizes `第六条 等待期`.
- Recognizes `第 10 条 责任免除`.
- Recognizes `2.3 保险责任`.
- Recognizes standalone heading `保险金额`.
- Avoids or lowers confidence for directory-like lines.
- Returns low-confidence fallback when no heading is detected.

`chunker.py`:

- Preserves clause metadata.
- Uses high-confidence parsed heading before keyword inference.
- Fallback keeps existing `section_title` behavior.

`rule_reranker.py`:

- Waiting-period questions rank waiting-period chunks above insurance-period and hesitation-period chunks.
- Exclusion questions rank exemption chunks above coverage chunks.
- Number matches add score.
- Heading mismatch reduces score.
- Rerank reasons are included in explanations.
- Reranker failure falls back cleanly.

`citation_verifier.py`:

- Supports `等待期90天` when the same citation directly supports it.
- Blocks `等待期90天` when evidence is `等待期30天` plus `保险期间90天`.
- Supports responsibility exemption facts when citation directly supports them.
- Blocks built-in background content presented as user-policy fact.
- Detects subject confusion such as policyholder versus insured person.
- Produces warnings for partial support.

`answer_guard.py`:

- Calls citation verifier.
- Blocks when verifier returns block.
- Preserves answer when verifier returns warning.
- Fails closed on verifier runtime exception.

`evaluation.py`:

- Runs synthetic hard negative eval.
- Computes positive rank.
- Fails when hard negative outranks positive.
- Handles missing local documents explicitly.
- Includes rerank reasons and verifier result in reports.

`app.py`:

- Renders evidence verification expander.
- Renders warning and block states.
- Does not fail when verification result is absent.

### Regression Commands

Required:

```powershell
pytest tests -q -p no:cacheprovider
python scripts\evaluate_rag.py --synthetic
python scripts\evaluate_rag.py --hard-negative
```

Optional local:

```powershell
python scripts\evaluate_rag.py --local-hard-negative documents
```

## Acceptance Criteria

- Uploaded PDFs produce chunks with clause metadata where headings are detectable.
- The app reports heading confidence quality.
- Rule rerank is enabled by default and records score/reason explanations.
- Waiting-period, exemption, coverage, amount, period, and waiver questions benefit from rerank explanations.
- Answer payloads include citation verification results.
- UI shows `证据核验结果`.
- Severe unsupported facts are blocked.
- Mild uncertainty produces warnings.
- Synthetic hard negative eval is repo-contained and passes.
- Local hard negative eval can run against local `documents/` and writes an ignored report.
- Existing tests and synthetic eval still pass.
- No real local document content or local eval report is committed.

## Git and Privacy

Keep these ignored and uncommitted:

- `documents/`
- `eval_reports/`
- `.rag_eval_cache/`
- local hard negative reports
- any generated report containing real policy text

Only synthetic hard negative cases and source code should be committed.
