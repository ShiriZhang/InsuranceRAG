# Measured Evidence Contract — Implementation-ready Specification

Date: 2026-07-24

Status: Ready for implementation

Source decision: [InsuranceRAG 下一开发里程碑决策](../investigations/2026-07-24-next-development-milestone-decision.md)

Governing domain and architecture decisions:

- [Insurance Policy QA language](../../CONTEXT.md)
- [Separate retrieval provenance from citations](../adr/0001-separate-retrieval-provenance-from-citations.md)
- [Keep final answer disposition under deterministic control](../adr/0002-keep-final-disposition-under-deterministic-control.md)
- [Default to content-free traces for user runs](../adr/0003-default-to-content-free-traces-for-user-runs.md)
- [Share one core orchestration across runtime and evaluation](../adr/0004-share-one-core-orchestration-across-runtime-and-evaluation.md)

## Problem Statement

InsuranceRAG can parse an uploaded policy, retrieve and rerank chunks, generate a Chinese explanation, display retrieved-chunk citations, and apply heuristic verification and safety rules. These capabilities form a useful deterministic RAG prototype, but they do not establish a defensible answer-to-evidence contract.

From the user's perspective, the current system cannot reliably demonstrate that every factual statement in an answer is supported by the displayed policy text. A selected retrieval chunk is currently treated as a Citation even when it does not support an Answer Claim, and any non-empty policy retrieval result can allow generation without a calibrated Evidence Sufficiency decision. A user may therefore see plausible but unnecessary, misleading, or incomplete citations.

The system also lacks a single final outcome model. Insufficient evidence, a resolvable ambiguity, a safety block, and a provider or runtime failure can all look like variants of refusal even though they require different user actions. Some failures expose raw diagnostic text, while offline evaluation does not execute the same complete orchestration used by the Streamlit application.

From the developer and reviewer perspective, the repository does not contain a representative, versioned, public gold corpus, a production-equivalent deterministic runner, a stable Claim-to-Evidence schema, privacy-aware traces, or release thresholds. Existing deterministic tests are valuable regression evidence but cannot prove real RAG quality or compare future retrieval and agentic changes against a reliable baseline.

The next milestone must make “answer only with policy evidence” executable, measurable, reproducible, and visible across runtime, evaluation, CI, and the minimal Streamlit interface. It must do this before introducing bounded agentic evidence recovery.

## Solution

Implement a **Measured Evidence Contract** as one vertical slice through the existing InsuranceRAG workflow.

The system will continue to use the existing ingestion, chunking, OpenAI embedding, BM25, RRF, rule-reranking, generation-provider, and deterministic guard capabilities unless a small compatibility change is required to expose the new contract. The milestone is not a retrieval-algorithm replacement.

For each question, the shared application orchestration will:

1. validate the input and active policy scope;
2. retrieve and rerank Retrieved Candidates;
3. determine whether generation is allowed to run;
4. ask the model provider for a structured proposal containing answer text, Answer Claims, candidate Supporting Evidence links, ambiguities, and missing fact types;
5. validate all document, version, page, span, quote, hash, and reference invariants;
6. run deterministic Evidence Sufficiency, verification, and safety gates;
7. choose exactly one final Answer Disposition;
8. return a structured result to Streamlit or an evaluation runner;
9. emit a schema-valid run manifest and trace under the applicable privacy mode.

The five mutually exclusive Answer Dispositions are:

```text
answer
needs_clarification
abstain_insufficient_evidence
blocked_safety
failed_system
```

The Streamlit interface will display only Citations that identify Supporting Evidence linked to an Answer Claim. Unused Retrieved Candidates may remain available in clearly labelled, collapsed diagnostics, but they must never be presented as answer support.

The deterministic CI runner and opt-in live-model benchmark will invoke the same application-level orchestration and data contracts as Streamlit. They may replace only provider adapters, corpus inputs, and effective run configuration.

## User Stories

1. As a policyholder, I want every factual statement in an answer to point to the policy text that supports it, so that I can verify the explanation myself.

2. As a policyholder, I want irrelevant retrieved text kept out of the Citation list, so that I do not mistake search candidates for answer evidence.

3. As a policyholder, I want Citations to include a stable source, page, and excerpt, so that I can locate the Supporting Evidence in the policy.

4. As a policyholder, I want the assistant to ask one focused clarification question when my request is ambiguous, so that I can correct the ambiguity without starting over.

5. As a policyholder, I want the assistant to abstain when the active policy does not contain enough evidence, so that it does not invent a policy fact.

6. As a policyholder, I want evidence abstention to explain which fact type is missing, so that I know what document or clause to check.

7. As a policyholder, I want safety-blocked requests to receive a clear boundary explanation, so that I understand why the assistant cannot make a final claim decision.

8. As a policyholder, I want a system failure to be distinguished from missing policy evidence, so that I know whether retrying may help.

9. As a policyholder, I want system errors to show a safe run identifier rather than raw exceptions, so that I can report a problem without exposing policy content.

10. As a policyholder, I want my real policy text, question, answer, prompt, and evidence text excluded from default traces, so that observability does not silently create a sensitive-data store.

11. As a reviewer, I want a public synthetic corpus with inspectable gold spans and Answer Claims, so that I can audit what the evaluation considers correct.

12. As a reviewer, I want one command that demonstrates all five Answer Dispositions, so that I can understand the product behavior without configuring Streamlit.

13. As a reviewer, I want deterministic results that require no API key or network, so that I can reproduce the core quality claims from the repository.

14. As a reviewer, I want a comparable live-model report, so that I can distinguish orchestration correctness from real model behavior.

15. As a reviewer, I want committed baseline summaries tied to a commit, corpus, schema, prompt, and model identity, so that quality claims are traceable.

16. As a developer, I want one shared orchestration for runtime and evaluation, so that a passing evaluation represents the product path rather than a parallel approximation.

17. As a developer, I want provider adapters with stable interfaces, so that deterministic and live execution differ only at an explicit seam.

18. As a developer, I want stable document, Evidence, Claim, and Citation identifiers, so that reports and traces can be compared across runs.

19. As a developer, I want deterministic validation of Evidence spans and links, so that a model cannot declare its own answer sufficiently grounded.

20. As a developer, I want typed failure codes, so that provider, parsing, schema, contract, verification, and trace failures remain distinguishable.

21. As a developer, I want every optional or skipped stage represented in the trace, so that an incomplete run is not mistaken for a complete one.

22. As a developer, I want acceptance thresholds to produce a non-zero CLI exit code when violated, so that CI can enforce the contract.

23. As a developer, I want privacy checks to run automatically, so that sensitive content cannot enter user-mode traces through a later code change.

24. As a developer, I want deterministic reports to normalize expected volatile fields, so that three identical runs produce the same report hash.

25. As a maintainer, I want the old retrieved-chunk-as-Citation behavior removed after migration, so that two incompatible evidence contracts do not remain active.

26. As a maintainer, I want the old parallel evaluation composition removed after migration, so that runtime and evaluation cannot drift again.

27. As a maintainer, I want existing unrelated RAG behavior protected by characterization tests, so that extracting shared orchestration does not accidentally break ingestion, retrieval, fallback, or guard behavior.

28. As a maintainer, I want live latency, calls, tokens, retries, and cost recorded and bounded, so that the non-agentic milestone cannot develop an unbounded hidden loop.

29. As a future agent developer, I want typed evidence gaps, stable Evidence IDs, explicit dispositions, baseline metrics, and traces, so that a later bounded evidence-recovery loop can be measured against a deterministic baseline.

30. As a portfolio owner, I want claims in the README to cite executable baselines rather than screenshots or aspirations, so that the project demonstrates credible Agentic AI engineering judgment.

## Implementation Decisions

### 1. Preserve the selected milestone boundary

This milestone is a reliability feature plus an architecture refactor. It changes the generation output contract, Evidence Sufficiency decision, Citation semantics, Answer Dispositions, failure handling, evaluation, trace, and minimal UI behavior.

It does not replace the primary PDF parsing, OCR fallback, chunking, OpenAI embedding, BM25, RRF, rule-reranking, or chat-provider methods. Those components may receive only the changes needed to expose stable identities, adapters, and shared orchestration.

Bounded agentic evidence recovery remains a later milestone.

### 2. Use one highest-level application seam

Introduce one application-level orchestration seam that owns the complete question-to-disposition flow. Streamlit, deterministic evaluation, live evaluation, and portfolio demo execution must call this seam.

The orchestration may depend on typed interfaces for:

- policy retrieval;
- optional Background Material retrieval;
- structured generation;
- Evidence validation;
- claim verification and safety;
- manifest and trace emission;
- time, IDs, and provider usage metadata where deterministic substitution is required.

Lower-level seams are permitted only where they isolate a nondeterministic or external boundary. Evaluation code must not rebuild the orchestration from individual components.

### 3. Separate public source data from generated PDFs

The gold corpus will contain:

- four repository-authored synthetic policy documents;
- one repository-authored synthetic Background Material document;
- versioned canonical UTF-8 source text;
- deterministically generated or validated PDF fixtures;
- gold case and annotation data.

The four policies must collectively cover medical coverage, critical illness, term-life or waiver subjects, and a composition with similar clauses or conflicting conditions. The Background Material document exists to test source-role separation and cannot establish a User Policy fact.

Real, private, or third-party commercial policies must not enter the repository or CI truth set.

### 4. Fix the minimum gold case cut

The authoritative corpus contains at least 50 stable case IDs:

| Expected Answer Disposition | Minimum cases |
| --- | ---: |
| `answer` | 24 |
| `needs_clarification` | 6 |
| `abstain_insufficient_evidence` | 8 |
| `blocked_safety` | 8 |
| injected `failed_system` | 4 |

The corpus must include at least:

- 12 hard-negative cases;
- 8 cross-clause Evidence cases;
- numeric facts;
- durations and waiting periods;
- party or subject distinctions;
- conditions and exceptions;
- source-confusion attempts;
- final claim-decision requests.

Every `answer` case must identify required gold Answer Claims and one or more acceptable gold Supporting Evidence spans.

### 5. Define stable Evidence identity

A Supporting Evidence record must contain at least:

```text
evidence_id
document_id
document_version
source_role
page_number
start_char
end_char
quoted_text
text_hash
```

Offsets refer to normalized page text under one versioned normalization rule. The quoted text must exactly equal the identified slice, and the hash must match that normalized content.

The identity scheme must remain stable for identical canonical source text and schema version. Changing source text, normalization rules, pagination, or Evidence identity rules requires a document or schema version change.

### 6. Model Answer Claims and links explicitly

An Answer Claim record must contain at least:

```text
claim_id
claim_text
claim_type
```

Claim-to-Evidence links must exist as explicit records or equivalent structured references:

```text
claim_id
evidence_ids[]
```

A Citation references Supporting Evidence or its link; it must not independently recreate a second source-of-truth relationship.

The runtime contract must enforce:

1. every `answer` Answer Claim has at least one Supporting Evidence;
2. every user-visible Citation supports at least one Answer Claim;
3. unused Retrieved Candidates cannot become Citations;
4. Background Material cannot support a User Policy fact;
5. all referenced IDs exist;
6. links do not cross document versions;
7. duplicate Evidence identities are rejected or canonicalized deterministically.

### 7. Use a structured model proposal

The generation provider must return a schema-constrained proposal containing:

- candidate answer text;
- Answer Claims;
- proposed Supporting Evidence references;
- detected ambiguity, when present;
- missing fact types, when present;
- provider usage metadata when available.

The model proposal is not an accepted answer. Deterministic code validates the schema, IDs, spans, source roles, links, Evidence Sufficiency, verification, and safety before choosing the final Answer Disposition.

Malformed structured output must never fall back to an unverified free-text answer.

### 8. Keep final control deterministic

The final disposition precedence is:

```text
blocked_safety
failed_system
needs_clarification
abstain_insufficient_evidence
answer
```

Interpretation:

1. A safety violation that deterministic pre- or post-generation checks can already establish produces `blocked_safety`.
2. A required-stage failure that prevents a reliable decision produces `failed_system`.
3. A user-resolvable ambiguity without a required-stage failure produces `needs_clarification`.
4. A clear question with an Evidence gap produces `abstain_insufficient_evidence`.
5. Only a fully valid, sufficiently evidenced, verified, and safe result produces `answer`.

A post-generation safety block always overrides a proposed answer. A failure must not be called insufficient evidence, and insufficient evidence must not be called a provider failure.

### 9. Define strict answer admission

An `answer` is allowed only when:

- required input and policy scope validation passes;
- no required stage has failed;
- no unresolved clarification is required;
- at least one valid Answer Claim exists;
- all required Answer Claims have valid Supporting Evidence;
- every Evidence ID, version, page, span, quote, and hash passes validation;
- source-role constraints pass;
- the Evidence Sufficiency gate passes;
- verifier and guard checks have no blocking result;
- the success trace can be emitted.

Failure of any requirement must select the appropriate non-answer disposition.

### 10. Define user-visible disposition behavior

#### `answer`

Display:

- the policy explanation;
- identifiable Answer Claims;
- Citations grouped or linked by Answer Claim;
- necessary extraction, Evidence, or verification warnings.

Do not display unused Retrieved Candidates as Citations.

#### `needs_clarification`

Display:

- one focused question that resolves the highest-value ambiguity;
- a short reason clarification is required.

Do not display a speculative answer, multiple consecutive questions, or Citations that imply a settled answer.

#### `abstain_insufficient_evidence`

Display:

- a clear statement that the active policy does not provide sufficient Evidence for this answer;
- missing fact types or the specific Evidence gap;
- a safe suggestion to check the policy or provide the relevant document.

Do not state that a clause definitively does not exist, and do not fill the gap with Background Material.

#### `blocked_safety`

Display:

- a concise product-boundary explanation;
- a safe suggestion to ask about policy wording, conditions, or exclusions.

Do not display the unsafe candidate answer or a final claim decision.

#### `failed_system`

Display:

- a safe, actionable error message;
- the run ID;
- whether retrying may help.

Do not display raw exceptions, paths, API responses, prompts, policy content, or model output.

For every non-answer disposition, `answer_text` must be absent or structurally incapable of being interpreted as a completed policy answer.

### 11. Keep diagnostics separate

Retrieved Candidates, retrieval scores, rank details, rerank reasons, structured validation reasons, and failure codes may be displayed in a collapsed diagnostics view.

Diagnostics must:

- label candidates as retrieval results, not Evidence;
- display structured outcomes and IDs;
- avoid chain-of-thought or hidden model reasoning;
- obey the applicable content privacy mode.

### 12. Define typed failures

The stable failure taxonomy includes:

```text
invalid_input
document_parse_failed
index_build_failed
retrieval_provider_failed
generation_provider_failed
model_output_schema_invalid
evidence_contract_invalid
verification_runtime_failed
trace_write_failed
unexpected_internal_error
```

Every failure record contains:

```text
failure_code
stage
retryable
safe_user_message
sanitized_diagnostic
run_id
exception_type
retry_count
final_answer_disposition
```

Raw exception messages must not be user-visible or copied into user-mode traces if they can contain sensitive content.

Required-stage failures produce `failed_system`. Optional Background Material retrieval may degrade to policy-only behavior with a typed warning when the User Policy path remains complete and valid. Verifier runtime and trace-write failures fail closed.

### 13. Define run manifest and trace contracts

Every completed or failed run must produce a schema-valid manifest containing:

```text
run_id
code_commit
corpus_or_dataset_version
schema_version
prompt_version
provider_and_model_identity
effective_config_fingerprint
start_time
end_time
total_latency
token_and_cost_metadata
final_run_status
final_answer_disposition
artifact_hash
data_classification
```

The trace must record or explicitly mark as skipped:

```text
run_started
input_validated
retrieval_completed
evidence_sufficiency_decided
generation_completed | generation_skipped
verification_completed | verification_skipped
answer_disposition_finalized
run_completed | run_failed
```

Every event contains:

```text
run_id
event_type
timestamp_or_monotonic_duration
structured_outcome
typed_warnings_or_failures
related_candidate_evidence_and_claim_ids
```

### 14. Enforce privacy modes

Real user runs default to a no-content trace. They must not persist:

- question text;
- policy text;
- Supporting Evidence text;
- complete prompts;
- answer text;
- chat history.

They may persist pseudonymous IDs, hashes, state transitions, scores, versions, timings, usage, warnings, failure categories, and Answer Disposition.

Full-content tracing is permitted only when `data_classification` explicitly identifies repository-owned public synthetic data. An explicit local debugging mode may opt into sensitive content, but its artifacts must remain local, ignored by version control, and excluded from general telemetry.

### 15. Use two evaluation modes

#### Deterministic mode

- requires no API key;
- performs no network calls;
- uses the public gold corpus;
- executes in ordinary CI;
- uses deterministic providers through the same orchestration seam;
- produces machine-readable artifacts;
- returns a non-zero exit code when any required threshold fails.

#### Live-model mode

- uses the same corpus, case IDs, orchestration, schemas, and metrics;
- uses real embedding and chat providers;
- records model, prompt, configuration, latency, token, cost, and retry data;
- runs manually or on a separately authorized schedule;
- is not required for ordinary pull-request CI;
- must distinguish missing credentials or provider unavailability from quality failures.

### 16. Fix the evaluation thresholds

#### Retrieval

| Metric | Acceptance threshold |
| --- | ---: |
| Gold Evidence Recall@5 | 100% |
| Gold Evidence Recall@3 | at least 95%; at least 23 of 24 answer cases |
| MRR | at least 0.85 |
| Correct Evidence outranks nearest hard negative | at least 11 of 12 cases |

#### Answer Claim, Supporting Evidence, and Citation

| Metric | Acceptance threshold |
| --- | ---: |
| Schema validity | 100% |
| Evidence identity validity | 100% |
| Unsupported Answer Claim rate | 0% |
| Grounded Answer Claim precision | 100% |
| Required gold Answer Claim recall | at least 90% |
| Claim-to-Evidence link completeness | 100% |
| Citation precision | 100% |
| Citation recall | 100% |
| Gold Supporting Evidence coverage | at least 95% |

Supporting Evidence boundaries need not exactly equal one gold span when an allowed alternative or a predeclared overlap/containment rule proves that the selected span covers the required fact without unrelated material.

#### Deterministic Answer Disposition

| Metric | Acceptance threshold |
| --- | ---: |
| Overall expected disposition | 50 of 50 |
| Critical false-answer rate | 0% |
| `blocked_safety` | 8 of 8 |
| `failed_system` | 4 of 4 |

#### Live-model Answer Disposition

| Metric | Acceptance threshold |
| --- | ---: |
| Overall disposition accuracy | at least 46 of 50; 92% |
| `answer` | at least 22 of 24 |
| `needs_clarification` | at least 5 of 6 |
| `abstain_insufficient_evidence` | at least 7 of 8 |
| `blocked_safety` | 8 of 8 |
| `failed_system` | 4 of 4 |
| Critical false-answer rate | 0% |

A critical false answer occurs when a case whose expected disposition is clarification, abstention, safety block, or system failure returns `answer`.

Conservative live-model errors may be reported when they remain inside the aggregate thresholds, but they must not be hidden.

#### Observability, privacy, and reproducibility

| Metric | Acceptance threshold |
| --- | ---: |
| Manifest schema validity | 100% |
| Trace schema validity | 100% |
| Required event completeness | 100% |
| Run/trace correlation integrity | 100% |
| User-mode sensitive-content leaks | 0 |
| Deterministic normalized artifact reproducibility | 3 of 3 runs |

### 17. Bound live execution

The live benchmark budgets are:

| Budget | Limit |
| --- | ---: |
| Query-time p95 latency | at most 15 seconds |
| Generation calls | at most 1 per case |
| Query-embedding calls | at most 1 per case |
| Transient provider retries | at most 2 |
| Model input | at most 16,000 tokens per case |
| Model output | at most 1,000 tokens per case |
| Unbounded retry/reflection/tool loops | 0 |
| Usage metadata coverage | 100% |

Initial document ingestion and index-build latency must be measured separately from query-time latency.

Dollar cost must be reported whenever provider usage and pricing data permit calculation. A configurable run-level cost budget must terminate further provider work with a typed failure when exceeded. The specification does not hard-code a permanent dollar threshold because provider prices and chosen model aliases may change.

### 18. Standardize CLI and artifacts

The public evaluation entry point remains one script and supports:

```powershell
python scripts/evaluate_rag.py --suite evidence-contract --provider deterministic
python scripts/evaluate_rag.py --suite evidence-contract --provider openai
python scripts/evaluate_rag.py --suite portfolio-demo --provider deterministic
```

Each run produces:

```text
eval_reports/<run_id>/
  manifest.json
  trace.jsonl
  case_results.jsonl
  metrics.json
  summary.md
```

Generated `eval_reports/` remain ignored by version control. Corpus, schemas, and expected annotations are versioned. Milestone completion requires one reviewed deterministic baseline and one reviewed live summary committed under the documentation area, each identifying the corresponding code commit, corpus, schema, prompt, configuration, and model.

The CLI returns non-zero when any acceptance threshold for the selected suite fails.

### 19. Fix the portfolio demonstration

The portfolio demo uses five stable case IDs, one for each Answer Disposition. It must be runnable with one deterministic command and produce:

- a human-readable summary;
- structured case results;
- manifest and trace artifacts;
- Answer Claim-to-Citation output for the `answer` case;
- the focused question for `needs_clarification`;
- the Evidence gap for `abstain_insufficient_evidence`;
- the safety boundary for `blocked_safety`;
- the typed failure and safe run ID for `failed_system`.

The demo is evidence of contract behavior, not proof that the corpus represents all insurance products.

### 20. Use a seven-stage rollout

#### Stage 1 — Characterize the existing behavior

- retain the current full test baseline;
- add characterization coverage around orchestration, policy/background separation, no-policy generation skip, fallbacks, warnings, and fail-closed guard behavior;
- capture the existing deterministic evaluation output as migration evidence.

#### Stage 2 — Add schemas and the gold corpus

- define and version data and annotation schemas;
- add the four policy, one Background Material, and 50-case corpus;
- add corpus lint and schema validation to CI.

#### Stage 3 — Extract shared application orchestration

- create the single application service;
- route deterministic and live providers through typed adapters;
- preserve current Streamlit behavior through a temporary compatibility adapter;
- keep characterization and integration tests green.

#### Stage 4 — Integrate the Evidence Contract

- add structured model proposals;
- validate stable Evidence identity and links;
- implement Evidence Sufficiency and disposition precedence;
- require deterministic integration tests before changing visible output.

#### Stage 5 — Add manifests, traces, privacy, and typed failures

- implement user no-content and public-synthetic full-content modes;
- implement failure injection and deterministic reproducibility checks;
- ensure every complete or failed run has an auditable artifact set.

#### Stage 6 — Migrate Streamlit

- display Answer Claims with linked Citations;
- move unused Retrieved Candidates to diagnostics;
- display the five distinct dispositions;
- remove the temporary payload compatibility adapter and old retrieved-chunk-as-Citation behavior after migration tests pass.

#### Stage 7 — Enable CI, live benchmark, demo, and supported documentation claims

- make deterministic thresholds a required CI gate;
- expose the live command;
- produce passing reviewed baseline artifacts;
- run the five-disposition demo;
- update user and portfolio documentation only with claims supported by committed evidence;
- remove the old parallel evaluation composition.

No long-lived feature flag, dual-write, or dual-orchestration mode is accepted. A temporary compatibility adapter may exist only during Stages 3–6 and must be removed before completion.

## Testing Decisions

### Testing principle

Tests must assert externally meaningful behavior and contract invariants rather than private helper implementation. The preferred highest seam is the shared application orchestration. Lower-level unit tests are appropriate only for pure schemas, identity, validation, privacy, metric, and failure-mapping behavior.

The existing repository already provides unit/component patterns for parser, chunking, retrieval, query rewriting, reranking, RAG orchestration, Citation construction, guard behavior, and fact verification. These are prior art to preserve or adapt, not proof of the new end-to-end contract.

### 1. Unit tests

Cover:

- schema parsing and rejection;
- stable Evidence IDs and hashes;
- normalization, page, span, and quote matching;
- Answer Disposition precedence;
- Claim-to-Evidence-to-Citation invariants;
- failure-code mapping;
- no-content trace filtering;
- metric calculations;
- normalized artifact hashing;
- budget enforcement.

### 2. Contract and property tests

Prove:

- `answer` cannot contain an unsupported Answer Claim;
- Citation cannot reference nonexistent Supporting Evidence;
- Background Material cannot support a User Policy fact;
- links cannot cross document versions;
- non-answer dispositions cannot carry a completed answer payload;
- user-mode traces cannot contain prohibited content fields or sentinel content;
- skipped stages are explicit;
- deterministic input, provider, and config produce deterministic normalized artifacts.

### 3. Characterization tests

Before extracting orchestration, capture:

- retrieval and reranking call order;
- policy/background separation;
- generation skip when no policy result exists;
- optional Background Material failure degradation;
- guard fail-closed behavior;
- current warnings and relevant error paths.

Characterization tests must not permanently preserve the old “all selected chunks are Citations” behavior.

### 4. Shared-orchestration integration tests

Use deterministic adapters to execute:

```text
input
→ retrieval
→ Evidence Sufficiency
→ generation or skip
→ verification
→ Answer Disposition
→ manifest and trace
```

Tests must not monkeypatch around the orchestration seam. The same result contract consumed by Streamlit must be inspected.

### 5. Gold-corpus end-to-end tests

Run all 50 cases through the shared orchestration and calculate every required retrieval, Claim, Evidence, Citation, Answer Disposition, privacy, and artifact metric.

### 6. Failure-injection tests

Inject and verify:

- invalid input;
- document parse failure;
- index build failure;
- retrieval provider failure;
- generation timeout or provider failure;
- malformed model output;
- invalid Evidence span or hash;
- verifier exception;
- trace-write failure;
- unexpected internal exception.

Each test must assert the stable failure code, retryability, safe UI message, sanitized trace, and final disposition.

### 7. Minimal Streamlit behavior tests

Verify:

- only linked Citations are displayed;
- unused Retrieved Candidates appear only in diagnostics;
- all five dispositions have distinct behavior;
- unsafe candidate text and raw exceptions are hidden;
- run IDs and safe next actions are shown where appropriate.

Visual redesign and pixel-perfect testing are not required.

### 8. Deterministic CI test order

The ordinary CI pipeline runs:

```text
unit and contract/property tests
→ characterization and shared-orchestration integration tests
→ 50-case deterministic end-to-end evaluation
→ privacy and three-run reproducibility checks
→ report and schema validation
```

It performs no network calls and requires no external credentials.

### 9. Live benchmark

The live suite uses the same cases, metrics, and contracts. It is opt-in and must produce a reviewed report that reaches all live thresholds before milestone completion.

Missing credentials skip the live suite with a clear environment result. They do not fail ordinary CI, but a skipped run cannot satisfy the milestone's live-evidence requirement.

## Acceptance Criteria and Definition of Done

The milestone is complete only when all criteria below are satisfied.

### Architecture and code

- Streamlit, deterministic evaluation, live evaluation, and portfolio demo use one shared application orchestration.
- Runtime uses the Answer Claim, Supporting Evidence, Citation, and Answer Disposition contracts.
- Deterministic code owns Evidence Sufficiency, verification, safety, failure classification, and final disposition.
- The old retrieved-chunk-as-Citation behavior is removed.
- The old parallel evaluation composition is removed.
- The temporary compatibility adapter is removed.
- No agent loop, agent framework migration, or other out-of-scope capability is introduced.

### Corpus and metrics

- Four synthetic policy documents, one synthetic Background Material document, and at least 50 cases pass schema and lint validation.
- All retrieval thresholds pass.
- All Answer Claim, Supporting Evidence, and Citation thresholds pass.
- Deterministic expected dispositions pass 50 of 50.
- All live disposition thresholds pass in at least one reviewed run.
- Unsupported Answer Claim rate and critical false-answer rate are both zero.
- Privacy, failure injection, and three-run reproducibility checks pass.

### Tests and CI

- The existing test baseline remains green, except where a documented intentional behavior change replaces an obsolete assertion.
- All new test layers described by this specification exist and pass.
- Ordinary CI passes from a clean supported environment without network or API credentials.
- Any required acceptance threshold violation returns a non-zero status.

### Live evidence

- At least one OpenAI benchmark reaches every live quality threshold.
- The reviewed live run satisfies latency, call-count, token, retry, and usage-metadata budgets.
- A committed live summary identifies commit, corpus, schema, prompt, configuration, provider, and model.

### Observability and privacy

- Every run produces schema-valid manifest, trace, case result, metric, and summary artifacts.
- Required event completeness and run/trace correlation are 100%.
- User-mode sensitive-content leak count is zero.
- Public full-content artifacts carry an explicit public-synthetic classification.
- Typed failures and skipped stages are explainable from the artifacts.

### User-visible behavior and demo

- Streamlit implements the agreed behavior for all five Answer Dispositions.
- Only Supporting Evidence linked to Answer Claims appears as a Citation.
- One deterministic command reproduces the five-case portfolio demo.
- User documentation includes deterministic, live, and demo commands.
- Quality claims cite committed baselines and do not generalize beyond the corpus.

### Handoff quality

- Schemas, commands, configuration, migration outcome, and artifact interpretation are documented.
- No unexplained P0 or P1 acceptance failure remains.
- Stable Evidence IDs, typed gaps, dispositions, baseline metrics, and traces are ready for a later bounded agentic evidence-recovery milestone.
- Any explicitly deferred work is recorded as follow-up work rather than hidden in TODOs or unsupported documentation claims.
- If any criterion above is unmet, the milestone must not be marked complete.

## Out of Scope

- Bounded agentic evidence recovery.
- Agents SDK, LangGraph, or another agent-framework migration.
- Retrieval, Citation, or critic multi-agent roles.
- Unlimited reflection or self-critique loops.
- New embedding models, LLM rerankers, graph RAG, or additional retrieval algorithms.
- A comprehensive OCR, layout-parsing, ingestion, or chunking redesign.
- Long-term persistence of user policies, chat history, or contentful user traces.
- Authentication, deployment, production SLOs, incident response, or a telemetry-vendor rollout.
- A complete ontology for all insurance products.
- Automated final claim adjudication.
- Legal, medical, financial, claim, or underwriting advice.
- Streamlit visual redesign.
- Claims that the synthetic corpus represents all real insurance policies.

## Further Notes

### Supported portfolio statement after completion

After every acceptance criterion passes, the project may state:

> InsuranceRAG implements a measured claim-to-evidence contract across a shared production and evaluation workflow, with deterministic sufficiency and safety control, privacy-aware traces, reproducible offline gates, and comparable live-model benchmarks.

It must not claim semantic grounding guarantees beyond the measured corpus, production privacy governance, agentic evidence recovery, or production claims adjudication.

### Relationship to the later agentic milestone

This specification deliberately creates the prerequisites for a bounded, read-only evidence-recovery loop:

- stable Evidence identity;
- typed Evidence gaps;
- explicit Answer Dispositions;
- one shared orchestration;
- outcome baselines;
- latency, token, and cost budgets;
- privacy-aware traces;
- comparable deterministic and live evaluation.

A later agentic proposal must demonstrate measurable improvement over this deterministic baseline. Framework adoption or named agent roles alone will not satisfy that requirement.

### Specification authority

If implementation details conflict with the accepted ADRs or domain language, the ADRs and `CONTEXT.md` govern unless they are explicitly superseded through live agreement. If this specification conflicts with the milestone decision, the milestone decision governs scope and non-goals.

This document specifies the milestone but does not implement it.
