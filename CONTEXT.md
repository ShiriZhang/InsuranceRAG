# Insurance Policy QA

This context describes how InsuranceRAG relates generated insurance-policy explanations to the policy text that supports them.

## Language

**Answer Claim（答案断言）**:
A factual proposition in a generated answer that can be evaluated independently against policy text.
_Avoid_: Claim, insurance claim, answer sentence

**Retrieved Candidate（检索候选）**:
A policy-text chunk returned by retrieval or reranking for possible use; relevance alone does not make it evidence for an Answer Claim.
_Avoid_: Evidence, citation, verified fact

**Supporting Evidence（支持证据）**:
A precisely located span of policy text that actually supports an Answer Claim.
_Avoid_: Retrieved Candidate, retrieval score, matched term

**Silver Supporting Evidence（银标支持证据）**:
An exact policy source span selected by an LLM as benchmark reference evidence without human validation; it is a proxy label rather than ground truth.
_Avoid_: Gold Supporting Evidence, human-validated evidence, ground truth

**Citation（来源引用）**:
The user-visible source location and excerpt for Supporting Evidence linked to an Answer Claim.
_Avoid_: Retrieved Candidate, retrieval provenance, context chunk

**Answer Disposition（回答处置）**:
The single final outcome of a question: answer, request clarification, abstain for insufficient evidence, block for safety, or report a system failure.
_Avoid_: Refusal, guard status, error result

**Policy Clause（保险条款）**:
A logically continuous section of policy text governed by a clause heading; it may extend across PDF page boundaries.
_Avoid_: Page, chunk, Retrieved Candidate
