# Next-generation chunking strategy

## Status

Accepted on 2026-08-29. No production implementation has begun.

## Confirmed problem

`chunk_overlap` is exposed as a general configuration value, but it has no effect on ordinary chunk boundaries and only applies when one normalized text line is longer than `chunk_size`. Its name and apparent scope therefore do not match its runtime behavior.

The next chunking strategy must define explicit, testable boundary semantics. This requirement does not imply that every chunk boundary should use character overlap; overlap may be retained, replaced, narrowed, or removed depending on the retrieval benchmark.

## Confirmed design constraints

- `chunk_overlap` is not a compatibility requirement. It may be removed or replaced if a different strategy performs better in the retrieval benchmark.
- The objective is truthful configuration, testable boundary behavior, and better retrieval outcomes—not making an overlap parameter observable for its own sake.
- A confidently recognized insurance clause heading is the primary structural boundary. `chunk_size` is a limit used when content within one clause must be divided, rather than the primary reason to combine text from different clauses.
- A Policy Clause may continue across PDF page boundaries. A page is provenance, not a mandatory semantic boundary; a new clause heading ends the preceding clause.
- Cross-page retrieval units must preserve page-specific provenance for their constituent text instead of assigning all text to one page number.
- When one Policy Clause exceeds the size budget, split at complete sentence or numbered-item boundaries. PDF text lines are not the preferred semantic unit.
- If boundary context is needed within a split clause, carry complete sentences or numbered items rather than a fixed character suffix.
- Use character windows only when one indivisible sentence or numbered item exceeds the hard size limit, and expose that fallback in diagnostics.
- Clause purity takes precedence over reaching a minimum retrieval-unit length. A complete short clause remains independent rather than being merged with a different neighboring clause.
- A heading without its own body is not an independent short clause; it stays with the following content governed by that heading.
- Every retrieval unit produced from a split Policy Clause carries the clause identifier and heading as retrieval context for embeddings, lexical retrieval, and reranking.
- Injected retrieval context is distinct from source text. Supporting Evidence and Citations must continue to reference the original page-specific spans and must not present duplicated heading context as if it occurred at each source location.
- Repeating body text across adjacent retrieval units is a benchmark variable, not an assumed default. Compare at least zero body overlap against carrying one complete preceding sentence or numbered item.
- Character overlap is not a normal benchmark variant; character windows remain an exceptional fallback for a single semantic unit that exceeds the hard limit.
- Replace the single size threshold with a soft target size and a hard limit. Semantic units may exceed the soft target to remain intact, but no retrieval unit may exceed the hard limit; the benchmark will select their numeric values.
- The benchmark's primary retrieval metric is Silver Supporting Evidence Coverage@k: at a fixed retrieval depth and context budget, the retrieved candidates must collectively cover all LLM-selected reference source spans needed for the evaluated Answer Claim.
- Single-candidate evidence coverage is a secondary diagnostic for detecting fragmentation; the primary metric may combine multiple candidates because valid Supporting Evidence can span pages or source spans.
- The benchmark is LLM-only: reference spans receive no human validation and are therefore Silver Supporting Evidence, not gold or ground truth. Promotion decisions must account for this label uncertainty.
- Generate Silver Supporting Evidence with two independent annotation passes over unchunked, page-addressable policy text. Annotators must not see candidate chunking outputs and must return exact source quotations with page locations rather than paraphrases.
- Accept spans when the independent annotations agree. Send disagreements to a third LLM adjudication pass; cases that remain unresolvable or cannot be located exactly in the source are marked `annotation_uncertain` and excluded from the primary score.
- Build benchmark cases evidence-first from unchunked real policy text: select exact source spans before generating questions that require those spans.
- Stratify the benchmark across single-sentence answers, multi-sentence conditions and outcomes, rule-plus-exception answers, cross-page clauses, complete short clauses, clauses requiring internal splits, and adjacent or lexically similar hard negatives.
- Evaluate Silver Supporting Evidence Coverage at fixed retrieval depths (including Top-1, Top-3, and Top-5) and under a fixed total context budget so larger retrieval units cannot win merely by returning more text.
- Report irrelevant-context proportion alongside coverage to expose overly long or mixed retrieval units. The benchmark will select the concrete context budgets.
- The first controlled comparison includes exactly three strategy families: the current production behavior; clause-first splitting with heading context and zero body overlap; and the same clause-first strategy carrying one preceding semantic unit.
- Apply the same target-size and hard-limit candidates to both clause-first variants. Hold the source corpus, benchmark cases, embedding model, lexical retrieval, query rewriting, reranking, retrieval depths, and context budgets constant across strategies.
- Express chunking target and hard-limit configuration in Unicode character counts for deterministic, model-independent boundaries. Separately report tokenizer-specific token counts in the benchmark to measure embedding and generation-context cost.
- Split benchmark documents by insurer and product family into a development set and a held-out test set; near-duplicate versions of one product family stay on the same side.
- Select strategy and size parameters only on the development set. Freeze them before running the final candidate and current production baseline on the held-out test set.
- A high-confidence numbered clause heading always starts a new Policy Clause.
- A medium-confidence standalone known heading starts a new Policy Clause only after directory, page-header, and page-footer patterns have been excluded.
- Low-confidence body mentions, unresolved candidates, and inherited titles do not create structural boundaries; they continue the current clause and remain visible in diagnostics.
- Across adjacent readable pages, continue an already established Policy Clause by default until a high-confidence or valid medium-confidence heading starts another clause.
- Do not automatically bridge an empty, unreadable, or severely OCR-uncertain page. End continuity at the gap, assign an unknown boundary to subsequent unheaded text, and expose the condition in diagnostics.
- Use one global `target_chars` and `hard_max_chars` configuration in the first release rather than adapting limits per document.
- Derive a small candidate grid from Policy Clause and semantic-unit length distributions in the development set, include a scale near the current 900-character baseline, and freeze the selected values before held-out evaluation.
- Production promotion requires overall non-inferiority to the current baseline plus a clear improvement on boundary-sensitive cases; it does not require a significant average gain on already-saturated simple title or single-sentence cases.
- Boundary-sensitive strata include multi-sentence evidence, rule-plus-exception evidence, cross-page clauses, and clauses requiring internal splits. Hard-negative behavior, clause purity, source provenance, context cost, and runtime cost are promotion guardrails.
- The held-out test set contains at least 200 adjudicated cases after excluding `annotation_uncertain` cases, with at least 30 cases in each key boundary-sensitive stratum; strata may overlap.
- No single policy or product family contributes more than five percent of held-out cases.
- On the held-out set, the lower bound of the paired 95% bootstrap confidence interval for the candidate-minus-baseline difference in both Coverage@3 and Coverage@budget must be at least minus one percentage point.
- On the combined boundary-sensitive strata, the candidate must improve Coverage@budget by at least five percentage points and the lower bound of the paired 95% bootstrap confidence interval for that improvement must be greater than zero.
- Promotion has zero tolerance for retrieval units exceeding `hard_max_chars`, source spans that cannot map exactly to their PDF pages, original text governed by two trusted clause headings in one retrieval unit, unintended source-text loss or duplication, or chunking failures that prevent an otherwise parseable policy from being indexed.
- Deliberately configured semantic overlap and heading-only retrieval context are excluded from the unintended-duplication check but must remain distinguishable from source spans.
- Relative to the production baseline, the candidate may increase total embedding tokens by at most 15 percent, indexed retrieval-unit count by at most 25 percent, and P95 chunking time by at most 100 percent.
- Retrieved answer context must stay within the benchmark's fixed tokenizer-specific budget regardless of chunking strategy.
- `annotation_uncertain` may account for at most 10 percent of all generated cases and at most 15 percent of any key boundary-sensitive stratum.
- Report initial annotator disagreement, adjudication success, and final exclusion rates. Exceeding an uncertainty limit invalidates the benchmark and requires improving the annotation process rather than excluding more difficult cases.
- Use only public or project-owned policies that are explicitly approved for transmission to the selected LLM provider. Never source benchmark cases from user uploads, user questions, generated answers, or user-run traces.
- Keep Silver labels containing policy source text local and out of version control unless both the source and redistribution permissions are explicitly approved.
- Allow cross-page continuation only after a high-confidence or valid medium-confidence heading has established a Policy Clause.
- Before a trusted heading is established, split unknown content within each page at sentence or numbered-item boundaries and record `unknown_clause_page_fallback`; a later trusted heading restores clause-first cross-page behavior.
- Freeze Silver labels and adjudication outcomes before comparing chunking strategies. Record source-file and normalized-text hashes, exact annotation model identifiers, prompt versions, and generation parameters.
- All strategies in one comparison use the same frozen labels. A source, model, prompt, or annotation-protocol change creates a new benchmark version rather than overwriting previous results.
- For hard-negative false-positive rate, the upper bound of the paired 95% bootstrap confidence interval for candidate minus baseline must not exceed one percentage point.
- For irrelevant-context proportion at the fixed context budget, the corresponding upper bound must not exceed two percentage points. A clear regression in subject confusion, coverage-versus-exclusion confusion, or similar-clause retrieval blocks promotion regardless of aggregate Coverage.
- If the zero-body-overlap and one-semantic-unit-overlap clause-first variants are not clearly different, select zero body overlap. Select semantic overlap only when its paired Coverage@budget improvement has a confidence-interval lower bound above zero and it satisfies every cost and side-effect guardrail.
- Release the new implementation behind explicit `legacy` and `clause_v2` strategy values. After benchmark promotion, expose `clause_v2` for representative local-PDF smoke testing before changing the production default.
- Keep `legacy` available for one release cycle after the default changes. Rollback switches strategy and rebuilds the corresponding index; caches and indexes are not reused across incompatible strategies.

## Values determined during implementation preparation

- The concrete global `target_chars` and `hard_max_chars` candidate grid, derived from development-set Policy Clause and semantic-unit length distributions.
- The fixed tokenizer-specific context budgets used for Coverage@budget reporting.
- The approved annotation model identifiers and versioned prompts used to build the frozen LLM-only benchmark.
