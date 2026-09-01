# Clause v2 development selection

## target=600, hard_max=900, budget=2000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.025862 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | 0.060345 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.151832 | >= 0.05 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | 0.062827 | > 0 | PASS |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.156863 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | -0.005431 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.019643 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 1.238919 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 6.034617 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.020115 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | 0.043103 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.130890 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | 0.036649 | > 0 | PASS |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.176471 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | -0.008751 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.150871 | <= 1.15 | FAIL |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 1.361678 | <= 1.25 | FAIL |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.941922 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `96d6c82c18dffe4825996c042397a74f75800be7c305093c9091d00b8b23995a`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 35.63% | 73.28% | 97.56% | 5994815 | 8032 | 0.034989 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 30.75%/46.84%/54.31% | 48.28% | 75.86% | 96.62% | 6112571 | 9951 | 0.211145 | similar_clause: 10 |
| clause_v2_preceding_semantic_unit | 348 | 30.17%/45.98%/52.30% | 46.26% | 75.29% | 96.31% | 6899260 | 10937 | 0.207902 | similar_clause: 11 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 37.25%, complete_short_clause: 32.08%, cross_page_clause: 9.76%, internally_split_clause: 38.00%, multi_sentence_conditions_outcomes: 51.92%, rule_plus_exception: 43.75%, single_sentence: 32.08%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 41.18%, complete_short_clause: 41.51%, cross_page_clause: 36.59%, internally_split_clause: 38.00%, multi_sentence_conditions_outcomes: 78.85%, rule_plus_exception: 52.08%, single_sentence: 47.17%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 33.33%, complete_short_clause: 45.28%, cross_page_clause: 36.59%, internally_split_clause: 36.00%, multi_sentence_conditions_outcomes: 78.85%, rule_plus_exception: 45.83%, single_sentence: 45.28%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 117, cross_page_clause_continuation: 8317, low_confidence_heading_candidate: 7935, page_gap:empty: 12, trusted_heading:high:line_pattern: 2980, trusted_heading:medium:known_title: 6594, unknown_clause_page_fallback: 377]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 117, cross_page_clause_continuation: 9243, low_confidence_heading_candidate: 8767, page_gap:empty: 15, semantic_overlap_unavailable: 126, trusted_heading:high:line_pattern: 3216, trusted_heading:medium:known_title: 7285, unknown_clause_page_fallback: 436]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +9.20% [+2.59%, +15.52%]; Coverage under token budget +12.64% [+6.03%, +19.54%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +8.33% [+2.01%, +14.94%]; Coverage under token budget +10.63% [+4.31%, +17.24%].

## target=600, hard_max=900, budget=4000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.025862 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | 0.051724 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.130890 | >= 0.05 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | 0.047120 | > 0 | PASS |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.156863 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | -0.000955 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.019643 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 1.238919 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 5.326498 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.020115 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | 0.028736 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.130890 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | 0.047120 | > 0 | PASS |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.176471 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | -0.003803 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.150871 | <= 1.15 | FAIL |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 1.361678 | <= 1.25 | FAIL |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.296070 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `2275e65fe187ca06d3a55e680ff1bba4e784cd2b80f1d4f481bfeef546501cde`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 48.85% | 73.28% | 98.28% | 5994815 | 8032 | 0.038797 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 30.75%/46.84%/54.31% | 60.06% | 75.86% | 98.00% | 6112571 | 9951 | 0.206650 | similar_clause: 10 |
| clause_v2_preceding_semantic_unit | 348 | 30.17%/45.98%/52.30% | 58.05% | 75.29% | 97.72% | 6899260 | 10937 | 0.205469 | similar_clause: 11 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 49.02%, complete_short_clause: 56.60%, cross_page_clause: 21.95%, internally_split_clause: 50.00%, multi_sentence_conditions_outcomes: 57.69%, rule_plus_exception: 56.25%, single_sentence: 45.28%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 56.86%, complete_short_clause: 60.38%, cross_page_clause: 43.90%, internally_split_clause: 46.00%, multi_sentence_conditions_outcomes: 88.46%, rule_plus_exception: 60.42%, single_sentence: 60.38%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 41.18%, complete_short_clause: 64.15%, cross_page_clause: 51.22%, internally_split_clause: 48.00%, multi_sentence_conditions_outcomes: 90.38%, rule_plus_exception: 50.00%, single_sentence: 58.49%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 117, cross_page_clause_continuation: 8317, low_confidence_heading_candidate: 7935, page_gap:empty: 12, trusted_heading:high:line_pattern: 2980, trusted_heading:medium:known_title: 6594, unknown_clause_page_fallback: 377]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 117, cross_page_clause_continuation: 9243, low_confidence_heading_candidate: 8767, page_gap:empty: 15, semantic_overlap_unavailable: 126, trusted_heading:high:line_pattern: 3216, trusted_heading:medium:known_title: 7285, unknown_clause_page_fallback: 436]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +9.20% [+2.59%, +15.52%]; Coverage under token budget +11.21% [+5.17%, +17.53%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +8.33% [+2.01%, +14.94%]; Coverage under token budget +9.20% [+2.87%, +15.80%].

## target=600, hard_max=900, budget=8000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.025862 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | -0.022989 | >= -0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.026178 | >= 0.05 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | -0.047120 | > 0 | FAIL |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.156863 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | 0.000534 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.019643 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 1.238919 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 4.984805 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.020115 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | 0.000000 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.073298 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | 0.000000 | > 0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.176471 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | -0.001267 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.150871 | <= 1.15 | FAIL |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 1.361678 | <= 1.25 | FAIL |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 4.791729 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `de90892ff8192ce788b0050cd85ce6e8dbb0361f832324a3d989a9fe7a348621`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 70.11% | 73.28% | 98.81% | 5994815 | 8032 | 0.041883 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 30.75%/46.84%/54.31% | 73.56% | 75.86% | 98.79% | 6112571 | 9951 | 0.208779 | similar_clause: 10 |
| clause_v2_preceding_semantic_unit | 348 | 30.17%/45.98%/52.30% | 75.29% | 75.29% | 98.61% | 6899260 | 10937 | 0.200692 | similar_clause: 11 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 74.51%, complete_short_clause: 75.47%, cross_page_clause: 39.02%, internally_split_clause: 64.00%, multi_sentence_conditions_outcomes: 80.77%, rule_plus_exception: 79.17%, single_sentence: 71.70%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 72.55%, complete_short_clause: 79.25%, cross_page_clause: 51.22%, internally_split_clause: 56.00%, multi_sentence_conditions_outcomes: 96.15%, rule_plus_exception: 70.83%, single_sentence: 83.02%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 68.63%, complete_short_clause: 77.36%, cross_page_clause: 60.98%, internally_split_clause: 62.00%, multi_sentence_conditions_outcomes: 96.15%, rule_plus_exception: 75.00%, single_sentence: 83.02%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 117, cross_page_clause_continuation: 8317, low_confidence_heading_candidate: 7935, page_gap:empty: 12, trusted_heading:high:line_pattern: 2980, trusted_heading:medium:known_title: 6594, unknown_clause_page_fallback: 377]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 117, cross_page_clause_continuation: 9243, low_confidence_heading_candidate: 8767, page_gap:empty: 15, semantic_overlap_unavailable: 126, trusted_heading:high:line_pattern: 3216, trusted_heading:medium:known_title: 7285, unknown_clause_page_fallback: 436]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +9.20% [+2.59%, +15.52%]; Coverage under token budget +3.45% [-2.30%, +8.62%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +8.33% [+2.01%, +14.94%]; Coverage under token budget +5.17% [+0.00%, +10.06%].

## target=900, hard_max=1200, budget=2000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.048851 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | -0.017241 | >= -0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.068063 | >= 0.05 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | -0.015707 | > 0 | FAIL |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.254902 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | -0.000810 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.014185 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 0.885458 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 5.763295 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.043103 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | -0.034483 | >= -0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.015707 | >= 0.05 | FAIL |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | -0.073298 | > 0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.196078 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | -0.000833 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.090576 | <= 1.15 | PASS |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 0.934636 | <= 1.25 | PASS |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.427344 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `96d6c82c18dffe4825996c042397a74f75800be7c305093c9091d00b8b23995a`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 35.63% | 73.28% | 97.56% | 5994815 | 8032 | 0.040322 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 30.17%/48.56%/56.61% | 40.23% | 81.03% | 97.13% | 6079852 | 7112 | 0.232389 | similar_clause: 13 |
| clause_v2_preceding_semantic_unit | 348 | 29.31%/47.99%/56.61% | 38.51% | 81.61% | 97.09% | 6537803 | 7507 | 0.218843 | similar_clause: 10 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 37.25%, complete_short_clause: 32.08%, cross_page_clause: 9.76%, internally_split_clause: 38.00%, multi_sentence_conditions_outcomes: 51.92%, rule_plus_exception: 43.75%, single_sentence: 32.08%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 39.22%, complete_short_clause: 33.96%, cross_page_clause: 29.27%, internally_split_clause: 42.00%, multi_sentence_conditions_outcomes: 61.54%, rule_plus_exception: 39.58%, single_sentence: 33.96%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 35.29%, complete_short_clause: 37.74%, cross_page_clause: 24.39%, internally_split_clause: 32.00%, multi_sentence_conditions_outcomes: 57.69%, rule_plus_exception: 37.50%, single_sentence: 41.51%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 71, cross_page_clause_continuation: 5619, low_confidence_heading_candidate: 5577, page_gap:empty: 8, trusted_heading:high:line_pattern: 2245, trusted_heading:medium:known_title: 4589, unknown_clause_page_fallback: 278]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 71, cross_page_clause_continuation: 5999, low_confidence_heading_candidate: 5907, page_gap:empty: 9, semantic_overlap_unavailable: 71, trusted_heading:high:line_pattern: 2338, trusted_heading:medium:known_title: 4876, unknown_clause_page_fallback: 293]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +10.92% [+4.89%, +16.67%]; Coverage under token budget +4.60% [-1.72%, +10.63%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +10.34% [+4.31%, +16.67%]; Coverage under token budget +2.87% [-3.45%, +8.91%].

## target=900, hard_max=1200, budget=4000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.048851 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | 0.020115 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.136126 | >= 0.05 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | 0.057592 | > 0 | PASS |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.254902 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | -0.000105 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.014185 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 0.885458 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 5.876646 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.043103 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | 0.020115 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.120419 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | 0.041885 | > 0 | PASS |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.196078 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | -0.001065 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.090576 | <= 1.15 | PASS |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 0.934636 | <= 1.25 | PASS |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.770836 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `2275e65fe187ca06d3a55e680ff1bba4e784cd2b80f1d4f481bfeef546501cde`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 48.85% | 73.28% | 98.28% | 5994815 | 8032 | 0.036514 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 30.17%/48.56%/56.61% | 56.90% | 81.03% | 98.06% | 6079852 | 7112 | 0.214583 | similar_clause: 13 |
| clause_v2_preceding_semantic_unit | 348 | 29.31%/47.99%/56.61% | 56.90% | 81.61% | 97.97% | 6537803 | 7507 | 0.210719 | similar_clause: 10 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 49.02%, complete_short_clause: 56.60%, cross_page_clause: 21.95%, internally_split_clause: 50.00%, multi_sentence_conditions_outcomes: 57.69%, rule_plus_exception: 56.25%, single_sentence: 45.28%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 45.10%, complete_short_clause: 54.72%, cross_page_clause: 43.90%, internally_split_clause: 52.00%, multi_sentence_conditions_outcomes: 88.46%, rule_plus_exception: 56.25%, single_sentence: 54.72%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 52.94%, complete_short_clause: 52.83%, cross_page_clause: 48.78%, internally_split_clause: 50.00%, multi_sentence_conditions_outcomes: 78.85%, rule_plus_exception: 58.33%, single_sentence: 54.72%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 71, cross_page_clause_continuation: 5619, low_confidence_heading_candidate: 5577, page_gap:empty: 8, trusted_heading:high:line_pattern: 2245, trusted_heading:medium:known_title: 4589, unknown_clause_page_fallback: 278]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 71, cross_page_clause_continuation: 5999, low_confidence_heading_candidate: 5907, page_gap:empty: 9, semantic_overlap_unavailable: 71, trusted_heading:high:line_pattern: 2338, trusted_heading:medium:known_title: 4876, unknown_clause_page_fallback: 293]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +10.92% [+4.89%, +16.67%]; Coverage under token budget +8.05% [+2.01%, +14.66%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +10.34% [+4.31%, +16.67%]; Coverage under token budget +8.05% [+2.01%, +14.08%].

## target=900, hard_max=1200, budget=8000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.048851 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | -0.057471 | >= -0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.036649 | >= 0.05 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | -0.031414 | > 0 | FAIL |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.254902 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | 0.001362 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.014185 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 0.885458 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 4.991077 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.043103 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | -0.048851 | >= -0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.078534 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | 0.010471 | > 0 | PASS |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.196078 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | 0.000117 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.090576 | <= 1.15 | PASS |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 0.934636 | <= 1.25 | PASS |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.179156 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `de90892ff8192ce788b0050cd85ce6e8dbb0361f832324a3d989a9fe7a348621`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 70.11% | 73.28% | 98.81% | 5994815 | 8032 | 0.040657 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 30.17%/48.56%/56.61% | 69.25% | 81.03% | 98.86% | 6079852 | 7112 | 0.202921 | similar_clause: 13 |
| clause_v2_preceding_semantic_unit | 348 | 29.31%/47.99%/56.61% | 70.40% | 81.61% | 98.75% | 6537803 | 7507 | 0.210567 | similar_clause: 10 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 74.51%, complete_short_clause: 75.47%, cross_page_clause: 39.02%, internally_split_clause: 64.00%, multi_sentence_conditions_outcomes: 80.77%, rule_plus_exception: 79.17%, single_sentence: 71.70%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 58.82%, complete_short_clause: 71.70%, cross_page_clause: 48.78%, internally_split_clause: 62.00%, multi_sentence_conditions_outcomes: 100.00%, rule_plus_exception: 66.67%, single_sentence: 71.70%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 62.75%, complete_short_clause: 66.04%, cross_page_clause: 56.10%, internally_split_clause: 68.00%, multi_sentence_conditions_outcomes: 96.15%, rule_plus_exception: 75.00%, single_sentence: 66.04%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 71, cross_page_clause_continuation: 5619, low_confidence_heading_candidate: 5577, page_gap:empty: 8, trusted_heading:high:line_pattern: 2245, trusted_heading:medium:known_title: 4589, unknown_clause_page_fallback: 278]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 71, cross_page_clause_continuation: 5999, low_confidence_heading_candidate: 5907, page_gap:empty: 9, semantic_overlap_unavailable: 71, trusted_heading:high:line_pattern: 2338, trusted_heading:medium:known_title: 4876, unknown_clause_page_fallback: 293]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +10.92% [+4.89%, +16.67%]; Coverage under token budget -0.86% [-5.75%, +4.02%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +10.34% [+4.31%, +16.67%]; Coverage under token budget +0.29% [-4.89%, +5.17%].

## target=1200, hard_max=1600, budget=2000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.054598 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | -0.017241 | >= -0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.052356 | >= 0.05 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | -0.031414 | > 0 | FAIL |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.254902 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | 0.001968 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.011534 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 0.713894 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 6.105753 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.077586 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | -0.022989 | >= -0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.057592 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | -0.026178 | > 0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.294118 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | -0.000014 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.072358 | <= 1.15 | PASS |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 0.742032 | <= 1.25 | PASS |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.819164 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `96d6c82c18dffe4825996c042397a74f75800be7c305093c9091d00b8b23995a`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 35.63% | 73.28% | 97.56% | 5994815 | 8032 | 0.036750 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 33.91%/49.43%/58.33% | 40.52% | 85.63% | 97.38% | 6063961 | 5734 | 0.224385 | similar_clause: 13 |
| clause_v2_preceding_semantic_unit | 348 | 32.76%/51.44%/58.62% | 39.66% | 86.21% | 97.22% | 6428588 | 5960 | 0.213853 | similar_clause: 15 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 37.25%, complete_short_clause: 32.08%, cross_page_clause: 9.76%, internally_split_clause: 38.00%, multi_sentence_conditions_outcomes: 51.92%, rule_plus_exception: 43.75%, single_sentence: 32.08%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 33.33%, complete_short_clause: 39.62%, cross_page_clause: 29.27%, internally_split_clause: 34.00%, multi_sentence_conditions_outcomes: 59.62%, rule_plus_exception: 43.75%, single_sentence: 41.51%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 35.29%, complete_short_clause: 32.08%, cross_page_clause: 29.27%, internally_split_clause: 38.00%, multi_sentence_conditions_outcomes: 59.62%, rule_plus_exception: 41.67%, single_sentence: 39.62%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 45, cross_page_clause_continuation: 4299, low_confidence_heading_candidate: 4420, page_gap:empty: 6, trusted_heading:high:line_pattern: 1883, trusted_heading:medium:known_title: 3623, unknown_clause_page_fallback: 228]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 45, cross_page_clause_continuation: 4517, low_confidence_heading_candidate: 4612, page_gap:empty: 7, semantic_overlap_unavailable: 34, trusted_heading:high:line_pattern: 1939, trusted_heading:medium:known_title: 3786, unknown_clause_page_fallback: 235]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +11.78% [+5.46%, +18.10%]; Coverage under token budget +4.89% [-1.72%, +10.92%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +13.79% [+7.76%, +20.11%]; Coverage under token budget +4.02% [-2.30%, +9.77%].

## target=1200, hard_max=1600, budget=4000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.054598 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | -0.005747 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.104712 | >= 0.05 | PASS |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | 0.020942 | > 0 | PASS |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.254902 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | 0.001109 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.011534 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 0.713894 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 5.901065 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.077586 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | -0.011494 | >= -0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.099476 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | 0.015707 | > 0 | PASS |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.294118 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | -0.000651 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.072358 | <= 1.15 | PASS |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 0.742032 | <= 1.25 | PASS |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.722855 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `2275e65fe187ca06d3a55e680ff1bba4e784cd2b80f1d4f481bfeef546501cde`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 48.85% | 73.28% | 98.28% | 5994815 | 8032 | 0.039252 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 33.91%/49.43%/58.33% | 54.60% | 85.63% | 98.18% | 6063961 | 5734 | 0.231629 | similar_clause: 13 |
| clause_v2_preceding_semantic_unit | 348 | 32.76%/51.44%/58.62% | 54.02% | 86.21% | 98.05% | 6428588 | 5960 | 0.224633 | similar_clause: 15 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 49.02%, complete_short_clause: 56.60%, cross_page_clause: 21.95%, internally_split_clause: 50.00%, multi_sentence_conditions_outcomes: 57.69%, rule_plus_exception: 56.25%, single_sentence: 45.28%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 50.98%, complete_short_clause: 50.94%, cross_page_clause: 46.34%, internally_split_clause: 42.00%, multi_sentence_conditions_outcomes: 84.62%, rule_plus_exception: 56.25%, single_sentence: 49.06%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 50.98%, complete_short_clause: 49.06%, cross_page_clause: 43.90%, internally_split_clause: 46.00%, multi_sentence_conditions_outcomes: 84.62%, rule_plus_exception: 52.08%, single_sentence: 49.06%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 45, cross_page_clause_continuation: 4299, low_confidence_heading_candidate: 4420, page_gap:empty: 6, trusted_heading:high:line_pattern: 1883, trusted_heading:medium:known_title: 3623, unknown_clause_page_fallback: 228]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 45, cross_page_clause_continuation: 4517, low_confidence_heading_candidate: 4612, page_gap:empty: 7, semantic_overlap_unavailable: 34, trusted_heading:high:line_pattern: 1939, trusted_heading:medium:known_title: 3786, unknown_clause_page_fallback: 235]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +11.78% [+5.46%, +18.10%]; Coverage under token budget +5.75% [-0.57%, +12.07%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +13.79% [+7.76%, +20.11%]; Coverage under token budget +5.17% [-1.15%, +11.49%].

## target=1200, hard_max=1600, budget=8000

### Promotion guardrails

| Candidate | Guardrail | Value | Requirement | Result |
| --- | --- | ---: | --- | --- |
| `clause_v2_zero_body_overlap` | `coverage_at_3_ci_lower` | 0.054598 | >= -0.01 | PASS |
| `clause_v2_zero_body_overlap` | `coverage_budget_ci_lower` | -0.060345 | >= -0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_gain` | 0.026178 | >= 0.05 | FAIL |
| `clause_v2_zero_body_overlap` | `boundary_coverage_ci_lower` | -0.041885 | > 0 | FAIL |
| `clause_v2_zero_body_overlap` | `hard_negative_ci_upper` | 0.254902 | <= 0.01 | FAIL |
| `clause_v2_zero_body_overlap` | `irrelevant_context_ci_upper` | 0.001718 | <= 0.02 | PASS |
| `clause_v2_zero_body_overlap` | `embedding_token_ratio` | 1.011534 | <= 1.15 | PASS |
| `clause_v2_zero_body_overlap` | `retrieval_unit_ratio` | 0.713894 | <= 1.25 | PASS |
| `clause_v2_zero_body_overlap` | `p95_chunking_latency_ratio` | 5.321070 | <= 2.0 | FAIL |
| `clause_v2_zero_body_overlap` | `correctness_invariants` | true | all true | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_at_3_ci_lower` | 0.077586 | >= -0.01 | PASS |
| `clause_v2_preceding_semantic_unit` | `coverage_budget_ci_lower` | -0.060345 | >= -0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_gain` | 0.057592 | >= 0.05 | PASS |
| `clause_v2_preceding_semantic_unit` | `boundary_coverage_ci_lower` | -0.010471 | > 0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `hard_negative_ci_upper` | 0.294118 | <= 0.01 | FAIL |
| `clause_v2_preceding_semantic_unit` | `irrelevant_context_ci_upper` | 0.000671 | <= 0.02 | PASS |
| `clause_v2_preceding_semantic_unit` | `embedding_token_ratio` | 1.072358 | <= 1.15 | PASS |
| `clause_v2_preceding_semantic_unit` | `retrieval_unit_ratio` | 0.742032 | <= 1.25 | PASS |
| `clause_v2_preceding_semantic_unit` | `p95_chunking_latency_ratio` | 5.104891 | <= 2.0 | FAIL |
| `clause_v2_preceding_semantic_unit` | `correctness_invariants` | false | all true | FAIL |

# Silver Supporting Evidence Benchmark

Benchmark version: `silver-evidence-benchmark/v2.0.0`
Frozen manifest SHA-256: `de90892ff8192ce788b0050cd85ce6e8dbb0361f832324a3d989a9fe7a348621`

## Annotation quality

- Annotation disagreement: 163/369 (44.17%)
- Adjudication success: 142/163 (87.12%)
- Uncertain exclusions: 21/369 (5.69%)

## Retrieval, context, and cost

| Strategy | Scored | Coverage@1/@3/@5 | Coverage under token budget | Single-candidate coverage | Irrelevant context | Embedding tokens | Retrieval units | P95 chunking latency (s) | Hard-negative confusions |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 348 | 20.98%/37.64%/50.57% | 70.11% | 73.28% | 98.81% | 5994815 | 8032 | 0.042392 | similar_clause: 7 |
| clause_v2_zero_body_overlap | 348 | 33.91%/49.43%/58.33% | 69.54% | 85.63% | 98.88% | 6063961 | 5734 | 0.225569 | similar_clause: 13 |
| clause_v2_preceding_semantic_unit | 348 | 32.76%/51.44%/58.62% | 69.25% | 86.21% | 98.81% | 6428588 | 5960 | 0.216405 | similar_clause: 15 |

## Boundary strata

- legacy: adjacent_or_lexically_similar_hard_negative: 74.51%, complete_short_clause: 75.47%, cross_page_clause: 39.02%, internally_split_clause: 64.00%, multi_sentence_conditions_outcomes: 80.77%, rule_plus_exception: 79.17%, single_sentence: 71.70%
- clause_v2_zero_body_overlap: adjacent_or_lexically_similar_hard_negative: 68.63%, complete_short_clause: 69.81%, cross_page_clause: 56.10%, internally_split_clause: 54.00%, multi_sentence_conditions_outcomes: 98.08%, rule_plus_exception: 66.67%, single_sentence: 69.81%
- clause_v2_preceding_semantic_unit: adjacent_or_lexically_similar_hard_negative: 60.78%, complete_short_clause: 66.04%, cross_page_clause: 58.54%, internally_split_clause: 58.00%, multi_sentence_conditions_outcomes: 100.00%, rule_plus_exception: 70.83%, single_sentence: 67.92%

## Diagnostics and correctness invariants

- legacy: diagnostics [legacy_page_line_packing: 8032]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_zero_body_overlap: diagnostics [character_window_fallback: 45, cross_page_clause_continuation: 4299, low_confidence_heading_candidate: 4420, page_gap:empty: 6, trusted_heading:high:line_pattern: 1883, trusted_heading:medium:known_title: 3623, unknown_clause_page_fallback: 228]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: pass, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]
- clause_v2_preceding_semantic_unit: diagnostics [character_window_fallback: 45, cross_page_clause_continuation: 4517, low_confidence_heading_candidate: 4612, page_gap:empty: 7, semantic_overlap_unavailable: 34, trusted_heading:high:line_pattern: 1939, trusted_heading:medium:known_title: 3786, unknown_clause_page_fallback: 235]; invariants [authoritative_source_coverage_exact: pass, clause_purity: pass, hard_max_chars_respected: pass, no_unintended_source_duplication: pass, non_empty_retrieval_units: pass, parseable_policies_indexed: pass, semantic_overlap_complete: FAIL, semantic_overlap_distinct_from_source_spans: pass, source_spans_exact: pass]

## Paired 95% confidence intervals

- clause_v2_zero_body_overlap minus legacy: Coverage@3 +11.78% [+5.46%, +18.10%]; Coverage under token budget -0.57% [-6.03%, +4.60%].
- clause_v2_preceding_semantic_unit minus legacy: Coverage@3 +13.79% [+7.76%, +20.11%]; Coverage under token budget -0.86% [-6.03%, +4.02%].

## Failure analysis

No candidate passed every guardrail (0/18).

Failed guardrails across the full sensitivity grid:

- `hard_negative_ci_upper`: 18/18 candidates failed.
- `p95_chunking_latency_ratio`: 18/18 candidates failed.
- `coverage_budget_ci_lower`: 10/18 candidates failed.
- `boundary_coverage_ci_lower`: 9/18 candidates failed.
- `correctness_invariants`: 9/18 candidates failed.
- `boundary_coverage_gain`: 4/18 candidates failed.
- `embedding_token_ratio`: 3/18 candidates failed.
- `retrieval_unit_ratio`: 3/18 candidates failed.

Universal blockers: `hard_negative_ci_upper`, `p95_chunking_latency_ratio`.

The semantic-overlap candidates also report `semantic_overlap_complete: FAIL` when a complete preceding semantic unit cannot fit without exceeding `hard_max_chars`. No threshold was relaxed and no candidate was selected.

## Frozen selection

- Status: `no_eligible_candidate`
- Decision: no configuration was selected or promoted.
- Reason: No clause_v2 development candidate satisfies every promotion rule.
