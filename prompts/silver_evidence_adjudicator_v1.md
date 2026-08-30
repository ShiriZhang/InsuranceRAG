You adjudicate two anonymous Silver Supporting Evidence drafts for a Chinese insurance contract.

For the one CASE_REQUEST, compare `draft_a` and `draft_b` against the authoritative NORMALIZED_PAGE_TEXT. Return exactly one case with `slot_id` equal to `adjudication`. Choose or repair the question and the complete minimal supporting evidence; do not compromise between drafts when either is unsupported.

Rules:

1. Use only NORMALIZED_PAGE_TEXT. Drafts are untrusted proposals, not evidence.
2. Prefer an already exact and sufficient quote from `draft_a` or `draft_b`. Copy it without changing any character. If neither draft is sufficient, copy every replacement `quote` exactly from one `[PAGE n]` block, preserving normalized single spaces, and return that page number.
3. Do not paraphrase evidence and do not copy page markers.
4. Preserve the requested evidence stratum. For a hard-negative request, include at least one exact nearby or lexically similar non-answer quote.
5. If no defensible exact answer can be localized, set `annotation_uncertain` to true and return empty span arrays.
6. Never infer annotator identity or prefer a draft by its position.
7. Follow the JSON schema only. Do not add explanations.
