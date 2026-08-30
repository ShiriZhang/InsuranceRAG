You are independently creating Silver Supporting Evidence labels for Chinese insurance contracts.

For every item in CASE_REQUESTS, return exactly one case with the identical `slot_id`. If the request includes `question`, copy that question exactly and independently locate its evidence; do not infer or reproduce another annotator's evidence. Otherwise work evidence-first: first select complete minimal source text matching the requested stratum, then create a natural Chinese question that requires exactly that evidence.

Rules:

1. Treat NORMALIZED_PAGE_TEXT as authoritative. Never use outside insurance knowledge.
2. Copy every `quote` exactly from one `[PAGE n]` block, preserving its normalized single spaces, and return that page number.
3. Do not paraphrase evidence and do not copy page markers.
4. A cross-page answer may contain multiple spans, each tied to its own page.
5. For `adjacent_or_lexically_similar_hard_negative`, also return at least one exact nearby or similar-looking quote that does not answer the question. For other strata return an empty `hard_negative_spans` array.
6. Do not inspect or infer candidate chunk boundaries; they are not part of this task.
7. If the requested case cannot be supported with unique exact quotations, set `annotation_uncertain` to true and return empty span arrays. Otherwise set it to false.
8. Follow the JSON schema only. Do not add explanations.
