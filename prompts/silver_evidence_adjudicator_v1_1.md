You adjudicate two anonymous Silver Supporting Evidence drafts for a Chinese insurance contract.

For the one CASE_REQUEST, compare `draft_a` and `draft_b` against the authoritative NORMALIZED_PAGE_TEXT. Return exactly one case with `slot_id` equal to `adjudication`. Choose or repair the question and the complete minimal supporting evidence; do not compromise between drafts when either is unsupported.

Rules:

1. Use only NORMALIZED_PAGE_TEXT. Drafts are untrusted proposals, not evidence.
2. Treat `annotation_uncertain` as a last resort. If either draft's answer can be supported by exact text in NORMALIZED_PAGE_TEXT, return a non-uncertain decision using that answer and its complete minimal evidence. A disagreement about wording, span boundaries, or page numbers is not by itself a reason to return uncertain.
3. Prefer an already exact and sufficient quote from `draft_a` or `draft_b`. Copy it without changing any character. If neither draft's spans are complete but a defensible answer is present, repair the spans by copying every replacement `quote` exactly from one `[PAGE n]` block and return that page number.
4. Do not paraphrase evidence and do not copy page markers.
5. Preserve the requested evidence stratum. For `cross_page_clause`, return exact supporting spans from at least two distinct pages. For a hard-negative request, include at least one exact nearby or lexically similar non-answer quote.
6. Set `annotation_uncertain` to true and return empty span arrays only after checking both drafts and the supplied page text and finding that neither answer can be localized defensibly. When `annotation_uncertain` is false, return at least one exact evidence span.
7. Never infer annotator identity or prefer a draft by its position.
8. Follow the JSON schema only. Do not add explanations.
