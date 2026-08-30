---
status: accepted
---

# Prioritize policy clause boundaries for retrieval units

InsuranceRAG will build retrieval units from trusted Policy Clause boundaries rather than treating PDF pages or fixed character counts as the primary structure: an established clause may span pages with page-specific provenance, complete short clauses remain independent, and long clauses split at sentence or numbered-item boundaries with a character-window hard-limit fallback. Each split unit carries clause heading context for retrieval while source spans remain authoritative for Supporting Evidence and Citations; body overlap is enabled only if the frozen Silver Supporting Evidence benchmark proves a clear benefit, with zero body overlap as the tie-breaker. This replaces the current page-and-line packing semantics, accepting more structural parsing and provenance complexity in exchange for truthful configuration, clause purity, testable boundaries, and evidence-oriented retrieval evaluation.
