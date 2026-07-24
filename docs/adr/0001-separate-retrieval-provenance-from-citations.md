---
status: accepted
---

# Separate retrieval provenance from citations

InsuranceRAG will treat retrieved chunks as candidates, not citations: a user-visible Citation must identify Supporting Evidence that actually supports an Answer Claim, while unused Retrieved Candidates may appear only in internal diagnostics. This replaces the current convention of displaying every selected chunk as a citation, accepting additional claim-to-span modeling and evaluation cost in exchange for measurable attribution and less misleading output.
