---
status: accepted
---

# Keep final answer disposition under deterministic control

Models may propose structured Answer Claims, Supporting Evidence links, ambiguities, and missing fact types, but deterministic code owns the Evidence Sufficiency gate and final Answer Disposition. This prevents a model from declaring its own output sufficiently grounded or bypassing verification and safety controls, trading some semantic flexibility for reproducible termination, testing, and future agent boundaries.
