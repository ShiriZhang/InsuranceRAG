---
status: accepted
---

# Share one core orchestration across runtime and evaluation

The Streamlit runtime, deterministic CI runner, and opt-in live-model benchmark will invoke the same application-level orchestration and Answer Claim, Supporting Evidence, Citation, and Answer Disposition contracts; only provider adapters, corpus inputs, and run configuration may vary. This requires a deliberate shared seam instead of preserving the current parallel evaluation composition, trading some refactoring cost for production-equivalent evidence and lower behavioral drift.
