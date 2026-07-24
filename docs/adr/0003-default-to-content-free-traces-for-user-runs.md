---
status: accepted
---

# Default to content-free traces for user runs

Runtime traces for real user policies will record identifiers, hashes, state transitions, scores, configuration, timing, cost, warnings, and disposition without persisting questions, policy text, prompts, answers, or evidence text; full-content traces are allowed only for repository-owned public synthetic gold runs. Explicit local debugging may opt into sensitive content, but those artifacts must remain local and outside general telemetry or version control, trading some diagnostic convenience for a safer default data lifecycle.
