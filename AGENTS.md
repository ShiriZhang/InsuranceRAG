## Agent skills

### Issue tracker

Issues and PRDs are tracked in this repository's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context domain-document layout. See `docs/agents/domain.md`.

## User communication preferences

- The user is a beginner in AI Agent development and has basic Python programming experience.
- Explain in Chinese by default; keep code, commands, file names, and API names in English.
- When a technical term first appears, provide its English name and a plain-language Chinese explanation.
- Do not assume the user understands Agent architecture, RAG, tool calling, state management, multi-agent collaboration, or evaluation frameworks.
- Before executing a task, explain the current goal, why the proposed approach is appropriate, and which files are expected to change.
- Break complex tasks into smaller steps and state which step is currently in progress.
- After modifying code, explain the key code, the data flow, and how to verify the change.
- When an error occurs, first explain what the error means, then distinguish confirmed causes from hypotheses that still need verification.
- When multiple approaches are available, compare their advantages and disadvantages, then recommend the approach best suited to a beginner and the current project.
- Avoid unnecessary complex abstractions, over-engineering, and large-scale refactoring.
- Do not agree blindly with the user; clearly point out possible misunderstandings or incorrect assumptions.
- After each meaningful step, summarize what was completed, which files changed, how to verify the result, and what comes next.
- These communication requirements also apply when using Matt Pocock Skills.
