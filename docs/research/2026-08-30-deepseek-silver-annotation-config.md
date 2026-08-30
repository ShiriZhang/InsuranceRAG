# DeepSeek configuration for Silver Supporting Evidence annotation

Date: 2026-08-30

Issue: #16

## Frozen project decision

The approved policy PDFs under `documents/` will be annotated through the DeepSeek Responses API. `OPENAI_API_KEY` and `GROQ_API_KEY` are not used by this workflow. The local secret is read only from `DEEPSEEK_API_KEY` and must remain in the git-ignored `.env` file.

The formal three-pass configuration is:

| Pass | Model | Reasoning effort |
| --- | --- | --- |
| Annotation A | `deepseek-v4-flash` | `low` |
| Annotation B | `deepseek-v4-flash` | `none` |
| Adjudication C | `deepseek-v4-flash` | `high` |

All passes use the Responses API and strict JSON Schema output. `temperature`, `top_p`, and `seed` are omitted: DeepSeek documents that sampling parameters do not affect thinking mode, and its current Responses API reference does not define a seed parameter. Annotation A reserves 8,192 output tokens, Annotation B reserves 4,096, and the high-effort adjudicator reserves 16,384. A one-source pilot showed that a 4,096-token adjudication budget could be consumed entirely by reasoning and return no final JSON, so the larger C budget is a measured reliability requirement rather than a change to the requested reasoning setting.

The provider and annotation protocol change advances the benchmark and release to `silver-evidence-benchmark/v2.0.0` and `clause-v2-silver/v2.0.0`. DeepSeek checkpoints use a separate `deepseek_checkpoints_v2` directory and must not be mixed with Groq v1 checkpoints.

## Reproducibility and quality controls

`deepseek-v4-flash` is a mutable alias. Every response therefore records the returned model metadata available through the API, `system_fingerprint`, response ID, request timestamp, token usage, retry count, prompt hash, schema hash, and the complete frozen generation parameters. A future backend change requires a new run record and may require a new benchmark version.

Strict JSON Schema constrains syntax but does not prove that quotations are supported by the contract. The local exact-span mapper remains authoritative: missing, repeated, or non-exact quotations become uncertain and disagreements go to adjudication. The release remains invalid if overall uncertainty exceeds 10% or any key stratum exceeds 15%.

The existing deterministic complete-page windows, progress reporting, retry handling, and local checkpoints remain enabled. The model's larger context capacity does not justify changing evidence-transport windows without a measured pilot.

## Primary references

- DeepSeek model and pricing: https://api-docs.deepseek.com/quick_start/pricing/
- Responses API and JSON Schema: https://api-docs.deepseek.com/api/create-response/
- Thinking mode: https://api-docs.deepseek.com/guides/thinking_mode/
- Rate limits: https://api-docs.deepseek.com/quick_start/rate_limit/
- Model updates: https://api-docs.deepseek.com/updates/
