# Groq configuration for Silver Supporting Evidence annotation

Date checked: 2026-08-30

Scope: a reproducible, three-pass workflow over approved Chinese insurance-policy PDFs: two independent evidence annotations followed by adjudication of disagreements. This note separates provider facts from project decisions. Groq's model catalog and limits are mutable, so re-check the linked pages before creating a new benchmark version.

## Verified provider facts

### Models and output guarantees

- Groq currently lists `openai/gpt-oss-120b` and `openai/gpt-oss-20b` as **Production Models**. Both have a 131,072-token context window and a 65,536-token maximum completion. The current catalog lists 120B at $0.15/M input tokens and $0.60/M output tokens, and 20B at $0.075/M input tokens and $0.30/M output tokens. [Groq Supported Models](https://console.groq.com/docs/models)
- Groq describes 120B as the stronger multilingual model (MMMLU 81.3% average and support across 81+ languages) and 20B as a lower-cost multilingual model (MMMLU 75.7% average). Those aggregate benchmarks are not evidence of accuracy on Chinese insurance contracts; no first-party benchmark for that domain was found. [Groq GPT-OSS 120B](https://console.groq.com/docs/model/openai/gpt-oss-120b), [Groq GPT-OSS 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b)
- OpenAI says GPT-OSS was trained on a **mostly English**, text-only dataset. Consequently, the multilingual figures above do not remove the need for a project-specific Chinese pilot. [OpenAI: Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)
- Groq Structured Outputs with `response_format.type="json_schema"` and `strict: true` uses constrained decoding and currently supports only `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. Every field must be required, every object must set `additionalProperties: false`, and optional values must be represented with a `null` union. Strict mode guarantees schema shape, **not semantic correctness or exact quotations**. Streaming and tool use are not supported with Structured Outputs. [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs)
- `qwen/qwen3.6-27b` and `qwen/qwen3.8-27b` are currently **Preview Models**, which Groq says are evaluation-only and may be discontinued at short notice. Their model pages advertise strong multilingual support, but the current structured-output support table does not include them in strict JSON Schema mode. They are therefore unsuitable for a frozen production benchmark despite being plausible Chinese-language pilot comparators. [Groq Supported Models](https://console.groq.com/docs/models), [Groq Qwen 3.6 27B](https://console.groq.com/docs/model/qwen/qwen3.6-27b), [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs)
- Previously common Groq IDs such as `llama-3.3-70b-versatile`, `qwen/qwen3-32b`, and Llama 4 Scout are deprecated/shut down for free and developer tiers. They must not be frozen into this release. [Groq Model Deprecations](https://console.groq.com/docs/deprecations)

### Sampling, reasoning, and repeatability

- Chat Completions accepts `temperature` from 0 to 2 (default 1), `top_p` from 0 to 1 (default 1), and recommends changing one rather than both. It accepts `seed`, but Groq explicitly describes determinism as best effort, not guaranteed. Responses include `system_fingerprint`, which should be recorded with the seed to detect backend changes. [Groq API Reference](https://console.groq.com/docs/api-reference)
- GPT-OSS 20B and 120B accept `reasoning_effort` values `low`, `medium`, and `high`; `medium` is the default. `reasoning_format` can be `hidden`, `raw`, or `parsed`, and is mutually exclusive with `include_reasoning`. [Groq Reasoning](https://console.groq.com/docs/reasoning), [Groq API Reference](https://console.groq.com/docs/api-reference)
- `max_completion_tokens` is the current parameter; `max_tokens` is deprecated. Groq currently supports only `n=1`. Frequency penalty, presence penalty, logit bias, and log probabilities are not supported by current models and should not be added to the frozen config. [Groq API Reference](https://console.groq.com/docs/api-reference)

### Capacity and data handling

- Rate limits are organization-wide and may include RPM, daily requests, TPM, and separate input/output token limits. Groq says the public tables are high-level and the account Limits page is authoritative. On `429`, use `retry-after`; other limit/reset headers are returned on responses. The model catalog currently advertises developer-plan limits of 250K TPM and 1K RPM for each GPT-OSS model, while actual plans and organizations may differ. [Groq Rate Limits](https://console.groq.com/docs/rate-limits), [Groq Supported Models](https://console.groq.com/docs/models)
- By default, ordinary inference inputs/outputs are not retained, except temporary logging for reliability or suspected abuse for up to 30 days. All customers can enable Zero Data Retention (ZDR); when enabled, Groq says inference customer data is not retained for those purposes. Usage metadata is always retained but does not contain inputs or outputs. Any retained customer data is in US GCP buckets. [Groq: Your Data](https://console.groq.com/docs/your-data)
- Batch input/output files are retained for up to 30 days unless deleted earlier. Groq's agreement says inputs and outputs are not used to train or fine-tune models unless the customer explicitly grants permission or instructs Groq to do so. [Groq: Your Data](https://console.groq.com/docs/your-data), [Groq Services Agreement](https://console.groq.com/docs/legal/services-agreement)

## Frozen annotation configuration

These are project decisions, not Groq guarantees.

| Pass | Model | Reasoning | Sampling | Output cap | Purpose |
|---|---|---:|---|---:|---|
| Annotator A | `openai/gpt-oss-120b` | `medium` | `temperature=0`, `top_p=1`, `seed=16001` | 4,096 | Highest-quality primary extraction |
| Annotator B | `openai/gpt-oss-20b` | `medium` | `temperature=0`, `top_p=1`, `seed=16002` | 4,096 | Independent lower-cost model-size view; reduces identical-decoding agreement |
| Adjudicator C | `openai/gpt-oss-120b` | `medium` | `temperature=0`, `top_p=1`, `seed=16003` | 4,096 | Resolve disagreements within the measured 8,000 TPM account limit |

Common request settings:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "silver_evidence_annotation_v1",
      "strict": true,
      "schema": "<frozen closed JSON Schema>"
    }
  },
  "reasoning_format": "hidden",
  "stream": false,
  "n": 1,
  "citation_options": "disabled",
  "service_tier": "on_demand"
}
```

Rationale:

- Both annotators use production models with strict schema support. Using different model sizes provides more meaningful independence than calling one greedy decoder twice; it does not make errors statistically independent because the models share a family and training lineage.
- Temperature zero minimizes avoidable variance. Distinct seeds are still recorded, but at temperature zero they should be treated primarily as provenance, not as a guarantee of diverse or repeatable answers.
- The adjudicator uses the stronger model and `high` effort only where needed. If the Chinese pilot shows that 20B causes excessive disagreement or misses exact spans, use 120B for both annotators with separate calls and annotator identities; record that the remaining independence is procedural rather than model-diverse.
- Disable platform citations and tools. Page numbers and exact quotations must come from the locally extracted, page-addressable source supplied in the prompt, then be verified byte/character-wise against normalized local text. Provider-generated citations are not a substitute for the repository's `EvidenceSpan` validation.

## Prompt and schema versioning

Freeze two immutable prompt artifacts and one schema artifact:

- `silver-evidence-annotator/v1.0.0`: used unchanged by A and B so disagreement measures model judgment rather than different instructions.
- `silver-evidence-adjudicator/v1.0.0`: receives the authoritative page-addressable source plus the two drafts under neutral labels. It must never see candidate chunking output.
- `silver-evidence-schema/v1.0.0`: closed JSON Schema with every field required; nullable unions represent unavailable values.

Record in every run manifest: semantic version, prompt SHA-256, schema SHA-256, exact model ID, every request parameter, `seed`, returned `system_fingerprint`, Groq SDK version, source/normalized-text hashes, request timestamp, retry count, and response ID. Any prompt, schema, model, normalization, or annotation-protocol change creates a new benchmark version rather than overwriting the prior release.

For adjudication, replace model names and annotator identities with `draft_a`/`draft_b`. Derive their display order from the case hash and record the order to reduce fixed first-position bias. Do not pass prior cases or conversation history between calls. A retry must reuse the same frozen input and seed, retain failed-attempt metadata, and never silently substitute a model.

## Context-budget decisions

Two different budgets must not be conflated:

1. **Annotation request budget:** although GPT-OSS allows 131,072 total tokens, cap locally tokenized input at **96,000 tokens**, reserve up to 16,384 for adjudication output/reasoning, and keep the remainder as serialization/tokenization headroom. Do not truncate a page or evidence candidate silently. If a policy exceeds the cap, process deterministic, page-addressable windows with adjacent-page continuity and include a stable window ID; these windows are annotation transport units, not candidate retrieval chunks.
2. **Retrieval benchmark budget:** freeze `context_token_budgets=(2000, 4000, 8000)` for development sensitivity analysis, with **4,000 tokens as the primary promotion budget**. Report all three, but make selection and held-out promotion decisions against the predeclared 4,000-token primary result.

The annotation input cap is below the provider maximum, not a promise that the account can send 96K tokens in one request. Before generation, inspect the organization's actual Limits page/headers. If its TPM is below a request, reduce deterministic window size or obtain a higher limit; do not change model or prompt opportunistically.

## Chunk-size and overlap decisions

Freeze this small development grid in Unicode characters:

```python
size_grid = (
    (600, 900),
    (900, 1200),
    (1200, 1600),
)
overlap_variants = (
    "zero_body_overlap",
    "preceding_semantic_unit",
)
```

This retains the existing 900/1200 baseline while testing one smaller and one larger scale. It is a starting decision, not provider-derived evidence. Before freezing the held-out run, verify against development-set clause and semantic-unit length distributions as required by the design spec; if the distribution contradicts these candidates, changing the grid requires a new development benchmark version. Keep `zero_body_overlap` as the tie-breaker already required by the project design.

## Operational guardrails

- Use direct `/openai/v1/chat/completions` inference, not Batch, for the first release; this avoids Batch's required file retention. Enable ZDR in Groq Data Controls before sending policies if project governance requires no reliability/abuse-content logging.
- Extract PDF text locally and send only normalized, page-marked text needed for each deterministic window. GPT-OSS is text-only; the model does not read the PDF layout itself.
- Pace requests from live response headers and retry `429` using `retry-after` plus bounded jitter. Persist checkpoints by source/window so a restart never re-annotates completed work without an explicit flag.
- Run a pre-freeze pilot on a stratified Chinese sample and measure exact-span validation rate, A/B disagreement, adjudication success, and uncertainty rate. Strict JSON only guarantees format. Freeze the production release only if repository acceptance thresholds pass.

## Not confirmed by primary sources

- No first-party evidence was found establishing either GPT-OSS model's accuracy for Chinese insurance-policy clause extraction, exact quotation, or PDF page localization.
- Groq does not promise bit-for-bit repeatability, even with a fixed seed and parameters. Reproducibility therefore means immutable inputs/configuration plus recorded backend fingerprint and outputs, not guaranteed regeneration.
- The account's current plan, spend limit, model permissions, ZDR state, and exact rate limits cannot be inferred from public documentation or the presence of `GROQ_API_KEY`; they must be checked in the Groq Console or via actual response headers.

## Local corpus facts and approval decision

The project owner confirmed on 2026-08-30 that every PDF currently under `documents/` is approved for LLM annotation. Use the immutable approval reference `project-owner-approved-documents-for-llm-annotation/2026-08-30` in source manifests.

Local inspection found:

- 176 approved PDFs from 57 insurer directories;
- 176 unique product directories and no duplicate PDF-relative paths;
- every current file uses `documents/<insurer>/<product>/<pdf>`, while inventory code must also accept `documents/<insurer>/<pdf>`;
- all 176 PDFs opened successfully with PyMuPDF: 6,591 pages, 7,761,450 extracted characters, 73 empty pages (1.11%), median 38 pages, maximum 84 pages, and maximum 87,544 extracted characters in one policy.

Freeze corpus identities as follows:

- `insurer_family`: normalized first path component below `documents/`;
- `product_family`: SHA-256 of the normalized PDF-relative path, namespaced with `product-family/v1:`;
- `near_duplicate_family`: equal to `product_family` for this release because the owner confirmed that every PDF is a distinct insurance product;
- `source_id`: SHA-256 of the normalized PDF-relative path, namespaced with `benchmark-source/v1:`; source-content and normalized-text hashes remain separate manifest fields.

The document split is insurer-level, with no insurer crossing sides. Target held-out at 30% of policies using a deterministic, version-seeded subset selection over whole insurer groups; freeze the resulting assignments before any annotation request. The release generator must support both observed directory shapes even though the current snapshot uses only the nested-product form.

## Local Groq pilot and final project decisions

The live account model API on 2026-08-30 confirmed that both recommended GPT-OSS IDs are active with 131,072-token contexts and strict Structured Outputs support. A content-minimized pilot used normalized text from four non-empty pages in one policy, followed by a 10-policy sample spanning 10 insurers. It emitted no source text in logs and persisted no model output.

| Model | Requests | Schema-valid | Unique exact normalized quote | Uncertain | Request errors |
|---|---:|---:|---:|---:|---:|
| `openai/gpt-oss-120b` | 10 | 10 | 8 | 0 | 0 |
| `openai/gpt-oss-20b` | 10 | 9 | 9 of 9 successful | 0 | 1 |

The pilot does not measure legal interpretation correctness, but it confirms two operational facts: strict schema does not guarantee an exact quote, and neither model reliably self-identifies localization failure as uncertain. Therefore local validation is authoritative: zero or multiple normalized-quote matches turn that draft into an invalid/uncertain annotation and force adjudication; an adjudicator result that still cannot map exactly becomes `annotation_uncertain`.

Freeze these project versions:

- document split: `silver-document-split/v1.0.0`;
- benchmark: `silver-evidence-benchmark/v1.0.0`;
- release: `clause-v2-silver/v1.0.0`;
- annotator prompt: `silver-evidence-annotator/v1.0.0` for both A and B;
- adjudicator prompt: `silver-evidence-adjudicator/v1.0.0`;
- response schema: `silver-evidence-schema/v1.0.0`;
- normalized page text: `normalized-page-text/v1.0.0`.

Final executable model/parameter decision:

- A: `openai/gpt-oss-120b`, `reasoning_effort=medium`, `temperature=0`, `top_p=1`, `seed=16001`, `max_completion_tokens=4096`;
- B: `openai/gpt-oss-20b`, `reasoning_effort=medium`, `temperature=0`, `top_p=1`, `seed=16002`, `max_completion_tokens=4096`;
- C: `openai/gpt-oss-120b`, `reasoning_effort=medium`, `temperature=0`, `top_p=1`, `seed=16003`, `max_completion_tokens=4096`;
- all: strict JSON Schema, hidden reasoning, `stream=false`, `n=1`, no tools, direct on-demand inference, record response ID and `system_fingerprint`.

Freeze `size_grid=((600,900),(900,1200),(1200,1600))`, development retrieval budgets `(2000,4000,8000)` with 4,000 primary, and overlap variants `("zero_body_overlap", "preceding_semantic_unit")`. These are development selection inputs; held-out evaluation must use the selected values without reopening this grid.

### Development-only size-grid check

The frozen insurer split was applied before this analysis, and only the 123 development policies were inspected. A reproducible local heuristic used the current trusted-heading recognizer for approximate Policy Clause boundaries and Chinese terminal punctuation for semantic units; no held-out outcome or label was consulted.

| Unit | Count | P50 | P75 | P90 | P95 | P99 | >900 | >1200 | >1600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Semantic unit | 86,099 | 42 | 66 | 98 | 129 | 260 | 93 | 46 | 25 |
| Approximate Policy Clause | 2,138 | 229 | 1,239 | 4,348 | 12,044 | 39,319 | 646 | 544 | 471 |

The distribution supports keeping complete semantic units intact while splitting many long clauses. `(900,1200)` preserves the existing 900-character comparison point; `(600,900)` tests a smaller retrieval unit without cutting ordinary semantic units; `(1200,1600)` brackets the approximate clause P75 while retaining a finite hard limit. These are benchmark candidates, not a preselected production winner.

### Account-limit adaptation from the end-to-end run

The configured Groq organization reported an on-demand TPM limit of 8,000. A full-policy request was rejected at 53,782 requested tokens. Groq also counted reserved `max_completion_tokens` toward the request limit: a 1,400-character complete-page window with an 8,192-token completion reservation was rejected at 9,527 requested tokens. The executable v1 config therefore transports one case per deterministic complete-page window (1,400 normalized characters maximum), never silently truncates a page, and uses the 4,096-token completion cap above. This replaces the earlier whole-policy multi-case reservation of 8,192/16,384. A real `high`-effort adjudication then exhausted the same cap and returned no schema-valid JSON, so C was empirically reduced to `medium`; the model IDs, seeds, prompts, schema, and retrieval grids are unchanged. The organization-wide 8,000 TPM throughput remains an external constraint on the duration of the full 176-policy run.
