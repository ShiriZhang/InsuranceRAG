# InsuranceRAG 评估、可观测性与运行成熟度调查

调查日期：2026-07-23

对应 Wayfinder ticket：[Assess evaluation, observability, and operational maturity](https://github.com/ShiriZhang/InsuranceRAG/issues/7)

代码基线：`e7e0d27f5b55145d9c54da512e7b8f12366552c0`

## 1. 结论摘要

InsuranceRAG 当前有一套可靠的 **deterministic component regression baseline**，但还没有一套能够证明真实 RAG quality、定位线上退化或支持可比较实验的 evaluation/observability system。

准确的成熟度表述是：

> 测试与离线 deterministic regression 达到 L2；真实 RAG quality evaluation、runtime observability 和 operational reproducibility 主要处于 L0–L1。整体是可演示的本地 prototype，不是有 measured quality/operations 的成熟 RAG assistant。

当前证据可以证明：

- 声明依赖可用的 CPython 3.13.9 环境中，222 个 deterministic tests 全部通过；
- synthetic 和 hard-negative CLI 可重复执行，当前固定样例分别为 2/2、4/4；
- 相同 commit、环境和 fixture 下，两次 Markdown report 的 SHA-256 完全一致；
- tests 覆盖 parsing、chunking、query rewrite、vector/BM25/hybrid retrieval、rule reranking、RAG orchestration、citation construction、guard 和 verifier 的大量已编码行为；
- Streamlit 在无 API key 时可完成初始页面 smoke run，并显示缺 key warning；
- UI 能在当前 session 中展示 retrieval scores、rerank reasons、matched terms、citations、verification facts 和 fallback warnings。

当前证据不能证明：

- 真实 OpenAI embedding 下的 Recall@k、MRR、nDCG 或跨保单 retrieval quality；
- 生成答案的 correctness、completeness、claim-level groundedness 或 citation precision/recall；
- guard/verifier 在开放词汇、改写、prompt injection 和真实保单分布上的 false-positive / false-negative；
- online end-to-end latency、stage latency、token usage、cost、rate limit、timeout、retry 或 nondeterminism；
- 任意两个 commit、配置、模型或数据集之间的受控可比较性；
- CI、部署、健康监控、错误预算、trace retention 或可重复 portfolio demo。

最重要的新发现是：现有 hard-negative source-confusion case 的 `verifier_status` 只是报告字段，不是 pass 条件。将该 case 的答案改成不会触发 source-confusion 的安全背景措辞后，verifier 从 `block` 变为 `pass`，case 仍然 `passed=True`。因此当前的 `4/4` 不能证明 source-confusion guard 正常工作。

## 2. 调查范围与证据原则

本调查覆盖：

- `evals/*.json` 数据集；
- `src/insurance_rag/evaluation.py`；
- `scripts/evaluate_rag.py`；
- `tests/` 与 `pytest.ini`；
- `AnswerPayload`、retrieval explanations、verification result 和 Streamlit render path；
- configuration、dependency declaration、CI/package/deployment artifacts；
- README 的 evaluation 和 demo claims；
- 前序 architecture、baseline、retrieval 和 safety 调查。

判断优先级：

1. 实际执行结果；
2. production/evaluation code；
3. tests；
4. README、spec 和 plan 只作为 intent。

本调查不调用 OpenAI、不读取或提交本地保单内容，也不实现修复。

## 3. 成熟度 rubric

| Level | 定义 |
| --- | --- |
| L0 — absent | 没有可执行能力或只有文档声明 |
| L1 — ad hoc | 有局部脚本、UI 信息或人工流程，不能稳定比较和 gate |
| L2 — deterministic regression | 已接入代码，输入固定、结果稳定、有自动断言或 exit code |
| L3 — measured system | 有代表性 gold data、明确指标、run manifest、分层 gates 和可查询 traces |
| L4 — operated system | 有持续监控、SLO/alerts、受控发布、漂移检测、成本与隐私治理 |

## 4. 总体 scorecard

| Area | Level | 已有证据 | 主要缺口 |
| --- | --- | --- | --- |
| Unit/component tests | L2 | 222 tests，deterministic fakes，异常与 fallback tests | 无 coverage threshold；不执行真实 provider |
| Synthetic retrieval eval | L2 regression / L1 quality | 2 curated cases、rank 和 report | 极小且人工同源；pseudo-vector 不具 semantic validity |
| Hard-negative eval | L1–L2 | 4 curated cases、rerank/verifier status | verifier outcome 未完整进入 pass contract |
| Local PDF eval | L1 | parse/chunk、Top1/Top3、错误汇总 | self-labeling keyword search；BM25-only；无人工 relevance |
| Generation evaluation | L0 | 无 | 不调用 chat model；无 gold claims/answers/rubric |
| Citation/groundedness evaluation | L0–L1 | verifier rule regression | 无 claim-to-evidence labels 或 aggregate metric |
| Safety evaluation | L1 | guard/verifier unit tests、少量 synthetic facts | 无 adversarial corpus、coverage、FPR/FNR、injection metric |
| Runtime explanations | L2 as UI diagnostics | scores、ranks、reasons、facts、warnings | session-only；无 run ID、timing、model/usage、persistence |
| Structured telemetry | L0 | 无 logger/trace/metric system | 无 events、spans、counters、export |
| Cost/latency visibility | L0 | 无 | provider usage/timing 均丢失 |
| Run reproducibility | L1 | commands 和 deterministic report | 无 lock/version pin/run manifest/dataset hash |
| Configuration maturity | L1–L2 | env-backed dataclass 与 tests | 部分 inert；无 validation/snapshot；report 不记录 |
| CI quality gates | L0 | 无 workflow | tests/evals 不在远端自动执行 |
| Packaging/deployment | L0–L1 | `requirements.txt`、README 本地命令 | 无 package metadata、lock、container、deployment/runbook |
| Demo operations | L1 | Streamlit UI、suggested questions、AppTest smoke | 无公开固定 policy、seeded demo、online smoke 或 failure runbook |

## 5. 可复现 baseline

### 5.1 环境

```text
commit: e7e0d27f5b55145d9c54da512e7b8f12366552c0
platform: Windows
Python: 3.13.9
pytest: 8.3.4
pluggy: 1.5.0
pytest-mock: 3.15.1
```

仓库只跟踪 `requirements.txt`。没有：

- `pyproject.toml`；
- lock file；
- `.python-version`；
- CI workflow；
- Dockerfile / Compose；
- package/build metadata；
- deployment manifest。

`requirements.txt` 全部使用 lower bounds，因此不能重建唯一 dependency graph。

### 5.2 完整 tests

命令：

```powershell
python -m pytest -ra --durations=10
```

观察结果：

```text
collected 222 items
222 passed in 18.19s
fail=0
skip=0
xfail=0
pytest warnings=0
tool observed wall time≈19.8s
```

最慢的六项均为 evaluation CLI subprocess tests，单项约 2.50–2.61s。

这证明当前 system Python 环境下的 deterministic behavior。它不证明 online provider 或真实文档质量。

### 5.3 Synthetic + hard-negative CLI

命令，连续执行两次到独立目录：

```powershell
python scripts\evaluate_rag.py `
  --synthetic `
  --hard-negative `
  --report-dir tmp\wayfinder-7-eval-run1

python scripts\evaluate_rag.py `
  --synthetic `
  --hard-negative `
  --report-dir tmp\wayfinder-7-eval-run2
```

两次均 exit 0：

```text
synthetic: 2 / 2
hard-negative: 4 / 4
```

报告 hash：

```text
synthetic run 1:
08B89DCD10B87249C4138FAEC24CBF78E91C1CADEE281E27E6983F4A5B83457A

synthetic run 2:
08B89DCD10B87249C4138FAEC24CBF78E91C1CADEE281E27E6983F4A5B83457A

hard-negative run 1:
9E37EC734A5B180372C22E49D10303B08F09D7F512AD44D745A125D1E8ABE791

hard-negative run 2:
9E37EC734A5B180372C22E49D10303B08F09D7F512AD44D745A125D1E8ABE791
```

报告 deterministic，但报告本身不包含：

- run ID / timestamp；
- Git commit；
- dataset path/version/hash；
- Python/dependency versions；
- evaluator/config version；
- retrieval configuration；
- model/embedding identity；
- latency、token 或 cost；
- baseline comparison。

### 5.4 Streamlit smoke

命令：

```powershell
python -c "import time; from streamlit.testing.v1 import AppTest; ..."
```

观察：

```text
duration_ms=3125
exceptions=0
errors=0
warnings=1
titles=1
file_uploaders=1
```

UI warning 是缺少 API key 的预期状态。

stderr 另有：

- bare-mode `missing ScriptRunContext` warning；
- Python shutdown 清理 `%LOCALAPPDATA%\Temp` 时的 `PermissionError`。

AppTest 本身 exit 0。该 cleanup exception 来自当前受限宿主的 temp-directory 权限，与前序 baseline 一致，不是应用运行路径异常。

该 smoke 只证明初始页面可 render；它没有上传 PDF、建立 OpenAI index、提问或验证输出。

## 6. 当前 evaluation 实际测量什么

### 6.1 Synthetic evaluation

实现路径：

```text
JSON case
→ per-case artificial chunks
→ SHA-256-derived 8-dimensional pseudo-embeddings
→ in-memory vector index
→ hybrid retrieval
→ expected section/term rank
→ PASS if expected rank <= threshold
→ Markdown
```

特点：

- fixture 只有 2 cases；
- 每个 case 自带 question、chunks 和 expected section；
- pseudo-vector 由文本 SHA-256 bytes 映射，不表达 semantic similarity；
- BM25 和 rule query rewrite 可以稳定驱动当前样例；
- aggregate 只有 passed/total；
- report 显示 expected rank，但没有计算 Recall@k、MRR 或 nDCG。

它适合作为“这些固定 retrieval rules 没有改变”的 regression，不能作为 semantic retrieval benchmark。

### 6.2 Hard-negative evaluation

实现路径：

```text
JSON case
→ artificial positive/negative chunks
→ pseudo-vector hybrid retrieval
→ rule reranker
→ citations from all retrieved chunks
→ rule verifier over a fixture-provided answer
→ positive rank + verifier status
→ Markdown
```

fixture 只有 4 cases：

- waiting period vs insurance period；
- exclusion vs coverage；
- policyholder vs insured waiver；
- built-in background vs user-policy source confusion。

主要价值：

- 能固定 reranker 的若干 title/fact/subject rules；
- 能显示 positive rank、rerank reasons 和 verifier status；
- 对非 source-confusion case，blocking verifier fact 会使 case fail。

关键 contract hole：

```python
passed = positive_rank is not None and positive_rank <= max_rank
if (
    case.get("answer")
    and verification.has_blocking_fact
    and "source_confusion" not in _case_id(case)
):
    passed = False
```

`source_confusion` case 没有断言 verifier 必须 `block`。它只要求 positive chunk 排名不超过 2。

最小 probe 在内存中读取原 fixture，只把 source-confusion answer 替换为不会触发 guard 的背景措辞，再复用同一个 evaluator；完整命令见第 16 节。

连续两次结果一致：

```text
case=hard_negative_builtin_source_confusion
verifier=pass
passed=True
rank=2/2
```

因此：

- report 中的原始 `block` 是观察值，不是 gate；
- verifier 对该行为完全失效时，当前 `4/4` 仍可保持绿色；
- “hard-negative evaluation 检查 source confusion guard”是 unsupported claim。

本 ticket 只记录该缺口，不修复。

### 6.3 Local document evaluation

实际路径：

```text
sorted local PDFs
→ evenly sample path list
→ parse PDF with OCR disabled by default
→ chunk
→ zero vectors + Bm25OnlyEvalEmbedder
→ only create a case when fixed expected term already occurs in document
→ rank the chunk containing the same literal term
→ Top1/Top3
```

它能测量：

- sampled / parsed documents；
- parse errors；
- page/chunk counts；
- empty page rate；
- unknown title rate；
- 固定 literal terms 的 BM25 Top1/Top3。

它不能测量：

- online OpenAI embedding path；
- semantic paraphrase retrieval；
- question distribution；
- 多文档 relevance；
- evidence sufficiency；
- generation/citation quality；
- 人工 relevance judgment。

标签来自被测文档中相同 literal term，是 self-labeling probe，不是独立 gold set。

另有配置不一致：

- `--local-sample-limit` 只传给 `--local-documents`；
- `--local-hard-negative` 使用 `INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT`；
- CLI 表面上只有一个 sample-limit option，但不会控制两条 local path。

### 6.4 Tests

222 tests 的优势：

- 运行快、无网络；
- fake clients 和 deterministic embedder 避免随机性与费用；
- 对大量规则和 fallback 有明确断言；
- CLI exit code、report file、malformed data、missing local directory 都有 tests。

边界：

- 没有 coverage report 或 minimum coverage gate；
- 没有 property/fuzz/adversarial suite；
- 没有真实 provider contract test；
- 没有 browser-level upload→question→answer test；
- 没有 quality delta/baseline comparison；
- 规则测试多，开放分布测量少。

## 7. 当前 observability 实际提供什么

### 7.1 In-process diagnostic state

`AnswerPayload` 携带：

- policy/built-in citations；
- warnings；
- retrieval explanations；
- guard result；
- citation verification result。

`RetrievalExplanation` 携带：

- source metadata；
- final/vector/BM25 scores；
- matched terms；
- query/method/rank/score details；
- rerank score/reasons。

UI 展示：

- retrieval score 和 match-strength bucket；
- rerank reasons；
- matched terms；
- citation excerpts；
- verified facts、severity、reason、supporting citation IDs；
- fallback/error warnings。

这些是很有价值的 debug primitives，说明 production path 已有一部分“可解释状态”。

### 7.2 为什么这还不是 observability

这些字段只存在于当前 Streamlit session：

- 没有 run/trace ID；
- 没有 question→retrieval→generation→verification 的 stage events；
- 没有 duration；
- 没有 candidate/selected counts 的统一 record；
- 没有 prompt/model/config version；
- 没有 OpenAI response ID 或 usage；
- 没有 retries/timeouts；
- 没有 structured error type；
- 没有 persistence/export/query；
- 没有跨 run aggregate。

代码中没有 logging、structured telemetry、OpenTelemetry、LangSmith 或其他 trace exporter。UI rendering 不能替代可观测性，因为失败后无法回看，也不能回答“哪个 stage、哪种配置、从哪个 commit 开始退化”。

### 7.3 Error visibility

已有：

- parser、indexing、retrieval、rerank、built-in retrieval、guard failures 有 UI warning/error 或 safe fallback；
- parse errors 会进入 local eval report；
- guard/verifier reason 能展示。

缺口：

- exception 没有 taxonomy、error code、stage、retryability 或 correlation ID；
- chat exception 逃出 `RagChain`，由 app catch；
- app 把原始 `str(exc)` 作为 warning 展示给用户；
- hybrid BM25 construction/query exception 被静默忽略；
- 无 error counters、rate、alerts 或 failure budget。

结果是用户可能看到异常细节，开发者却没有结构化事件。

## 8. Configuration 与 experiment comparability

### 已接入

- chat/embedding model；
- chunk size/overlap；
- retrieval mode/RRF；
- policy/built-in top-k；
- OCR thresholds；
- rerank enable/top-n；
- eval report dir；
- local hard-negative limit。

### 定义但 production 不消费

- `answer_guard_llm`；
- `verifier_enabled`；
- `verifier_strictness`；
- `heading_confidence_warn_threshold`。

`query_rewrite_llm` 被读取，但只产生“未启用”warning，不调用 LLM。

### 可比较性问题

- `AppConfig.from_env()` 没有 value/range/cross-field validation；
- runtime 和 eval report 不记录 effective config；
- 没有 config fingerprint；
- 模型是可变 alias 字符串，不记录 provider response 的 resolved identity；
- temperature 固定在代码中，不在 config/report；
- 没有 seed 或 repeated-online-run policy；
- fixture/report 没有 schema version。

因此，即使两次 report 分数不同，也无法仅凭 artifact 判断变化来自代码、数据、配置、依赖还是模型。

## 9. CI、packaging 和 demo operations

### 已实现

- README 有 Windows local setup、API key、OCR 和启动说明；
- `requirements.txt` 声明运行与测试依赖；
- Streamlit 是单一 UI entry point；
- CLI 用非零 exit code表达部分 evaluation failure；
- `.gitignore` 排除 `.env`、documents、eval reports、venv 和 temp artifacts；
- UI 有固定 suggested questions。

### 缺失

- CI workflow；
- dependency lock 与 Python pin；
- package installation metadata；
- lint/type/security gates；
- coverage gate；
- automated synthetic/hard-negative gate；
- online smoke schedule；
- public、版本化 demo policy；
- container/deployment artifact；
- health/readiness beyond Streamlit framework endpoint；
- startup/config validation；
- operational runbook；
- backup/retention/incident policy；
- trace redaction policy。

### Portfolio consequence

当前 demo 能展示：

- 本地 PDF ingestion；
- hybrid retrieval explanations；
- guarded answer UI；
- deterministic tests。

但 reviewer 无法仅从 repo 重复一个真实、质量可测的 end-to-end scenario，也无法比较两个 retrieval/generation changes。这个缺口比增加“agent”角色更影响 Agentic AI Developer 可信度。

## 10. README claims 审核

| Claim | 判断 | 证据 |
| --- | --- | --- |
| synthetic eval 可复现 | 支持，但限定当前环境/fixture | 两次 report hash 相同 |
| 检查 expected rank | 支持 | report 和 pass contract |
| 检查 Recall@k | 不支持 | 无 aggregate Recall@k 实现 |
| 检查 MRR | 不支持 | 无 MRR 实现 |
| 检查引用覆盖 | 不支持 | synthetic path 不生成答案或 claim attribution |
| 检查回答守卫行为 | 部分支持 | hard-negative 调 verifier，但 source-confusion status 未 gate |
| local eval 是不外传检索评测 | 支持 | 本地 parse/chunk/BM25-only |
| local eval 代表真实 RAG retrieval quality | 不支持 | zero vector/BM25-only/self-labeling |
| UI 可查看 retrieval details | 支持 | `AnswerPayload`→Streamlit expander |
| 项目有可观测 RAG flow | 不支持为强声明 | 只有 session diagnostic state，无 traces/metrics |

## 11. 需要建立的 evaluation spine

### Layer A：deterministic regression gate

保留当前 tests 和小型 fixtures，但明确定位为 regression。

每个 PR 必跑：

- unit/component tests；
- synthetic retrieval；
- hard-negative safety；
- CLI exit-code contracts；
- schema validation。

修正 pass contract 时，应让每个 case 显式声明：

```text
expected_retrieval
expected_verifier_status
expected_guard_status
expected_answer_action
```

任何展示在 report 中的关键安全 outcome 都必须进入 pass/fail contract。

### Layer B：版本化 public gold corpus

建立不含私人保单内容、可提交 Git 的小型多文档保险 fixture：

- text PDF；
- scanned/OCR PDF；
- 多 section / heading boundary；
- 相邻数字、期限、主体和责任条款；
- answerable / unanswerable；
- policy fact / general background；
- adversarial instruction；
- degraded document。

独立 gold labels：

- query intent；
- relevant chunk IDs / supporting spans；
- required claims；
- prohibited claims；
- claim→evidence links；
- expected citation source role；
- expected refusal/clarification；
- safety category。

gold label 不能由被测文档中的 literal term自动生成。

### Layer C：分层 metrics

Ingestion：

- parse success rate；
- OCR activation/success rate；
- empty-page rate；
- unknown-heading rate；
- chunk boundary defect rate。

Retrieval：

- Recall@k；
- MRR；
- nDCG@k；
- evidence sufficiency rate；
- hard-negative intrusion rate。

Generation：

- required-claim recall；
- unsupported-claim rate；
- answer correctness/completeness rubric；
- abstention correctness。

Citation：

- citation precision/recall；
- claim attribution coverage；
- supporting-span faithfulness；
- policy/background source-role accuracy。

Safety：

- guard/verifier false-positive / false-negative；
- unsafe final-claim rate；
- source-confusion rate；
- prompt-injection success rate；
- correct refusal/clarification rate。

Operations：

- end-to-end/stage p50/p95 latency；
- input/output/embedding tokens；
- cost per query/document；
- retry/rate-limit/error rate；
- fallback frequency。

每个 metric 都需要定义 denominator、threshold 和 failure interpretation。

### Layer D：bounded online evaluation

在固定 public corpus 上运行真实 embedding/chat：

- pin model/config where provider permits；
- 记录实际 returned model/response ID；
- temperature/seed/retry policy显式化；
- 对 nondeterministic answer 重复运行；
- 使用预算上限；
- 保存 redacted structured result；
- online eval 不替代 deterministic PR gate。

建议：

- deterministic suite：每个 PR；
- small online smoke：手动或受保护 PR；
- larger online benchmark：nightly/release；
- 没有 credential 时明确 `skipped`, 不伪装 pass。

### Layer E：run manifest 与 baseline comparison

每次 eval 产生：

```text
run_id
started_at / duration
git_commit / dirty_state
dataset_name / schema_version / sha256
evaluator_version
effective_config + fingerprint
Python / dependency snapshot
model / embedding identifiers
metric definitions
per-case results
aggregate results
errors / skips
baseline_run_id
delta / threshold verdict
```

输出至少包括 machine-readable JSON 和 human-readable Markdown。Markdown 由同一 JSON render，避免两个事实源。

固定文件名不应静默覆盖历史；run directory 或 content-addressed artifact 更适合比较。

### Layer F：CI gates

PR gate：

- environment install；
- tests；
- deterministic eval；
- artifact upload；
- 与 checked-in baseline 比较；
- threshold regression fail。

Nightly/release：

- online smoke/benchmark；
- latency/cost budgets；
- provider error分类；
- trend artifact。

CI 必须区分：

- pass；
- quality fail；
- environment error；
- credential unavailable / skip；
- provider unavailable。

## 12. 需要建立的 observability spine

### 12.1 Trace model

每个 upload 和 question 生成 pseudonymous IDs：

```text
session_id
document_id = hash/fingerprint, not raw filename
question_run_id
trace_id
```

stage spans：

```text
parse
ocr
chunk
embed/index
query_rewrite
policy_retrieve
rerank
builtin_retrieve
prompt_build
chat_generate
claim_extract
verify
guard
render
```

每个 span 记录：

- start/end/duration/status；
- input/output counts，不默认记录原文；
- config/model/version；
- candidate/selected evidence IDs；
- fallback/error code；
- token/usage/cost；
- verification/guard decision。

### 12.2 Privacy boundary

默认不记录：

- raw PDF；
- full extracted text；
- full prompt/answer；
- API key；
-真实文件路径/用户名。

可记录：

- salted/pseudonymous document ID；
- page/chunk counts；
- text length；
- source role；
- redacted claim/evidence IDs；
- normalized error type；
- metrics。

如果为 debug 明确启用 content capture，需要：

- consent；
- redaction；
- retention；
- access control；
- deletion path。

### 12.3 Operational signals

最小 dashboard/summary：

- request count / success / refusal / block；
- stage latency；
- provider tokens/cost；
- retrieval no-evidence；
- rerank/built-in/guard fallbacks；
- error taxonomy；
- citation/verification status distribution；
- model/config/dataset version。

portfolio 规模不要求部署大型 telemetry stack；结构化 JSONL trace + evaluator ingestion 就能可信展示设计。关键是 schema、correlation、privacy 和可比较性，而不是 vendor 数量。

## 13. 优先级

### P0

1. Evaluation pass contract 与显示字段不一致

   source-confusion verifier 回归可以保持绿色。

2. 没有代表性、独立标注的 gold corpus

   无法测真实 retrieval/generation/citation quality。

3. 没有 run manifest

   结果不能归因到 commit/data/config/model/environment。

4. 没有 claim/evidence labels

   无法测上一 ticket 发现的核心 groundedness gap。

### P1

5. 没有 runtime stage trace、latency、usage/cost 和 error taxonomy。

6. 没有 CI，现有 222 tests 和 CLI eval 不会自动 gate changes。

7. 在线路径与 offline evaluator 不同构。

8. 配置存在 inert fields，effective config 不可观察。

### P2

9. 缺少 dependency lock、Python pin、package/deployment artifact。

10. Demo 没有版本化 public policy、expected outputs 和 runbook。

11. Reports 只有 Markdown、固定文件名、无 trend comparison。

## 14. Agentic AI Developer portfolio 判断

可信的 agentic 能力需要显式状态、工具边界、验证、有限恢复和人类升级；这些都依赖 evaluation 与 traces。

建议未来的 portfolio demonstration：

```text
question
→ evidence selection
→ structured claims
→ claim/evidence verification
→ if failed and budget remains: targeted retrieval/repair
→ else abstain or request clarification
→ emit trace + metrics
```

演示时应能回答：

- 为什么触发 repair？
- repair 使用了哪个工具和新 evidence？
- 哪个 claim 在哪个 citation span 得到支持？
- 预算何时耗尽？
- 为什么最终 abstain？
- 此行为在 gold suite 上提高了什么 metric，代价是多少 latency/cost？

没有 gold cases、pass contract 和 trace 时，添加 planner、critic 或多个 persona 只会增加不可测 nondeterminism。evaluation/observability spine 因此是可信 agentic milestone 的前置基础，而不是附加 polish。

## 15. 对后续 Wayfinder tickets 的输入

### Glossary 输入

- evaluation case；
- gold label；
- relevant evidence；
- selected evidence；
- supporting span；
- required/prohibited claim；
- abstention；
- run manifest；
- trace/span；
- fallback；
- quality gate；
- baseline/delta；
- source-role accuracy。

### ADR 候选

1. 是否使用一个 versioned public gold corpus 作为质量事实源；
2. offline/online evaluation 的分层与 CI cadence；
3. claim/evidence/citation schema；
4. runtime trace schema 和默认 no-content privacy boundary；
5. machine-readable run artifact 与 baseline comparison；
6. provider telemetry、content capture 和 retention policy。

这些是 hard-to-reverse、跨 evaluation/runtime/portfolio 的决策，值得后续 ticket 判断是否正式写 ADR。

### Milestone 输入

最高价值不是“增加更多 metrics”或“接一个 observability vendor”，而是建立一条纵向 tracer bullet：

```text
public gold case
→ production-equivalent RAG run
→ structured claim/evidence result
→ deterministic verifier outcome
→ redacted trace
→ metric + CI verdict
```

它同时关闭：

- quality 不可测；
- claim attribution 不可测；
- agent recovery 不可解释；
- change 不可比较；
- demo 不可复现。

## 16. 可复核命令

```powershell
# 环境与 commit
python --version
python -m pytest --version
git rev-parse HEAD

# 完整 deterministic tests
python -m pytest -ra --durations=10

# 两次离线评估
python scripts\evaluate_rag.py `
  --synthetic `
  --hard-negative `
  --report-dir tmp\wayfinder-7-eval-run1

python scripts\evaluate_rag.py `
  --synthetic `
  --hard-negative `
  --report-dir tmp\wayfinder-7-eval-run2

# 报告 hash
Get-FileHash -Algorithm SHA256 `
  tmp\wayfinder-7-eval-run1\synthetic_eval_report.md,`
  tmp\wayfinder-7-eval-run2\synthetic_eval_report.md,`
  tmp\wayfinder-7-eval-run1\hard_negative_eval_report.md,`
  tmp\wayfinder-7-eval-run2\hard_negative_eval_report.md

# 数据集规模
python -c "import json,pathlib; p=pathlib.Path('evals/synthetic_cases.json'); print(len(json.loads(p.read_text(encoding='utf-8')))); p=pathlib.Path('evals/hard_negative_cases.json'); print(len(json.loads(p.read_text(encoding='utf-8'))))"

# source-confusion pass-contract probe
$env:PYTHONPATH='src'
python -c "import json; from pathlib import Path; import insurance_rag.evaluation as e; cases=json.loads(Path('evals/hard_negative_cases.json').read_text(encoding='utf-8')); target=next(x for x in cases if 'source_confusion' in x['case_id']); target['answer']='这是一般背景定义。'; e._load_cases=lambda _: cases; r=e.evaluate_hard_negative_cases(Path('unused')); x=next(y for y in r.results if 'source_confusion' in y.case_id); print(f'case={x.case_id} verifier={x.verifier_status} passed={x.passed} rank={x.positive_rank}/{x.max_expected_rank}')"

# telemetry/packaging/CI inventory
rg -n "logging|logger|trace|span|token|usage|latency|duration" app.py src scripts tests
git ls-files ".github/**" "Dockerfile*" "pyproject.toml" ".python-version" "*lock*"
```

## 17. 最终判断

InsuranceRAG 已经有一个比普通 demo 更扎实的 deterministic regression foundation，尤其是大量 guard/verifier 和 retrieval rule tests。真正的成熟度瓶颈不是测试数量，而是：

1. 评测 contract 没有覆盖所有声称测量的 outcome；
2. fixture 太小且不是独立 gold data；
3. offline path 不等价于 online RAG；
4. report 不可归因；
5. runtime diagnostic state 没有变成 privacy-aware traces；
6. tests/evals 没有进入 CI；
7. 没有一个可公开复现、可度量、可比较的 end-to-end demo。

下一阶段若想体现 Agentic AI Developer 能力，evaluation/observability 应成为 agent workflow 的骨架：每次 evidence selection、verification、repair 和 abstention 都必须能被 gold case 测量、被 trace 解释、被预算约束。否则 agentic behavior 只是不可验证的 orchestration。
