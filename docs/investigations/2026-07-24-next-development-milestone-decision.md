# InsuranceRAG 下一开发里程碑决策

日期：2026-07-24

代码基准：`a747d8592c3a4c1c5ab23ed8d1e8e1b81c3386c4`

对应 Wayfinder ticket：[Choose and justify the next development milestone](https://github.com/ShiriZhang/InsuranceRAG/issues/10)

决策方式：基于完整审计证据，经 2026-07-24 live HITL grilling 与 repository owner 逐项确认。

## 1. 决策

InsuranceRAG 的下一个开发里程碑是：

> **Measured Evidence Contract（可测的证据契约纵向切片）**

该里程碑把“有证据才回答”从 prompt/README 承诺变成贯穿 production runtime、evaluation、trace、CI 和 UI 的结构化契约。它优先修复当前最高价值的可靠性缺口：系统尚不能可靠表示、测量或证明一个答案断言由哪段保单证据支持，也不能在同一 production-equivalent 路径中区分证据不足、安全阻断和系统失败。

真正的 **bounded agentic evidence-recovery loop** 明确排在本里程碑之后。只有本里程碑建立 baseline、Evidence Sufficiency、Answer Disposition、trace 和评估后，下一阶段才有能力证明动态 tool choice 是否改善结果，以及付出了多少 latency、cost 和 failure surface。

## 2. 为什么它是第一优先级

### 2.1 当前最严重的问题不是组件数量，而是质量不可证

既有调查一致表明：

- 在线系统已经有 ingestion、hybrid retrieval、rule reranking、generation、retrieved-chunk citations、heuristic verification 和 guard；
- retrieval 与 safety 组件大体达到 integrated prototype / heuristic layer；
- 当前 evaluation 直接拼装部分组件，不是在线 `RagChain.answer()` 的完整镜像；
- synthetic `2/2` 和 hard-negative `4/4` 是有价值的 deterministic regression，但不能证明真实 RAG quality；
- 当前所有 selected chunks 都被显示为 citations，无法表达 Claim 与 Supporting Evidence 的对应关系；
- “有任意 policy result”即允许 generation，不是 Evidence Sufficiency；
- runtime 没有可比较的 run manifest、privacy-aware trace 或 production/evaluation parity。

因此再增加 model、reranker、agent persona 或 framework，只会让一个尚不可测的系统变得更复杂。

### 2.2 该里程碑同时关闭多个 P0 缺口

一个纵向 tracer bullet 可以同时建立：

```text
public gold policy case
→ shared production orchestration
→ Retrieved Candidates
→ Answer Claims
→ source-bound Supporting Evidence
→ user-visible Citations
→ deterministic Evidence Sufficiency and safety gates
→ one final Answer Disposition
→ privacy-aware trace and run manifest
→ CI/live evaluation
```

它比孤立增加 metrics、修一个 retriever 或包装一个 agent 更能形成可审查、可重复的 portfolio evidence。

### 2.3 它是可信 agentic 能力的前置基础

未来 bounded evidence recovery 需要回答：

- initial evidence 为什么不充分；
- 模型选择了哪个 read-only retrieval action；
- evidence set 是否取得进展；
- 第二次行动是否提高 grounded answer rate；
- 为什么最终 answer、clarify 或 abstain；
- 增加了多少 latency、token、cost 和失败概率。

没有本里程碑提供的 contracts、gold cases 和 traces，这些问题无法客观回答。推迟 agent loop 不是放弃 Agentic AI，而是拒绝装饰性 agentification。

## 3. 里程碑范围

### 3.1 Authoritative public gold corpus

建立由仓库自行编写、可公开提交的 synthetic insurance policies 和人工 gold annotations。

Corpus 必须能够表达代表性保险 QA 难点，包括：

- 等待期、保险责任、责任免除和保费豁免；
- 投保人、被保险人等主体差异；
- 数值、期限、条件、例外和跨条款关系；
- 词语相似但事实不相关的 hard negatives；
- 证据缺失、问题歧义、source confusion 和高风险理赔措辞；
- 能触发全部五种 Answer Disposition 的 cases。

真实、私人或第三方商业保单：

- 不进入仓库；
- 不作为 CI 的质量事实源；
- 只允许作为 opt-in local exploratory evaluation；
- 不得把内容或派生敏感 artifacts 提交到版本控制。

### 3.2 Structured evidence contract

Runtime 和 evaluation 采用同一组核心概念：

- **Answer Claim（答案断言）**：答案中可独立核验的事实性命题；
- **Retrieved Candidate（检索候选）**：retrieval/reranking 返回的候选 chunk，不自动构成证据；
- **Supporting Evidence（支持证据）**：实际支持一个 Answer Claim、具有稳定来源定位的精确 policy-text span；
- **Citation（来源引用）**：向用户展示的 Supporting Evidence 来源位置和 excerpt。

必须满足：

```text
Answer Claim
→ one or more Supporting Evidence spans
→ one or more user-visible Citations
```

没有支持任何 Answer Claim 的 Retrieved Candidate 只能进入 diagnostics，不得作为 Citation 展示。

### 3.3 Evidence Sufficiency 与 Answer Disposition

Milestone 建立结构化 Evidence Sufficiency gate，并输出且只输出以下五种互斥最终 disposition：

```text
answer
needs_clarification
abstain_insufficient_evidence
blocked_safety
failed_system
```

语义：

| Disposition | 含义 |
| --- | --- |
| `answer` | 已有足以形成受约束解释的 Supporting Evidence。 |
| `needs_clarification` | 问题存在用户能够解决的歧义，需要定向澄清。 |
| `abstain_insufficient_evidence` | 问题明确，但 User Policy 没有足够支持证据。 |
| `blocked_safety` | Candidate answer 触犯最终理赔判断、source authority 或其他安全 postcondition。 |
| `failed_system` | Provider、解析、schema 或 runtime 故障导致本次运行无法完成。 |

`abstain`、`block` 和 `failure` 不得共用一个模糊“拒答”状态。

### 3.4 Hybrid control allocation

模型可以通过 structured output 提议：

- Answer Claims；
- Claim-to-Evidence links；
- 问题歧义；
- 缺失的事实类型；
- 其他需要语义理解的候选判断。

Deterministic code 必须拥有：

- schema validation；
- source role 与 policy scope；
- span existence/identity；
- Evidence Sufficiency gate；
- verifier/guard；
- provider/runtime failure classification；
- 最终 Answer Disposition。

模型不得宣布自己的输出“证据充分”，不得绕过 verifier/guard，也不得直接拥有最终 termination authority。

### 3.5 One shared core orchestration

以下三个入口必须调用同一条 application-level orchestration 和同一组 evidence/disposition contracts：

1. Streamlit runtime；
2. deterministic CI runner；
3. opt-in live-model benchmark。

入口之间只允许替换：

- provider adapters；
- corpus/input；
- effective run configuration。

不得继续维护三条“功能相似但行为不等价”的 pipelines。允许为此从当前 `RagChain` 提取共享 seam，但不要求更换 UI 或 agent framework。

### 3.6 Two-layer evaluation

#### Deterministic CI gate

- 不需要 API key；
- 不访问网络；
- 每个 PR 可稳定执行；
- 使用 public gold corpus；
- 检查 schema、evidence links、dispositions、安全结果和 regression metrics；
- 产生 machine-readable report 与 run manifest。

#### Opt-in live-model benchmark

- 使用与 deterministic runner 相同的 cases、contracts、case IDs 和 metrics；
- 调用真实 embedding/chat provider；
- 记录 model identity、prompt/config、latency、token、cost 和 outcome；
- 手动或定期执行；
- 不作为普通 PR 的必需检查；
- provider 波动或缺少 credentials 必须归类为 environment/provider condition，不得伪装成质量失败或证据不足。

### 3.7 Privacy-aware manifest and trace

真实用户运行默认使用 **no-content trace**，只记录：

- run/correlation IDs；
- dataset/policy/index hashes 或 pseudonymous IDs；
- candidate/evidence/span IDs；
- state transitions、scores、warnings 和 disposition；
- model/config/prompt/schema versions；
- latency、token 和 cost metadata；
- typed failure category。

默认不记录：

- 用户问题原文；
- 保单原文；
- Supporting Evidence 文本；
- 完整 prompt；
- 模型回答原文；
- 聊天历史。

完整内容 trace 只允许用于 repository-owned public synthetic gold runs。显式 local debugging 可以 opt in 敏感内容，但 artifacts 必须保持本地、被 Git 忽略且不得进入通用 telemetry。

### 3.8 Minimal Streamlit integration

Streamlit 必须消费新的 evidence/disposition contract：

- 用户答案只显示与 Answer Claims 关联的 Citations；
- Retrieved Candidates 仅在明确标注的 diagnostics 中显示；
- UI 区分 clarification、evidence abstention、safety block 和 system failure；
- 保留必要的 retrieval/verification 可解释信息。

UI 只做支持新行为所需的最小修改，不做视觉重设计。

### 3.9 Reproducible portfolio demonstration

基于同一 public gold corpus，提供从仓库可重复运行的 end-to-end demonstration，至少展示：

- `answer`：每个 Answer Claim 映射到精确 Supporting Evidence 和 Citation；
- `needs_clarification`：存在可解决歧义；
- `abstain_insufficient_evidence`：问题明确但保单无足够证据；
- `blocked_safety`：最终理赔判断、source confusion 或其他安全违规；
- `failed_system`：provider/runtime failure 与证据不足明确分离；
- 每次运行都有 schema-valid manifest/trace；
- deterministic baseline 和 live result 可按相同 case IDs、metrics 与 dispositions 比较。

## 4. 明确 non-goals

本里程碑不包含：

- bounded agentic evidence-recovery loop；
- Agents SDK、LangGraph 或其他 framework migration；
- retrieval/citation/critic 多代理角色；
- unlimited reflection 或 self-critique loop；
- OCR、layout parsing 或 ingestion 的全面改造；
- 长期用户保单、聊天或完整 trace 持久化；
- 登录、部署、SLO、incident response 等 production platform 工作；
- 覆盖所有保险产品的完整 ontology；
- 自动理赔判断；
- 法律、医疗、财务、理赔或核保建议；
- 在 gold baseline 建立前继续堆叠 retrieval/reranking 算法；
- Streamlit 视觉重设计。

## 5. 依赖

### 必须已有

- 当前可执行 ingestion/retrieval/reranking/generation/guard components；
- 已建立的 test/runtime baseline；
- source-role 和 safety 边界；
- repo 内可运行的 deterministic test environment；
- 对 public synthetic corpus 的明确 ownership。

### 里程碑内需要建立

- versioned gold data/annotation schema；
- stable Claim、Evidence、Citation 和 Disposition schemas；
- provider adapter seam；
- shared application orchestration；
- deterministic and live runner interfaces；
- run manifest 与 trace schema；
- CI workflow 与 artifact policy；
- public/local data separation。

### 后续 agentic milestone 的依赖输出

- measurable initial Evidence Sufficiency；
- typed evidence gaps；
- stable evidence IDs；
- explicit run state 与 dispositions；
- baseline outcome/latency/cost metrics；
- trace ingestion；
- cases where a second retrieval action could be evaluated against the deterministic baseline。

## 6. 主要风险与处理方向

| 风险 | 影响 | 处理方向 |
| --- | --- | --- |
| Synthetic corpus 代表性有限 | 对真实商业保单的外推不足 | 覆盖多文档结构和 hard negatives；明确不作 production-quality 外推。 |
| Gold annotation 偏差 | Metrics 可能奖励错误 contract | 采用显式 schema、review checklist 和 versioning；保留可审查 spans。 |
| Claim-to-span modeling 复杂 | Generation/schema failure 增加 | 从小型纵向 corpus 开始；所有 schema failure 显式归类。 |
| Shared orchestration refactor regression | UI 与 eval 行为短期波动 | 先固定现有 baseline，以 characterization/integration tests 保护 seam。 |
| Sufficiency gate 过严 | Answer rate 下降、false abstention | 同时测 grounded answer 与 abstention；不以 answer rate 单指标优化。 |
| Sufficiency gate 过松 | Unsupported claims 继续输出 | Gold unsupported/irrelevant/source-confusion cases 必须进入 gate。 |
| Live model 漂移 | 结果跨时间不可直接比较 | 记录 model/config/prompt，区分 CI deterministic baseline 与 live observations。 |
| Trace 泄露敏感内容 | 隐私与治理风险 | 用户运行默认 no-content；full content 仅限 public synthetic fixtures。 |
| Milestone scope 膨胀 | 推迟可交付 tracer bullet | 遵守 non-goals；threshold 和 task decomposition 在后续 specification ticket 定义。 |

## 7. Completion evidence 的类别

本 ticket 选择 milestone，不为后续 specification 预设最终数值阈值。下一 ticket 必须将以下类别转换为 measurable acceptance criteria：

- retrieval relevance；
- Evidence Sufficiency accuracy；
- grounded Answer Claim rate；
- unsupported Claim rate；
- claim-to-Citation precision/recall；
- correct Answer Disposition；
- source-confusion/final-claim safety outcomes；
- schema validity；
- production/evaluation parity；
- deterministic reproducibility；
- live latency/token/cost；
- trace completeness 与 privacy checks。

完成证明必须来自可执行 reports、traces、CI results 和 demo cases，而不是 README 声明或截图。

## 8. Portfolio value

该里程碑能够可信展示：

- **AI contract design**：自由文本模型输出被转换为结构化 Claim/Evidence/Disposition contract；
- **Control allocation**：模型处理语义，deterministic code 持有 scope、verification、safety 和 termination；
- **Evaluation engineering**：public gold corpus、deterministic gate 与 live benchmark 分层；
- **Production parity**：runtime 与 evaluation 共享 orchestration；
- **Observability**：版本化 manifest、typed trace 和 failure semantics；
- **Privacy judgment**：真实用户运行默认 no-content；
- **Safety**：insufficient evidence、unsafe output 与 system failure 不再混为一谈；
- **Agentic restraint**：在没有可测 baseline 前，不引入 decorative agent complexity；
- **Future agent readiness**：为 bounded read-only evidence recovery 提供 state、tools、feedback 和 evaluation foundation。

准确的 portfolio 表述是：

> InsuranceRAG implements a measured claim-to-evidence contract across a shared production and evaluation workflow, with deterministic sufficiency and safety control, privacy-aware traces, reproducible offline gates, and comparable live-model benchmarks.

本里程碑完成前，不应声称：

- semantic grounding 已得到完整保证；
- evaluation 代表全部真实保险产品；
- 当前系统已是 agent；
- bounded recovery 已改善质量；
- production privacy/operations 已完成。

## 9. 为什么其他候选必须排后

| 候选 | 为什么不先做 |
| --- | --- |
| Bounded agentic evidence recovery | 没有 Evidence Sufficiency、gold cases 和 trace 时，无法证明第二次 action 的必要性或收益。 |
| Agent framework migration | Framework 不是 capability；当前没有需要它承载的已测 agent loop。 |
| 多代理 decomposition | 高度共享同一 evidence，只增加 coordination、token、latency 和 debugging surface。 |
| 新 retrieval/reranking 算法 | 当前没有代表性 gold baseline，无法判断 improvement。 |
| OCR/layout overhaul | 是重要但较窄的 ingestion reliability 工作，不先解决回答证据不可测的问题。 |
| Production deployment/telemetry vendor | 会扩大治理和运维范围，但仍不能补齐 Claim/Evidence contract。 |
| UI redesign | 提升展示，不提升 grounded answer 的可证性。 |

## 10. 已固化的领域与架构决定

Live agreement 已同时产生以下 authoritative docs：

- [`CONTEXT.md`](../../CONTEXT.md)：Answer Claim、Retrieved Candidate、Supporting Evidence、Citation、Answer Disposition；
- [`ADR-0001`](../adr/0001-separate-retrieval-provenance-from-citations.md)：retrieval provenance 与 Citation 分离；
- [`ADR-0002`](../adr/0002-keep-final-disposition-under-deterministic-control.md)：deterministic code 拥有最终 disposition；
- [`ADR-0003`](../adr/0003-default-to-content-free-traces-for-user-runs.md)：真实用户运行默认 no-content trace；
- [`ADR-0004`](../adr/0004-share-one-core-orchestration-across-runtime-and-evaluation.md)：runtime/evaluation 共享核心 orchestration。

## 11. 下一步

下一 Wayfinder ticket 应把本决策转化为 implementation-ready specification，明确：

- schema fields 与 invariants；
- gold corpus cut；
- test matrix；
- quantitative thresholds；
- CI/live commands；
- failure taxonomy；
- trace/manifest schemas；
- UI behavior；
- rollout/migration sequence；
- demonstration cases；
- Definition of Done。

该 specification 不应重新打开本文件已经 live-agreed 的 milestone 选择和 non-goals，除非新的可执行证据证明存在冲突。

## 12. 本决策未做的事项

- 未实现 milestone；
- 未修改 application、tests、dependencies、README 或 CI；
- 未运行或调用真实模型；
- 未读取本地真实保单；
- 未认领或解决后续 specification ticket；
- 未处理其他 Wayfinder ticket。
