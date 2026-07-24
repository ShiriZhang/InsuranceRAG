# InsuranceRAG 领域词汇与 ADR 输入登记册

日期：2026-07-23

代码基准：`86ecf17f9cd4f7baef7a99e0b3d52132fdae95ab`

对应 Wayfinder ticket：[Prepare the insurance-policy QA glossary and ADR input register](https://github.com/ShiriZhang/InsuranceRAG/issues/9)

## 1. 结论摘要

本仓库当前没有 `CONTEXT.md`、`CONTEXT-MAP.md` 或 `docs/adr/`。调查确认：

- 产品边界已有一致证据：InsuranceRAG 是“基于用户上传保单的条款解释助手”，不是理赔、法律、医疗、财务或核保决策系统；
- 用户保单与内置背景资料承担不同 source roles：前者是用户保单事实的主要依据，后者只能补充术语或背景；
- 当前代码把“检索到的 chunk”直接展示为 citation，并用 citation excerpt 做有限的 rule-based fact verification；这不等于 claim-level supporting evidence；
- 当前固定顺序的 `RagChain` 是 deterministic RAG workflow，不是 agentic workflow；
- session-only 数据生命周期、source authority、claim/evidence/citation contract，以及未来 bounded evidence-recovery 的控制边界，都是值得进一步决策的架构事项；
- BM25、RRF、Streamlit、OpenAI client、regex guard、具体 top-k 等是实现选择，不应进入领域词汇表，也不满足当前 ADR 门槛。

本轮没有发生对具体术语定义或架构选项的 live agreement。因此，本文件只登记可供确认的准确输入：

- **没有创建 `CONTEXT.md`**；
- **没有创建 ADR**；
- “ready for agreement”表示证据充分、建议定义已成形，但仍不是 canonical language；
- “requires clarification”表示存在会改变产品行为或评估契约的歧义，不能由调查者代替 owner 决定。

## 2. 方法与证据边界

### 2.1 判定规则

词汇候选必须同时满足：

1. 是 InsuranceRAG 业务或产品上下文特有概念，不是通用编程术语；
2. 在可执行代码、测试或当前产品承诺中有具体证据；
3. 能给出一到两句、排他的定义，并指出应避免的近义词；
4. 若定义会改变行为，则必须等待 live agreement 后才写入 `CONTEXT.md`。

ADR 候选必须同时通过：

1. **Hard to reverse**：改变后会牵动多个数据、接口、评估或安全边界；
2. **Surprising without context**：只看代码时，后继维护者很可能误以为是偶然限制并“修正”它；
3. **Real trade-off**：存在可信替代方案，选择带来明确收益与代价。

### 2.2 主要证据

- `README.md:3-13,45-52,98-105,147`：产品定位、source authority、外部数据流、session-only 承诺和禁止最终理赔判断；
- `src/insurance_rag/rag_chain.py:28-39,61-83,108-177`：source role、无 policy result 时拒答、prompt、generation、retrieved-chunk citation 和 guard 编排；
- `src/insurance_rag/models.py:14-146`：chunk、citation、verified fact、guard result 和 answer payload 的当前结构；
- `src/insurance_rag/answer_guard.py:12-177`：最终理赔措辞、source confusion、policy fact support、warning/block 规则；
- `src/insurance_rag/citation_verifier.py:96-200`：closed-ontology、regex/字符串匹配式 fact verification；
- `app.py:27-44,104-183,233-310`：session state 与 answer/citation/verification 展示；
- [端到端架构与代码状态调查](./2026-07-23-end-to-end-architecture-code-state.md)；
- [生成、引用、安全与失败处理调查](./2026-07-23-generation-citation-safety-maturity.md)；
- [评估、可观测性与运行成熟度调查](./2026-07-23-evaluation-observability-operational-maturity.md)；
- [Agentic workflow 与 portfolio credibility 调查](./2026-07-23-agentic-workflow-portfolio-credibility.md)。

本调查没有读取 `documents/` 中的本地 PDF，没有调用真实 OpenAI 服务，也没有以设计文档代替可执行证据。

## 3. 可进入 live agreement 的词汇输入

下表中的定义与现有产品边界一致，适合作为首次 `CONTEXT.md` 讨论稿。它们尚未被写入 authoritative glossary。

| 建议 canonical term | 建议定义 | `_Avoid_` | 证据与限制 |
| --- | --- | --- | --- |
| **User Policy（用户保单）** | 用户为当前问答提供、并被系统作为该保单事实主要依据的保险合同资料。 | 上传文件、用户文档、policy corpus | README 与 prompt 一致支持“主要依据”；但“上传内容是否一定是有效合同”仍需输入校验决策，见第 4 节。 |
| **Background Material（背景资料）** | 为解释通用保险术语或背景而使用的内置资料；它不能单独证明 User Policy 的具体事实。 | 内置保单、第二保单、权威保单证据 | `should_use_builtin_context()` 仅在已有 policy results 且问题像定义问题时触发；prompt、guard 和 UI 都分离 source role。 |
| **Policy Fact（保单事实）** | 对 User Policy 内容作出的、可独立核对真假的具体陈述，例如期间、金额、责任、除外或主体。 | 普通回答、模型意见 | guard/verifier 已按数字、期限、责任、免责和主体检查部分事实；当前识别范围有限。 |
| **Claim（答案断言）** | 生成答案中一个可独立验证的事实性命题。 | 整段回答、检索结果、claim adjudication | `VerifiedFact` 已表达 fact text/status/severity；必须与保险“理赔申请/claim”区分，建议中文始终写“答案断言”。 |
| **Retrieved Candidate（检索候选）** | 检索或重排后进入候选集合的 policy/background chunk；它表示相关性候选，不自动表示事实支持。 | 证据、引用、已验证事实 | 当前 `HybridSearchResult`/`DocumentChunk` 是实际检索单位；审计已证明非空结果不能代表 evidence sufficiency。 |
| **Citation（来源引用）** | 向用户展示某个来源位置及节选的 provenance 记录；在当前实现中，它说明该 chunk 被检索并送入回答流程，不保证支持某条 Claim。 | 支持证据、verified citation、inline citation | `build_citation()` 为每个 selected chunk 构造 citation；当前没有 answer-to-citation mapping。 |
| **Supporting Evidence（支持证据）** | User Policy 中能够支持某一 Claim 的具体文本 span，并保留稳定来源定位。 | 任意 citation、matched term、retrieval score | 当前只有 heuristic supporting citation ID，没有 span offset 或稳定 chunk/span ID；这是目标 contract，不是已成熟能力。 |
| **Evidence Sufficiency（证据充分性）** | 对当前问题而言，已选 Supporting Evidence 是否足以形成受约束解释的判定。 | 有检索结果、top-k 非空、低/高 RRF score | 当前未实现 calibrated sufficiency；“找到任意 policy chunk”只是一道 generation gate。 |
| **Source Confusion（来源混淆）** | 把 Background Material 或其他来源的内容表述为 User Policy 事实。 | 检索错误、一般 hallucination | verifier/guard 有独立规则，但只覆盖已编码措辞与事实类型。 |
| **Policy Explanation（保单条款解释）** | 基于 User Policy 条款，对其含义、条件和限制作出的通俗说明，不替代合同原文或专业最终判断。 | 保险建议、理赔建议、最终解释 | README、system prompt 和免责声明一致支持该产品边界。 |
| **Final Claim Decision（最终理赔判断）** | 对具体事故、疾病或费用是否必然赔付、拒赔或报销作出的最终结论；它不属于 InsuranceRAG 的输出范围。 | 保单解释、条件式说明 | system prompt、answer guard 与 README 都明确禁止；规则覆盖并不完整。 |
| **Abstention（证据性拒答）** | 因没有足够 User Policy 证据、核验失败或无法安全回答而不生成实质性解释的 disposition。 | Block、error、免责声明 | 当前无检索结果和 guard block 都会替换答案，但尚未形成统一 disposition schema。 |

### 建议的首次 glossary cut

若 owner 同意上述语义，首次 `CONTEXT.md` 应只纳入以下九个最稳定的产品概念：

```text
User Policy
Background Material
Policy Fact
Claim（答案断言）
Retrieved Candidate
Citation
Supporting Evidence
Policy Explanation
Final Claim Decision
```

`Evidence Sufficiency`、`Source Confusion` 和 `Abstention` 也有充分价值，但应在第 4 节中的行为 taxonomy 确认后再定稿，避免 glossary 先于 runtime contract。

## 4. 必须由 owner 澄清的术语与场景

### 4.1 “Policy”指合同、上传文件，还是已验证的合同版本？

场景：用户上传产品说明书、投保建议书、批单与主合同的组合 PDF，当前系统都会把解析结果标成 `source_type="policy"`。

必须决定：

- `User Policy` 是否只是用户指定的问答权威来源；
- 还是必须通过文档类型、合同主体和版本有效性校验后才能称为 User Policy；
- 批单、附加险和主险冲突时，谁拥有 authority。

建议：当前 glossary 采用“用户为本次问答指定的合同资料”这一窄定义，同时明确系统不验证法律有效性；未来多文档/版本 contract 另行决策。

### 4.2 Citation 是 provenance，还是对 Claim 的支持关系？

场景：回答由一个 chunk 支持，但 UI 展示全部 retrieved chunks；用户可能把每条 citation 都理解为该答案的依据。

必须决定：

- 保留 `Citation = provenance`，另设 `Supporting Evidence`；
- 或将 `Citation` 升级为严格 claim-to-span attribution，其他检索结果只能叫 Retrieved Candidate。

建议：选择第二种用户语义：对外的 Citation 必须指向 Supporting Evidence；内部仍可保留 retrieval provenance diagnostics。当前实现未达到该定义，必须在文档中标为 gap。

### 4.3 Evidence 是 chunk、excerpt、span，还是 verifier 的匹配结果？

场景：model 看完整 chunk，用户和 verifier 通常只看前 180 字 excerpt；支持事实可能位于截断位置之后。

必须决定：

- evidence 的最小身份与稳定 ID；
- span 是否必须落在用户可见 excerpt；
- 一个 Claim 是否允许跨多个 spans/clauses 组合支持；
- retrieval score 和 matched terms 是否永远不得当作事实支持。

建议：`Supporting Evidence = source-bound span`；chunk 只是容器，score/term 只是 retrieval signals。

### 4.4 Verification 是规则检查，还是语义支持证明？

场景：当前 regex/词典能够验证部分数字、期限、主体与责任措辞，但 OOV、改写、否定、条件、例外和跨条款推理可能漏检或误判。

必须决定：

- 对外是否只称 `Rule-based Verification`；
- 未来 `Claim Verification` 的完成标准是否要求 claim schema、span links、gold labels 和 false-positive/false-negative metrics。

建议：当前能力统一称 `Rule-based Claim Check`，不要使用暗示完备性的“事实核验保证”。

### 4.5 Answer、Clarification、Abstention 与 Block 的关系

场景：当前无检索结果时返回 refusal；识别到高风险输出时返回 blocked answer；没有结构化 clarification 状态。

必须决定统一 disposition taxonomy：

```text
answer
needs_clarification
abstain_insufficient_evidence
blocked_safety
failed_provider
```

建议：`Abstention` 表示系统根据证据状态主动不答；`Block` 表示 candidate answer 违反安全 postcondition；`Failure` 表示系统无法完成运行。三者不能共用一个“拒答”标签。

### 4.6 Coverage（保障）是否允许推导个案赔付？

场景：用户问“这份保单保障什么”属于条款解释；问“我这次住院能赔吗”需要个案事实、有效合同、等待期、除外与理赔审核。

必须决定：产品是否永远只解释 coverage conditions，还是未来允许收集个案材料形成非最终的 eligibility explanation。

建议：当前 context 只定义 Policy Explanation，不引入 Coverage Decision；个案赔付自动判断继续明确排除。

### 4.7 “Agentic”是否是产品领域词汇？

场景：当前 pipeline 固定顺序执行，未来建议只有 evidence gap 时才允许 bounded read-only recovery loop。

建议结论：`agentic workflow`、tool、state、budget、stop reason 属于工程架构语言，不进入保险 QA 的 `CONTEXT.md`。它们可以出现在 ADR、设计和 portfolio claim 中。

## 5. 不应进入 `CONTEXT.md` 的名称

| 名称 | 分类 | 原因 |
| --- | --- | --- |
| BM25、vector search、RRF、rule reranker | 实现技术 | 可替换的 retrieval 策略，不是保险 QA 领域概念。 |
| chunk、top-k、embedding、token | 通用 RAG/LLM 技术 | 可以出现在接口和实现文档，不属于 ubiquitous domain language。 |
| Streamlit session state、dataclass、OpenAI client | 框架/代码结构 | 一般编程概念，且可能随实现替换。 |
| PASS/WARN/BLOCK、SUPPORTED/UNSUPPORTED/UNCERTAIN | 当前 enum/value | 在行为 taxonomy 达成共识前只是实现状态；不能反向定义产品语义。 |
| policy retriever、background retriever | 组件名 | 应由 source roles 和 evidence contract 定义组件，而不是把组件名写入领域模型。 |
| “retrieval agent”“citation agent”“critic agent” | decorative agent naming | 当前没有独立 decision authority、tools/state 或 handoff；会错误夸大 architecture。 |

## 6. ADR 输入登记册

状态说明：

- **Ready for decision**：三项 ADR 测试全部通过，但尚未得到 live agreement；
- **Defer**：可能成为 ADR，但依赖后续 milestone 或产品范围选择；
- **Reject as ADR**：当前是可逆实现选择、事实记录或尚无真实 trade-off。

### ADR-A：User Policy 是保单事实的唯一默认 authority

**状态：Ready for decision**

建议决策：

> User Policy 是用户特定保单事实的唯一默认 authority。Background Material 只能解释通用术语和背景；当 User Policy 没有支持时，系统不得用背景资料补齐为“你的保单”事实。

三项测试：

| 测试 | 结果 | 依据 |
| --- | --- | --- |
| Hard to reverse | 通过 | 影响 ingestion source role、retrieval、prompt、claim/evidence schema、citation UI、安全评测和未来 tools。 |
| Surprising | 通过 | 仓库确实索引内置 PDF；维护者很容易把它当作第二知识库直接回答。 |
| Real trade-off | 通过 | 严格 authority 降低覆盖率与回答率，但防止跨保单/source confusion；宽松融合相反。 |

待确认替代方案：

1. strict user-policy-only facts（建议）；
2. background 可填补但必须显式标为 general information；
3. 允许多级权威来源并建立冲突规则。

### ADR-B：保留 session-only 的敏感数据生命周期

**状态：Ready for decision**

建议决策：

> 默认不持久化用户上传 PDF、解析文本、embeddings 或聊天历史；需要可恢复 workflow 时，只允许保存最小化、可配置保留期且经过 redaction 的状态，并与原文持久化分开决策。

三项测试：

| 测试 | 结果 | 依据 |
| --- | --- | --- |
| Hard to reverse | 通过 | 一旦引入持久化，会牵动身份、授权、加密、删除、retention、备份、审计和数据迁移。 |
| Surprising | 通过 | 未来 resumable/HITL/observability 很容易诱导开发者直接持久化完整 policy 与 prompt。 |
| Real trade-off | 通过 | session-only 降低隐私与治理风险，但失去 resume、longitudinal eval、support diagnostics 和多设备连续性。 |

待确认：portfolio milestone 是否仍保持完全 session-only，还是允许只持久化 pseudonymous IDs、hash、metrics 和 redacted workflow state。

### ADR-C：分离 Retrieved Candidate、Supporting Evidence、Claim 与 Citation

**状态：Ready for decision**

建议决策：

> Runtime 和 evaluation contract 必须区分检索候选、选定的 source-bound evidence spans、答案断言及其 citation links。对外 Citation 只引用 Supporting Evidence；未被使用的 Retrieved Candidates 只进入 diagnostics。

三项测试：

| 测试 | 结果 | 依据 |
| --- | --- | --- |
| Hard to reverse | 通过 | 会决定核心 schema、prompt/output format、verifier、UI、gold labels、trace 和 metrics。 |
| Surprising | 通过 | 当前 `Citation` 是“所有 retrieved chunks”；未来维护者可能沿用名称并错误声称 faithfulness。 |
| Real trade-off | 通过 | 严格 attribution 增加 schema、generation 和标注成本，但获得可测 citation precision/recall 与用户可解释性。 |

待确认：Citation 对外采用严格语义，还是保留 provenance 语义并另设 `ClaimEvidenceLink`。建议前者。

### ADR-D：用 deterministic safety shell 包围 bounded agentic evidence recovery

**状态：Defer，等待下一 milestone 明确**

潜在决策：

> 只有 evidence sufficiency 明确失败时，single agent 才能在少量 read-only retrieval actions 中做有限选择；scope、attempt/cost budget、no-progress、verification、guard、privacy 和最终 disposition 始终由 deterministic code 控制。

三项测试：

| 测试 | 结果 | 依据 |
| --- | --- | --- |
| Hard to reverse | 通过 | 决定 workflow state、tool contracts、control allocation、trace/eval 和 safety boundary。 |
| Surprising | 通过 | portfolio 项目常把所有 stages 交给 agent；这里刻意限制 model authority。 |
| Real trade-off | 通过 | bounded loop提高复杂 query 的恢复能力，但增加 latency、cost、state 和 trajectory evaluation。 |

暂缓原因：现有代码仍是 deterministic A1 workflow；issue #10 尚未选择下一 milestone。此 ADR 只能在 owner 选择 agentic seam 后提出，不能把建议写成既成决策。

### ADR-E：建立版本化 public gold corpus 作为质量 truth source

**状态：Defer，等待 milestone scope 与数据许可**

潜在决策：

> 质量声明以可提交、版本化、无私人保单内容的多文档 gold corpus 和 run manifest 为依据；本地 `documents/` 与自标注 BM25 reports 不作为 release truth source。

三项测试：

| 测试 | 结果 | 依据 |
| --- | --- | --- |
| Hard to reverse | 部分通过 | corpus/schema 会影响长期趋势和 release gates，但早期仍可版本演进。 |
| Surprising | 通过 | 仓库已有大量本地 PDF 与 eval CLI，容易被误认为真实质量基线。 |
| Real trade-off | 通过 | public corpus 可复现、可审查，但标注成本高、代表性有限并需处理授权。 |

暂缓原因：需先决定下一 milestone 是否以 evaluation spine 为核心，并明确 fixture 的来源许可和标注责任。

## 7. 当前不应创建的 ADR

| 候选 | 判定 | 原因 |
| --- | --- | --- |
| 使用 Streamlit | Reject as ADR | 当前 demo UI，替换成本有限，没有证据表明已做长期平台 trade-off。 |
| 使用 OpenAI Chat Completions / embeddings | Reject as ADR | 是现状，不等于已完成 provider lock-in 决策；模型/API 迁移边界尚未设计。 |
| 使用 BM25 + vector + RRF | Reject as ADR | 有测试的 retrieval 实现，但仍是可实验、可比较的算法选择。 |
| 使用 rule reranker / regex verifier | Reject as ADR | 当前实现和成熟度限制，未形成应该长期保留的架构承诺。 |
| 单份 PDF、top-k、chunk size、180 字 excerpt | Reject as ADR | 配置或 MVP 限制；没有通过 hard-to-reverse 测试。 |
| 当前 workflow 不是 agent | Reject as ADR | 这是代码事实，不是 trade-off 决策。 |
| 直接迁移 Agents SDK 或 LangGraph | Reject as ADR | 没有先选出需要 framework 支持的 capability；迁移本身不是产品决策。 |
| 创建 retrieval/citation/critic 多代理 | Reject as ADR | 不解决当前 evidence contract，且属于 decorative agent decomposition。 |

## 8. 建议的 live agreement 顺序

为避免一次讨论同时混入产品、数据、安全和 framework 选择，建议依次确认：

1. **产品 authority**：User Policy 与 Background Material 的边界；
2. **证据语言**：Claim、Retrieved Candidate、Supporting Evidence 与 Citation；
3. **输出 taxonomy**：answer、clarify、abstain、block、failure；
4. **数据生命周期**：session-only 是否作为长期默认；
5. **下一 milestone**：先建 evaluation/evidence contract，还是同时加入 bounded evidence recovery；
6. 仅对确认且通过三项测试的决策创建顺序编号 ADR。

一旦 owner 对第 1–3 项给出明确同意，可以按 `domain-modeling` 格式创建根目录 `CONTEXT.md`；一旦对 ADR-A/B/C 中的具体选项达成选择，再创建 `docs/adr/0001-*.md` 起始文件。调查结果本身不能替代这一步 agreement。

## 9. 对后续 Wayfinder 的准确输入

- issue #10 应把“先建立 claim/evidence/citation 与 evaluation spine”视为 agentic milestone 的前置条件，而不是把 framework migration 或多代理命名当作 milestone；
- 若 issue #10 选择 bounded evidence recovery，应显式决定 ADR-D，并把 `Evidence Sufficiency` 与 disposition taxonomy 纳入 runtime state；
- issue #11 在更新 `CONTEXT.md` 或 ADR 时，应引用本登记册并记录 owner 的 live agreement，而不是直接把全部候选复制为 accepted；
- portfolio 可准确声称当前具有 source-role separation、retrieval provenance、selected rule-based claim checks 和 deterministic fail-closed guard；
- portfolio 不应声称 semantic citation faithfulness、完整事实核验、真实 RAG quality、agentic workflow 或 production privacy governance。

## 10. 本调查未做的事项

- 未创建或修改 `CONTEXT.md`；
- 未创建 ADR；
- 未修改 application、tests、README、依赖或配置；
- 未读取本地真实保单；
- 未调用外部模型；
- 未替 owner 决定模糊术语或架构 trade-off；
- 未处理或解决其他 Wayfinder ticket。
