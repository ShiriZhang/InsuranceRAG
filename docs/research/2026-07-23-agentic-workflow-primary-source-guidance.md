# Agentic workflow 成熟度：第一方设计与评测指导

日期：2026-07-23

用途：为 InsuranceRAG Wayfinder ticket “Assess agentic workflow maturity and portfolio credibility” 提供外部判据。

范围：只采用 OpenAI 与 Anthropic 的官方开发者文档或第一方工程文章；本文不是对当前仓库实现的审计结论。

## Executive summary

对一个保险 policy QA assistant 而言，可信的 Agentic AI 能力不等于增加多个角色、handoff 或自我反思提示。第一方资料支持一条更克制的路线：

1. 对固定、可预测的 RAG 阶段使用 code-orchestrated workflow；只有当下一步无法可靠预先编码、模型必须根据中间证据动态选择动作时，才引入 agent loop。
2. 将工具作为有 schema、有权限边界、有错误语义的能力接口；模型只决定真正需要语义判断的部分。
3. 把 workflow state、证据、tool result、停止原因和人工审批状态作为显式、可恢复的数据，而不是藏在自然语言对话里。
4. 对循环设置 turns、retries、time、cost 和 stopping criteria；失败后应进入受控 fallback、abstention 或 human escalation，而不是无限“反思”。
5. guardrails、HITL、tracing 和 evals 是独立但互补的控制面。它们必须能覆盖最终输出和有副作用的工具调用，并以端到端轨迹验证。

因此，portfolio-quality 的证明应是：一个必要、有限、可观测、可评测的 evidence-driven workflow，而不是装饰性的 multi-agent topology。

## 1. Agents 与 deterministic workflows 的选择

OpenAI 将 agent 描述为由 LLM 管理 workflow execution、识别完成状态、必要时纠错，并动态选择工具的系统；若 LLM 没有控制 workflow execution，则简单 chatbot、single-turn LLM 或 classifier 不构成 agent。OpenAI 建议优先考虑传统规则难以覆盖的复杂判断、难维护规则和大量非结构化数据；否则 deterministic solution 可能已经足够。[OpenAI, *A practical guide to building agents*](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)

OpenAI Agents SDK 的 orchestration 指导进一步区分：

- LLM orchestration 适合 open-ended task，让模型规划、选工具或 delegation。
- Code orchestration 在 speed、cost 和 performance 上更 deterministic、predictable，并可通过 structured outputs、固定 chaining、并行执行等普通代码结构完成。
- 两者可以混合，而不是全局二选一。[OpenAI Agents SDK, *Agent orchestration*](https://openai.github.io/openai-agents-python/multi_agent/)

Anthropic 的第一方工程指导给出相同的复杂度门槛：workflow 是由预定义代码路径编排 LLM 与工具；agent 则由 LLM 动态决定过程和工具使用。应从最简单的解法开始，只在结果有可测改善时增加复杂度；well-defined task 优先 workflow，必须灵活、无法硬编码路径的 open-ended task 才适合 agent。[Anthropic, *Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents)

### 对 InsuranceRAG 审计的判据

以下内容本身不需要 agent：

- 文档解析、chunking、embedding、index update；
- 固定 hybrid retrieval、fusion、reranking；
- citation formatting、schema validation、确定性 guard；
- 已知异常到用户可读 fallback 的映射。

只有存在如下证据时，模型驱动的 loop 才有技术价值：

- 单次 retrieval 无法决定是否已有充分证据；
- 不同问题需要动态选择 policy scope、query rewrite 或检索策略；
- verifier 能给出结构化、可行动的缺口；
- 第二轮行动能在代表性 eval 上显著提高 grounded answer quality；
- 相对 deterministic baseline 的额外 latency、cost 和 failure surface 有记录。

若没有这些证明，称现有 pipeline 为“agent”或加入多个命名角色不构成可信的 Agentic AI portfolio signal。

## 2. Tool use：能力接口，而不是函数数量

OpenAI 的 function-calling 文档将 tool 定义为应用暴露给模型的数据或动作接口。官方建议：

- 使用清晰、详细的 function name、parameter description 和 instruction；
- 说明函数用途、参数格式、返回值语义，以及何时使用或不使用；
- 遵守软件工程原则，避免让模型补填应用已知参数；
- 控制初始 tool surface，官方软建议是单 turn 起始时少于 20 个函数，并对大型或低频工具使用 deferred loading/tool search；
- 推荐 strict schema，使工具参数符合声明的 JSON Schema。[OpenAI API, *Function calling*](https://developers.openai.com/api/docs/guides/function-calling)

Anthropic 同样把清晰、经过测试的 agent-computer interface 和 tool documentation 视为 agent 可靠性的核心，而不是框架复杂度。[Anthropic, *Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents)

### 对 InsuranceRAG 审计的判据

可信工具应有：

- 单一且可说明的职责，例如 `retrieve_policy_evidence`、`verify_claims`；
- typed input/output，尤其是 `policy_id`、query、evidence span、claim、confidence/decision；
- 明确的 timeout、空结果、partial result 和 provider failure 语义；
- read-only 与 side-effecting capability 的权限分离；
- 可单独测试的 deterministic implementation；
- trace 中可辨认的调用、输入摘要、输出摘要和失败类型。

将现有 Python 函数简单包装成 tools，但不让模型依据中间结果做必要决策，也不形成上述契约，只是 API restyling，不是 agentic maturity。

## 3. Explicit state 与可恢复执行

OpenAI 的结果与状态文档强调，一次 agent run 的结果不仅是 final answer，还包括 history、最后活跃 agent、pending approvals 和 resumable state。审批中断时可能没有 final output；系统应保存 interruptions 和 state，并在批准或拒绝后从该 state 恢复。[OpenAI API, *Results and state*](https://developers.openai.com/api/docs/guides/agents/results)

OpenAI Agents SDK 的 HITL 文档进一步说明，`RunState` 可以序列化并在较晚时间恢复；长时间 pending 的审批还应保存 agent/tool definition 的版本信息，以避免恢复时定义已经变化。该文档也提醒 serialized context 可能包含 secrets，应按持久化数据对待。[OpenAI Agents SDK, *Human-in-the-loop*](https://openai.github.io/openai-agents-python/human_in_the_loop/)

### 对 InsuranceRAG 审计的判据

如果引入 agentic retrieval/verification loop，至少应显式记录：

- request/run ID、policy corpus/version、index version；
- normalized question 与当前阶段；
- executed retrieval strategies 和去重后的 evidence IDs；
- candidate claims、claim-to-evidence links、verifier decisions；
- iteration/turn/retry/cost budget；
- stop reason：`sufficient_evidence`、`no_evidence`、`guard_blocked`、`budget_exhausted`、`human_required` 或 typed failure；
- final answer、citation set 和所有降级警告。

仅靠 prompt 文本或 Streamlit session history 隐式携带这些信息，不足以证明可恢复、可审计的 agent workflow。

## 4. Guardrails 与 human escalation

OpenAI 将两类控制明确分开：

- guardrails 自动验证 input、output 或 tool behavior；
- human review 在敏感工具动作前暂停，由人或 policy 批准/拒绝。

官方选择表建议：disallowed request 用 input guardrail；输出发布前的验证/脱敏用 output guardrail；tool arguments/results 用 tool guardrail；取消、编辑、shell、敏感 MCP action 等 side effect 用 HITL approval。[OpenAI API, *Guardrails and human review*](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals)

OpenAI 的实务指南还建议 layered guardrails，并指出 human intervention 特别适用于 guardrail failure 或超过 failure threshold，以及高风险、敏感、不可逆 actions。[OpenAI, *A practical guide to building agents*](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)

### 对 InsuranceRAG 审计的判据

纯 policy QA 当前通常没有真实 side effect，因此不应为展示 HITL 而虚构 approval 流程。更有价值的 escalation 是：

- evidence 不足或不同 policy/version 冲突时明确 abstain；
- 涉及 personalized coverage determination、claim adjudication 或法律/医疗判断时转人工；
- guard/verifier 失败超过固定阈值时停止自动回答；
- 将 escalation reason、已有证据和未解决问题一起交给 reviewer。

未来若加入 claim submission、policy update、email 或 case-system write，则必须在 tool 层设置 approval boundary；不能只在 prompt 中要求模型“先询问”。

## 5. Bounded loops 与停止条件

Anthropic 指出 agent 应通过 tool result 或 code execution 等环境反馈获取每一步的“ground truth”，在 checkpoint/blocker 处寻求人类反馈，并设置最大迭代次数等 stopping conditions。自主性会增加成本并造成错误累积，因此应在 sandbox 中充分测试并配置 guardrails。[Anthropic, *Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents)

OpenAI Agents SDK 的 runner loop 会在 final output、handoff 或 tool call 之间推进，并在超过 `max_turns` 时抛出 `MaxTurnsExceeded`；SDK 也支持将 max-turn、model refusal 和 invalid final output 映射为受控的 application-specific fallback。[OpenAI Agents SDK, *Running agents*](https://openai.github.io/openai-agents-python/running_agents/)

OpenAI orchestration 文档展示 evaluator/feedback loop，但没有把无限循环视为最佳实践；code orchestration 的价值正是可预测地控制流程。[OpenAI Agents SDK, *Agent orchestration*](https://openai.github.io/openai-agents-python/multi_agent/)

### 对 InsuranceRAG 审计的判据

一个可信的 repair loop 应同时具备：

- 最大 retrieval/generation/verification iteration；
- 每类 transient failure 的有限 retry 和 backoff；
- latency、token 与 monetary budget；
- evidence-set 无变化时的 no-progress detection；
- verifier 的明确 pass/fail/insufficient contract；
- 达到上限后的 deterministic abstention，而不是输出最后一次未验证答案；
- 防止重复 tool side effect 的 idempotency 设计。

“让模型不断自我反思直到满意”没有 objective stopping rule、独立证据或 eval gain，不应作为 portfolio milestone。

## 6. Tracing 与 observability

OpenAI Agents SDK tracing 会记录一个 run 中的 model generations、tool calls、handoffs、guardrails 和 custom events；trace 是端到端 operation，span 表示其中有开始/结束时间的步骤，并可通过 trace/group IDs 关联 workflow 和 conversation。[OpenAI Agents SDK, *Tracing*](https://openai.github.io/openai-agents-python/tracing/)

官方同时警告 generation/tool spans 可能包含敏感 input/output，且敏感数据记录在该 SDK 中默认开启；应用需要显式决定是否关闭或替换 trace processor。ZDR 组织不能使用该官方 tracing surface。[OpenAI Agents SDK, *Tracing — Sensitive data*](https://openai.github.io/openai-agents-python/tracing/#sensitive-data)

OpenAI 的 observability 指导建议在调优之前先检查 runtime 中实际发生的行为；built-in tracing 用于调试 prompts、tools、handoffs 与 approvals，并为后续 formal evals 生成高信号轨迹。[OpenAI API, *Integrations and observability*](https://developers.openai.com/api/docs/guides/agents/integrations-observability)

### 对 InsuranceRAG 审计的判据

最小有用 trace 应覆盖：

- ingestion/index version；
- retrieval query、strategy、latency、candidate count；
- reranking/fusion 决策与最终 evidence IDs；
- model/provider、prompt/version、usage 和 latency；
- claims、citations、verification/guard outcomes；
- retries、fallbacks、stop reason 和最终 disposition。

保险材料和用户问题可能包含个人或敏感信息，因此 trace 设计必须同时说明 redaction、retention、access、export destination 和是否记录原文。只有日志数量而没有 correlation IDs、stage semantics 和 privacy policy，不构成成熟 observability。

## 7. Evaluation：同时评最终结果和执行轨迹

OpenAI 的 evaluation best practices 建议：

- eval-driven development，早期且持续评测；
- 使用反映真实生产分布的 task-specific dataset；
- 开发期间记录数据，从 logs 挖掘失败用例；
- 自动评分与 human judgment 结合，并校准二者的一致性；
- 每次变更持续运行 eval，逐步扩充覆盖；
- 避免通用指标、偏离生产分布的数据集、vibe-based eval 和忽略人工反馈。[OpenAI API, *Evaluation best practices*](https://developers.openai.com/api/docs/guides/evaluation-best-practices)

对 document Q&A，OpenAI 明确建议结合 production questions/user satisfaction、domain-expert authored correct answers 和历史 logs，并衡量是否能精准回答、召回所需上下文和满足用户需要。[OpenAI API, *Evaluation best practices — Q&A over docs*](https://developers.openai.com/api/docs/guides/evaluation-best-practices#example-qa-over-docs)

OpenAI 的 agent eval 指导区分：

- 调试行为时先看 traces，并用 structured graders 检查 tool choice、handoff、instruction/safety violation 和 routing change；
- 当“good”已有定义后，再用 repeatable datasets/eval runs 比较 prompts 和 workflow changes。[OpenAI API, *Evaluate agent workflows*](https://developers.openai.com/api/docs/guides/agent-evals)

Anthropic 的 agent-eval 工程文章补充了三个适合本项目的判据：

- grader 应按任务组合 code-based、model-based 和 human grading；model grader 需要用专家判断校准；
- capability eval 应保留提升空间，而 regression eval 应接近全通过以防止 backsliding；
- research/RAG 类结果应分别评 groundedness、coverage 和 source quality，并检查完整 transcript/trajectory，以发现 grader、harness 或任务定义本身的问题。[Anthropic, *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

### 对 InsuranceRAG 审计的判据

要证明 agentic loop 优于当前 RAG baseline，应在同一组真实或专家构造 policy questions 上比较：

- answer correctness、completeness、abstention correctness；
- retrieval recall/precision 与 evidence sufficiency；
- claim-level citation entailment 和 attribution；
- policy/source confusion、prompt injection 和 contradictory evidence；
- tool selection、query rewrite usefulness、unnecessary turns；
- guard/verifier decision 与 human labels 的 agreement；
- end-to-end success、latency、tokens、cost 和 failure recovery。

必须同时评分 final answer 与 trajectory。最终答案碰巧正确，不代表 tool selection、citation 或 safety path 正确；反之，少一次 tool call 也只有在最终质量仍通过同一 eval 时才算改进。

## 8. Multi-agent：需要额外的收益证明

OpenAI 将 manager/agents-as-tools 用于“一个 agent 保持最终回答和共享 guardrail 控制，specialists 只完成 bounded subtasks”；handoff 则用于 specialist 应直接接管下一段交互的情形。[OpenAI Agents SDK, *Agent orchestration*](https://openai.github.io/openai-agents-python/multi_agent/)

Anthropic 对其生产 research system 的第一方复盘显示，multi-agent 适合 open-ended、path-dependent、可高度并行且价值足以覆盖成本的研究任务；其内部数据中 agent 通常约为普通 chat 的 4 倍 token，multi-agent 约为 15 倍。共享上下文很多或依赖关系密集的任务并不适合该结构。[Anthropic, *How we built our multi-agent research system*](https://www.anthropic.com/engineering/multi-agent-research-system)

### 对 InsuranceRAG 审计的判据

InsuranceRAG 的默认问题通常围绕一个 policy corpus 和一个用户问题，retrieval、generation、verification 强依赖共享证据。这更支持：

- 单一 code-orchestrated workflow；
- 必要时一个受限 model-driven evidence-repair loop；
- deterministic verifier/guard 作为独立边界；
- 单一 user-facing answer owner。

只有在跨多个独立 policy/coverage domain 的并行调查获得可测收益时，才值得研究 manager + bounded specialist tools。为每个 pipeline stage 创建“agent persona”会增加 coordination、cost 和 debugging surface，却没有自然的独立决策权或并行收益。

## Portfolio credibility checklist

| 可信能力 | 应展示的可执行证据 | 弱/装饰性信号 |
|---|---|---|
| Agent necessity | 与 deterministic baseline 对比，证明动态决策提高任务成功率 | README 自称“agentic” |
| Tool engineering | typed schema、clear semantics、timeouts/errors、权限边界、unit/integration tests | 将任意函数包装为 tool |
| Explicit state | 可检查/序列化的阶段、证据、budget、stop reason | 自然语言 scratchpad |
| Bounded autonomy | turns/retries/time/cost/no-progress limits 与确定性 fallback | 无上限 reflection loop |
| Safety | input/output/tool guardrails；高风险动作 HITL；abstention/escalation tests | prompt 中一句“be safe” |
| Observability | end-to-end trace、correlation、stage spans、privacy controls | 零散 print/log |
| Evaluation | representative gold cases、trajectory + outcome grading、human calibration、regression gates | 少量 happy-path demo |
| Multi-agent justification | 独立可并行 subtasks，质量/成本/延迟收益可测 | 多 persona、互相 review |

## 建议用作本 ticket 的审计问题

1. 当前 workflow 中，究竟有哪些 step 由模型根据 intermediate evidence 动态决定？
2. 如果移除所谓 agent abstraction，是否仍是完全相同的固定 RAG pipeline？
3. state 是否显式包含 evidence、verification、budgets、stop reason，还是只存在于 prompt/session？
4. 每个 tool 是否有 schema、权限、typed failures 和独立测试？
5. 是否存在有限且有 objective progress signal 的 loop？
6. 哪些情况会 abstain、fallback 或 escalate，且这些路径是否被 eval？
7. trace 能否重建一次 run，又是否避免泄露 policy/customer 敏感数据？
8. eval 是否证明 trajectory 和 final answer 都正确，并能比较 deterministic baseline？
9. multi-agent 是否解决真正可分解、可并行的问题，其收益是否覆盖额外成本与失败面？
10. portfolio narrative 是否准确区分“已实现并测量”“仅设计”“尚未支持”？

## Source inventory

- [OpenAI — A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [OpenAI Agents SDK — Agent orchestration](https://openai.github.io/openai-agents-python/multi_agent/)
- [OpenAI API — Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI API — Guardrails and human review](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals)
- [OpenAI API — Results and state](https://developers.openai.com/api/docs/guides/agents/results)
- [OpenAI Agents SDK — Human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
- [OpenAI Agents SDK — Running agents](https://openai.github.io/openai-agents-python/running_agents/)
- [OpenAI Agents SDK — Tracing](https://openai.github.io/openai-agents-python/tracing/)
- [OpenAI API — Integrations and observability](https://developers.openai.com/api/docs/guides/agents/integrations-observability)
- [OpenAI API — Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- [OpenAI API — Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)
- [Anthropic — Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Anthropic — Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

## Method note

本次研究首先尝试按 OpenAI Docs skill 安装官方 Developer Docs MCP，但本地 `codex.exe` 因 access denied 无法执行。随后仅通过官方 OpenAI/Anthropic 域名检索并打开原始页面；未采用二手博客或聚合摘要。文中结论为上述第一方资料的归纳，针对 InsuranceRAG 的条目明确标为“审计判据”，不冒充官方对该项目的判断。
