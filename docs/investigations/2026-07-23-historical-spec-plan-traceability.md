# InsuranceRAG 历史规格、计划与当前系统追溯矩阵

日期：2026-07-23

代码基准：`e7c66d5694fd530dfa5f0b8f9bff003de49c55e9`

对应 Wayfinder ticket：[Reconcile historical specifications and plans with the live system](https://github.com/ShiriZhang/InsuranceRAG/issues/4)

## 1. 调查结论

InsuranceRAG 的历史文档大体记录了三次已经落地的能力扩展和一次尚未落地的 benchmark 提案：

1. 2026-06-08/09：单保单 Streamlit MVP；
2. 2026-06-11：query rewrite、BM25/RRF hybrid retrieval、answer guard 和 offline evaluation；
3. 2026-06-14/16：clause metadata、rule reranking、citation verifier 和 hard-negative evaluation；
4. 2026-07-09：真实 embedding 的 vector vs hybrid-reranked comparison，目前只有未跟踪的 spec/plan。

前三期的 34 个明确计划文件路径当前全部存在，Git history 也能看到对应 feature commits；但是部分 acceptance criteria 只完成了类型、配置或局部 component，没有接入产品执行路径。第四期的三个目标产物都不存在。

最重要的文档真相是：

- 当前产品不是 June MVP 的原样实现，而是经过两轮增强后的 deterministic RAG pipeline；
- README 对核心在线架构的描述基本正确，但对 evaluation 的能力存在明显 overclaim；
- `answer_guard_llm`、`verifier_enabled`、`verifier_strictness` 和 `heading_confidence_warn_threshold` 是“文档和配置已存在、运行时行为未接线”的 weak integration；
- local hard-negative evaluation 的实现远弱于 advanced design：它没有构造可审计的 positive/hard-negative candidate pairs，也没有报告 hard-negative outrank rate、heading-confidence distribution 或 verifier counts；
- 2026-07-09 retrieval comparison 是 documentation-only / 当前分支中已搁置的计划，不能作为 portfolio capability；
- 历史 implementation plans 中共 204 个 checkbox 全部仍为 unchecked，因此 checkbox 不能作为完成状态来源；
- current code、tests、可执行基线和 Git commits 应是状态依据，历史 spec/plan 只能表示当时 intent。

本调查没有修复文档或产品代码，也没有执行真实 OpenAI/Tesseract/private-policy workflow。

## 2. 证据与状态定义

证据优先级：

1. 当前生产入口与实际调用关系；
2. 当前 tests、offline CLI 和既有 runtime baseline；
3. Git history 中的实现演进；
4. README；
5. dated spec/plan。

状态定义：

| 状态 | 含义 |
| --- | --- |
| **complete** | 当前代码中生产可达，或作为明确 offline entry point 可执行，并有测试/运行证据 |
| **partial** | 主体存在，但缺少重要场景、外部验证、指标或 acceptance criterion |
| **superseded** | 后续架构已替代原先默认实现；旧表面可能仍作为底层或兼容层存在 |
| **duplicated** | 同一领域规则、模型或文档真相在多处独立维护 |
| **inconsistent** | 当前文档陈述、配置含义或计划 acceptance criterion 与实际行为不同 |
| **weakly integrated** | 模块/字段/config 存在，但没有贯通主要执行路径或可观测结果 |
| **documentation-only / abandoned in current branch** | 只有文档 intent，没有目标代码、入口、测试或 report；不代表存在正式放弃决议 |
| **intentional non-goal** | 历史文档明确排除，不应误报为缺失 |

## 3. 文档清单与可信边界

| 文档 | Git 状态 | 当前用途 | 可信边界 |
| --- | --- | --- | --- |
| `README.md` | tracked | 当前用户说明 | runtime 命令与产品边界可参考；evaluation claim 必须以代码校正 |
| `2026-06-08-insurance-policy-rag-design.md` | tracked | 初始 MVP design | 历史 intent；多处已被后续 retrieval/safety 架构 supersede |
| `2026-06-09-insurance-policy-rag.md` | tracked | MVP implementation recipe | 内含旧代码 snapshot 和已经失效的 Git constraint |
| `2026-06-11-rag-quality-enhancement-design.md` | tracked | 第一次质量增强 design | 大部分已实现；错误处理和 optional LLM 部分未完全兑现 |
| `2026-06-11-rag-quality-enhancement.md` | tracked | 第一次质量增强计划 | 计划路径已落地，但 checkbox 没有维护 |
| `2026-06-14-rag-accuracy-advanced-design.md` | tracked | advanced accuracy design | parser/reranker/verifier 主体已实现；config/UI/local eval 多处 partial |
| `2026-06-16-rag-accuracy-advanced.md` | tracked | advanced accuracy plan | 实现采取了比 design 更小的 local hard-negative scope |
| `2026-07-09-retrieval-baseline-comparison-design.md` | **untracked** | benchmark proposal | 没有进入 repository history，也没有对应 executable |
| `2026-07-09-retrieval-baseline-comparison.md` | **untracked** | benchmark implementation plan | 11 个步骤均 unchecked；目标代码不存在 |
| 两份 2026-07-23 investigation | tracked | 当前 executable ground truth | 分别给出代码路径 ledger 与可复现 runtime baseline |

计划 checkbox 统计：

| Plan | unchecked | checked |
| --- | ---: | ---: |
| 2026-06-09 MVP | 63 | 0 |
| 2026-06-11 quality enhancement | 60 | 0 |
| 2026-06-16 advanced accuracy | 70 | 0 |
| 2026-07-09 retrieval comparison | 11 | 0 |
| **合计** | **204** | **0** |

这不是“前三期没有实现”的证据，而是计划文档从未被当作 execution-state ledger 维护。

## 4. 演进追溯

### Phase A：单保单 MVP

初始 design/plan 承诺 Streamlit upload、PDF extraction、OCR fallback、chunking、OpenAI embedding、in-memory index、Chinese answer、citations、built-in background 和 session-only state。

Git history 中存在连续实现 commits：

- `c07a69d`：PDF loader；
- `5904ef7`：policy chunking；
- `83950b0`：in-memory retriever；
- `a1b18a7`：RAG answer chain；
- `fd0070e`：Streamlit app；
- `e64f9de`、`844f3cf`、`bb6049c`：built-in background 与 lazy indexing；
- `1840566`、`8cf1d21`：README/OCR 说明。

当前代码保留了这一产品边界，但 vector-only default、简单 heading inference 和 prompt-only safety 已被后续 phase 增强或 supersede。

### Phase B：RAG quality enhancement

2026-06-11 spec/plan 的核心路径对应：

- `5ff7b2d`：rule-based query rewrite；
- `f9fdaea`：hybrid BM25/vector retrieval；
- `599c3fe`：answer guard；
- `1d617c1`：RAG chain integration；
- `39eab75`：UI retrieval details；
- `ff08972`：offline evaluation；
- `41b937b`：README。

这些模块当前仍生产可达。主要偏差是 optional LLM flags 没有实现、BM25 failure 没有传播 warning，以及 evaluation 从未达到 README 所称的完整质量指标。

### Phase C：advanced accuracy

2026-06-14/16 design/plan 对应：

- `38fe41e`：clause parser；
- `5d36672`：chunk metadata；
- `128f896`：rule reranking；
- `4cf3bf1`：RAG chain reranking；
- `1b52aae`：citation verifier；
- `90b73a2`：verification UI；
- `f6c2cec`：synthetic hard-negative eval；
- `9e9bb8e`：local hard-negative eval；
- `5bc9aca`：README。

parser、policy reranker、heuristic verifier 和 UI 主体已落地。heading-quality summary、verifier switches/strictness 和 design 级 local hard-negative metrics 没有落地。

### Phase D：retrieval baseline comparison

2026-07-09 spec/plan 提议使用真实 `text-embedding-3-small`，在同一批 20 个本地 PDF/chunks/queries 上比较：

- pure vector；
- production hybrid + reranking；
- Top-3 hit rate；
- hard-negative misretrieval rate；
- raw counts、absolute delta、relative change。

计划目标：

- `src/insurance_rag/retrieval_comparison.py`
- `scripts/compare_retrieval_modes.py`
- `eval_reports/retrieval_mode_comparison.md`

当前三个路径都不存在，计划中的 `compare_document_chunks` 和 `RetrievalComparisonReport` 也没有定义。状态是 **documentation-only / abandoned in current branch**。由于 spec/plan 本身也未跟踪，不能把它描述为已承诺交付或部分实现。

## 5. 产品与 runtime traceability matrix

| 历史承诺 | 来源 | 当前 executable evidence | 状态 | 准确解释 |
| --- | --- | --- | --- | --- |
| 本地 Streamlit、单次会话上传一份 PDF | MVP design/README | `app.py:243-310`；file uploader、chat input、session state | **complete** | 单保单 session flow 已实现 |
| 分阶段 upload progress | MVP design | `app.py:193-240` 显示 receive、parse/OCR、chunk、embedding/index、complete | **complete** | 是阶段级 progress，不是 page/job observability |
| Text PDF extraction | MVP design/README | `document_loader.py:63-99`；PDF tests | **complete** | PyMuPDF 内存解析可执行 |
| Simple OCR fallback | MVP design/README | `document_loader.py:51-88` | **partial / external** | fallback 代码完整；Tesseract executable 与 `chi_sim+eng` 不随仓库提供，当前机器未安装 |
| OpenAI embeddings + chat RAG | MVP design/README | `retriever.py:15-22`；`rag_chain.py:145-151` | **implemented / external-unverified** | 代码可达、fake tests 覆盖；没有真实 API baseline |
| Temporary in-memory policy index | MVP design | `retriever.py:25-72`；`app.py:233-239` | **complete** | 无 persistent vector database |
| Session-only file/text/index/chat privacy | MVP design/README | objects 只在 `st.session_state`；`.gitignore` 忽略 `.env`、`documents/`、reports | **complete for repo code** | OpenAI data transmission 已在 README 披露；provider-side retention 不由本仓库控制 |
| User-policy-first evidence | 全部三期 | `rag_chain.py:101-143` | **complete** | 无 policy chunks 时拒答；built-in 只在 policy result 存在时使用 |
| Built-in dataset 只做背景 | MVP/quality design/README | `should_use_builtin_context()` + separate prompt/citations | **partial / weakly integrated** | source boundary 存在，但只按 definition keywords gate；索引选择只是排序后的前 8 个 PDF，失败后本 session 不重试 |
| Built-in content 按问题相关性选择 | MVP design 的“when useful” | `app.py:56` 固定 `limit=8` | **inconsistent / weakly integrated** | 不是 corpus-level semantic selection |
| Chinese answer、不得做最终理赔判断 | 全部三期 | prompt + `answer_guard.py` | **complete as layered heuristic** | prompt 和 deterministic guard 均存在；不是法律/理赔正确性证明 |
| Citations 带页码、条款标题、excerpt，并分 policy/built-in | MVP design/README | `rag_chain.py:28-39,152-156`；`app.py:164-183` | **complete as retrieved-chunk citations** | citation 表明 chunk 被检索并送入 prompt，不表示 model claim 与该 citation 已精确对齐 |
| “每个重要陈述”有来源 | MVP design | 所有 retrieved chunks 自动成为 citations；verifier 检查部分事实类型 | **partial / inconsistent** | 没有 model-produced inline citation alignment，也没有完整 claim-to-citation coverage |
| Missing evidence 时拒答 | MVP/quality design | `rag_chain.py:106-131` | **complete** | policy search exception 或无 result 均返回固定 refusal |
| API failure 保留 parsed state | MVP design | upload index failure保留 parse/chunks；question failure由 `app.py:294-307` 捕获 | **partial** | state 通常保留，但 online API error 路径没有真实 service test |
| `config.py` 与 `utils.py` 分担 shared helpers | MVP design | `config.py` 存在；`utils.py` 不存在 | **superseded** | helpers 留在各模块；没有证据表明缺少 `utils.py` 影响功能 |
| temporary-directory settings | MVP design | 无对应 config；用户数据只在内存/session | **superseded by in-memory design** | 当前路径不创建 user temp artifacts |
| `mixed` extraction method | MVP data model | loader 只产生 `text` 或 `ocr` | **abandoned detail** | 没有 mixed page/chunk path，也没有当前需求证据 |

## 6. Retrieval 与结构化条款 traceability matrix

| 历史承诺 | 来源 | 当前 executable evidence | 状态 | 偏差 |
| --- | --- | --- | --- | --- |
| Vector retrieval | MVP | `InMemoryVectorIndex` + `OpenAIEmbedder` | **complete but superseded as default** | 仍是 hybrid 的 vector substrate；默认不再是 vector-only |
| Rule-based query expansion | quality design | `query_rewriter.py:8-75`；RAG chain 每次调用 | **complete** | hard-coded insurance intents |
| Optional LLM query rewrite | quality design/config | `query_rewrite_llm` 传入；`use_llm=True` 只产生 warning | **placeholder / weakly integrated** | `used_llm` 永远为 false |
| BM25 + vector + RRF | quality design/README | `hybrid_retriever.py:130-285` | **complete** | 默认 production retrieval |
| BM25/vector failure 显式降级并 warning | quality design | vector `ValueError` 和 BM25 exception 被内部吞掉 | **partial / inconsistent** | 降级存在，但调用方拿不到 warning |
| Retrieval explanation：vector/BM25/fusion/matched terms | quality design/README | structured model + `app.py:104-142` | **complete** | UI 还显示 rank details 与 rerank |
| Clause numbers/headings metadata | advanced design/README | `clause_parser.py` + `chunker.py` | **complete** | 支持 high-frequency patterns，不是完整 document tree |
| Heading-confidence quality summary/warning | advanced design | metadata 存在；`heading_confidence_warn_threshold` 只在 config/tests | **partial / weakly integrated** | upload summary 不统计 high/medium/low，不使用 threshold |
| Policy rule reranking | advanced design/README | `rag_chain.py:101-128`；`rule_reranker.py` | **complete** | 有 failure fallback 和解释 |
| Built-in reranking | advanced data-flow 的笼统表述 | built-in search 后直接截取 | **not integrated** | 当前只 rerank policy candidates |
| Rerank enable/top-N config | advanced design | `rerank_enabled`、`rerank_top_n` 在 RAG chain 使用 | **complete** | 实际可切换 |
| `RerankExplanation` shared model | advanced design | class 只被 model tests 构造 | **duplicated / unused compatibility surface** | production 直接用 `HybridSearchResult.rerank_score/reasons` 和 `RetrievalExplanation` fields |
| 旧 keyword heading inference | MVP | `infer_section_title()` 仅 tests 调用 | **superseded / duplicated** | production 已调用 `parse_clause_metadata()`；旧函数保留兼容表面 |

## 7. Generation、safety 与 verification traceability matrix

| 历史承诺 | 来源 | 当前 executable evidence | 状态 | 偏差 |
| --- | --- | --- | --- | --- |
| Post-generation answer guard | quality design/README | `rag_chain.py:157-177`；`answer_guard.py` | **complete heuristic implementation** | 不是 semantic judge |
| Final claim wording block | quality/advanced design | deterministic term/regex checks | **complete** | 有大量 regression tests |
| Built-in source confusion block | quality/advanced design | answer guard + citation verifier | **complete heuristic implementation** | 仅覆盖词典/规则可识别表达 |
| Unsupported numeric/amount/period fact verification | advanced design | `citation_verifier.py` | **complete for encoded rules** | 不等于开放域 entailment |
| Coverage/exclusion/subject fact verification | advanced design | verifier dictionaries/regex + tests | **partial but substantial** | 覆盖已编码保险术语；不能声称逐条理解任意事实 |
| Verification UI | advanced design | `app.py:145-161` | **complete** | 没有 dedicated Streamlit UI tests；现有 baseline 仅做 app smoke |
| Verifier runtime failure fail-closed | advanced design | `rag_chain.py:166-168` | **complete** | 这条后期决策 supersede 2026-06-11 “保留原 answer 并 warning”的早期规则 |
| `verifier_enabled` | advanced config | config/tests only | **inconsistent / inert** | 即使设 false，guard 仍调用 verifier |
| `verifier_strictness` | advanced config | config/tests only | **inconsistent / inert** | `strict/balanced/warn_only` 不改变行为 |
| `answer_guard_llm` | quality config | config/tests only | **placeholder / inert** | 没有 LLM guard interface 或调用 |
| Verification status taxonomy | advanced design | design 写 `partially_supported`；code 使用 `supported/unsupported/uncertain` | **superseded terminology** | code enum 是当前真相 |

## 8. Evaluation 与 reproducibility traceability matrix

| 历史承诺 | 来源 | 当前 executable evidence | 状态 | 偏差 |
| --- | --- | --- | --- | --- |
| Repo-contained deterministic synthetic eval | quality design | 2 cases；SHA-256 pseudo-embedder；重复 report hash 一致 | **complete as regression fixture** | 不是真实 semantic retrieval |
| Synthetic report 含 ranks/scores/matched terms | quality design | `evaluation.py:298-326,535-557` | **complete** | 细节行包含 vector/BM25/fusion/matched terms |
| Synthetic eval 覆盖 answer guard | quality design/README | synthetic path只做 rewrite + hybrid retrieval | **inconsistent** | 不调用 reranker、generation、guard 或 verifier |
| README 所称 Recall@k、MRR、citation coverage | README | 只报告 case pass、expected rank 和 retrieved details | **inconsistent / overclaimed** | 无 aggregate Recall@k、MRR、generated-answer citation coverage |
| Optional local PDF evaluation | quality design | parse/chunk + fixed term cases | **partial** | 使用 zero-vector `Bm25OnlyEvalEmbedder`，不是 online OpenAI hybrid mirror |
| Synthetic hard-negative categories | advanced design | 4 cases：number、clause type、subject、source confusion | **complete as curated component regression** | 使用预写 answer，不调用 chat/RagChain/full guard |
| Local hard-negative 自动构造 confusion pairs | advanced design/README | 复用 local evaluator，仅在同一 document text 同时出现 expected 与 negative terms 时运行 fixed query | **weakly integrated / inconsistent** | 没有显式 positive/hard-negative chunks 或 pair artifact |
| Local hard-negative report：outrank rate、low-confidence rate、verifier counts | advanced design | 使用通用 local report，只给 Top1/Top3、unknown rate | **not implemented** | design acceptance 未兑现 |
| Evaluation batch case failure 继续 | quality design | validation/runner 多处直接抛出，CLI 返回 non-zero | **partial / inconsistent** | 没有通用 per-case exception capture |
| CI-style reproducibility | quality design | clean env 222 tests、synthetic 2/2、hard-negative 4/4 | **partial** | 无 CI workflow、lock file 或 Python pin |
| README setup 可重现 | README | clean install 成功；repo `.venv` 缺 `rank-bm25` | **partial / environment drift** | lower-bound requirements 每次可 resolve 不同版本 |
| Real vector vs hybrid-reranked benchmark | 2026-07-09 untracked spec/plan | 无 module、CLI、test、report | **documentation-only / abandoned in current branch** | portfolio 中不能声称已有 baseline comparison |
| Production latency/token/cost/model metadata | 未被前三期实现 | 无 instrumentation/report fields | **not implemented** | 属于后续 maturity gap，不是历史已完成承诺 |

## 9. 重复、漂移与弱集成清单

### 9.1 重复领域词表

保险标题、intent 和事实词独立分散在：

- `chunker.py:7-19`
- `clause_parser.py:8-31`
- `query_rewriter.py:8-39`
- `hybrid_retriever.py:14-29`
- `rule_reranker.py:8-41`
- `answer_guard.py:14-87`
- `citation_verifier.py:6-57`

这些代码都不是 dead code，但同一概念由 parsing、retrieval、rerank 和 verification 各自维护，已经产生 duplication 和 vocabulary drift 风险。

### 9.2 重复/遗留模型

- `infer_section_title()` 已被 production clause parser supersede，只剩 direct unit tests；
- `RerankExplanation` 与 production 使用的 `rerank_score/reasons` fields 重叠，只在 model test 中出现；
- local document 与 local hard-negative 共用同一个 report/data model，使后者无法表达其 design 中独有的 negative candidates、outrank rate 和 verifier aggregation。

### 9.3 历史文档复制代码

三个 tracked implementation plan 包含大量完整 code snapshot。后续 bugfix commits 已改变这些实现，但 plan 没有同步；例如 verifier 在 2026-06-19 至 06-21 之间经历多次 false-positive、source-confusion 和 fallback 修正。

这些 plan 应作为 historical decision trail，而不应作为当前实现说明或可复制安装手册。

### 9.4 配置与行为分离

以下 environment variables 能被解析、也有 config tests，却不能改变 production behavior：

- `INSURANCE_RAG_ANSWER_GUARD_LLM`
- `INSURANCE_RAG_VERIFIER_ENABLED`
- `INSURANCE_RAG_VERIFIER_STRICTNESS`
- `INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD`

`INSURANCE_RAG_QUERY_REWRITE_LLM` 稍有不同：运行时会读取，但只让 UI 收到“LLM 尚未启用”warning，不会执行 LLM rewrite。

## 10. README accuracy ledger

### 当前准确

- 单保单 session-local Streamlit assistant；
- PyMuPDF text extraction 与 external Tesseract requirement；
- OpenAI embedding/chat data flow；
- policy-first、built-in-background-only boundary；
- BM25 + vector + RRF + rule rewrite + rule rerank；
- retrieval details UI；
- programmatic answer guard 和 heuristic fact verification；
- local data/report Git ignore；
- offline CLI 命令本身。

### 需要降级或改写

1. “合成评测检查 Recall@k、MRR、引用覆盖和回答守卫行为”不准确。当前 synthetic runner 没有 aggregate Recall/MRR、generation、citation coverage 或 guard。
2. “local hard-negative 构造正负查询对”过度表述。当前实现只用固定 term/negative-term presence gate，并运行 BM25-only local retrieval。
3. “逐条检查答案中的具体事实”应限定为“对规则可识别的数字、条款、主体和 source-confusion facts 做 heuristic verification”。
4. 安装说明可以得到 clean environment，但没有 lock/Python pin；不应暗示 resolved dependency versions 可长期复现。
5. 当前没有 real OpenAI、real OCR 或 representative policy corpus 的公开质量结果；README 不应把 offline fixture pass 当作 production-quality evidence。

## 11. 明确不是缺失的 non-goals

以下能力在历史 design 中被明确排除，当前没有实现是符合范围，而不是 abandoned feature：

- multi-file user upload；
- authentication/accounts；
- persistent user document library；
- policy comparison/recommendation；
- click-to-jump PDF viewer；
- full production deployment；
- full clause tree 或 complex table reconstruction；
- LLM/cross-encoder reranking；
- final claim/legal/medical/financial/underwriting decisions；
- agent framework 或 decorative multi-agent workflow。

## 12. 后续 ticket 应采用的文档真相

1. 用当前 code、tests 和两份 2026-07-23 investigation 作为 executable truth；dated plans 仅表示 historical intent。
2. 将前三个 June phase 视为“主体落地但有 integration debt”，不要把它们重新列成从零实现的 milestone。
3. 将 2026-07-09 retrieval comparison 视为未实施 proposal，而不是已有 benchmark。
4. 不要用 README 的 Recall/MRR/citation-coverage 表述作为 evaluation maturity 证据。
5. 后续 retrieval ticket 应重点检查 built-in corpus selection、silent degradation、distributed term dictionaries、heading-quality observability 和真实 vector baseline。
6. 后续 generation/safety ticket 应区分 prompt constraint、retrieved citation、heuristic verified fact 和 semantic faithfulness。
7. 后续 evaluation/observability ticket 应以缺失的 gold dataset、real-model E2E、latency/token/cost、run metadata 和 CI/lock governance 为事实起点。
8. 后续 glossary/ADR 输入应把 `user policy evidence`、`built-in background`、`retrieved chunk citation` 和 `verified fact` 定义为不同概念。
9. “Agentic AI”仍是 portfolio destination 的评价维度，不是当前架构描述；当前 `RagChain` 是固定顺序 pipeline。

## 13. 可复核命令

```powershell
# 当前历史文档
rg --files -g "*.md" docs\superpowers README.md

# 计划步骤是否被维护
rg -n "^- \[[ xX]\]" docs\superpowers\plans

# June 三期明确计划路径是否存在
rg --files app.py src tests scripts evals

# July comparison 目标符号是否存在
rg -n "compare_document_chunks|RetrievalComparisonReport|compare_retrieval_modes" .

# inert/placeholder config 是否有 production consumers
rg -n "answer_guard_llm|verifier_enabled|verifier_strictness|heading_confidence_warn_threshold|query_rewrite_llm" app.py src tests

# implementation history
git log --reverse --date=short --pretty=format:"%h`t%ad`t%s"
```

截至本基准：

- June 三期列出的 34 个明确路径：34 present / 0 missing；
- July comparison 三个目标路径：0 present / 3 missing；
- clean runtime baseline：222 tests passed；
- synthetic：2/2；
- hard-negative：4/4；
- 这些数字证明代码和 deterministic regression 当前可执行，不证明真实 RAG quality。
