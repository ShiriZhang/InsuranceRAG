# InsuranceRAG 可复现测试、评估与运行基线

日期：2026-07-23

基准提交：`d68194f3623cafc649e466cebeb140ebe92b4f96`

对应 Wayfinder ticket：[#3 建立可复现的测试、评估和运行基线](https://github.com/ShiriZhang/InsuranceRAG/issues/3)

## 1. 结论摘要

- 在 Windows 11、CPython 3.13.9 的全新虚拟环境中，`requirements.txt` 可以安装，`pip check` 成功，完整测试为 **222 passed / 0 failed / 0 skipped / 0 xfailed，22.34s**。
- 仓库已有 `.venv` 不是可靠基线：它缺少 `rank-bm25`，导致 pytest 在 collection 阶段产生 **4 errors**，只收集到 158 个 item，3.56s 后退出。系统 Python 恰好安装了该依赖，因此完整测试为 **222 passed，19.47s**。这说明“本机能跑”依赖未记录的环境状态。
- 离线 synthetic 与 hard-negative CLI 可复现运行，分别为 **2/2** 和 **4/4**；重复运行的 Markdown report SHA-256 完全一致。
- Streamlit server 可以启动，`/_stcore/health` 在 3.357s 后返回 `200 ok`；无 API key 时页面仍可渲染，但不能执行真实的上传后 embedding、retrieval 和 answer generation。
- OCR Python 依赖已安装，但外部 `tesseract` executable 不存在。空白 PDF 会安全退回原始 text extraction，产生 1 条 warning，而不是崩溃。
- 当前 evaluation suite 是有价值的 deterministic component/integration regression suite，但**不是**真实 RAG 质量评测：它不调用真实 OpenAI embedding/chat model，不评估生成答案质量、真实 citation faithfulness、端到端 latency/cost，也没有带人工 gold answer 的代表性保险保单数据集。
- 本调查没有修改产品代码、依赖或测试，也没有使用或输出 `documents/` 中的私有保单内容。

## 2. 支持的环境与入口

### 2.1 文档支持的安装方式

README 给出的 Windows 安装流程是：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

仓库没有 `pyproject.toml`、lock file、`.python-version` 或 CI workflow。`requirements.txt` 只有 lower bounds，没有 Python 版本约束，因此它能描述“最低声明版本”，不能固定未来安装出来的 dependency graph。

核心 direct dependencies：

```text
streamlit>=1.35
openai>=1.30
PyMuPDF>=1.24
numpy>=1.26
python-dotenv>=1.0
pydantic>=2.7
rank-bm25>=0.2.2
pytest>=8.2
pytest-mock>=3.14
Pillow>=10.3
pytesseract>=0.3
```

### 2.2 可执行入口

| 目的 | 入口 | 是否需要外部条件 |
| --- | --- | --- |
| Web UI | `python -m streamlit run app.py` | 页面启动不需要 key；上传后 indexing/QA 需要 OpenAI API key 和网络 |
| 完整测试 | `python -m pytest -ra --durations=10` | 不需要 OpenAI、真实 PDF 或 Tesseract |
| synthetic eval | `python scripts\evaluate_rag.py --synthetic` | 无外部条件 |
| hard-negative eval | `python scripts\evaluate_rag.py --hard-negative` | 无外部条件 |
| local document eval | `python scripts\evaluate_rag.py --local-documents <dir>` | 需要本地 PDF；默认关闭 OCR；不需要 OpenAI |
| local hard-negative eval | `python scripts\evaluate_rag.py --local-hard-negative <dir>` | 需要本地 PDF；默认关闭 OCR；不需要 OpenAI |
| OCR fallback | PDF parser，默认 `INSURANCE_RAG_OCR_ENABLED=true` | 需要 `pytesseract`、Pillow、Tesseract executable 和 `chi_sim+eng` language data |

配置只通过 `os.getenv` 读取。仓库根目录存在包含非空 `OPENAI_API_KEY` entry 的 `.env`，但代码没有调用 `load_dotenv()`；本次所有 probe 都确认 process environment 中没有该 key，也没有读取或发送其值。`python-dotenv` 目前是声明但未集成的 dependency。

## 3. 实际环境矩阵

宿主环境：

- OS：Windows 11 (`Windows-11-10.0.26200-SP0`)
- system Python：`D:\Anaconda3\python.exe`，CPython 3.13.9
- repo `.venv`：CPython 3.13.9
- clean baseline env：CPython 3.13.9
- `tesseract`：command not found
- `documents/`：存在 176 个 PDF；为避免泄露私有保单，本调查不读取这些文件

| Package | system Python | repo `.venv` | clean env（2026-07-23 resolve） |
| --- | ---: | ---: | ---: |
| streamlit | 1.45.1 | 1.58.0 | 1.60.0 |
| openai | 2.41.0 | 2.41.1 | 2.48.0 |
| PyMuPDF | 1.27.2.3 | 1.27.2.3 | 1.28.0 |
| numpy | 2.2.6 | 2.4.6 | 2.5.1 |
| python-dotenv | 1.1.0 | 1.2.2 | 1.2.2 |
| pydantic | 2.11.10 | 2.13.4 | 2.13.4 |
| rank-bm25 | 0.2.2 | **missing** | 0.2.2 |
| pytest | 8.3.4 | 9.0.3 | 9.1.1 |
| pytest-mock | 3.15.1 | 3.15.1 | 3.15.1 |
| Pillow | 11.3.0 | 12.2.0 | 12.3.0 |
| pytesseract | 0.3.13 | 0.3.13 | 0.3.13 |

全新环境的实际命令：

```powershell
python -m venv tmp\baseline-clean-py313-escalated
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe -m pip install -r requirements.txt
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe -m pip check
```

结果：

- venv 创建成功；
- install exit 0，耗时 156.1s；
- `pip check` exit 0，输出 `No broken requirements found.`；
- 在 Codex filesystem sandbox 内创建 venv 时，`ensurepip` 因无法写入 `%LOCALAPPDATA%\Temp` 而失败；允许标准 venv 命令在 sandbox 外执行后成功。这是 execution harness 权限条件，不是仓库代码缺陷。

需要注意，repo `.venv` 的 `pip check` 也返回成功。`pip check` 只验证已安装 distributions 之间的 dependency constraints，不会验证当前环境是否完整满足一个尚未执行安装的 `requirements.txt`，因此它没有发现缺失的 direct dependency `rank-bm25`。

## 4. 测试基线

精确命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -ra --durations=10
python -m pytest -ra --durations=10
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe -m pytest -ra --durations=10
```

| Python environment | pytest / pluggy | 结果 | skip / xfail | warnings | duration |
| --- | --- | --- | ---: | ---: | ---: |
| repo `.venv` | 9.0.3 / 1.6.0 | exit 1；`158 items / 4 errors`，collection 中止 | 0 / 0 | 0 | 3.56s |
| system Python | 8.3.4 / 1.5.0 | exit 0；**222 passed** | 0 / 0 | 0 | 19.47s |
| clean env | 9.1.1 / 1.6.0 | exit 0；**222 passed** | 0 / 0 | 0 | 22.34s |

repo `.venv` 的四个 collection errors：

```text
tests/test_evaluation.py       ModuleNotFoundError: No module named 'rank_bm25'
tests/test_hybrid_retriever.py ModuleNotFoundError: No module named 'rank_bm25'
tests/test_rag_chain.py        ModuleNotFoundError: No module named 'rank_bm25'
tests/test_rule_reranker.py    ModuleNotFoundError: No module named 'rank_bm25'
```

clean env 中最慢的六个测试都是 evaluation CLI subprocess tests，单项约 3.07–3.23s。测试没有调用真实 OpenAI service：retrieval 使用 fake/deterministic embedder，RAG chain 使用 `FakeChatClient` 并 monkeypatch `OpenAI`。

## 5. 离线 evaluation 基线

### 5.1 Synthetic

```powershell
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe scripts\evaluate_rag.py `
  --synthetic `
  --report-dir tmp\baseline-results\synthetic
```

结果：exit 0，3253ms，2/2 passed：

- `synthetic_waiting_period`：expected rank 1
- `synthetic_exclusion`：expected rank 1

### 5.2 Hard negative

```powershell
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe scripts\evaluate_rag.py `
  --hard-negative `
  --report-dir tmp\baseline-results\hard-negative
```

结果：exit 0，3165ms，4/4 passed：

- waiting period number disambiguation
- exclusion vs coverage
- waiver subject disambiguation
- built-in source confusion；verifier 按预期 block

### 5.3 参数与缺失数据的失败签名

```powershell
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe scripts\evaluate_rag.py `
  --report-dir tmp\baseline-results\none
```

结果：exit 2，3143ms：

```text
No evaluation selected. Use --synthetic, --hard-negative, --local-documents, or --local-hard-negative.
```

```powershell
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe scripts\evaluate_rag.py `
  --local-documents tmp\does-not-exist-baseline `
  --report-dir tmp\baseline-results\missing-local
```

结果：exit 1，3083ms：

```text
Skipping local document evaluation: tmp\does-not-exist-baseline does not exist.
```

虽然消息使用 “Skipping”，但在只选择 local evaluation 时 CLI 把缺失目录视为失败并返回 1。

### 5.4 重复运行一致性

synthetic 和 hard-negative 各重复一次，结果都为 exit 0。两次生成的报告 hash 分别完全一致：

```text
synthetic:
08B89DCD10B87249C4138FAEC24CBF78E91C1CADEE281E27E6983F4A5B83457A

hard-negative:
9E37EC734A5B180372C22E49D10303B08F09D7F512AD44D745A125D1E8ABE791
```

这说明当前离线 fixture 在同一代码与 dependency environment 中是 deterministic 的；由于 dependencies 未锁定，不能由此推出跨时间、跨版本 hash 稳定。

## 6. 最小安全 local PDF workflow

为避免读取 `documents/` 中的私有保单，使用 PyMuPDF 生成一个只包含人工条款的单页 PDF。可复现输入：

```powershell
New-Item -ItemType Directory -Force -Path tmp\baseline-safe-local-20260723 | Out-Null
@'
from pathlib import Path
import fitz

root = Path(r"tmp/baseline-safe-local-20260723")
text = (
    "\u7b2c\u516d\u6761 \u7b49\u5f85\u671f\n"
    "\u7b49\u5f85\u671f\u4e3a\u4e5d\u5341\u65e5\u3002\n"
    "\u7b2c\u4e03\u6761 \u4fdd\u9669\u671f\u95f4\n"
    "\u4fdd\u9669\u671f\u95f4\u4e3a\u4e00\u5e74\u3002\n"
    "\u7b2c\u516b\u6761 \u8d23\u4efb\u514d\u9664\n"
    "\u9152\u540e\u9a7e\u9a76\u5c5e\u4e8e\u8d23\u4efb\u514d\u9664\u3002\n"
    "\u7b2c\u4e5d\u6761 \u4fdd\u9669\u8d23\u4efb\n"
    "\u672c\u5408\u540c\u627f\u62c5\u91cd\u5927\u75be\u75c5\u4fdd\u9669\u8d23\u4efb\u3002"
)
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), text, fontname="china-s", fontsize=11)
(root / "sample.pdf").write_bytes(doc.tobytes())
doc.close()
'@ | .\tmp\baseline-clean-py313-escalated\Scripts\python.exe -
```

运行：

```powershell
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe scripts\evaluate_rag.py `
  --local-documents tmp\baseline-safe-local-20260723 `
  --local-sample-limit 1 `
  --report-dir tmp\baseline-results\local-safe-2

.\tmp\baseline-clean-py313-escalated\Scripts\python.exe scripts\evaluate_rag.py `
  --local-hard-negative tmp\baseline-safe-local-20260723 `
  --report-dir tmp\baseline-results\local-hard-safe-2
```

| Workflow | exit | duration | parse/chunk | retrieval |
| --- | ---: | ---: | --- | --- |
| local document | 0 | 3167ms | 1/1 PDF，1 page，1 chunk，0 empty，0 unknown title | Top1 4/4，Top3 4/4 |
| local hard-negative | 0 | 3204ms | 1/1 PDF，1 page，1 chunk，0 empty，0 unknown title | Top1 3/3，Top3 3/3 |

这些结果证明 PDF parsing → chunking → BM25-only retrieval → report/exit-code 路径可执行。它们**不能**证明 retrieval quality：输入只有一个 chunk，所有命中的 `section_title` 都是该 chunk 的首个标题“等待期”，因此 rank 1 是一个退化的单候选结果。

## 7. Streamlit、OpenAI 与 OCR 运行边界

### 7.1 Streamlit

使用 clean env 启动 README 所述入口，并增加 headless、loopback address 和固定端口参数：

```powershell
.\tmp\baseline-clean-py313-escalated\Scripts\python.exe -m streamlit run app.py `
  --server.headless true `
  --server.address 127.0.0.1 `
  --server.port 8517 `
  --browser.gatherUsageStats false
```

观察：

- process 保持运行；
- `http://127.0.0.1:8517/_stcore/health` 在 3357ms 后返回 `200 ok`；
- `streamlit.testing.v1.AppTest.from_file("app.py").run(timeout=20)`：exit 0，5198ms，0 exception，0 error，1 UI warning，1 title，1 file uploader；
- UI warning 是缺少 OpenAI API key 的预期状态。

Codex sandbox 中 `AppTest` 还在 stderr 产生一条 bare-mode `missing ScriptRunContext` warning，并在 Python shutdown 清理 `%LOCALAPPDATA%\Temp` 时遇到 `PermissionError`；AppTest 本身已 exit 0。这与创建 venv 时的行为一致，属于测试宿主 temp-directory 权限，不是应用异常。

### 7.2 OpenAI online path

真实 upload/index/ask workflow 没有执行，原因是：

- process environment 中没有 `OPENAI_API_KEY`；
- 根目录 `.env` 不会被当前代码自动加载；
- 执行会把人工或用户保单文本发送给外部 OpenAI embedding/chat endpoints，并产生网络与费用影响。

因此当前基线只能证明 UI server 和 offline paths，不能证明：

- OpenAI credentials 有效；
- model names 当前可用；
- embedding dimensions 与现有 index 一致；
- API rate limits、timeout、retry 和网络错误行为；
- 真实 answer generation 与 citation rendering 的端到端结果。

这些是缺失 external credential/service baseline，不是本次观察到的 code defect。

### 7.3 OCR

`pytesseract` 和 Pillow 已安装，但 `tesseract --list-langs` 无法执行，因为系统找不到 Tesseract executable。使用空白单页 PDF 和默认 OCR threshold 的 probe：

```text
pages=1 method=text text_len=0 warnings=1 quality_notes=1
exit=0 elapsed=3415ms
```

解释：parser 识别到页面需要 OCR，调用外部 runtime 失败，捕获异常并保留原始空文本。这证明 fallback 可运行；不证明 OCR 识别质量。要建立 OCR 质量基线，仍需固定 Tesseract 版本和 `chi_sim+eng` language data。

## 8. 失败分类

| 观察 | 分类 | 判断依据 |
| --- | --- | --- |
| repo `.venv` collection 4 errors | 本地环境漂移 / missing dependency | `rank-bm25` 已在 `requirements.txt` 声明；clean install 后 222 tests 全通过 |
| clean env 222 passed | 当前代码在声明依赖下可执行 | 无 fail、skip、xfail、warning |
| Tesseract command missing | missing external runtime/model data | Python packages 存在；外部 executable 与 language data 不存在 |
| online RAG 未执行 | missing credential/service baseline | process env 无 key；需要网络、model access 和费用授权 |
| `.env` 不被加载 | 配置/依赖集成不一致 | config 只用 `os.getenv`；`python-dotenv` 已声明但无 `load_dotenv` |
| missing local directory exit 1 | 预期 CLI input validation | 有明确消息和稳定非零 exit |
| local one-chunk 4/4、3/3 | 可执行性证据，不是质量证据 | 单候选导致所有 expected rank 都是 1 |
| Codex temp PermissionError | execution harness limitation | sandbox 外相同 venv 命令成功；AppTest 主流程 exit 0 |
| 当前调查发现的确定产品 code failure | **无** | 声明依赖的 clean environment 中所有现有测试和 offline workflows 通过 |

## 9. Evaluation suite 到底测量什么

### 9.1 实际测量

- 222 个 deterministic tests 覆盖 parser fallback、chunking、query rewriting、vector/BM25/hybrid retrieval、rule reranking、RAG orchestration、citation construction、answer guard 和 citation verifier 的 unit/component behavior。
- synthetic eval 有 2 个 curated cases，使用 SHA-256 派生的 deterministic pseudo-vector，检查 expected section/rank、expected terms 和 report output。
- hard-negative eval 有 4 个 curated cases，使用同一 deterministic pseudo-vector，加上 reranker 与 rule-based verifier，检查相邻条款、主体和 source-confusion regression。
- local eval 对真实或人工 PDF 执行 parsing/chunking，但 embedding 全为零，实际退化为 BM25-only；只对文档中出现的固定关键词创建 case。
- report 中可看到 expected rank、Top1/Top3、matched terms 和 verifier status，适合做稳定的 regression gate。

### 9.2 没有测量

- 真实 OpenAI embedding 的 semantic retrieval quality；
- 真实 chat model 的 correctness、completeness、groundedness、abstention 和 language quality；
- 生成答案中的 claim-level citation precision/recall 或人工 citation faithfulness；
- 代表性保险产品、扫描质量、表格、跨页条款、长保单和多文档冲突；
- production latency、token usage、cost、rate limit、retry、timeout 和 nondeterminism；
- adversarial prompt injection、PII handling 或 policy-content data leakage；
- 有独立人工 gold answer 的 Recall@k、MRR、answer accuracy benchmark。

README 将 synthetic evaluation 描述为检查 “Recall@k、MRR、expected rank、引用覆盖和回答守卫行为”，但当前实现只在极小 curated fixture 上报告 pass/rank；没有 aggregate Recall@k 或 MRR 数值，也没有运行生成模型后再计算 citation coverage。因此不应把 2/2 或 4/4 表述为“真实 RAG 质量已验证”。

综合判断：当前 suite 超过纯 unit testing，因为 CLI 确实串联了多个 offline RAG components；但它仍属于 deterministic regression baseline，而非 mature RAG evaluation system。

## 10. 后续 ticket 必须沿用的 ground truth

1. 以 `d68194f...`、Windows 11、CPython 3.13.9 clean env 的 **222 passed / 22.34s** 作为代码回归基线。
2. 不得把 repo `.venv` 当作可复现环境；它缺少 declared dependency。任何后续测试都应先从 clean install 或明确锁定环境开始。
3. synthetic **2/2**、hard-negative **4/4** 只作为 deterministic regression gate；不能作为 portfolio 中的真实 RAG quality claim。
4. local eval 默认是 BM25-only；单文档单 chunk 的满分只证明 pipeline execution。
5. online OpenAI、真实 OCR 和私有 `documents/` corpus 目前没有可分享、可复现的基准结果。
6. 后续若建立成熟评估，最低应加入版本化且可公开的多文档保险 fixture、gold retrieval relevance、gold answers/claims、claim-to-citation labels，以及真实模型的 latency/token/cost metadata；所有 online runs 还应记录 model snapshot、seed/temperature（若支持）和 retry policy。
7. 在依赖治理 ticket 解决前，记录每次环境的 resolved versions；lower-bound `requirements.txt` 本身不足以保证跨时间复现。

## 11. 一键复核顺序

在不使用真实 key、Tesseract 或私有 PDF 的前提下：

```powershell
python -m venv tmp\baseline-clean
.\tmp\baseline-clean\Scripts\python.exe -m pip install -r requirements.txt
.\tmp\baseline-clean\Scripts\python.exe -m pip check
.\tmp\baseline-clean\Scripts\python.exe -m pytest -ra --durations=10
.\tmp\baseline-clean\Scripts\python.exe scripts\evaluate_rag.py --synthetic
.\tmp\baseline-clean\Scripts\python.exe scripts\evaluate_rag.py --hard-negative
.\tmp\baseline-clean\Scripts\python.exe -m streamlit run app.py
```

预期 ground truth：

- install 与 `pip check` exit 0；
- pytest 222 passed；
- synthetic 2/2；
- hard-negative 4/4；
- Streamlit health 为 `ok`，页面显示 missing-key warning；
- 不应声称 online RAG 或 OCR quality 已经验证。
