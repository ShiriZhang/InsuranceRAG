# InsuranceRAG

InsuranceRAG 是一个中文本地保单解释助手。用户上传一份保险 PDF 后，可以用中文提问，系统会基于保单条款给出通俗解释，并展示页码、条款标题和原文引用。

## 功能范围

- 支持单次会话上传一份 PDF。
- 支持文字型 PDF 抽取。
- 支持简单 OCR fallback。
- 使用 OpenAI embeddings 和 chat model 做 RAG。
- 用户保单是主要依据。
- `documents/` 内置资料库只用于术语和背景解释。
- 不做最终理赔判断。
- 不长期保存用户上传的 PDF、解析文本、embeddings 或聊天历史。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 因执行策略阻止虚拟环境激活，可以在当前终端运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

也可以不激活虚拟环境，直接使用虚拟环境里的 Python 启动应用：

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

## 配置 OpenAI API Key

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

使用 OpenAI API 时，请注意以下数据流：

- 上传和索引阶段，抽取出的用户保单文本片段会发送给 OpenAI，用于生成 embeddings。
- 检索阶段，用户问题会发送给 OpenAI，用于生成 query embedding。
- 回答生成阶段，用户问题和被检索到的文本片段会发送给 OpenAI chat model。
- 如果本地存在 `documents/`，且用户问题触发了内置背景支持，选中的内置资料库文本片段也可能发送给 OpenAI 用于生成 embeddings，并可能作为明确标注的背景上下文参与回答。
- `documents/` 仍由 Git 忽略，不会上传到 GitHub，也不应提交到仓库。

## OCR 说明

OCR fallback 使用 `pytesseract`，它只是 Python wrapper。扫描版中文 PDF 的 OCR 需要额外安装外部 Tesseract OCR runtime、中文语言数据 `chi_sim`，并确保 Tesseract 可以在 PATH 中被找到。如果没有安装 Tesseract 或中文语言数据，应用仍会处理文字型 PDF，但扫描页可能无法识别。

## 启动

```powershell
streamlit run app.py
```

## Demo 问题

- 这份保单主要保障什么？
- 等待期是多少？
- 哪些情况不赔？
- 保险责任包括哪些？
- 保险期间是多久？
- 保险金额在哪里说明？
- 重大疾病定义在哪里？
- 这份保单有没有提到豁免保险费？

## RAG 检索增强

系统默认使用混合检索，以提高中文保单问答在“语义相近”和“关键词精确命中”两类问题上的召回质量：

- 语义检索使用 OpenAI embeddings，将用户问题和条款片段转换为向量后检索相近内容。
- 关键词检索使用 `rank-bm25`，补充召回保险条款中的精确词、同义表达和短句匹配。
- 混合结果使用 RRF（Reciprocal Rank Fusion）融合排序，让语义结果和关键词结果共同影响最终上下文。
- 系统包含规则式查询扩展。例如用户问“赔不赔”时，会扩展检索“保险责任”“责任免除”“赔付条件”“除外责任”等相关表达。

可以通过环境变量调整检索行为：

```powershell
$env:INSURANCE_RAG_RETRIEVAL_MODE="vector"
$env:INSURANCE_RAG_RRF_K="60"
```

`INSURANCE_RAG_RETRIEVAL_MODE="vector"` 可切换为纯向量检索；`INSURANCE_RAG_RRF_K` 用于调整 RRF 融合排序参数。

## 回答自检

生成回答后，系统会对答案进行程序化自检，降低无依据结论和过度承诺的风险：

- 如果回答陈述了具体保单事实，但没有用户上传保单中的证据支持，系统会阻断该回答。
- 如果回答使用“肯定赔”“一定不赔”等最终理赔结论式措辞，系统会阻断该回答。
- 如果回答把内置资料库内容表述成“你的保单写明”的用户保单事实，系统会阻断该回答。
- 如果引用数量较少、检索分数偏低，或 OCR 质量可能影响条款识别，系统会给出风险提示。

本项目只解释和整理保单条款，不提供法律、医疗、财务、理赔或核保建议，也不替代保险公司、合同原文或专业人士的最终判断。

## 本地资料库说明

`documents/` 是本地内置资料库目录，已故意由 Git 忽略，应保持为本地资料，不要提交到仓库。

## 离线评测

可以运行可复现的合成评测，检查检索和回答质量：

```powershell
python scripts\evaluate_rag.py --synthetic
```

评测报告会写入 `eval_reports/`，该目录已由 Git 忽略，不应提交到仓库。

如需对本地真实 PDF 做不外传数据的检索评测，可以运行：

```powershell
python scripts\evaluate_rag.py --local-documents documents --local-sample-limit 20
```

本地文档评测会解析 PDF、统计页数/chunk 数/未识别标题比例，并对“等待期”“责任免除”“保险责任”等精确条款词计算 Top1/Top3 命中。真实本地文档和评测报告都应保留在本地，不要提交到 Git。

## 免责声明

本项目只用于保单条款解释和学习演示，不构成法律、医疗、财务、保险理赔或核保建议。最终解释和理赔结论应以保险公司、合同原文和专业人士意见为准。
