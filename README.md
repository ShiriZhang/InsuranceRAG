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

## 配置 OpenAI API Key

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

使用 OpenAI API 时，用户问题、被检索到的保单片段和 embeddings 输入会发送给 OpenAI。

## OCR 说明

OCR fallback 使用 `pytesseract`。如果本机没有安装 Tesseract OCR 程序，应用仍会处理文字型 PDF，但扫描页可能无法识别。

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

## 本地资料库说明

`documents/` 是本地内置资料库目录，已故意由 Git 忽略，应保持为本地资料，不要提交到仓库。

## 免责声明

本项目只用于保单条款解释和学习演示，不构成法律、医疗、财务、保险理赔或核保建议。最终解释和理赔结论应以保险公司、合同原文和专业人士意见为准。
