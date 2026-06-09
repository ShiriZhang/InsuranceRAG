# Insurance Policy RAG Design

Date: 2026-06-08

## Overview

The MVP is a Chinese local Streamlit web app for individual users who want to understand their own insurance policy PDF. A user uploads one insurance PDF, waits for parsing and indexing, then asks questions such as:

- 这份保单主要保障什么？
- 等待期是多少？
- 哪些情况不赔？
- 如何理解重大疾病定义？

The system acts as a policy explanation assistant. It explains policy clauses in plain Chinese and cites the original source. It must not make final claim approval, denial, legal, medical, or financial conclusions.

The project already contains a built-in insurance document dataset under `documents/`, currently 176 Chinese insurance clause PDFs. In this MVP, that dataset is only used for terminology and background explanation. The uploaded user policy remains the primary source for answers.

## Product Scope

### In Scope

- Local Streamlit web app.
- Upload one user insurance PDF per session.
- Support text-based PDF extraction.
- Support simple OCR fallback for scanned or low-text pages.
- Build a temporary RAG index for the uploaded policy.
- Ask and answer policy explanation questions in Chinese.
- Show answer citations with page number, inferred clause title, and original excerpt.
- Use built-in insurance documents only for terminology or background context.
- Clearly separate user policy citations from built-in dataset citations.
- Maintain chat history during the current Streamlit session.
- Show upload status, parsing progress, document quality hints, answer citations, and user-facing error messages.

### Out of Scope

- Long-term storage of uploaded user files.
- Multi-user accounts or authentication.
- Final claim approval or denial judgment.
- Legal, medical, financial, or underwriting advice.
- Policy comparison across products.
- Multi-file user document management.
- Click-to-jump PDF citation viewer.
- Full production deployment.

## User Flow

1. The user opens the local Streamlit app.
2. The user uploads one insurance PDF.
3. The app shows parsing progress:
   - file received
   - text extraction
   - OCR fallback for low-text pages
   - chunk generation
   - temporary vector index construction
4. The app shows a document summary and quality notes.
5. The app enables a chat interface with suggested questions.
6. The user asks policy explanation questions.
7. The app retrieves relevant user policy chunks first.
8. The app optionally retrieves built-in dataset chunks only when terminology or background explanation is useful.
9. The app returns a Chinese answer with citations.
10. The user can expand citation sections to inspect page numbers, clause titles, and original excerpts.
11. When the app restarts, uploaded user files, parsed user text, temporary indexes, and chat history are cleared.

## Architecture

The application is split into focused modules.

### `app.py`

Streamlit UI and session state.

Responsibilities:

- File upload.
- Progress display.
- Document summary display.
- Chat UI.
- Suggested question buttons.
- Citation rendering.
- User-facing error messages.
- Current-session chat history.

It should not contain complex PDF parsing, retrieval, or prompt orchestration logic.

### `document_loader.py`

PDF parsing and OCR fallback.

Responsibilities:

- Read PDF pages.
- Extract text from text-based PDFs.
- Detect pages with too little text or poor text quality.
- Render low-text pages to images.
- Run simple OCR when available.
- Return normalized page objects with page number, text, extraction method, and quality notes.

Recommended tools:

- `PyMuPDF` for PDF text extraction and page rendering.
- `pytesseract` or `easyocr` for OCR fallback.

### `chunker.py`

Chunking and clause title detection.

Responsibilities:

- Split parsed page text into retrieval chunks.
- Preserve page number metadata.
- Infer likely section titles such as:
  - 保险责任
  - 责任免除
  - 等待期
  - 重大疾病定义
  - 保险期间
  - 保险金额
  - 犹豫期
  - 解除合同
- Attach citation metadata to every chunk.

Each chunk should contain:

- `text`
- `page_number`
- `section_title`
- `source_type`
- `source_name`
- `extraction_method`
- `quality_notes`

### `retriever.py`

Embedding, temporary vector indexing, and retrieval.

Responsibilities:

- Build embeddings for uploaded policy chunks using OpenAI embeddings.
- Store the uploaded policy index only for the current session.
- Search the uploaded policy index for each user question.
- Search the built-in dataset only as terminology or background support.
- Return ranked chunks with metadata for answer generation and citation display.

Recommended first version:

- Use an in-memory vector index or FAISS.
- Avoid persistent vector databases for uploaded user documents in the MVP.

### `rag_chain.py`

RAG orchestration, prompt construction, and answer generation.

Responsibilities:

- Decide retrieval scope for each query.
- Prioritize user policy evidence.
- Add built-in dataset context only for terminology or background.
- Build the model prompt.
- Enforce Chinese explanation style.
- Enforce citation requirements.
- Prevent final claim decisions and unsupported conclusions.
- Return structured answer data for the UI.

### `config.py` and `utils.py`

Configuration and shared helpers.

Responsibilities:

- OpenAI API key loading.
- Model names.
- Chunk size and overlap.
- Retrieval top-k values.
- OCR settings.
- Temporary directory settings.
- Text quality heuristics.

## Data Model

### `DocumentPage`

Represents one parsed PDF page.

Fields:

- `page_number`: 1-based page number.
- `text`: extracted or OCR text.
- `extraction_method`: `text`, `ocr`, or `mixed`.
- `quality_notes`: list of human-readable quality notes.

### `DocumentChunk`

Represents one retrievable text chunk.

Fields:

- `chunk_id`: stable ID within the current session.
- `text`: chunk text.
- `page_number`: source page number when available.
- `section_title`: inferred clause title or `未识别条款标题`.
- `source_type`: `user_policy` or `built_in_dataset`.
- `source_name`: uploaded filename or built-in document name.
- `extraction_method`: `text`, `ocr`, or `mixed`.
- `quality_notes`: page or chunk quality notes.

### `Citation`

Represents one displayed source.

Fields:

- `source_type`: `user_policy` or `built_in_dataset`.
- `source_name`: filename or document name.
- `page_number`: page number when available.
- `section_title`: inferred clause title.
- `excerpt`: original text excerpt.
- `quality_notes`: citation-level quality notes.

## RAG Behavior

The user policy is always the primary evidence source.

For every question:

1. Retrieve top policy chunks from the uploaded policy.
2. Check whether retrieved evidence is sufficient.
3. If the question asks about a general insurance term, or if the uploaded policy evidence needs background explanation, optionally retrieve from the built-in dataset.
4. Generate a Chinese answer using the retrieved evidence.
5. Attach citations grouped by source type.

The model prompt must require:

- Use plain Chinese.
- Explain, do not decide claims.
- Do not invent policy clauses.
- Do not state that a claim will be paid or denied.
- If evidence is insufficient, say so directly.
- For every important statement, rely on provided context.
- Distinguish user policy content from built-in dataset background.
- Include OCR quality warnings when cited evidence came from OCR text.

If retrieval does not find enough evidence, the app should answer:

> 这份保单中没有找到足够明确的依据。你可以换一种问法，或确认上传的保单是否完整。

The answer may then explain what kind of clause the user should look for, but it must not fabricate an answer.

## Citation Rules

User policy citations are displayed as:

> 第 X 页｜条款标题｜原文片段

Built-in dataset citations are displayed as:

> 内置资料库｜文件或产品名｜第 X 页｜原文片段

Citation groups:

- 用户保单引用
- 内置资料库引用

When citing OCR text, show:

> 该引用来自 OCR 识别文本，请核对 PDF 原文。

## Built-In Dataset Role

The built-in `documents/` dataset is not the main answer source in the MVP.

Allowed uses:

- Explain common insurance terminology.
- Provide background context for common clause patterns.
- Help users understand terms such as 等待期, 责任免除, 保险责任, 重大疾病.

Disallowed uses:

- Override the uploaded user policy.
- Imply that a clause exists in the user's policy when it only appears in built-in documents.
- Compare the user's policy against other products.
- Give product recommendations.

## Privacy and Storage

Uploaded user documents are treated as session-only data.

MVP behavior:

- Do not persist uploaded user PDFs.
- Do not persist parsed user text.
- Do not persist user policy embeddings.
- Do not persist chat history.
- Store temporary artifacts only in Streamlit session state or a temporary directory.
- Clear temporary user artifacts when the app restarts.

OpenAI API usage:

- User questions and retrieved document chunks are sent to OpenAI for answer generation.
- Policy chunks are sent to OpenAI for embedding generation.
- The UI should make this dependency clear in setup or README documentation.

## UI Design

The app is a single-page Chinese Streamlit interface.

### Sidebar

Contains:

- PDF upload control.
- Parsing status.
- Document summary.
- Quality warnings.
- Basic settings if needed:
  - OCR enabled/disabled.
  - Retrieval top-k.

### Main Area

Before upload:

- Show a concise explanation of what the app does.
- Show privacy and API usage notes.

During parsing:

- Show step-by-step progress.
- Show page-level OCR fallback status when available.

After parsing:

- Show suggested question buttons:
  - 这份保单主要保障什么？
  - 等待期是多少？
  - 哪些情况不赔？
  - 保险责任包括哪些？
  - 重大疾病定义在哪里？
- Show chat interface.
- Show expandable citation sections under each answer.

## Error Handling

All user-facing errors should be in clear Chinese.

Required cases:

- Missing OpenAI API key:
  - Tell the user how to set the environment variable.
- PDF cannot be opened:
  - Tell the user the file may be encrypted, corrupted, or unsupported.
- Text extraction fails:
  - Fall back to OCR when possible.
- OCR is unavailable:
  - Continue with text-based pages and warn that scanned pages cannot be recognized.
- OCR quality is low:
  - Mark affected pages and citations.
- Retrieval evidence is insufficient:
  - Refuse to answer beyond the evidence.
- OpenAI API request fails:
  - Tell the user to retry later and keep the parsed document state.

Developer stack traces should not be shown in the main UI.

## Testing Strategy

### Unit Tests

Cover:

- Text quality heuristic for OCR fallback.
- PDF page normalization.
- Chunk metadata generation.
- Clause title detection.
- Citation object formatting.
- Prompt input construction.

### Integration Tests

Cover:

- Parse a sample PDF.
- Generate chunks with page numbers and section titles.
- Build an in-memory index.
- Retrieve relevant chunks for common questions.
- Produce answer payloads with grouped citations.

### Manual Demo Tests

Prepare 5-8 questions against one sample policy:

- 这份保单主要保障什么？
- 等待期是多少？
- 哪些情况不赔？
- 保险期间是多久？
- 保险金额在哪里说明？
- 什么情况下合同会解除？
- 重大疾病定义在哪里？
- 这份保单有没有提到豁免保险费？

Check:

- Answers are understandable in Chinese.
- Answers avoid final claim decisions.
- Important claims have citations.
- Citations show page number, title, and excerpt.
- OCR citations show OCR warning.
- Built-in dataset citations are clearly separated.

## Implementation Approach

Use the recommended approach: complete product experience with controlled depth.

This means:

- Build a polished local Streamlit experience.
- Include upload progress, parsing progress, chat history, citations, and error handling.
- Keep OCR as a simple fallback, not a full document intelligence system.
- Keep the built-in dataset as background support, not a comparison engine.
- Keep uploaded user data temporary.

## Open Questions for Later Phases

These are intentionally outside the MVP:

- Multi-file user uploads.
- Persisted local document library.
- Clickable PDF viewer citations.
- Product comparison against the built-in dataset.
- Deployment beyond local use.
- Authentication and user accounts.
- More robust OCR layout reconstruction.
