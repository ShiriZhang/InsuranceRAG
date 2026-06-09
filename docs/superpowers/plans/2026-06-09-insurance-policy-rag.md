# Insurance Policy RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Chinese local Streamlit app that lets a personal user upload one insurance PDF, ask policy explanation questions, and inspect cited page-level evidence.

**Architecture:** The app separates UI, document loading, chunking, retrieval, and RAG orchestration into focused Python modules under `src/insurance_rag/`. Uploaded user documents are session-only; the built-in `documents/` dataset is available only as clearly labeled terminology/background support.

**Tech Stack:** Python, Streamlit, PyMuPDF, optional pytesseract OCR, OpenAI embeddings/chat completions, NumPy in-memory vector search, pytest.

---

## Git Constraint

Current workspace `D:\GitHub_ShiriZhang\InsuranceRAG` is not a git repository. Do not run `git init` or `git commit` unless the user explicitly asks for git setup. Each task includes a checkpoint step that replaces commits for this workspace.

## File Structure

- Create: `requirements.txt`
  - Runtime and test dependencies.
- Create: `.gitignore`
  - Local Python, cache, Streamlit, and temporary artifact ignores.
- Create: `README.md`
  - Setup, OpenAI API key, OCR dependency notes, app launch command, privacy notes.
- Create: `src/insurance_rag/__init__.py`
  - Package marker.
- Create: `src/insurance_rag/config.py`
  - Environment-driven configuration for OpenAI models, chunking, retrieval, and OCR.
- Create: `src/insurance_rag/models.py`
  - Dataclasses for `DocumentPage`, `DocumentChunk`, `Citation`, `AnswerPayload`, and `ParseResult`.
- Create: `src/insurance_rag/document_loader.py`
  - PDF text extraction, low-text detection, optional OCR fallback, and page normalization.
- Create: `src/insurance_rag/chunker.py`
  - Paragraph-aware chunking and clause title inference.
- Create: `src/insurance_rag/retriever.py`
  - OpenAI embedding client wrapper plus NumPy cosine-search index.
- Create: `src/insurance_rag/builtin_dataset.py`
  - Lightweight built-in dataset discovery and optional indexing helper.
- Create: `src/insurance_rag/rag_chain.py`
  - Query classification, retrieval orchestration, prompt construction, OpenAI answer generation, and citation grouping.
- Create: `app.py`
  - Streamlit single-page UI with upload, progress, chat, suggested questions, and citations.
- Create: `tests/conftest.py`
  - Shared pytest fixtures.
- Create: `tests/test_config.py`
  - Config defaults and environment parsing.
- Create: `tests/test_document_loader.py`
  - Text quality heuristic and extraction fallback behavior.
- Create: `tests/test_chunker.py`
  - Chunk metadata and clause title inference.
- Create: `tests/test_retriever.py`
  - Deterministic vector search behavior.
- Create: `tests/test_rag_chain.py`
  - Retrieval scope, refusal behavior, prompt constraints, and citation formatting.
- Create: `tests/test_builtin_dataset.py`
  - Built-in dataset discovery from existing `documents/` layout.

## Task 1: Project Skeleton and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/insurance_rag/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create dependency file**

Write `requirements.txt`:

```txt
streamlit>=1.35
openai>=1.30
PyMuPDF>=1.24
numpy>=1.26
python-dotenv>=1.0
pydantic>=2.7
pytest>=8.2
pytest-mock>=3.14
```

- [ ] **Step 2: Create ignore file**

Write `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.streamlit/secrets.toml
.env
.venv/
venv/
tmp/
temp/
.insurance_rag_tmp/
```

- [ ] **Step 3: Create package marker**

Write `src/insurance_rag/__init__.py`:

```python
"""Insurance policy RAG package."""
```

- [ ] **Step 4: Create pytest import fixture**

Write `tests/conftest.py`:

```python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 5: Install dependencies**

Run: `python -m pip install -r requirements.txt`

Expected: packages install without resolver errors.

- [ ] **Step 6: Run empty test suite**

Run: `pytest -q`

Expected: pytest starts successfully and reports no tests or zero failures.

- [ ] **Step 7: Checkpoint**

Run: `git rev-parse --is-inside-work-tree`

Expected in current workspace: `fatal: not a git repository`. Do not commit. Report: "Task 1 checkpoint complete; git is not initialized."

## Task 2: Configuration and Data Models

**Files:**
- Create: `src/insurance_rag/config.py`
- Create: `src/insurance_rag/models.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Write `tests/test_config.py`:

```python
from insurance_rag.config import AppConfig


def test_config_defaults_are_mvp_values(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = AppConfig.from_env()

    assert config.openai_api_key is None
    assert config.chat_model == "gpt-4o-mini"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.chunk_size == 900
    assert config.chunk_overlap == 150
    assert config.policy_top_k == 6
    assert config.builtin_top_k == 3
    assert config.ocr_enabled is True


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("INSURANCE_RAG_CHAT_MODEL", "custom-chat")
    monkeypatch.setenv("INSURANCE_RAG_OCR_ENABLED", "false")

    config = AppConfig.from_env()

    assert config.openai_api_key == "test-key"
    assert config.chat_model == "custom-chat"
    assert config.ocr_enabled is False
```

- [ ] **Step 2: Run failing config tests**

Run: `pytest tests/test_config.py -q`

Expected: FAIL because `insurance_rag.config` does not exist.

- [ ] **Step 3: Implement config**

Write `src/insurance_rag/config.py`:

```python
from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 900
    chunk_overlap: int = 150
    policy_top_k: int = 6
    builtin_top_k: int = 3
    min_page_text_chars: int = 80
    max_garbled_ratio: float = 0.25
    ocr_enabled: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chat_model=os.getenv("INSURANCE_RAG_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv(
                "INSURANCE_RAG_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            chunk_size=int(os.getenv("INSURANCE_RAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("INSURANCE_RAG_CHUNK_OVERLAP", "150")),
            policy_top_k=int(os.getenv("INSURANCE_RAG_POLICY_TOP_K", "6")),
            builtin_top_k=int(os.getenv("INSURANCE_RAG_BUILTIN_TOP_K", "3")),
            min_page_text_chars=int(os.getenv("INSURANCE_RAG_MIN_PAGE_TEXT_CHARS", "80")),
            max_garbled_ratio=float(os.getenv("INSURANCE_RAG_MAX_GARBLED_RATIO", "0.25")),
            ocr_enabled=_env_bool("INSURANCE_RAG_OCR_ENABLED", True),
        )
```

- [ ] **Step 4: Implement data models**

Write `src/insurance_rag/models.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str
    extraction_method: str
    quality_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    page_number: int | None
    section_title: str
    source_type: str
    source_name: str
    extraction_method: str
    quality_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    source_type: str
    source_name: str
    page_number: int | None
    section_title: str
    excerpt: str
    quality_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    policy_citations: tuple[Citation, ...] = ()
    builtin_citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParseResult:
    filename: str
    pages: tuple[DocumentPage, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
```

- [ ] **Step 5: Run config tests**

Run: `pytest tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 6: Checkpoint**

Report: "Task 2 checkpoint complete; config and data models are available."

## Task 3: PDF Text Extraction and OCR Fallback

**Files:**
- Create: `src/insurance_rag/document_loader.py`
- Test: `tests/test_document_loader.py`

- [ ] **Step 1: Write failing loader tests**

Write `tests/test_document_loader.py`:

```python
from insurance_rag.config import AppConfig
from insurance_rag.document_loader import (
    PageExtraction,
    garbled_ratio,
    needs_ocr,
    normalize_page_text,
)


def test_garbled_ratio_counts_replacement_characters():
    assert garbled_ratio("abc��") == 0.4


def test_needs_ocr_for_short_text():
    config = AppConfig(openai_api_key=None, min_page_text_chars=20)

    assert needs_ocr("短", config) is True


def test_needs_ocr_for_garbled_text():
    config = AppConfig(openai_api_key=None, max_garbled_ratio=0.2)

    assert needs_ocr("正常文字����", config) is True


def test_normalize_page_text_returns_quality_note_for_ocr():
    page = normalize_page_text(
        PageExtraction(page_number=3, text=" 等待期 是 90 天 ", extraction_method="ocr"),
    )

    assert page.page_number == 3
    assert page.text == "等待期 是 90 天"
    assert page.extraction_method == "ocr"
    assert "该页来自 OCR 识别，可能有误。" in page.quality_notes
```

- [ ] **Step 2: Run failing loader tests**

Run: `pytest tests/test_document_loader.py -q`

Expected: FAIL because `insurance_rag.document_loader` does not exist.

- [ ] **Step 3: Implement loader helpers and PDF parser**

Write `src/insurance_rag/document_loader.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import fitz

from insurance_rag.config import AppConfig
from insurance_rag.models import DocumentPage, ParseResult


@dataclass(frozen=True)
class PageExtraction:
    page_number: int
    text: str
    extraction_method: str


def garbled_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad_chars = sum(1 for char in text if char in {"�", "\ufffd"})
    return bad_chars / len(text)


def needs_ocr(text: str, config: AppConfig) -> bool:
    cleaned = text.strip()
    if len(cleaned) < config.min_page_text_chars:
        return True
    return garbled_ratio(cleaned) > config.max_garbled_ratio


def normalize_page_text(extraction: PageExtraction) -> DocumentPage:
    text = " ".join(extraction.text.split())
    notes: list[str] = []
    if extraction.extraction_method == "ocr":
        notes.append("该页来自 OCR 识别，可能有误。")
    if not text:
        notes.append("该页未提取到可用文本。")
    return DocumentPage(
        page_number=extraction.page_number,
        text=text,
        extraction_method=extraction.extraction_method,
        quality_notes=tuple(notes),
    )


def _ocr_page(page: fitz.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OCR 依赖不可用，请安装 pytesseract 和 Pillow。") from exc

    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    return pytesseract.image_to_string(image, lang="chi_sim+eng")


def parse_pdf_bytes(pdf_bytes: bytes, filename: str, config: AppConfig) -> ParseResult:
    warnings: list[str] = []
    pages: list[DocumentPage] = []

    with NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(pdf_bytes)
        temp_file.flush()
        try:
            document = fitz.open(temp_file.name)
        except Exception as exc:
            raise ValueError("PDF 无法打开，文件可能已加密、损坏或格式不受支持。") from exc

        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            method = "text"
            if config.ocr_enabled and needs_ocr(raw_text, config):
                try:
                    raw_text = _ocr_page(page)
                    method = "ocr"
                except RuntimeError as exc:
                    warnings.append(str(exc))
            pages.append(
                normalize_page_text(
                    PageExtraction(
                        page_number=index,
                        text=raw_text,
                        extraction_method=method,
                    )
                )
            )

    return ParseResult(filename=Path(filename).name, pages=tuple(pages), warnings=tuple(warnings))
```

- [ ] **Step 4: Add optional OCR dependencies**

Append to `requirements.txt`:

```txt
Pillow>=10.3
pytesseract>=0.3
```

- [ ] **Step 5: Run loader tests**

Run: `pytest tests/test_document_loader.py -q`

Expected: PASS.

- [ ] **Step 6: Checkpoint**

Report: "Task 3 checkpoint complete; PDF parser has text extraction and OCR fallback hooks."

## Task 4: Chunking and Clause Title Detection

**Files:**
- Create: `src/insurance_rag/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing chunker tests**

Write `tests/test_chunker.py`:

```python
from insurance_rag.chunker import infer_section_title, chunk_pages
from insurance_rag.models import DocumentPage


def test_infer_section_title_matches_known_clause_heading():
    text = "责任免除\n因下列情形之一导致被保险人发生疾病的，本公司不承担保险责任。"

    assert infer_section_title(text, current_title="未识别条款标题") == "责任免除"


def test_chunk_pages_preserves_page_and_source_metadata():
    pages = (
        DocumentPage(
            page_number=2,
            text="保险责任\n本合同保障重大疾病。\n等待期\n等待期为九十日。",
            extraction_method="text",
        ),
    )

    chunks = chunk_pages(pages, source_name="user.pdf", source_type="user_policy", chunk_size=20, overlap=5)

    assert chunks
    assert chunks[0].page_number == 2
    assert chunks[0].section_title == "保险责任"
    assert chunks[0].source_type == "user_policy"
    assert chunks[0].source_name == "user.pdf"
```

- [ ] **Step 2: Run failing chunker tests**

Run: `pytest tests/test_chunker.py -q`

Expected: FAIL because `insurance_rag.chunker` does not exist.

- [ ] **Step 3: Implement chunker**

Write `src/insurance_rag/chunker.py`:

```python
import re

from insurance_rag.models import DocumentChunk, DocumentPage


KNOWN_TITLES = (
    "保险责任",
    "责任免除",
    "等待期",
    "重大疾病定义",
    "保险期间",
    "保险金额",
    "犹豫期",
    "解除合同",
    "合同解除",
    "保险费",
    "豁免保险费",
)


def infer_section_title(text: str, current_title: str) -> str:
    first_lines = [line.strip() for line in text.splitlines()[:5] if line.strip()]
    for line in first_lines:
        normalized = re.sub(r"^[第\d一二三四五六七八九十百、\.\s条款章节]+", "", line)
        for title in KNOWN_TITLES:
            if title in normalized:
                return title
    for title in KNOWN_TITLES:
        if title in text[:120]:
            return title
    return current_title


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{1,}|\r\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap :]
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(
    pages: tuple[DocumentPage, ...],
    source_name: str,
    source_type: str,
    chunk_size: int,
    overlap: int,
) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    current_title = "未识别条款标题"
    for page in pages:
        if not page.text:
            continue
        for part in _split_text(page.text, chunk_size=chunk_size, overlap=overlap):
            current_title = infer_section_title(part, current_title)
            chunk_id = f"{source_type}:{source_name}:p{page.page_number}:c{len(chunks) + 1}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=part,
                    page_number=page.page_number,
                    section_title=current_title,
                    source_type=source_type,
                    source_name=source_name,
                    extraction_method=page.extraction_method,
                    quality_notes=page.quality_notes,
                )
            )
    return tuple(chunks)
```

- [ ] **Step 4: Run chunker tests**

Run: `pytest tests/test_chunker.py -q`

Expected: PASS.

- [ ] **Step 5: Checkpoint**

Report: "Task 4 checkpoint complete; chunks carry citation metadata."

## Task 5: In-Memory Embedding Retriever

**Files:**
- Create: `src/insurance_rag/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write failing retriever tests**

Write `tests/test_retriever.py`:

```python
from insurance_rag.models import DocumentChunk
from insurance_rag.retriever import InMemoryVectorIndex


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        page_number=1,
        section_title="保险责任",
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
    )


def test_vector_index_returns_most_similar_chunk():
    chunks = (make_chunk("a", "等待期为九十日"), make_chunk("b", "责任免除条款"))
    index = InMemoryVectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = index.search([0.9, 0.1], top_k=1)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].score > 0.9
```

- [ ] **Step 2: Run failing retriever tests**

Run: `pytest tests/test_retriever.py -q`

Expected: FAIL because `insurance_rag.retriever` does not exist.

- [ ] **Step 3: Implement retriever**

Write `src/insurance_rag/retriever.py`:

```python
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from insurance_rag.models import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class InMemoryVectorIndex:
    def __init__(self, chunks: tuple[DocumentChunk, ...], matrix: np.ndarray) -> None:
        self.chunks = chunks
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    @classmethod
    def from_embeddings(
        cls,
        chunks: tuple[DocumentChunk, ...],
        embeddings: list[list[float]],
    ) -> "InMemoryVectorIndex":
        matrix = np.array(embeddings, dtype=np.float32)
        return cls(chunks=chunks, matrix=matrix)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm
        scores = self.matrix @ query
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked
        ]


def build_index(chunks: tuple[DocumentChunk, ...], embedder: OpenAIEmbedder) -> InMemoryVectorIndex:
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    return InMemoryVectorIndex.from_embeddings(chunks, embeddings)
```

- [ ] **Step 4: Run retriever tests**

Run: `pytest tests/test_retriever.py -q`

Expected: PASS.

- [ ] **Step 5: Checkpoint**

Report: "Task 5 checkpoint complete; in-memory vector retrieval is available."

## Task 6: Built-In Dataset Discovery

**Files:**
- Create: `src/insurance_rag/builtin_dataset.py`
- Test: `tests/test_builtin_dataset.py`

- [ ] **Step 1: Write failing built-in dataset tests**

Write `tests/test_builtin_dataset.py`:

```python
from pathlib import Path

from insurance_rag.builtin_dataset import discover_builtin_pdfs


def test_discover_builtin_pdfs_reads_nested_pdf_layout(tmp_path):
    pdf_path = tmp_path / "公司A" / "产品A" / "公司A_产品A_条款书.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4")

    docs = discover_builtin_pdfs(tmp_path)

    assert len(docs) == 1
    assert docs[0].company_name == "公司A"
    assert docs[0].product_name == "产品A"
    assert docs[0].path == pdf_path
```

- [ ] **Step 2: Run failing built-in dataset tests**

Run: `pytest tests/test_builtin_dataset.py -q`

Expected: FAIL because `insurance_rag.builtin_dataset` does not exist.

- [ ] **Step 3: Implement built-in dataset discovery**

Write `src/insurance_rag/builtin_dataset.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuiltInPdf:
    path: Path
    company_name: str
    product_name: str

    @property
    def display_name(self) -> str:
        return f"{self.company_name}｜{self.product_name}"


def discover_builtin_pdfs(root: Path) -> tuple[BuiltInPdf, ...]:
    if not root.exists():
        return ()
    docs: list[BuiltInPdf] = []
    for path in sorted(root.rglob("*.pdf")):
        relative = path.relative_to(root)
        parts = relative.parts
        company = parts[0] if len(parts) >= 1 else "未知保险公司"
        product = parts[1] if len(parts) >= 2 else path.stem
        docs.append(BuiltInPdf(path=path, company_name=company, product_name=product))
    return tuple(docs)
```

- [ ] **Step 4: Run built-in dataset tests**

Run: `pytest tests/test_builtin_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Add manual dataset count check**

Run: `python -c "from pathlib import Path; from insurance_rag.builtin_dataset import discover_builtin_pdfs; print(len(discover_builtin_pdfs(Path('documents'))))"`

Expected in current workspace: `176`.

- [ ] **Step 6: Checkpoint**

Report: "Task 6 checkpoint complete; built-in PDF discovery matches current dataset."

## Task 7: RAG Prompting and Answer Payloads

**Files:**
- Create: `src/insurance_rag/rag_chain.py`
- Test: `tests/test_rag_chain.py`

- [ ] **Step 1: Write failing RAG tests**

Write `tests/test_rag_chain.py`:

```python
from insurance_rag.models import DocumentChunk
from insurance_rag.rag_chain import (
    REFUSAL_ANSWER,
    build_citation,
    build_messages,
    should_use_builtin_context,
)


def make_chunk(source_type: str = "user_policy") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="c1",
        text="等待期为九十日。",
        page_number=4,
        section_title="等待期",
        source_type=source_type,
        source_name="user.pdf" if source_type == "user_policy" else "内置条款.pdf",
        extraction_method="text",
    )


def test_should_use_builtin_context_for_term_question():
    assert should_use_builtin_context("什么是等待期？", policy_result_count=2) is True


def test_should_not_use_builtin_context_for_specific_policy_question():
    assert should_use_builtin_context("这份保单等待期是多少？", policy_result_count=3) is False


def test_build_citation_uses_chunk_metadata():
    citation = build_citation(make_chunk())

    assert citation.page_number == 4
    assert citation.section_title == "等待期"
    assert citation.excerpt == "等待期为九十日。"


def test_build_messages_include_no_claim_decision_rule():
    messages = build_messages("等待期是多少？", [make_chunk()], [])
    joined = "\n".join(message["content"] for message in messages)

    assert "不得做最终理赔判断" in joined
    assert "用户保单资料" in joined


def test_refusal_answer_is_evidence_limited():
    assert "没有找到足够明确的依据" in REFUSAL_ANSWER
```

- [ ] **Step 2: Run failing RAG tests**

Run: `pytest tests/test_rag_chain.py -q`

Expected: FAIL because `insurance_rag.rag_chain` does not exist.

- [ ] **Step 3: Implement RAG chain helpers and OpenAI answer client**

Write `src/insurance_rag/rag_chain.py`:

```python
from openai import OpenAI

from insurance_rag.config import AppConfig
from insurance_rag.models import AnswerPayload, Citation, DocumentChunk
from insurance_rag.retriever import InMemoryVectorIndex, OpenAIEmbedder


REFUSAL_ANSWER = "这份保单中没有找到足够明确的依据。你可以换一种问法，或确认上传的保单是否完整。"

TERM_KEYWORDS = ("什么是", "是什么意思", "如何理解", "定义", "概念")


def should_use_builtin_context(question: str, policy_result_count: int) -> bool:
    if policy_result_count == 0:
        return True
    return any(keyword in question for keyword in TERM_KEYWORDS)


def build_citation(chunk: DocumentChunk, max_chars: int = 180) -> Citation:
    excerpt = chunk.text.strip().replace("\n", " ")
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "..."
    return Citation(
        source_type=chunk.source_type,
        source_name=chunk.source_name,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        excerpt=excerpt,
        quality_notes=chunk.quality_notes,
    )


def _format_context(title: str, chunks: list[DocumentChunk]) -> str:
    if not chunks:
        return f"{title}：无"
    lines = [f"{title}："]
    for index, chunk in enumerate(chunks, start=1):
        page = f"第 {chunk.page_number} 页" if chunk.page_number is not None else "页码未知"
        lines.append(
            f"[{index}] {chunk.source_name}｜{page}｜{chunk.section_title}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def build_messages(
    question: str,
    policy_chunks: list[DocumentChunk],
    builtin_chunks: list[DocumentChunk],
) -> list[dict[str, str]]:
    system = (
        "你是中文保险保单解释助手。"
        "你只能解释条款含义，不得做最终理赔判断，不得给法律、医疗、财务建议。"
        "用户保单资料是主要依据；内置资料库只能用于术语或背景解释。"
        "如果资料不足，必须明确说明没有找到足够明确的依据。"
        "回答要通俗、简洁，并能对应引用来源。"
    )
    user = (
        f"用户问题：{question}\n\n"
        f"{_format_context('用户保单资料', policy_chunks)}\n\n"
        f"{_format_context('内置资料库背景', builtin_chunks)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class RagChain:
    def __init__(
        self,
        config: AppConfig,
        policy_index: InMemoryVectorIndex,
        embedder: OpenAIEmbedder,
        builtin_index: InMemoryVectorIndex | None = None,
    ) -> None:
        if not config.openai_api_key:
            raise ValueError("缺少 OPENAI_API_KEY。")
        self.config = config
        self.policy_index = policy_index
        self.builtin_index = builtin_index
        self.embedder = embedder
        self.client = OpenAI(api_key=config.openai_api_key)

    def answer(self, question: str) -> AnswerPayload:
        query_embedding = self.embedder.embed_texts([question])[0]
        policy_results = self.policy_index.search(query_embedding, top_k=self.config.policy_top_k)
        policy_chunks = [result.chunk for result in policy_results]
        if not policy_chunks:
            return AnswerPayload(answer=REFUSAL_ANSWER)

        builtin_chunks: list[DocumentChunk] = []
        if self.builtin_index and should_use_builtin_context(question, len(policy_chunks)):
            builtin_results = self.builtin_index.search(query_embedding, top_k=self.config.builtin_top_k)
            builtin_chunks = [result.chunk for result in builtin_results]

        messages = build_messages(question, policy_chunks, builtin_chunks)
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content or REFUSAL_ANSWER
        return AnswerPayload(
            answer=answer,
            policy_citations=tuple(build_citation(chunk) for chunk in policy_chunks),
            builtin_citations=tuple(build_citation(chunk) for chunk in builtin_chunks),
        )
```

- [ ] **Step 4: Run RAG tests**

Run: `pytest tests/test_rag_chain.py -q`

Expected: PASS.

- [ ] **Step 5: Checkpoint**

Report: "Task 7 checkpoint complete; RAG prompt and answer payload rules are test-covered."

## Task 8: Streamlit App

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create Streamlit app**

Write `app.py`:

```python
from pathlib import Path

import streamlit as st

from insurance_rag.chunker import chunk_pages
from insurance_rag.config import AppConfig
from insurance_rag.document_loader import parse_pdf_bytes
from insurance_rag.models import AnswerPayload
from insurance_rag.rag_chain import RagChain
from insurance_rag.retriever import OpenAIEmbedder, build_index


st.set_page_config(page_title="保单解释助手", page_icon="📄", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("parse_result", None)
    st.session_state.setdefault("policy_index", None)
    st.session_state.setdefault("embedder", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("policy_chunks", ())


def render_citations(payload: AnswerPayload) -> None:
    if payload.policy_citations:
        with st.expander("用户保单引用", expanded=True):
            for citation in payload.policy_citations:
                page = f"第 {citation.page_number} 页" if citation.page_number else "页码未知"
                st.markdown(f"**{page}｜{citation.section_title}**")
                st.write(citation.excerpt)
                for note in citation.quality_notes:
                    st.warning(note)
    if payload.builtin_citations:
        with st.expander("内置资料库引用", expanded=False):
            for citation in payload.builtin_citations:
                page = f"第 {citation.page_number} 页" if citation.page_number else "页码未知"
                st.markdown(f"**{citation.source_name}｜{page}｜{citation.section_title}**")
                st.write(citation.excerpt)


def process_upload(uploaded_file, config: AppConfig) -> None:
    if not config.openai_api_key:
        st.error("缺少 OPENAI_API_KEY。请先在本地环境变量中配置 OpenAI API key。")
        return

    progress = st.progress(0, text="接收文件")
    pdf_bytes = uploaded_file.getvalue()

    progress.progress(20, text="解析 PDF 文本，必要时使用 OCR")
    parse_result = parse_pdf_bytes(pdf_bytes, uploaded_file.name, config)

    progress.progress(50, text="生成检索片段")
    chunks = chunk_pages(
        parse_result.pages,
        source_name=parse_result.filename,
        source_type="user_policy",
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )

    progress.progress(75, text="生成 embeddings 并建立临时索引")
    embedder = OpenAIEmbedder(api_key=config.openai_api_key, model=config.embedding_model)
    policy_index = build_index(chunks, embedder)

    st.session_state.parse_result = parse_result
    st.session_state.policy_chunks = chunks
    st.session_state.policy_index = policy_index
    st.session_state.embedder = embedder
    st.session_state.messages = []
    progress.progress(100, text="解析完成")


def main() -> None:
    init_state()
    config = AppConfig.from_env()

    st.title("保单解释助手")
    st.caption("上传一份保险 PDF，用中文提问，并查看页码、条款标题和原文引用。")

    with st.sidebar:
        st.header("上传保单")
        uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"])
        if uploaded_file and st.button("解析保单", type="primary"):
            process_upload(uploaded_file, config)

        parse_result = st.session_state.parse_result
        if parse_result:
            st.success(f"已解析：{parse_result.filename}")
            st.write(f"页数：{len(parse_result.pages)}")
            st.write(f"检索片段：{len(st.session_state.policy_chunks)}")
            for warning in parse_result.warnings:
                st.warning(warning)

    if not st.session_state.policy_index:
        st.info("请先上传并解析一份保险 PDF。用户上传内容只在当前会话中使用。")
        st.warning("使用 OpenAI API 时，问题和被检索到的保单片段会发送给 OpenAI 用于生成回答。")
        return

    suggested = [
        "这份保单主要保障什么？",
        "等待期是多少？",
        "哪些情况不赔？",
        "保险责任包括哪些？",
        "重大疾病定义在哪里？",
    ]
    cols = st.columns(len(suggested))
    for col, question in zip(cols, suggested):
        if col.button(question):
            st.session_state.pending_question = question

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("payload"):
                render_citations(message["payload"])

    question = st.chat_input("请输入你想了解的保单问题")
    question = st.session_state.pop("pending_question", None) or question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索保单并生成解释"):
                try:
                    chain = RagChain(
                        config=config,
                        policy_index=st.session_state.policy_index,
                        embedder=st.session_state.embedder,
                    )
                    payload = chain.answer(question)
                except Exception as exc:
                    payload = AnswerPayload(answer=f"处理问题时出错：{exc}")
                st.write(payload.answer)
                render_citations(payload)
        st.session_state.messages.append(
            {"role": "assistant", "content": payload.answer, "payload": payload}
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run import check**

Run: `python -c "import app; print('app import ok')"`

Expected: prints `app import ok` without import errors.

- [ ] **Step 3: Run test suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 4: Start local app**

Run: `streamlit run app.py`

Expected: Streamlit prints a local URL such as `http://localhost:8501`.

- [ ] **Step 5: Manual UI smoke test**

Open the local URL, confirm:

- The page title is `保单解释助手`.
- Upload control accepts PDF.
- Missing API key produces a Chinese error.
- Before upload, the app shows privacy and OpenAI API notes.

- [ ] **Step 6: Checkpoint**

Report: "Task 8 checkpoint complete; Streamlit UI is wired to the RAG pipeline."

## Task 9: Built-In Dataset Background Support

**Files:**
- Modify: `src/insurance_rag/builtin_dataset.py`
- Modify: `app.py`
- Test: `tests/test_builtin_dataset.py`

- [ ] **Step 1: Add failing test for capped built-in sample loading**

Append to `tests/test_builtin_dataset.py`:

```python
from insurance_rag.builtin_dataset import select_background_pdfs


def test_select_background_pdfs_caps_count(tmp_path):
    docs = []
    for index in range(5):
        pdf_path = tmp_path / f"公司{index}" / f"产品{index}" / "条款书.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4")
    discovered = discover_builtin_pdfs(tmp_path)

    selected = select_background_pdfs(discovered, limit=2)

    assert len(selected) == 2
```

- [ ] **Step 2: Run failing built-in test**

Run: `pytest tests/test_builtin_dataset.py -q`

Expected: FAIL because `select_background_pdfs` does not exist.

- [ ] **Step 3: Implement capped selection**

Append to `src/insurance_rag/builtin_dataset.py`:

```python
def select_background_pdfs(docs: tuple[BuiltInPdf, ...], limit: int = 8) -> tuple[BuiltInPdf, ...]:
    return docs[:limit]
```

- [ ] **Step 4: Wire optional built-in indexing in UI**

Modify `app.py` imports:

```python
from insurance_rag.builtin_dataset import discover_builtin_pdfs, select_background_pdfs
```

Modify `init_state()`:

```python
    st.session_state.setdefault("builtin_index", None)
```

Add helper function:

```python
def build_builtin_background_index(config: AppConfig, embedder: OpenAIEmbedder):
    docs = select_background_pdfs(discover_builtin_pdfs(Path("documents")), limit=8)
    chunks = []
    for doc in docs:
        try:
            parsed = parse_pdf_bytes(doc.path.read_bytes(), doc.path.name, config)
        except ValueError:
            continue
        chunks.extend(
            chunk_pages(
                parsed.pages,
                source_name=doc.display_name,
                source_type="built_in_dataset",
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        )
    if not chunks:
        return None
    return build_index(tuple(chunks), embedder)
```

In `process_upload()`, after `policy_index = build_index(chunks, embedder)`, add:

```python
    progress.progress(90, text="准备内置资料库背景索引")
    builtin_index = build_builtin_background_index(config, embedder)
```

In `process_upload()`, after setting `st.session_state.policy_index`, add:

```python
    st.session_state.builtin_index = builtin_index
```

In `RagChain(...)` construction, pass:

```python
                        builtin_index=st.session_state.builtin_index,
```

- [ ] **Step 5: Run tests**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 6: Manual source separation check**

Ask the app: `什么是等待期？`

Expected:

- The answer remains Chinese.
- User policy citations appear under `用户保单引用`.
- If built-in context is used, citations appear under `内置资料库引用`.

- [ ] **Step 7: Checkpoint**

Report: "Task 9 checkpoint complete; built-in dataset is available as capped background support."

## Task 10: README and Demo Questions

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Write `README.md`:

```markdown
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

## 免责声明

本项目只用于保单条款解释和学习演示，不构成法律、医疗、财务、保险理赔或核保建议。最终解释和理赔结论应以保险公司、合同原文和专业人士意见为准。
```

- [ ] **Step 2: Run README smoke check**

Run: `Get-Content -Encoding UTF8 README.md | Select-Object -First 20`

Expected: Chinese text displays correctly.

- [ ] **Step 3: Checkpoint**

Report: "Task 10 checkpoint complete; README documents setup, privacy, OCR, and demo flow."

## Task 11: Final Verification

**Files:**
- No new files.
- Verify: full project.

- [ ] **Step 1: Run full tests**

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 2: Run module import checks**

Run: `python -c "from insurance_rag.config import AppConfig; from insurance_rag.document_loader import parse_pdf_bytes; from insurance_rag.rag_chain import RagChain; print('imports ok')"`

Expected: prints `imports ok`.

- [ ] **Step 3: Count built-in PDFs**

Run: `python -c "from pathlib import Path; from insurance_rag.builtin_dataset import discover_builtin_pdfs; print(len(discover_builtin_pdfs(Path('documents'))))"`

Expected: `176`.

- [ ] **Step 4: Launch app for manual demo**

Run: `streamlit run app.py`

Expected: app opens at a local Streamlit URL.

- [ ] **Step 5: Manual answer behavior check**

Upload a sample insurance PDF from `documents/`, then ask:

```text
这份保单主要保障什么？
```

Expected:

- Answer is in Chinese.
- Answer explains rather than decides claims.
- Citations include page number, clause title, and excerpt.

- [ ] **Step 6: Manual refusal behavior check**

Ask:

```text
这份保单是否一定会赔我的某次具体理赔？
```

Expected:

- App does not make a final claim decision.
- App explains it can only interpret clauses.
- App references relevant clauses if found.

- [ ] **Step 7: Final checkpoint**

Report:

```text
Implementation verification complete.
Tests: pytest -q passed.
Built-in documents discovered: 176.
Manual Streamlit smoke test completed.
Git commit skipped because workspace is not a git repository.
```

## Self-Review

- Spec coverage: The plan covers local Streamlit UI, user upload, text PDF parsing, OCR fallback hooks, session-only index behavior, citation metadata, built-in dataset background support, Chinese answer rules, error handling, tests, README, and manual demo verification.
- Placeholder scan: The plan contains concrete file paths, commands, code snippets, and expected results for each implementation task.
- Type consistency: `DocumentPage`, `DocumentChunk`, `Citation`, `AnswerPayload`, `ParseResult`, `AppConfig`, `InMemoryVectorIndex`, `OpenAIEmbedder`, and `RagChain` names are introduced before later tasks use them.
