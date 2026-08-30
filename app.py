from pathlib import Path
import sys

import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_rag.chunker import chunk_pages
from insurance_rag.builtin_dataset import (
    BuiltInPdf,
    discover_builtin_pdfs,
    select_background_pdfs,
)
from insurance_rag.config import AppConfig
from insurance_rag.document_loader import parse_pdf_bytes
from insurance_rag.hybrid_retriever import HybridRetriever
from insurance_rag.models import AnswerPayload, Citation, RetrievalExplanation
from insurance_rag.rag_chain import RagChain, should_use_builtin_context
from insurance_rag.retriever import OpenAIEmbedder, build_index


st.set_page_config(page_title="保单解释助手", page_icon="📄", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("parse_result", None)
    st.session_state.setdefault("policy_retriever", None)
    st.session_state.setdefault("builtin_retriever", None)
    st.session_state.setdefault("builtin_index_attempted", False)
    st.session_state.setdefault("embedder", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("policy_chunks", ())


def clear_policy_state() -> None:
    st.session_state.parse_result = None
    st.session_state.policy_retriever = None
    st.session_state.builtin_retriever = None
    st.session_state.builtin_index_attempted = False
    st.session_state.embedder = None
    st.session_state.messages = []
    st.session_state.policy_chunks = ()


@st.cache_data(show_spinner=False)
def discover_builtin_pdf_metadata() -> tuple[BuiltInPdf, ...]:
    return discover_builtin_pdfs(Path("documents"))


def build_builtin_background_index(
    config: AppConfig,
    embedder: OpenAIEmbedder,
) -> HybridRetriever | None:
    docs = select_background_pdfs(discover_builtin_pdf_metadata(), limit=8)
    chunks = []
    for doc in docs:
        try:
            parsed = parse_pdf_bytes(doc.path.read_bytes(), doc.path.name, config)
        except Exception:
            continue
        chunks.extend(
            chunk_pages(
                parsed.pages,
                source_name=doc.display_name,
                source_type="built_in_dataset",
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                strategy=config.chunking_strategy,
                target_chars=config.chunk_target_chars,
                hard_max_chars=config.chunk_hard_max_chars,
            )
        )
    if not chunks:
        if docs:
            st.warning("内置资料库背景索引未建立，系统仍可基于用户保单回答。")
        return None
    try:
        vector_index = build_index(tuple(chunks), embedder)
        return HybridRetriever(
            chunks=tuple(chunks),
            vector_index=vector_index,
            embedder=embedder,
            rrf_k=config.rrf_k,
            retrieval_mode=config.retrieval_mode,
        )
    except Exception as exc:
        st.warning(f"内置资料库背景索引建立失败，系统仍可基于用户保单回答：{exc}")
        return None


def maybe_build_builtin_background_index(question: str, config: AppConfig) -> None:
    if st.session_state.builtin_retriever is not None:
        return
    if st.session_state.builtin_index_attempted:
        return
    if not should_use_builtin_context(question, len(st.session_state.policy_chunks)):
        return

    st.session_state.builtin_index_attempted = True
    builtin_retriever = build_builtin_background_index(config, st.session_state.embedder)
    if builtin_retriever is not None:
        st.session_state.builtin_retriever = builtin_retriever


def render_retrieval_details(
    explanations: tuple[RetrievalExplanation, ...],
) -> None:
    if not explanations:
        return

    with st.expander("检索依据详情", expanded=False):
        for index, explanation in enumerate(explanations, start=1):
            page = (
                f"第 {explanation.page_number} 页"
                if explanation.page_number is not None
                else "页码未知"
            )
            st.markdown(
                f"**{index}. {explanation.source_name} - {page} - "
                f"{explanation.section_title}**"
            )
            st.write(f"匹配强度：{explanation.match_strength}")

            scores = [f"final_score={explanation.final_score:.4f}"]
            if explanation.vector_score is not None:
                scores.append(f"vector_score={explanation.vector_score:.4f}")
            if explanation.bm25_score is not None:
                scores.append(f"bm25_score={explanation.bm25_score:.4f}")
            st.caption("；".join(scores))

            if explanation.rerank_score is not None:
                st.caption(f"rerank_score={explanation.rerank_score:.4f}")
            if explanation.rerank_reasons:
                st.write("重排依据：" + "、".join(explanation.rerank_reasons))

            if explanation.matched_terms:
                st.write("匹配词：" + "、".join(explanation.matched_terms))

            for detail in explanation.rank_details or ():
                st.caption(
                    f"{detail.method} | rank={detail.rank} | "
                    f"score={detail.score:.4f} | query={detail.query}"
                )


def render_citation_verification(payload: AnswerPayload) -> None:
    verification = payload.citation_verification
    if verification is None or not verification.facts:
        return

    with st.expander("证据核验结果", expanded=False):
        for fact in verification.facts:
            if fact.severity == "block":
                st.error(f"未通过：{fact.fact_text}")
            elif fact.severity == "warn":
                st.warning(f"需核对：{fact.fact_text}")
            else:
                st.success(f"已支持：{fact.fact_text}")
            if fact.reason:
                st.caption(fact.reason)
            if fact.supporting_citation_ids:
                st.write("支持引用：" + "、".join(fact.supporting_citation_ids))


def render_citations(payload: AnswerPayload) -> None:
    for warning in payload.warnings:
        st.warning(warning)

    if payload.policy_citations:
        with st.expander("用户保单引用", expanded=True):
            for citation in payload.policy_citations:
                page = _format_citation_pages(citation)
                st.markdown(f"**{page}｜{citation.section_title}**")
                st.write(citation.excerpt)
                for note in citation.quality_notes:
                    st.warning(note)
    if payload.builtin_citations:
        with st.expander("内置资料库引用", expanded=False):
            for citation in payload.builtin_citations:
                page = _format_citation_pages(citation)
                st.markdown(f"**{citation.source_name}｜{page}｜{citation.section_title}**")
                st.write(citation.excerpt)
    render_citation_verification(payload)
    render_retrieval_details(payload.retrieval_explanations)


def _format_citation_pages(citation: Citation) -> str:
    if not citation.page_numbers:
        return "页码未知"
    pages = "、".join(str(page_number) for page_number in citation.page_numbers)
    return f"第 {pages} 页"


def process_upload(uploaded_file, config: AppConfig) -> None:
    clear_policy_state()

    if not config.openai_api_key:
        st.error("缺少 OPENAI_API_KEY。请先在本地环境变量中配置 OpenAI API key。")
        return

    progress = st.progress(0, text="接收文件")
    pdf_bytes = uploaded_file.getvalue()

    progress.progress(20, text="解析 PDF 文本，必要时使用 OCR")
    try:
        parse_result = parse_pdf_bytes(pdf_bytes, uploaded_file.name, config)
    except Exception as exc:
        st.error(f"解析 PDF 时出错：{exc}")
        return

    progress.progress(50, text="生成检索片段")
    chunks = chunk_pages(
        parse_result.pages,
        source_name=parse_result.filename,
        source_type="user_policy",
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
        strategy=config.chunking_strategy,
        target_chars=config.chunk_target_chars,
        hard_max_chars=config.chunk_hard_max_chars,
    )
    if not chunks:
        st.session_state.parse_result = parse_result
        st.error("这份 PDF 没有生成可检索的文本片段。请确认文件包含可读取的保单文字。")
        return

    progress.progress(75, text="生成 embeddings 并建立临时索引")
    try:
        embedder = OpenAIEmbedder(api_key=config.openai_api_key, model=config.embedding_model)
        policy_index = build_index(chunks, embedder)
        policy_retriever = HybridRetriever(
            chunks=chunks,
            vector_index=policy_index,
            embedder=embedder,
            rrf_k=config.rrf_k,
            retrieval_mode=config.retrieval_mode,
        )
    except Exception as exc:
        st.session_state.parse_result = parse_result
        st.session_state.policy_chunks = chunks
        st.error(f"建立检索索引时出错：{exc}")
        return

    st.session_state.parse_result = parse_result
    st.session_state.policy_chunks = chunks
    st.session_state.policy_retriever = policy_retriever
    st.session_state.builtin_retriever = None
    st.session_state.builtin_index_attempted = False
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

    if not st.session_state.policy_retriever:
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
                    maybe_build_builtin_background_index(question, config)
                    chain = RagChain(
                        config=config,
                        policy_retriever=st.session_state.policy_retriever,
                        builtin_retriever=st.session_state.builtin_retriever,
                    )
                    payload = chain.answer(question)
                except Exception as exc:
                    payload = AnswerPayload(answer="处理问题时出错。", warnings=(str(exc),))
                st.write(payload.answer)
                render_citations(payload)
        st.session_state.messages.append(
            {"role": "assistant", "content": payload.answer, "payload": payload}
        )


if __name__ == "__main__":
    main()
