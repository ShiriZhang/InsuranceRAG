from pathlib import Path
import sys

import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def clear_policy_state() -> None:
    st.session_state.parse_result = None
    st.session_state.policy_index = None
    st.session_state.embedder = None
    st.session_state.messages = []
    st.session_state.policy_chunks = ()


def render_citations(payload: AnswerPayload) -> None:
    for warning in payload.warnings:
        st.warning(warning)

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
    )
    if not chunks:
        st.session_state.parse_result = parse_result
        st.error("这份 PDF 没有生成可检索的文本片段。请确认文件包含可读取的保单文字。")
        return

    progress.progress(75, text="生成 embeddings 并建立临时索引")
    try:
        embedder = OpenAIEmbedder(api_key=config.openai_api_key, model=config.embedding_model)
        policy_index = build_index(chunks, embedder)
    except Exception as exc:
        st.session_state.parse_result = parse_result
        st.session_state.policy_chunks = chunks
        st.error(f"建立检索索引时出错：{exc}")
        return

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
                    payload = AnswerPayload(answer="处理问题时出错。", warnings=(str(exc),))
                st.write(payload.answer)
                render_citations(payload)
        st.session_state.messages.append(
            {"role": "assistant", "content": payload.answer, "payload": payload}
        )


if __name__ == "__main__":
    main()
