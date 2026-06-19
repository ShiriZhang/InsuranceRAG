from dataclasses import replace

from insurance_rag.config import AppConfig
from insurance_rag.hybrid_retriever import HybridSearchResult
from insurance_rag.models import DocumentChunk, GuardStatus, QueryRewriteResult
from insurance_rag.rag_chain import (
    REFUSAL_ANSWER,
    RagChain,
    build_citation,
    build_messages,
    should_use_builtin_context,
)


def make_chunk(
    source_type: str = "user_policy",
    *,
    chunk_id: str = "c1",
    quality_notes: tuple[str, ...] = ("扫描件文字可能不完整",),
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text="等待期为九十日。",
        page_number=4,
        section_title="等待期",
        source_type=source_type,
        source_name="user.pdf" if source_type == "user_policy" else "内置条款.pdf",
        extraction_method="text",
        quality_notes=quality_notes,
    )


class FakeHybridRetriever:
    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        error: Exception | None = None,
        matched_terms: tuple[str, ...] = ("等待期",),
    ) -> None:
        self.chunks = chunks or []
        self.error = error
        self.matched_terms = matched_terms
        self.calls: list[tuple[QueryRewriteResult, int]] = []

    def search(
        self,
        rewrite: QueryRewriteResult,
        top_k: int,
    ) -> list[HybridSearchResult]:
        self.calls.append((rewrite, top_k))
        if self.error:
            raise self.error
        return [
            HybridSearchResult(
                chunk=chunk,
                final_score=0.05,
                vector_score=0.8,
                bm25_score=1.2,
                matched_terms=self.matched_terms,
            )
            for chunk in self.chunks
        ]


class FakeChatClient:
    def __init__(self, answer: str = "等待期为九十日。") -> None:
        self.calls: list[dict[str, object]] = []
        self.chat = self
        self.completions = self
        self.answer = answer

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class Message:
            content = self.answer

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


def make_chain(
    monkeypatch,
    *,
    policy_retriever: FakeHybridRetriever | None = None,
    builtin_retriever: FakeHybridRetriever | None = None,
    chat_client: FakeChatClient | None = None,
) -> tuple[RagChain, FakeChatClient]:
    client = chat_client or FakeChatClient()
    monkeypatch.setattr("insurance_rag.rag_chain.OpenAI", lambda api_key: client)
    chain = RagChain(
        config=AppConfig(openai_api_key="test-key", policy_top_k=2, builtin_top_k=1),
        policy_retriever=policy_retriever or FakeHybridRetriever([make_chunk()]),
        builtin_retriever=builtin_retriever,
    )
    return chain, client


def test_should_use_builtin_context_for_term_question():
    assert should_use_builtin_context("什么是等待期？", policy_result_count=2) is True


def test_should_not_use_builtin_context_for_specific_policy_question():
    assert should_use_builtin_context("这份保单等待期是多少？", policy_result_count=3) is False


def test_should_not_use_builtin_context_when_policy_has_no_results():
    assert should_use_builtin_context("什么是等待期？", policy_result_count=0) is False


def test_build_citation_uses_chunk_metadata():
    citation = build_citation(make_chunk())

    assert citation.page_number == 4
    assert citation.section_title == "等待期"
    assert citation.excerpt == "等待期为九十日。"
    assert citation.quality_notes == ("扫描件文字可能不完整",)


def test_build_citation_normalizes_whitespace_and_truncates():
    chunk = make_chunk()
    long_text = " 等待期\n\n为\t九十日。" + "补充说明" * 80
    chunk = DocumentChunk(
        chunk_id=chunk.chunk_id,
        text=long_text,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        source_type=chunk.source_type,
        source_name=chunk.source_name,
        extraction_method=chunk.extraction_method,
    )

    citation = build_citation(chunk, max_chars=11)

    assert citation.excerpt == "等待期 为 九十日。补..."


def test_build_messages_include_no_claim_decision_rule():
    messages = build_messages("等待期是多少？", [make_chunk()], [])
    joined = "\n".join(message["content"] for message in messages)

    assert "不得做最终理赔判断" in joined
    assert "用户保单资料" in joined


def test_refusal_answer_is_evidence_limited():
    assert "没有找到足够明确的依据" in REFUSAL_ANSWER


def test_answer_policy_first_happy_path_includes_policy_and_builtin_citations(monkeypatch):
    policy_chunk = make_chunk()
    builtin_chunk = make_chunk(source_type="builtin")
    chain, client = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([policy_chunk]),
        builtin_retriever=FakeHybridRetriever([builtin_chunk]),
    )

    payload = chain.answer("什么是等待期？")

    assert payload.answer == "等待期为九十日。"
    assert payload.policy_citations[0].source_name == "user.pdf"
    assert payload.builtin_citations[0].source_name == "内置条款.pdf"
    assert len(client.calls) == 1
    assert "用户保单资料" in client.calls[0]["messages"][1]["content"]
    assert "内置资料库背景" in client.calls[0]["messages"][1]["content"]


def test_answer_refuses_when_policy_search_returns_no_results_without_chat(monkeypatch):
    chain, client = make_chain(monkeypatch, policy_retriever=FakeHybridRetriever([]))

    payload = chain.answer("什么是等待期？")

    assert payload.answer == REFUSAL_ANSWER
    assert payload.policy_citations == ()
    assert payload.builtin_citations == ()
    assert client.calls == []


def test_answer_refuses_when_policy_retriever_errors_without_chat(monkeypatch):
    chain, client = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever(error=ValueError("dimension mismatch")),
    )

    payload = chain.answer("等待期是多少？")

    assert payload.answer == REFUSAL_ANSWER
    assert any("保单检索失败" in warning for warning in payload.warnings)
    assert client.calls == []


def test_answer_degrades_to_policy_only_when_builtin_retriever_errors(monkeypatch):
    chain, client = make_chain(
        monkeypatch,
        builtin_retriever=FakeHybridRetriever(error=ValueError("dimension mismatch")),
    )

    payload = chain.answer("什么是等待期？")

    assert payload.answer == "等待期为九十日。"
    assert len(payload.policy_citations) == 1
    assert payload.builtin_citations == ()
    assert any("内置资料库检索失败" in warning for warning in payload.warnings)
    assert len(client.calls) == 1
    assert "内置资料库背景：无" in client.calls[0]["messages"][1]["content"]


def test_answer_reranks_policy_candidates_before_prompt(monkeypatch):
    period_chunk = DocumentChunk(
        chunk_id="period",
        text="保险期间为90天。",
        page_number=1,
        section_title="保险期间",
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
        heading_confidence="high",
    )
    waiting_chunk = DocumentChunk(
        chunk_id="waiting",
        text="等待期为90天。",
        page_number=2,
        section_title="等待期",
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
        heading_confidence="high",
    )
    policy_retriever = FakeHybridRetriever([period_chunk, waiting_chunk])
    chain, client = make_chain(
        monkeypatch,
        policy_retriever=policy_retriever,
        chat_client=FakeChatClient(answer="等待期为90天。"),
    )

    payload = chain.answer("等待期是多久？")
    prompt = client.calls[0]["messages"][1]["content"]

    assert policy_retriever.calls[0][1] == chain.config.rerank_top_n
    assert prompt.find("等待期为90天。") < prompt.find("保险期间为90天。")
    assert payload.policy_citations[0].section_title == "等待期"
    assert payload.retrieval_explanations[0].rerank_score is not None
    assert "title_intent_match" in payload.retrieval_explanations[0].rerank_reasons


def test_answer_preserves_policy_candidate_order_when_rerank_disabled(monkeypatch):
    first_chunk = DocumentChunk(
        chunk_id="first",
        text="第一段候选。",
        page_number=1,
        section_title="第一段",
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
        heading_confidence="high",
    )
    second_chunk = DocumentChunk(
        chunk_id="second",
        text="第二段候选。",
        page_number=2,
        section_title="第二段",
        source_type="user_policy",
        source_name="user.pdf",
        extraction_method="text",
        heading_confidence="high",
    )
    policy_retriever = FakeHybridRetriever([first_chunk, second_chunk])
    chain, client = make_chain(monkeypatch, policy_retriever=policy_retriever)
    chain.config = replace(chain.config, rerank_enabled=False)

    chain.answer("等待期是多久？")
    prompt = client.calls[0]["messages"][1]["content"]

    assert policy_retriever.calls[0][1] == chain.config.policy_top_k
    assert prompt.find("第一段候选。") < prompt.find("第二段候选。")


def test_answer_calls_query_rewriter_and_hybrid_retriever(monkeypatch):
    policy_retriever = FakeHybridRetriever(
        [make_chunk(quality_notes=())],
        matched_terms=("保险责任",),
    )
    chain, _ = make_chain(monkeypatch, policy_retriever=policy_retriever)

    payload = chain.answer("这个赔不赔？")

    rewrite, top_k = policy_retriever.calls[0]
    assert top_k == chain.config.rerank_top_n
    assert rewrite.original_query == "这个赔不赔？"
    assert "保险责任" in rewrite.expanded_queries
    assert payload.retrieval_explanations[0].matched_terms == ("保险责任",)


def test_answer_guard_block_replaces_model_answer(monkeypatch):
    chain, _ = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([make_chunk(quality_notes=())]),
        chat_client=FakeChatClient(answer="这种情况一定赔。"),
    )

    payload = chain.answer("这个赔不赔？")

    assert "不能直接给出该结论" in payload.answer
    assert payload.guard_result is not None
    assert payload.guard_result.status is GuardStatus.BLOCK
    assert any("最终理赔判断" in warning for warning in payload.warnings)


def test_answer_guard_warn_preserves_model_answer(monkeypatch):
    chain, _ = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([make_chunk(quality_notes=())]),
        chat_client=FakeChatClient(answer="根据条款，等待期为九十日。"),
    )

    payload = chain.answer("等待期是多少？")

    assert payload.answer == "根据条款，等待期为九十日。"
    assert payload.guard_result is not None
    assert payload.guard_result.status is GuardStatus.WARN
    assert any("用户保单引用较少" in warning for warning in payload.warnings)


def test_answer_blocks_model_answer_when_guard_runtime_fails(monkeypatch):
    def raise_guard_error(**_kwargs):
        raise RuntimeError("guard crashed")

    monkeypatch.setattr("insurance_rag.rag_chain.check_answer", raise_guard_error)
    chain, _ = make_chain(
        monkeypatch,
        policy_retriever=FakeHybridRetriever([make_chunk(quality_notes=())]),
        chat_client=FakeChatClient(answer="根据引用，等待期为九十日。"),
    )

    payload = chain.answer("等待期是多久？")

    assert "不能直接给出该结论" in payload.answer
    assert payload.guard_result is None
    assert any("自检未完成" in warning for warning in payload.warnings)
