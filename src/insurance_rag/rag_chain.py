from typing import Protocol

from openai import OpenAI

from insurance_rag.answer_guard import BLOCKED_ANSWER, check_answer
from insurance_rag.config import AppConfig
from insurance_rag.hybrid_retriever import HybridSearchResult
from insurance_rag.models import (
    AnswerPayload,
    Citation,
    DocumentChunk,
    GuardStatus,
    QueryRewriteResult,
)
from insurance_rag.query_rewriter import rewrite_query
from insurance_rag.rule_reranker import rerank_results


REFUSAL_ANSWER = "这份保单中没有找到足够明确的依据。你可以换一种问法，或确认上传的保单是否完整。"

TERM_KEYWORDS = ("什么是", "是什么意思", "如何理解", "定义", "概念")


def should_use_builtin_context(question: str, policy_result_count: int) -> bool:
    return policy_result_count > 0 and any(keyword in question for keyword in TERM_KEYWORDS)


def build_citation(chunk: DocumentChunk, max_chars: int = 180) -> Citation:
    excerpt = " ".join(chunk.authoritative_text.split())
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "..."
    return Citation(
        source_type=chunk.source_type,
        source_name=chunk.source_name,
        page_number=chunk.authoritative_page_number,
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
            f"[{index}] {chunk.source_name}｜{page}｜{chunk.section_title}\n{chunk.authoritative_text}"
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


class SearchRetriever(Protocol):
    def search(
        self,
        rewrite: QueryRewriteResult,
        top_k: int,
    ) -> list[HybridSearchResult]: ...


class RagChain:
    def __init__(
        self,
        config: AppConfig,
        policy_retriever: SearchRetriever,
        builtin_retriever: SearchRetriever | None = None,
    ) -> None:
        if not config.openai_api_key:
            raise ValueError("缺少 OPENAI_API_KEY。")
        self.config = config
        self.policy_retriever = policy_retriever
        self.builtin_retriever = builtin_retriever
        self.client = OpenAI(api_key=config.openai_api_key)

    def answer(self, question: str) -> AnswerPayload:
        warnings: list[str] = []
        rewrite = rewrite_query(question, use_llm=self.config.query_rewrite_llm)
        warnings.extend(rewrite.warnings)

        policy_search_top_k = (
            max(self.config.policy_top_k, self.config.rerank_top_n)
            if self.config.rerank_enabled
            else self.config.policy_top_k
        )
        try:
            policy_results = self.policy_retriever.search(
                rewrite,
                top_k=policy_search_top_k,
            )
        except Exception as error:
            return AnswerPayload(
                answer=REFUSAL_ANSWER,
                warnings=tuple(warnings + [f"保单检索失败：{error}"]),
            )
        if self.config.rerank_enabled:
            try:
                policy_results = rerank_results(
                    question=question,
                    rewrite=rewrite,
                    candidates=policy_results,
                    top_k=self.config.policy_top_k,
                )
            except Exception as error:
                warnings.append(f"规则重排未完成，已使用原始检索结果：{error}")
                policy_results = policy_results[: self.config.policy_top_k]
        else:
            policy_results = policy_results[: self.config.policy_top_k]
        policy_chunks = [result.chunk for result in policy_results]
        if not policy_chunks:
            return AnswerPayload(answer=REFUSAL_ANSWER, warnings=tuple(warnings))

        builtin_results = []
        builtin_chunks: list[DocumentChunk] = []
        if self.builtin_retriever and should_use_builtin_context(question, len(policy_chunks)):
            try:
                builtin_results = self.builtin_retriever.search(
                    rewrite,
                    top_k=self.config.builtin_top_k,
                )
                builtin_chunks = [result.chunk for result in builtin_results]
            except Exception as error:
                warnings.append(f"内置资料库检索失败，已仅使用用户保单资料回答：{error}")

        messages = build_messages(question, policy_chunks, builtin_chunks)
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content or REFUSAL_ANSWER
        policy_citations = tuple(build_citation(chunk) for chunk in policy_chunks)
        builtin_citations = tuple(build_citation(chunk) for chunk in builtin_chunks)
        retrieval_explanations = tuple(
            result.to_explanation() for result in [*policy_results, *builtin_results]
        )
        guard_result = None
        try:
            guard_result = check_answer(
                question=question,
                answer=answer,
                policy_citations=policy_citations,
                builtin_citations=builtin_citations,
                retrieval_explanations=retrieval_explanations,
            )
        except Exception as error:
            answer = BLOCKED_ANSWER
            warnings.append(f"回答自检未完成：{error}")
        else:
            warnings.extend(guard_result.warnings)
            if guard_result.status is GuardStatus.BLOCK:
                answer = BLOCKED_ANSWER
                if guard_result.block_reason:
                    warnings.append(guard_result.block_reason)
        citation_verification = (
            guard_result.citation_verification if guard_result is not None else None
        )

        return AnswerPayload(
            answer=answer,
            policy_citations=policy_citations,
            builtin_citations=builtin_citations,
            warnings=tuple(warnings),
            retrieval_explanations=retrieval_explanations,
            guard_result=guard_result,
            citation_verification=citation_verification,
        )
