from openai import OpenAI

from insurance_rag.config import AppConfig
from insurance_rag.models import AnswerPayload, Citation, DocumentChunk
from insurance_rag.retriever import InMemoryVectorIndex, OpenAIEmbedder


REFUSAL_ANSWER = "这份保单中没有找到足够明确的依据。你可以换一种问法，或确认上传的保单是否完整。"

TERM_KEYWORDS = ("什么是", "是什么意思", "如何理解", "定义", "概念")


def should_use_builtin_context(question: str, policy_result_count: int) -> bool:
    return policy_result_count > 0 and any(keyword in question for keyword in TERM_KEYWORDS)


def build_citation(chunk: DocumentChunk, max_chars: int = 180) -> Citation:
    excerpt = " ".join(chunk.text.split())
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "..."
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
        warnings: list[str] = []
        query_embeddings = self.embedder.embed_texts([question])
        if not query_embeddings:
            return AnswerPayload(
                answer=REFUSAL_ANSWER,
                warnings=("向量生成失败：没有返回可用于检索的问题向量。",),
            )
        query_embedding = query_embeddings[0]

        try:
            policy_results = self.policy_index.search(
                query_embedding,
                top_k=self.config.policy_top_k,
            )
        except Exception as error:
            return AnswerPayload(
                answer=REFUSAL_ANSWER,
                warnings=(f"保单检索失败：{error}",),
            )
        policy_chunks = [result.chunk for result in policy_results]
        if not policy_chunks:
            return AnswerPayload(answer=REFUSAL_ANSWER)

        builtin_chunks: list[DocumentChunk] = []
        if self.builtin_index and should_use_builtin_context(question, len(policy_chunks)):
            try:
                builtin_results = self.builtin_index.search(
                    query_embedding,
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
        return AnswerPayload(
            answer=answer,
            policy_citations=tuple(build_citation(chunk) for chunk in policy_chunks),
            builtin_citations=tuple(build_citation(chunk) for chunk in builtin_chunks),
            warnings=tuple(warnings),
        )
