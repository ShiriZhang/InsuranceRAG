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
        policy_results = self.policy_index.search(
            query_embedding,
            top_k=self.config.policy_top_k,
        )
        policy_chunks = [result.chunk for result in policy_results]
        if not policy_chunks:
            return AnswerPayload(answer=REFUSAL_ANSWER)

        builtin_chunks: list[DocumentChunk] = []
        if self.builtin_index and should_use_builtin_context(question, len(policy_chunks)):
            builtin_results = self.builtin_index.search(
                query_embedding,
                top_k=self.config.builtin_top_k,
            )
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
