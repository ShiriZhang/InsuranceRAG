from insurance_rag.models import (
    AnswerGuardResult,
    Citation,
    GuardStatus,
    RetrievalExplanation,
)


BLOCKED_ANSWER = "不能直接给出该结论。请先补充或核对用户保单原文后再判断。"

_SPECIFIC_POLICY_FACT_TERMS = (
    "这份保单",
    "你的保单",
    "保单写明",
    "条款显示",
    "合同约定",
)
_FINAL_CLAIM_DECISION_TERMS = (
    "一定赔",
    "肯定赔",
    "必须赔",
    "一定不赔",
    "肯定不赔",
    "不会赔",
    "保险公司必须",
)
_SAFE_FINAL_CLAIM_PREFIXES = (
    "不能直接判断",
    "不能判断",
    "无法判断",
    "不应写",
    "不能写",
    "不要写",
    "不能说",
    "不应说",
    "避免说",
)
_FINAL_CLAIM_PREFIX_WINDOW = 8
_SOURCE_CONFUSING_TERMS = (
    "你的保单",
    "这份保单写明",
)


def check_answer(
    *,
    question: str,
    answer: str,
    policy_citations: tuple[Citation, ...],
    builtin_citations: tuple[Citation, ...],
    retrieval_explanations: tuple[RetrievalExplanation, ...],
) -> AnswerGuardResult:
    if not policy_citations and _contains_any(answer, _SPECIFIC_POLICY_FACT_TERMS):
        return AnswerGuardResult(
            status=GuardStatus.BLOCK,
            block_reason="回答包含具体保单事实，但没有用户保单引用。",
        )

    if _contains_final_claim_decision(answer):
        return AnswerGuardResult(
            status=GuardStatus.BLOCK,
            block_reason="回答包含最终理赔判断，需改为基于条款的条件性说明。",
        )

    warnings: list[str] = []

    if builtin_citations and _contains_any(answer, _SOURCE_CONFUSING_TERMS):
        warnings.append("回答可能将内置资料库内容表述为用户保单，请改为引用用户保单原文。")

    if len(policy_citations) == 1:
        warnings.append("用户保单引用较少，请结合原文继续核对。")

    if retrieval_explanations and retrieval_explanations[0].final_score < 0.01:
        warnings.append("检索分数较低，请谨慎使用该证据并继续核对原文。")

    if any(citation.quality_notes for citation in policy_citations):
        warnings.append("用户保单引用存在OCR或文本提取质量提示，请结合原文继续核对。")

    if builtin_citations:
        warnings.append("内置资料库内容仅用于术语或背景解释，不能替代用户保单。")

    if warnings:
        return AnswerGuardResult(status=GuardStatus.WARN, warnings=tuple(warnings))

    return AnswerGuardResult(status=GuardStatus.PASS)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _contains_final_claim_decision(text: str) -> bool:
    for term in _FINAL_CLAIM_DECISION_TERMS:
        start = 0
        while True:
            index = text.find(term, start)
            if index == -1:
                break
            if not _has_safe_final_claim_prefix(text, index):
                return True
            start = index + len(term)
    return False


def _has_safe_final_claim_prefix(text: str, term_index: int) -> bool:
    prefix_start = max(0, term_index - _FINAL_CLAIM_PREFIX_WINDOW)
    prefix = text[prefix_start:term_index]
    return any(safe_prefix in prefix for safe_prefix in _SAFE_FINAL_CLAIM_PREFIXES)
