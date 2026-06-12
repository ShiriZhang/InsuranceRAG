import re

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
_DIRECT_FINAL_CLAIM_PATTERNS = (
    re.compile(r"(这种情况|该情况|本次|这次|此次|该费用|这笔费用|该事故|本次事故|该疾病|这种疾病).{0,8}(不赔|可以理赔|可理赔|不予赔付|可以报销|可报销|不可报销|不能报销)"),
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
_POLICY_SUPPORT_TERMS = (
    "等待期",
    "责任免除",
    "除外责任",
    "免责条款",
    "保险责任",
    "赔付条件",
    "保险金额",
    "保险期间",
    "生效日",
    "免赔额",
    "住院医疗费用",
    "住院医疗",
    "重大疾病",
    "豁免保险费",
    "合同约定",
    "给付",
    "报销",
    "赔付",
    "理赔",
)
_NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"(?:\d+|[零〇一二三四五六七八九十百千万两]+)(?:日|天|年|月|个月|元|万元|%)"
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
    if policy_citations and _contains_any(answer, _SPECIFIC_POLICY_FACT_TERMS):
        unsupported_terms = _unsupported_policy_fact_terms(
            answer,
            policy_citations,
            retrieval_explanations,
        )
        if unsupported_terms:
            terms = "、".join(unsupported_terms)
            return AnswerGuardResult(
                status=GuardStatus.BLOCK,
                block_reason=f"回答中的保单事实未被用户保单引用支持：{terms}。",
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
    for pattern in _DIRECT_FINAL_CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            if not _has_safe_final_claim_prefix(text, match.start()):
                return True
    return False


def _has_safe_final_claim_prefix(text: str, term_index: int) -> bool:
    prefix_start = max(0, term_index - _FINAL_CLAIM_PREFIX_WINDOW)
    prefix = text[prefix_start:term_index]
    return any(safe_prefix in prefix for safe_prefix in _SAFE_FINAL_CLAIM_PREFIXES)


def _unsupported_policy_fact_terms(
    answer: str,
    policy_citations: tuple[Citation, ...],
    retrieval_explanations: tuple[RetrievalExplanation, ...],
) -> tuple[str, ...]:
    terms = _extract_policy_support_terms(answer)
    if not terms:
        return ("具体保单事实",)

    evidence = _policy_evidence_text(policy_citations, retrieval_explanations)
    return tuple(term for term in terms if term not in evidence)


def _extract_policy_support_terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for term in _POLICY_SUPPORT_TERMS:
        if term in text:
            terms.append(term)
    terms.extend(_NUMBER_WITH_UNIT_PATTERN.findall(text))
    return _dedupe(terms)


def _policy_evidence_text(
    policy_citations: tuple[Citation, ...],
    retrieval_explanations: tuple[RetrievalExplanation, ...],
) -> str:
    parts: list[str] = []
    for citation in policy_citations:
        parts.extend((citation.section_title, citation.excerpt))
    for explanation in retrieval_explanations:
        if explanation.source_type in {"policy", "user_policy"}:
            parts.append(explanation.section_title)
            parts.extend(explanation.matched_terms)
    return "\n".join(part for part in parts if part)


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return tuple(deduped)
