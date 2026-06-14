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
_FACT_CLAIM_VERBS = (
    "是",
    "为",
    "包括",
    "包含",
    "承担",
    "属于",
    "适用",
    "给付",
    "赔付",
    "报销",
    "写明",
    "约定",
    "载明",
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
    if _contains_final_claim_decision(answer):
        return AnswerGuardResult(
            status=GuardStatus.BLOCK,
            block_reason="回答包含最终理赔判断，需改为基于条款的条件性说明。",
        )

    if builtin_citations and _contains_any(answer, _SOURCE_CONFUSING_TERMS):
        unsupported_terms = _unsupported_policy_fact_terms(
            answer,
            policy_citations,
            retrieval_explanations,
        )
        if not policy_citations or unsupported_terms:
            return AnswerGuardResult(
                status=GuardStatus.BLOCK,
                block_reason="回答可能将内置资料库内容表述为用户保单事实，需改为仅引用用户保单原文。",
            )

    if not policy_citations and _contains_policy_fact_claim(answer):
        return AnswerGuardResult(
            status=GuardStatus.BLOCK,
            block_reason="回答包含具体保单事实，但没有用户保单引用。",
        )
    if policy_citations and _contains_policy_fact_claim(answer):
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

    warnings: list[str] = []

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


def _contains_policy_fact_claim(text: str) -> bool:
    if _contains_any(text, _SPECIFIC_POLICY_FACT_TERMS):
        return True

    support_terms = _extract_policy_support_terms(text)
    if not support_terms:
        return False
    if any(_NUMBER_WITH_UNIT_PATTERN.fullmatch(term) for term in support_terms):
        return True

    compact_text = _compact(text)
    for term in _POLICY_SUPPORT_TERMS:
        compact_term = _compact(term)
        if compact_term not in compact_text:
            continue
        term_index = compact_text.find(compact_term)
        window = compact_text[term_index : term_index + len(compact_term) + 12]
        if any(verb in window for verb in _FACT_CLAIM_VERBS):
            return True
    return False


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
    evidence_segments = _policy_evidence_segments(policy_citations, retrieval_explanations)
    numeric_claims = _extract_numeric_fact_claims(answer)
    unsupported: list[str] = []
    supported_numeric_terms: set[str] = set()
    supported_numeric_values: set[str] = set()

    for policy_term, number_term in numeric_claims:
        if _numeric_fact_supported_by_evidence(policy_term, number_term, evidence_segments):
            supported_numeric_terms.add(policy_term)
            supported_numeric_values.add(number_term)
            continue
        unsupported.append(f"{policy_term}{number_term}")

    for term in terms:
        if term in supported_numeric_terms or term in supported_numeric_values:
            continue
        if not _term_supported_by_evidence(term, evidence):
            unsupported.append(term)

    return _dedupe(unsupported)


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
    return "\n".join(_policy_evidence_segments(policy_citations, retrieval_explanations))


def _policy_evidence_segments(
    policy_citations: tuple[Citation, ...],
    retrieval_explanations: tuple[RetrievalExplanation, ...],
) -> tuple[str, ...]:
    parts: list[str] = []
    for citation in policy_citations:
        parts.append(f"{citation.section_title}\n{citation.excerpt}")
    for explanation in retrieval_explanations:
        if explanation.source_type in {"policy", "user_policy"}:
            parts.append(
                "\n".join((explanation.section_title, *explanation.matched_terms))
            )
    return tuple(part for part in parts if part)


def _extract_numeric_fact_claims(text: str) -> tuple[tuple[str, str], ...]:
    claims: list[tuple[str, str]] = []
    for fragment in _split_fact_fragments(text):
        compact_fragment = _compact(fragment)
        for policy_term in _POLICY_SUPPORT_TERMS:
            compact_policy_term = _compact(policy_term)
            term_index = compact_fragment.find(compact_policy_term)
            if term_index == -1:
                continue
            for match in _NUMBER_WITH_UNIT_PATTERN.finditer(compact_fragment):
                if abs(match.start() - term_index) <= len(compact_policy_term) + 18:
                    claims.append((policy_term, match.group(0)))
    return _dedupe_pairs(claims)


def _numeric_fact_supported_by_evidence(
    policy_term: str,
    number_term: str,
    evidence_segments: tuple[str, ...],
) -> bool:
    normalized_number = _normalize_number_units(_compact(number_term))
    for segment in evidence_segments:
        for fragment in _split_fact_fragments(segment):
            compact_fragment = _compact(fragment)
            if _compact(policy_term) not in compact_fragment:
                continue
            normalized_fragment = _normalize_number_units(compact_fragment)
            if normalized_number and normalized_number in normalized_fragment:
                return True
    return False


def _split_fact_fragments(text: str) -> tuple[str, ...]:
    return tuple(fragment for fragment in re.split(r"[。；;，,\n]+", text) if fragment.strip())


def _term_supported_by_evidence(term: str, evidence: str) -> bool:
    compact_term = _compact(term)
    compact_evidence = _compact(evidence)
    if compact_term and compact_term in compact_evidence:
        return True

    normalized_term = _normalize_number_units(compact_term)
    normalized_evidence = _normalize_number_units(compact_evidence)
    return bool(normalized_term and normalized_term in normalized_evidence)


def _compact(text: str) -> str:
    return "".join(text.split())


def _normalize_number_units(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        number = match.group("number")
        unit = match.group("unit")
        if not number.isdigit():
            converted = _chinese_number_to_int(number)
            if converted is None:
                return match.group(0)
            number = str(converted)
        if unit == "日":
            unit = "天"
        return f"{number}{unit}"

    return re.sub(
        r"(?P<number>\d+|[零〇一二三四五六七八九十百千万两]+)(?P<unit>日|天|年|月|个月|元|万元|%)",
        replace,
        text,
    )


def _chinese_number_to_int(text: str) -> int | None:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text and all(char in digits or char == "十" for char in text):
        left, _, right = text.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if all(char in digits for char in text):
        value = 0
        for char in text:
            value = value * 10 + digits[char]
        return value
    return None


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return tuple(deduped)


def _dedupe_pairs(items: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return tuple(deduped)
