import re

from insurance_rag.models import Citation, CitationVerificationResult, VerifiedFact


POLICY_TERMS = (
    "等待期",
    "责任免除",
    "保险责任",
    "除外责任",
    "保险金额",
    "保险期间",
    "豁免保险费",
    "投保人",
    "被保险人",
    "重大疾病",
)

SOURCE_CONFUSION_TERMS = (
    "你的保单",
    "这份保单写明",
    "保单写明",
)

_NUMBER_WITH_UNIT_RE = re.compile(
    r"(?P<number>\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"
    r"\s*(?P<unit>个月|万元|周岁|日|天|年|月|岁|元|%)"
)
_FRAGMENT_SPLIT_RE = re.compile(r"[。！？；;!\?\n]+")
_CHINESE_DIGITS = {
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
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def verify_answer_facts(
    answer: str,
    policy_citations: tuple[Citation, ...],
    builtin_citations: tuple[Citation, ...],
) -> CitationVerificationResult:
    if _has_source_confusion(answer, policy_citations, builtin_citations):
        fact = VerifiedFact(
            fact_text="内置资料被表述为用户保单事实",
            fact_type="source_confusion",
            status="unsupported",
            severity="block",
            reason="答案将内置资料内容表述为你的保单事实，但没有用户保单引用。",
        )
        return CitationVerificationResult(
            facts=(fact,),
            block_reason="答案可能将内置资料误作用户保单事实，请先核对用户保单原文。",
        )

    facts = []
    unsupported_fact_texts = []

    answer_facts = _extract_policy_number_facts(answer)
    for answer_fact in answer_facts:
        support = _find_supporting_policy_citation(
            answer_fact["term"], answer_fact["normalized_number"], policy_citations
        )
        if support is None:
            fact_text = f"{answer_fact['term']}{answer_fact['display_number']}"
            unsupported_fact_texts.append(fact_text)
            facts.append(
                VerifiedFact(
                    fact_text=fact_text,
                    fact_type="number",
                    status="unsupported",
                    severity="block",
                    reason="未在同一个用户保单引用片段中找到相同条款和数值。",
                )
            )
        else:
            facts.append(
                VerifiedFact(
                    fact_text=f"{answer_fact['term']}{answer_fact['display_number']}",
                    fact_type="number",
                    status="supported",
                    severity="info",
                    supporting_citation_ids=(_citation_id(support),),
                    reason="同一个用户保单引用片段中包含相同条款和数值。",
                )
            )

    if not facts and policy_citations and _mentions_policy_term(answer):
        supporting_citation = policy_citations[0]
        facts.append(
            VerifiedFact(
                fact_text="政策条款概括性表述",
                fact_type="general_policy_reference",
                status="uncertain",
                severity="warn",
                supporting_citation_ids=(_citation_id(supporting_citation),),
                reason="答案提到保单条款但没有可精确核对的数值事实。",
            )
        )

    block_reason = None
    if unsupported_fact_texts:
        block_reason = "未找到用户保单引用支持以下事实：" + "、".join(
            unsupported_fact_texts
        )

    return CitationVerificationResult(facts=tuple(facts), block_reason=block_reason)


def _has_source_confusion(
    answer: str,
    policy_citations: tuple[Citation, ...],
    builtin_citations: tuple[Citation, ...],
) -> bool:
    if policy_citations or not builtin_citations:
        return False
    return any(term in answer for term in SOURCE_CONFUSION_TERMS)


def _extract_policy_number_facts(text: str) -> tuple[dict[str, str], ...]:
    facts = []
    seen = set()
    for fragment in _split_fragments(text):
        terms = [term for term in POLICY_TERMS if term in fragment]
        numbers = list(_extract_numbers(fragment))
        for term in terms:
            for number in numbers:
                key = (term, number["normalized"])
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "term": term,
                        "normalized_number": number["normalized"],
                        "display_number": number["display"],
                    }
                )
    return tuple(facts)


def _find_supporting_policy_citation(
    term: str,
    normalized_number: str,
    policy_citations: tuple[Citation, ...],
) -> Citation | None:
    for citation in policy_citations:
        citation_text = f"{citation.section_title}\n{citation.excerpt}"
        if term not in citation_text:
            continue
        citation_numbers = {
            number["normalized"] for number in _extract_numbers(citation_text)
        }
        if normalized_number in citation_numbers:
            return citation
    return None


def _extract_numbers(text: str) -> tuple[dict[str, str], ...]:
    numbers = []
    for match in _NUMBER_WITH_UNIT_RE.finditer(text):
        value = _normalize_number(match.group("number"))
        if value is None:
            continue
        unit = _normalize_unit(match.group("unit"))
        numbers.append(
            {
                "normalized": f"{value}{unit}",
                "display": f"{value}{unit}",
            }
        )
    return tuple(numbers)


def _normalize_number(raw_number: str) -> str | None:
    if re.fullmatch(r"\d+(?:\.\d+)?", raw_number):
        if raw_number.endswith(".0"):
            return raw_number[:-2]
        return raw_number

    value = _parse_chinese_number(raw_number)
    if value is None:
        return None
    return str(value)


def _parse_chinese_number(text: str) -> int | None:
    if not text:
        return None
    if text in _CHINESE_DIGITS:
        return _CHINESE_DIGITS[text]

    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
            continue
        unit = _CHINESE_UNITS.get(char)
        if unit is None:
            return None
        if unit == 10000:
            section = (section + number) or 1
            total += section * unit
            section = 0
        else:
            section += (number or 1) * unit
        number = 0
    return total + section + number


def _normalize_unit(unit: str) -> str:
    if unit == "日":
        return "天"
    return unit


def _split_fragments(text: str) -> tuple[str, ...]:
    return tuple(fragment for fragment in _FRAGMENT_SPLIT_RE.split(text) if fragment)


def _mentions_policy_term(text: str) -> bool:
    return any(term in text for term in POLICY_TERMS)


def _citation_id(citation: Citation) -> str:
    return f"{citation.source_name}:{citation.page_number}:{citation.section_title}"
