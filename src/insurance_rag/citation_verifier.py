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
    "免赔额",
    "赔付比例",
    "保费",
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
_FRAGMENT_SPLIT_RE = re.compile(r"[。！？；，、,;!\?\n]+")
_RELATION_BOUNDARY_RE = re.compile(r"(?:和|及|以及|并且|同时)")
_SOURCE_CONFUSING_FACT_RE = re.compile(
    r"(?:你的保单|这份保单写明|保单写明)\s*"
    r"(?:写明|约定|载明|显示)?\s*"
    r"(?P<term>[\u4e00-\u9fff]{2,12}?)"
    r"(?:是|为|写明|约定|载明|显示)?\s*"
    + _NUMBER_WITH_UNIT_RE.pattern
)
_MEANINGFUL_CLAIM_MIN_LENGTH = 8
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
    facts = []
    unsupported_fact_texts = []
    has_supported_policy_number_fact = False

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
            has_supported_policy_number_fact = True
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

    if (
        not has_supported_policy_number_fact
        and _has_source_confusion(answer, policy_citations, builtin_citations)
    ):
        facts.append(
            VerifiedFact(
                fact_text="内置资料被表述为用户保单事实",
                fact_type="source_confusion",
                status="unsupported",
                severity="block",
                reason="答案将内置资料内容表述为你的保单事实，但没有用户保单引用。",
            )
        )
        return CitationVerificationResult(
            facts=tuple(facts),
            block_reason="答案可能将内置资料误作用户保单事实，请先核对用户保单原文。",
        )

    if not facts and policy_citations and _mentions_policy_term(answer):
        supporting_citation = _find_term_matching_policy_citation(
            answer, policy_citations
        )
        supporting_citation_ids = ()
        if supporting_citation is not None:
            supporting_citation_ids = (_citation_id(supporting_citation),)
        facts.append(
            VerifiedFact(
                fact_text="政策条款概括性表述",
                fact_type="general_policy_reference",
                status="uncertain",
                severity="warn",
                supporting_citation_ids=supporting_citation_ids,
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
    if not builtin_citations:
        return False
    if not any(term in answer for term in SOURCE_CONFUSION_TERMS):
        return False

    mentioned_terms = _mentioned_policy_terms(answer)
    if not mentioned_terms:
        return True

    return not _policy_citations_support_source_confusing_claim(answer, policy_citations)


def _extract_policy_number_facts(text: str) -> tuple[dict[str, str], ...]:
    facts = []
    seen = set()
    for fragment in _split_fragments(text):
        for fact in _extract_known_term_number_facts(fragment):
            key = (fact["term"], fact["normalized_number"])
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)

    if any(term in text for term in SOURCE_CONFUSION_TERMS):
        for fact in _extract_source_confusing_number_facts(text):
            key = (fact["term"], fact["normalized_number"])
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
    return tuple(facts)


def _extract_known_term_number_facts(fragment: str) -> tuple[dict[str, str], ...]:
    facts = []
    term_matches = sorted(
        (
            match
            for term in POLICY_TERMS
            for match in re.finditer(re.escape(term), fragment)
        ),
        key=lambda match: match.start(),
    )
    for index, term_match in enumerate(term_matches):
        next_term_start = (
            term_matches[index + 1].start()
            if index + 1 < len(term_matches)
            else len(fragment)
        )
        relation_span = fragment[term_match.end() : next_term_start]
        boundary_match = _RELATION_BOUNDARY_RE.search(relation_span)
        if boundary_match is not None:
            relation_span = relation_span[: boundary_match.start()]

        number_match = _NUMBER_WITH_UNIT_RE.search(relation_span)
        if number_match is None:
            continue
        number = _number_from_match(number_match)
        if number is None:
            continue
        facts.append(
            {
                "term": term_match.group(0),
                "normalized_number": number["normalized"],
                "display_number": number["display"],
            }
        )
    return tuple(facts)


def _extract_source_confusing_number_facts(text: str) -> tuple[dict[str, str], ...]:
    facts = []
    for match in _SOURCE_CONFUSING_FACT_RE.finditer(text):
        number = _number_from_match(match)
        if number is None:
            continue
        facts.append(
            {
                "term": _normalize_source_confusing_term(match.group("term")),
                "normalized_number": number["normalized"],
                "display_number": number["display"],
            }
        )
    return tuple(facts)


def _normalize_source_confusing_term(term: str) -> str:
    return re.sub(r"^(?:写明|约定|载明|显示)+", "", term)


def _find_supporting_policy_citation(
    term: str,
    normalized_number: str,
    policy_citations: tuple[Citation, ...],
) -> Citation | None:
    for citation in policy_citations:
        for fragment in _citation_fragments(citation):
            citation_facts = _extract_known_term_number_facts(fragment)
            if any(
                fact["term"] == term
                and fact["normalized_number"] == normalized_number
                for fact in citation_facts
            ):
                return citation
    return None


def _extract_numbers(text: str) -> tuple[dict[str, str], ...]:
    numbers = []
    for match in _NUMBER_WITH_UNIT_RE.finditer(text):
        number = _number_from_match(match)
        if number is not None:
            numbers.append(number)
    return tuple(numbers)


def _number_from_match(match: re.Match[str]) -> dict[str, str] | None:
    value = _normalize_number(match.group("number"))
    if value is None:
        return None
    unit = _normalize_unit(match.group("unit"))
    return {
        "normalized": f"{value}{unit}",
        "display": f"{value}{unit}",
    }


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
    return tuple(
        subfragment.strip()
        for fragment in _FRAGMENT_SPLIT_RE.split(text)
        for subfragment in _RELATION_BOUNDARY_RE.split(fragment)
        if subfragment.strip()
    )


def _mentions_policy_term(text: str) -> bool:
    return any(term in text for term in POLICY_TERMS)


def _mentioned_policy_terms(text: str) -> tuple[str, ...]:
    return tuple(term for term in POLICY_TERMS if term in text)


def _find_term_matching_policy_citation(
    answer: str, policy_citations: tuple[Citation, ...]
) -> Citation | None:
    for term in _mentioned_policy_terms(answer):
        for citation in policy_citations:
            if _citation_mentions_term(citation, term):
                return citation
    return None


def _citation_mentions_term(citation: Citation, term: str) -> bool:
    return term in citation.section_title or term in citation.excerpt


def _policy_citations_support_source_confusing_claim(
    answer: str, policy_citations: tuple[Citation, ...]
) -> bool:
    claims = _source_confusing_claims(answer)
    if not claims:
        return False
    citation_texts = [
        _normalize_claim_text(citation.section_title + citation.excerpt)
        for citation in policy_citations
    ]
    return any(
        claim in citation_text
        for claim in claims
        for citation_text in citation_texts
    )


def _source_confusing_claims(answer: str) -> tuple[str, ...]:
    claims = []
    for fragment in _split_fragments(answer):
        claim = fragment
        for prefix in SOURCE_CONFUSION_TERMS:
            claim = claim.replace(prefix, "")
        claim = _normalize_claim_text(claim)
        if len(claim) >= _MEANINGFUL_CLAIM_MIN_LENGTH:
            claims.append(claim)
    return tuple(claims)


def _normalize_claim_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _citation_fragments(citation: Citation) -> tuple[str, ...]:
    title_fragments = _split_fragments(citation.section_title)
    excerpt_fragments = _split_fragments(citation.excerpt)
    combined_fragments = tuple(
        title_fragment + excerpt_fragment
        for title_fragment in title_fragments
        for excerpt_fragment in excerpt_fragments
    )
    return title_fragments + excerpt_fragments + combined_fragments


def _citation_id(citation: Citation) -> str:
    return f"{citation.source_name}:{citation.page_number}:{citation.section_title}"
