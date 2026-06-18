from dataclasses import replace
import re

from insurance_rag.hybrid_retriever import HybridSearchResult
from insurance_rag.models import QueryRewriteResult


_INTENT_TITLE_MATCHES: dict[str, tuple[str, ...]] = {
    "waiting_period": ("等待期",),
    "exclusion": ("责任免除", "除外责任", "免责条款", "免除责任"),
    "coverage": ("保险责任", "保险金给付", "给付条件"),
    "amount": ("保险金额", "基本保险金额"),
    "period": ("保险期间",),
    "waiver": ("豁免保险费",),
    "definition": ("释义", "疾病定义", "重大疾病定义"),
}

_NEGATIVE_TITLE_MATCHES: dict[str, tuple[str, ...]] = {
    "waiting_period": ("保险期间", "犹豫期", "宽限期"),
    "exclusion": ("保险责任",),
    "coverage": ("责任免除", "除外责任", "免责条款"),
}

_QUESTION_INTENT_TRIGGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("waiting_period", ("等待期", "等多久", "多久生效", "什么时候生效")),
    ("exclusion", ("不赔", "哪些情况不赔", "什么不赔", "免责", "除外")),
    ("coverage", ("保什么", "保障什么", "保障哪些", "保险责任", "赔不赔", "能不能赔")),
    ("amount", ("保险金额", "基本保险金额", "保额", "多少钱")),
    ("period", ("保险期间", "保障期间", "保多久")),
    ("waiver", ("豁免", "豁免保险费", "免交保费")),
    ("definition", ("什么是", "是什么意思", "定义", "释义", "如何理解")),
)

_FACT_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "exclusion": ("不赔", "免除", "除外", "免责"),
    "positive": ("承担", "给付", "赔付", "保障", "保险责任"),
}

_CLAUSE_HEADING_RE = re.compile(r"第[一二三四五六七八九十百千万0-9]+[条章节]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?|[一二三四五六七八九十百千万]+")
_RERANK_WEIGHT = 0.025
_RERANK_SCORE_LIMIT = 3.0


def rerank_results(
    question: str,
    rewrite: QueryRewriteResult,
    candidates: list[HybridSearchResult],
    top_k: int,
) -> list[HybridSearchResult]:
    if top_k <= 0:
        return []

    intents = _detect_intents(question, rewrite)
    question_terms = _query_terms(rewrite, intents)
    question_numbers = set(_NUMBER_RE.findall(question))

    reranked = [
        _rerank_candidate(candidate, intents, question_terms, question_numbers)
        for candidate in candidates
    ]

    return sorted(
        reranked,
        key=lambda candidate: (
            _combined_score(candidate),
            candidate.final_score,
        ),
        reverse=True,
    )[:top_k]


def _combined_score(candidate: HybridSearchResult) -> float:
    rerank_score = candidate.rerank_score if candidate.rerank_score is not None else 0.0
    bounded_score = max(-_RERANK_SCORE_LIMIT, min(_RERANK_SCORE_LIMIT, rerank_score))
    return candidate.final_score + (_RERANK_WEIGHT * bounded_score)


def _rerank_candidate(
    candidate: HybridSearchResult,
    intents: tuple[str, ...],
    question_terms: tuple[str, ...],
    question_numbers: set[str],
) -> HybridSearchResult:
    chunk = candidate.chunk
    title = chunk.section_title or ""
    text = chunk.text or ""
    combined_text = f"{title}\n{text}"

    score = 0.0
    reasons: list[str] = []
    title_intents = _title_match_intents(intents)

    for intent in title_intents:
        if _contains_any(title, _INTENT_TITLE_MATCHES.get(intent, ())):
            score += 2.0
            reasons.append("title_intent_match")
        if _contains_any(title, _NEGATIVE_TITLE_MATCHES.get(intent, ())):
            score -= 1.5
            reasons.append("negative_title_mismatch")

    if "exclusion" in intents and _contains_any(combined_text, _FACT_TYPE_TERMS["exclusion"]):
        score += 1.5
        reasons.append("exclusion_fact_type_match")

    if "coverage" in intents and _contains_any(combined_text, _FACT_TYPE_TERMS["positive"]):
        score += 1.0
        reasons.append("positive_fact_type_match")

    if any(term and term in combined_text for term in question_terms):
        score += 0.5
        reasons.append("exact_term_match")

    if question_numbers and any(number in combined_text for number in question_numbers):
        score += 0.25
        reasons.append("number_match")

    if _CLAUSE_HEADING_RE.search(title) or _CLAUSE_HEADING_RE.search(text[:40]):
        score += 0.25
        reasons.append("clause_heading_match")

    if chunk.heading_confidence == "low":
        score -= 0.5
        reasons.append("low_heading_confidence")

    if _is_directory_like_chunk(text):
        score -= 1.0
        reasons.append("directory_like_chunk")

    return replace(
        candidate,
        rerank_score=score,
        rerank_reasons=_dedupe_preserving_order(reasons),
    )


def _detect_intents(
    question: str, rewrite: QueryRewriteResult
) -> tuple[str, ...]:
    intents = list(rewrite.detected_intents)
    for intent, triggers in _QUESTION_INTENT_TRIGGERS:
        if any(trigger in question for trigger in triggers):
            intents.append(intent)
    return _dedupe_preserving_order(intents)


def _title_match_intents(intents: tuple[str, ...]) -> tuple[str, ...]:
    concrete_intents = tuple(intent for intent in intents if intent != "definition")
    return concrete_intents or intents


def _query_terms(
    rewrite: QueryRewriteResult, intents: tuple[str, ...]
) -> tuple[str, ...]:
    terms: list[str] = []
    for intent in _title_match_intents(intents):
        terms.extend(_INTENT_TITLE_MATCHES.get(intent, ()))
    terms.extend(query for query in rewrite.expanded_queries if len(query) <= 12)
    return _dedupe_preserving_order(terms)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _is_directory_like_chunk(text: str) -> bool:
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(stripped_lines) < 3:
        return False
    heading_like_lines = sum(
        1
        for line in stripped_lines
        if _CLAUSE_HEADING_RE.match(line) or re.match(r"^\d+(?:\.\d+)*\s+", line)
    )
    return heading_like_lines >= max(3, len(stripped_lines) // 2)


def _dedupe_preserving_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped)
