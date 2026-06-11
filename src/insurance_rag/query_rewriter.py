from insurance_rag.models import QueryRewriteResult


EMPTY_QUERY_WARNING = "问题为空，无法生成检索查询。"
LLM_NOT_ENABLED_WARNING = "LLM 查询改写尚未启用，已使用规则改写。"


_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "claim_condition",
        ("赔不赔", "能不能赔", "会不会赔", "是否赔", "理赔吗"),
        ("保险责任", "责任免除", "赔付条件", "除外责任"),
    ),
    (
        "exclusion",
        ("不赔", "哪些情况不赔", "什么不赔", "免责", "除外"),
        ("责任免除", "除外责任", "免责条款"),
    ),
    (
        "waiting_period",
        ("等待期", "等多久", "多久生效", "什么时候生效"),
        ("等待期", "生效日", "保险期间"),
    ),
    (
        "coverage",
        ("保什么", "保障什么", "保障哪些", "保险责任"),
        ("保险责任", "保障范围", "保险金额"),
    ),
    (
        "waiver",
        ("豁免", "豁免保险费", "免交保费"),
        ("豁免保险费", "投保人豁免", "被保险人豁免"),
    ),
    (
        "definition",
        ("什么是", "是什么意思", "定义", "如何理解"),
        ("释义", "定义", "术语解释"),
    ),
)


def rewrite_query(question: str, *, use_llm: bool = False) -> QueryRewriteResult:
    stripped_question = question.strip()
    warnings: list[str] = []

    if use_llm:
        warnings.append(LLM_NOT_ENABLED_WARNING)

    if not stripped_question:
        if not use_llm:
            warnings.append(EMPTY_QUERY_WARNING)
        else:
            warnings.insert(0, EMPTY_QUERY_WARNING)
        return QueryRewriteResult(
            original_query=question,
            expanded_queries=(),
            used_llm=False,
            warnings=tuple(warnings),
        )

    detected_intents: list[str] = []
    expanded_queries: list[str] = [stripped_question]

    for intent, triggers, additions in _RULES:
        if any(trigger in stripped_question for trigger in triggers):
            detected_intents.append(intent)
            expanded_queries.extend(additions)

    return QueryRewriteResult(
        original_query=question,
        expanded_queries=_deduplicate(expanded_queries),
        detected_intents=tuple(detected_intents),
        used_llm=False,
        warnings=tuple(warnings),
    )


def _deduplicate(queries: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduplicated: list[str] = []

    for query in queries:
        if query not in seen:
            seen.add(query)
            deduplicated.append(query)

    return tuple(deduplicated)
