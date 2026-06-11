import pytest

from insurance_rag.query_rewriter import rewrite_query


def test_rewrites_claim_question_to_coverage_and_exclusion_terms():
    result = rewrite_query("  住院医疗能不能赔？  ")

    assert result.original_query == "  住院医疗能不能赔？  "
    assert result.expanded_queries == (
        "住院医疗能不能赔？",
        "保险责任",
        "责任免除",
        "赔付条件",
        "除外责任",
    )
    assert result.detected_intents == ("claim_condition",)
    assert result.used_llm is False
    assert result.warnings == ()


def test_rewrites_waiting_period_question():
    result = rewrite_query("等待期过后多久生效")

    assert result.expanded_queries == (
        "等待期过后多久生效",
        "等待期",
        "生效日",
        "保险期间",
    )
    assert result.detected_intents == ("waiting_period",)


def test_rewrites_exclusion_question():
    result = rewrite_query("什么是免责条款，哪些情况不赔？")

    assert result.expanded_queries == (
        "什么是免责条款，哪些情况不赔？",
        "责任免除",
        "除外责任",
        "免责条款",
        "释义",
        "定义",
        "术语解释",
    )
    assert result.detected_intents == ("exclusion", "definition")


def test_rewrites_waiver_question():
    result = rewrite_query("如何理解投保人豁免保险费")

    assert result.expanded_queries == (
        "如何理解投保人豁免保险费",
        "豁免保险费",
        "投保人豁免",
        "被保险人豁免",
        "释义",
        "定义",
        "术语解释",
    )
    assert result.detected_intents == ("waiver", "definition")


def test_plain_question_keeps_original_without_duplicates():
    result = rewrite_query("  普通问题  ")

    assert result.original_query == "  普通问题  "
    assert result.expanded_queries == ("普通问题",)
    assert result.detected_intents == ()
    assert result.warnings == ()


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_empty_question_returns_warning(question):
    result = rewrite_query(question)

    assert result.original_query == question
    assert result.expanded_queries == ()
    assert result.detected_intents == ()
    assert result.used_llm is False
    assert result.warnings == ("问题为空，无法生成检索查询。",)


def test_use_llm_true_falls_back_to_rule_based_warning():
    result = rewrite_query("等待期什么时候生效", use_llm=True)

    assert result.expanded_queries == (
        "等待期什么时候生效",
        "等待期",
        "生效日",
        "保险期间",
    )
    assert result.detected_intents == ("waiting_period",)
    assert result.used_llm is False
    assert result.warnings == ("LLM 查询改写尚未启用，已使用规则改写。",)
