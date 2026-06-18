from insurance_rag.clause_parser import parse_clause_metadata


def test_parses_chinese_article_number_and_heading():
    metadata = parse_clause_metadata("第六条 等待期\n等待期为90天。")

    assert metadata.clause_id == "第六条"
    assert metadata.heading_text == "第六条 等待期"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "high"
    assert metadata.heading_source == "line_pattern"


def test_parses_spaced_arabic_article_number_and_heading():
    metadata = parse_clause_metadata("第 10 条 责任免除\n因酒后驾驶导致的事故不承担责任。")

    assert metadata.clause_id == "第10条"
    assert metadata.section_title == "责任免除"
    assert metadata.heading_confidence == "high"


def test_parses_decimal_clause_number_and_heading():
    metadata = parse_clause_metadata("2.3 保险责任\n本合同承担重大疾病保险责任。")

    assert metadata.clause_id == "2.3"
    assert metadata.section_title == "保险责任"
    assert metadata.heading_confidence == "high"


def test_parses_standalone_known_heading_as_medium_confidence():
    metadata = parse_clause_metadata("保险金额\n基本保险金额以保险单载明为准。")

    assert metadata.clause_id is None
    assert metadata.section_title == "保险金额"
    assert metadata.heading_confidence == "medium"
    assert metadata.heading_source == "known_title"


def test_parses_standalone_known_heading_with_colon_suffix_as_medium_confidence():
    metadata = parse_clause_metadata("等待期：90天")

    assert metadata.clause_id is None
    assert metadata.heading_text == "等待期：90天"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "medium"
    assert metadata.heading_source == "known_title"


def test_parses_standalone_known_heading_with_dash_suffix_as_medium_confidence():
    metadata = parse_clause_metadata("等待期-90天")

    assert metadata.clause_id is None
    assert metadata.heading_text == "等待期-90天"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "medium"
    assert metadata.heading_source == "known_title"


def test_parses_standalone_known_heading_with_parenthesized_suffix_as_medium_confidence():
    metadata = parse_clause_metadata("保险责任（一）")

    assert metadata.clause_id is None
    assert metadata.heading_text == "保险责任（一）"
    assert metadata.section_title == "保险责任"
    assert metadata.heading_confidence == "medium"
    assert metadata.heading_source == "known_title"


def test_standalone_known_title_sentence_like_mention_falls_back_to_current_title():
    metadata = parse_clause_metadata("等待期为90天", current_title="未识别条款标题")

    assert metadata.clause_id is None
    assert metadata.heading_text is None
    assert metadata.section_title == "未识别条款标题"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"


def test_directory_like_line_is_not_high_confidence():
    metadata = parse_clause_metadata("2.3 保险责任 ........ 5")

    assert metadata.clause_id is None
    assert metadata.heading_text is None
    assert metadata.section_title == "未识别条款标题"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"


def test_directory_like_line_before_real_heading_chooses_real_heading():
    metadata = parse_clause_metadata("2.3 保险责任 ........ 5\n第六条 等待期\n等待期为90天。")

    assert metadata.clause_id == "第六条"
    assert metadata.heading_text == "第六条 等待期"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "high"
    assert metadata.heading_source == "line_pattern"


def test_bare_page_number_directory_line_falls_back():
    metadata = parse_clause_metadata("2.3 保险责任 5")

    assert metadata.clause_id is None
    assert metadata.heading_text is None
    assert metadata.section_title == "未识别条款标题"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"


def test_bare_page_number_directory_variants_fall_back_to_current_title():
    for line in ("一、保险责任 5", "第六条 保险责任 5", "（一）保险责任 5", "1. 保险责任 5"):
        metadata = parse_clause_metadata(line, current_title="保险期间")

        assert metadata.clause_id is None
        assert metadata.heading_text is None
        assert metadata.section_title == "保险期间"
        assert metadata.heading_confidence == "low"
        assert metadata.heading_source == "fallback"


def test_bare_page_number_directory_before_real_heading_chooses_real_heading():
    metadata = parse_clause_metadata("第六条 保险责任 5\n第七条 等待期\n等待期为90天。")

    assert metadata.clause_id == "第七条"
    assert metadata.heading_text == "第七条 等待期"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "high"
    assert metadata.heading_source == "line_pattern"


def test_parses_no_space_article_heading():
    metadata = parse_clause_metadata("第六条等待期\n等待期为90天。")

    assert metadata.clause_id == "第六条"
    assert metadata.heading_text == "第六条等待期"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "high"
    assert metadata.heading_source == "line_pattern"


def test_parses_colon_article_heading():
    metadata = parse_clause_metadata("第六条：等待期\n等待期为90天。")

    assert metadata.clause_id == "第六条"
    assert metadata.heading_text == "第六条：等待期"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "high"
    assert metadata.heading_source == "line_pattern"


def test_parses_spaced_arabic_article_with_colon_heading():
    metadata = parse_clause_metadata("第 10 条：责任免除\n因酒后驾驶导致的事故不承担责任。")

    assert metadata.clause_id == "第10条"
    assert metadata.heading_text == "第 10 条：责任免除"
    assert metadata.section_title == "责任免除"
    assert metadata.heading_confidence == "high"
    assert metadata.heading_source == "line_pattern"


def test_longer_known_title_wins_over_shorter_substring():
    metadata = parse_clause_metadata("2.4 基本保险金额\n基本保险金额以保险单载明为准。")

    assert metadata.clause_id == "2.4"
    assert metadata.section_title == "基本保险金额"
    assert metadata.heading_confidence == "high"


def test_numbered_heading_accepts_known_title_prefix_with_heading_like_suffix():
    metadata = parse_clause_metadata("第六条 等待期：90天")

    assert metadata.clause_id == "第六条"
    assert metadata.heading_text == "第六条 等待期：90天"
    assert metadata.section_title == "等待期"
    assert metadata.heading_confidence == "high"


def test_numbered_heading_rejects_sentence_like_known_title_mentions():
    for line in ("一、等待期为90天", "第六条 本合同设有等待期"):
        metadata = parse_clause_metadata(line)

        assert metadata.clause_id is None
        assert metadata.heading_text is None
        assert metadata.section_title == "未识别条款标题"
        assert metadata.heading_confidence == "low"
        assert metadata.heading_source == "fallback"


def test_known_title_match_prefers_earliest_occurrence_before_length():
    metadata = parse_clause_metadata("2.5 保险金额与基本保险金额")

    assert metadata.clause_id == "2.5"
    assert metadata.section_title == "保险金额"
    assert metadata.heading_confidence == "high"


def test_fallback_uses_current_title_when_no_heading_found():
    metadata = parse_clause_metadata(
        "本合同自保险单载明的生效日零时起生效。",
        current_title="保险期间",
    )

    assert metadata.section_title == "保险期间"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"
