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


def test_directory_like_line_is_not_high_confidence():
    metadata = parse_clause_metadata("2.3 保险责任 ........ 5")

    assert metadata.section_title == "保险责任"
    assert metadata.heading_confidence != "high"


def test_fallback_uses_current_title_when_no_heading_found():
    metadata = parse_clause_metadata(
        "本合同自保险单载明的生效日零时起生效。",
        current_title="保险期间",
    )

    assert metadata.section_title == "保险期间"
    assert metadata.heading_confidence == "low"
    assert metadata.heading_source == "fallback"
