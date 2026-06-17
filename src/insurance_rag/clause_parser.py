import re

from insurance_rag.models import ClauseMetadata


UNKNOWN_SECTION_TITLE = "未识别条款标题"

KNOWN_SECTION_TITLES = (
    "等待期",
    "责任免除",
    "除外责任",
    "免责条款",
    "保险责任",
    "保险期间",
    "保险金额",
    "基本保险金额",
    "保险金给付",
    "给付条件",
    "豁免保险费",
    "犹豫期",
    "宽限期",
    "合同解除",
    "合同效力",
    "释义",
    "疾病定义",
    "重大疾病定义",
    "轻症疾病",
    "中症疾病",
    "身故保险金",
    "全残保险金",
)

_NUMBERED_HEADING_PATTERNS = (
    re.compile(r"^(第\s*[零〇一二三四五六七八九十百千万两\d]+\s*条)\s*[：:、.．]?\s*(.+)$"),
    re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$"),
    re.compile(r"^([（(][零〇一二三四五六七八九十百千万两\d]+[）)])\s*(.+)$"),
    re.compile(r"^([零〇一二三四五六七八九十百千万两]+、|\d+\.)\s*(.+)$"),
)
_DIRECTORY_LINE_PATTERN = re.compile(r"(?:\.{3,}|…{2,}|·{3,})\s*\d+\s*$")
_BARE_PAGE_NUMBER_DIRECTORY_PATTERN = re.compile(r"^\d+(?:\.\d+)+\s+.+\s+\d+\s*$")


def parse_clause_metadata(
    text: str,
    *,
    current_title: str = UNKNOWN_SECTION_TITLE,
) -> ClauseMetadata:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:80]:
        known_title = _find_known_title(line)
        if _is_directory_like(line):
            continue

        numbered_metadata = _parse_numbered_heading(line)
        if numbered_metadata is not None:
            return numbered_metadata

        if known_title is not None and line == known_title:
            return ClauseMetadata(
                heading_text=line,
                section_title=known_title,
                heading_confidence="medium",
                heading_source="known_title",
            )

    return ClauseMetadata(
        section_title=current_title,
        heading_confidence="low",
        heading_source="fallback",
    )


def _parse_numbered_heading(line: str) -> ClauseMetadata | None:
    for pattern in _NUMBERED_HEADING_PATTERNS:
        match = pattern.match(line)
        if not match:
            continue

        clause_id = re.sub(r"\s+", "", match.group(1))
        title = _find_known_title(match.group(2).strip())
        if title is None:
            return None

        return ClauseMetadata(
            clause_id=clause_id,
            heading_text=line,
            section_title=title,
            heading_confidence="high",
            heading_source="line_pattern",
        )

    return None


def _find_known_title(text: str) -> str | None:
    for title in sorted(KNOWN_SECTION_TITLES, key=len, reverse=True):
        if title in text:
            return title
    return None


def _is_directory_like(line: str) -> bool:
    if _DIRECTORY_LINE_PATTERN.search(line):
        return True

    if not _BARE_PAGE_NUMBER_DIRECTORY_PATTERN.match(line):
        return False

    without_page_number = re.sub(r"\s+\d+\s*$", "", line)
    return _find_known_title(without_page_number) is not None
