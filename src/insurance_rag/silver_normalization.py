from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from insurance_rag.models import DocumentPage


NORMALIZATION_VERSION = "normalized-page-text/v1.0.0"


@dataclass(frozen=True)
class NormalizedPage:
    page_number: int
    text: str
    normalized_to_raw: Mapping[int, int]
    raw_to_normalized: Mapping[int, int]
    raw_text: str


def normalize_page(page: DocumentPage) -> NormalizedPage:
    normalized_chars: list[str] = []
    normalized_to_raw: dict[int, int] = {}
    raw_to_normalized: dict[int, int] = {}
    previous_token_end: int | None = None
    for token_index, match in enumerate(re.finditer(r"\S+", page.text)):
        if token_index:
            normalized_to_raw[len(normalized_chars)] = previous_token_end or match.start()
            normalized_chars.append(" ")
        for offset, character in enumerate(match.group()):
            raw_index = match.start() + offset
            normalized_index = len(normalized_chars)
            normalized_to_raw[normalized_index] = raw_index
            raw_to_normalized[raw_index] = normalized_index
            normalized_chars.append(character)
        previous_token_end = match.end()
    return NormalizedPage(
        page_number=page.page_number,
        text="".join(normalized_chars),
        normalized_to_raw=normalized_to_raw,
        raw_to_normalized=raw_to_normalized,
        raw_text=page.text,
    )


def normalized_source_text(pages: tuple[DocumentPage, ...]) -> str:
    return "\n\f\n".join(
        f"page:{page.page_number}\n{normalize_page(page).text}" for page in pages
    )
