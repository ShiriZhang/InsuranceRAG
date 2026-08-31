from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import unicodedata

from insurance_rag.config import AppConfig
from insurance_rag.silver_annotation import EvidenceCaseRequest
from insurance_rag.models import DocumentPage
from insurance_rag.silver_benchmark import (
    BenchmarkSource,
    EvidenceSpan,
    SourceApproval,
    source_from_pdf_bytes,
)
from insurance_rag.silver_dataset import (
    REQUIRED_EVIDENCE_STRATA,
    DatasetSplit,
    DocumentSplitAssignment,
    FrozenDocumentSplit,
    freeze_document_split,
)


@dataclass(frozen=True)
class CorpusInventoryEntry:
    relative_path: str
    source_id: str
    insurer_family: str
    product_family: str


@dataclass(frozen=True)
class ApprovedCorpusInventory:
    documents_dir: Path
    approval_reference: str
    entries: tuple[CorpusInventoryEntry, ...]
    sources: tuple[BenchmarkSource, ...]


@dataclass(frozen=True)
class EvidenceCasePlan:
    split_by_source_id: dict[str, DatasetSplit]
    source_index_by_id: dict[str, int]
    strata_by_source_id: dict[str, tuple[str, ...]]
    development_cases_per_source: int
    held_out_cases_per_source: int

    def __call__(
        self,
        source: BenchmarkSource,
    ) -> tuple[EvidenceCaseRequest, ...]:
        split = self.split_by_source_id.get(source.source_id)
        if split is None:
            raise ValueError(f"Case plan received unknown source: {source.source_id}")
        case_count = (
            self.development_cases_per_source
            if split is DatasetSplit.DEVELOPMENT
            else self.held_out_cases_per_source
        )
        source_index = self.source_index_by_id[source.source_id]
        first_global_index = source_index * case_count
        requests = []
        for local_index in range(case_count):
            global_index = first_global_index + local_index
            stratum = self.strata_by_source_id[source.source_id][local_index]
            requests.append(
                EvidenceCaseRequest(
                    slot_id=(
                        f"{split.value}-{global_index:05d}-"
                        f"{_sha256(source.source_id)[:10]}"
                    ),
                    stratum=stratum,
                    hard_negative_category=(
                        "similar_clause"
                        if stratum
                        == "adjacent_or_lexically_similar_hard_negative"
                        else None
                    ),
                )
            )
        return tuple(requests)


@dataclass(frozen=True)
class AnnotationSourceWindow:
    original_source: BenchmarkSource
    window_id: str
    pages: tuple[DocumentPage, ...]

    @property
    def source_id(self) -> str:
        return f"{self.original_source.source_id}:{self.window_id}"

    @property
    def authoritative_pages(self) -> tuple[DocumentPage, ...]:
        return self.original_source.pages


def build_approved_corpus_inventory(
    documents_dir: Path,
    *,
    approval_reference: str,
    parse_config: AppConfig,
) -> ApprovedCorpusInventory:
    if not documents_dir.is_dir():
        raise ValueError(f"Documents directory does not exist: {documents_dir}")
    if not approval_reference.strip():
        raise ValueError("Approved corpus inventory requires an approval_reference.")

    pdf_paths = sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise ValueError("Approved corpus inventory requires at least one PDF.")

    entries: list[CorpusInventoryEntry] = []
    sources: list[BenchmarkSource] = []
    seen_relative_paths: set[str] = set()
    for pdf_path in pdf_paths:
        relative_path = _normalized_relative_path(pdf_path, documents_dir)
        path_parts = relative_path.split("/")
        if len(path_parts) < 2:
            raise ValueError(
                "Every approved PDF must be below an insurer directory: "
                f"{relative_path}"
            )
        if relative_path in seen_relative_paths:
            raise ValueError(f"Duplicate normalized PDF path: {relative_path}")
        seen_relative_paths.add(relative_path)

        insurer_family = path_parts[0]
        source_id = _stable_identifier("benchmark-source/v1", relative_path)
        product_family = _stable_identifier("product-family/v1", relative_path)
        source = source_from_pdf_bytes(
            pdf_path.read_bytes(),
            source_id=source_id,
            source_name=relative_path,
            approval=SourceApproval.PROJECT_OWNED,
            approval_reference=approval_reference,
            insurer_family=insurer_family,
            product_family=product_family,
            parse_config=parse_config,
        )
        entries.append(
            CorpusInventoryEntry(
                relative_path=relative_path,
                source_id=source_id,
                insurer_family=insurer_family,
                product_family=product_family,
            )
        )
        sources.append(source)

    return ApprovedCorpusInventory(
        documents_dir=documents_dir.resolve(),
        approval_reference=approval_reference,
        entries=tuple(entries),
        sources=tuple(sources),
    )


def freeze_insurer_document_split(
    inventory: ApprovedCorpusInventory,
    *,
    version: str,
    held_out_fraction: float,
    seed: str,
) -> FrozenDocumentSplit:
    if not 0 < held_out_fraction < 1:
        raise ValueError("held_out_fraction must be between zero and one.")
    if not seed.strip():
        raise ValueError("Insurer split seed cannot be empty.")

    source_count_by_insurer: dict[str, int] = {}
    for source in inventory.sources:
        source_count_by_insurer[source.insurer_family] = (
            source_count_by_insurer.get(source.insurer_family, 0) + 1
        )
    if len(source_count_by_insurer) < 2:
        raise ValueError("Insurer-level splitting requires at least two insurers.")

    total_sources = len(inventory.sources)
    target_held_out = min(
        total_sources - 1,
        max(1, round(total_sources * held_out_fraction)),
    )
    held_out_insurers = _select_insurer_subset(
        source_count_by_insurer,
        target_source_count=target_held_out,
        seed=seed,
    )
    assignments = tuple(
        DocumentSplitAssignment(
            source_id=source.source_id,
            split=(
                DatasetSplit.HELD_OUT
                if source.insurer_family in held_out_insurers
                else DatasetSplit.DEVELOPMENT
            ),
            near_duplicate_family=source.product_family,
        )
        for source in inventory.sources
    )
    return freeze_document_split(
        version=version,
        sources=inventory.sources,
        assignments=assignments,
    )


def build_evidence_case_plan(
    document_split: FrozenDocumentSplit,
    *,
    development_cases_per_source: int = 3,
    held_out_cases_per_source: int = 5,
) -> EvidenceCasePlan:
    if development_cases_per_source <= 0 or held_out_cases_per_source <= 0:
        raise ValueError("Evidence case counts per source must be positive.")
    split_by_source_id: dict[str, DatasetSplit] = {}
    source_index_by_id: dict[str, int] = {}
    strata_by_source_id: dict[str, tuple[str, ...]] = {}
    for split in DatasetSplit:
        sources = sorted(
            document_split.sources_for(split),
            key=lambda source: source.source_id,
        )
        case_count = (
            development_cases_per_source
            if split is DatasetSplit.DEVELOPMENT
            else held_out_cases_per_source
        )
        for source_index, source in enumerate(sources):
            split_by_source_id[source.source_id] = split
            source_index_by_id[source.source_id] = source_index
            first_global_index = source_index * case_count
            strata_by_source_id[source.source_id] = tuple(
                REQUIRED_EVIDENCE_STRATA[
                    (first_global_index + local_index)
                    % len(REQUIRED_EVIDENCE_STRATA)
                ]
                for local_index in range(case_count)
            )
        if split is DatasetSplit.HELD_OUT:
            _prioritize_cross_page_sources(sources, strata_by_source_id)
    return EvidenceCasePlan(
        split_by_source_id=split_by_source_id,
        source_index_by_id=source_index_by_id,
        strata_by_source_id=strata_by_source_id,
        development_cases_per_source=development_cases_per_source,
        held_out_cases_per_source=held_out_cases_per_source,
    )


def _prioritize_cross_page_sources(
    sources: tuple[BenchmarkSource, ...],
    strata_by_source_id: dict[str, tuple[str, ...]],
) -> None:
    cross_page = "cross_page_clause"
    assigned = {
        source.source_id
        for source in sources
        if cross_page in strata_by_source_id[source.source_id]
    }
    ranked = sorted(
        sources,
        key=lambda source: (
            -_best_cross_page_continuity_score(source.pages),
            source.source_id,
        ),
    )
    preferred = {source.source_id for source in ranked[: len(assigned)]}
    additions = [source_id for source_id in preferred if source_id not in assigned]
    removals = [source_id for source_id in assigned if source_id not in preferred]
    additions.sort()
    removals.sort()
    replacement_priority = {
        "complete_short_clause": 0,
        "single_sentence": 1,
        "adjacent_or_lexically_similar_hard_negative": 2,
        "internally_split_clause": 3,
        "rule_plus_exception": 4,
        "multi_sentence_conditions_outcomes": 5,
    }

    while removals:
        swapped = False
        for removal_index, removal_id in enumerate(removals):
            removal_strata = list(strata_by_source_id[removal_id])
            cross_index = removal_strata.index(cross_page)
            for addition_index, addition_id in enumerate(additions):
                addition_strata = list(strata_by_source_id[addition_id])
                candidates = sorted(
                    (
                        index
                        for index, stratum in enumerate(addition_strata)
                        if stratum not in removal_strata
                    ),
                    key=lambda index: (
                        replacement_priority.get(addition_strata[index], 99),
                        index,
                    ),
                )
                if not candidates:
                    continue
                addition_slot = candidates[0]
                replacement = addition_strata[addition_slot]
                removal_strata[cross_index] = replacement
                addition_strata[addition_slot] = cross_page
                strata_by_source_id[removal_id] = tuple(removal_strata)
                strata_by_source_id[addition_id] = tuple(addition_strata)
                removals.pop(removal_index)
                additions.pop(addition_index)
                swapped = True
                break
            if swapped:
                break
        if not swapped:
            raise ValueError(
                "Unable to preserve balanced strata while ranking cross-page cases."
            )


def _best_cross_page_continuity_score(
    pages: tuple[DocumentPage, ...],
) -> int:
    scores = [
        _cross_page_continuity_score(left.text[-700:], right.text[:700])
        for left, right in zip(pages, pages[1:])
    ]
    return max(scores, default=-100)


def build_annotation_source_window(
    source: BenchmarkSource,
    *,
    slot_id: str,
    slot_index: int,
    slot_count: int,
    max_chars: int,
    stratum: str | None = None,
) -> AnnotationSourceWindow:
    """Choose a deterministic transport window without dropping source text."""
    if slot_count <= 0 or not 0 <= slot_index < slot_count:
        raise ValueError("Annotation window slot index is outside the slot count.")
    if max_chars <= 0:
        raise ValueError("Annotation window max_chars must be positive.")
    if stratum == "cross_page_clause":
        boundary_windows = _cross_page_boundary_windows(source.pages, max_chars)
        if boundary_windows:
            if slot_count == 1:
                window_index = 0
            else:
                window_index = round(
                    slot_index * (len(boundary_windows) - 1) / (slot_count - 1)
                )
            return AnnotationSourceWindow(
                original_source=source,
                window_id=slot_id,
                pages=boundary_windows[window_index],
            )
    windows: list[tuple] = []
    current = []
    current_chars = 0
    for page in source.pages:
        segments = _page_transport_segments(page, max_chars)
        if len(segments) > 1:
            if current:
                windows.append(tuple(current))
                current = []
                current_chars = 0
            windows.extend((segment,) for segment in segments)
            continue
        transport_page = segments[0]
        page_chars = len(" ".join(transport_page.text.split()))
        separator_chars = 2 if current else 0
        if current and current_chars + separator_chars + page_chars > max_chars:
            windows.append(tuple(current))
            current = []
            current_chars = 0
            separator_chars = 0
        current.append(transport_page)
        current_chars += separator_chars + page_chars
    if current:
        windows.append(tuple(current))
    if stratum == "cross_page_clause":
        cross_page_windows = [
            window
            for window in windows
            if len({page.page_number for page in window}) >= 2
        ]
        if cross_page_windows:
            windows = cross_page_windows
    if stratum == "multi_sentence_conditions_outcomes":
        windows.sort(key=lambda window: -_multi_sentence_window_score(window))
        window_index = 0
    elif slot_count == 1:
        window_index = len(windows) // 2
    else:
        window_index = round(slot_index * (len(windows) - 1) / (slot_count - 1))
    return AnnotationSourceWindow(
        original_source=source,
        window_id=slot_id,
        pages=windows[window_index],
    )


def _cross_page_boundary_windows(
    pages: tuple[DocumentPage, ...],
    max_chars: int,
) -> tuple[tuple[DocumentPage, DocumentPage], ...]:
    if max_chars < 2:
        return ()
    left_budget = max_chars // 2
    right_budget = max_chars - left_budget
    ranked_windows = []
    for boundary_index, (left, right) in enumerate(zip(pages, pages[1:])):
        left_text = left.text[-left_budget:]
        right_text = right.text[:right_budget]
        if not left_text or not right_text:
            continue
        window = (
            type(left)(
                page_number=left.page_number,
                text=left_text,
                extraction_method=left.extraction_method,
            ),
            type(right)(
                page_number=right.page_number,
                text=right_text,
                extraction_method=right.extraction_method,
            ),
        )
        ranked_windows.append(
            (
                -_cross_page_continuity_score(left_text, right_text),
                boundary_index,
                window,
            )
        )
    ranked_windows.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked_windows)


def _cross_page_continuity_score(left_text: str, right_text: str) -> int:
    left = " ".join(left_text.split())
    right = " ".join(right_text.split())
    if not left or not right:
        return -100
    score = 0
    if left.endswith(("，", "、", "：", ",", ":", "（", "(")):
        score += 8
    elif not left.endswith(("。", "！", "？", "；", ".", "!", "?", ";")):
        score += 3
    continuation_prefixes = (
        "并",
        "且",
        "以及",
        "或者",
        "或",
        "但",
        "除",
        "若",
        "如",
        "则",
        "同时",
        "由",
        "给付",
    )
    if right.startswith(continuation_prefixes):
        score += 6
    if right.startswith(("，", "、", "；", ",", ";", ")", "）")):
        score += 4
    left_fragment = _boundary_fragment(left, from_end=True)
    right_fragment = _boundary_fragment(right, from_end=False)
    score += min(len(left_fragment) // 12, 5)
    score += min(len(right_fragment) // 12, 5)
    if len(left_fragment) < 6:
        score -= 8
    if len(right_fragment) < 6:
        score -= 4
    if right.startswith("目录") or (
        right.startswith("第") and "条" in right[:16]
    ):
        score -= 6
    return score


def _boundary_fragment(text: str, *, from_end: bool) -> str:
    terminal = "。！？；.!?;"
    if from_end:
        boundary = max((text.rfind(mark) for mark in terminal), default=-1)
        return text[boundary + 1 :].strip()
    indexes = [text.find(mark) for mark in terminal if text.find(mark) >= 0]
    boundary = min(indexes, default=len(text))
    return text[:boundary].strip()


def _multi_sentence_window_score(window: tuple[DocumentPage, ...]) -> int:
    text = " ".join(" ".join(page.text.split()) for page in window)
    sentence_count = sum(text.count(mark) for mark in "。！？；.!?;")
    condition_count = sum(
        text.count(marker)
        for marker in ("如果", "若", "当", "条件", "等待期", "情形", "情况下")
    )
    outcome_count = sum(
        text.count(marker)
        for marker in ("给付", "承担", "赔付", "豁免", "终止", "不承担")
    )
    return min(sentence_count, 8) * 2 + min(condition_count, 5) * 3 + min(
        outcome_count, 5
    ) * 3


def build_adjudication_source_window(
    source: BenchmarkSource,
    first_spans: tuple[EvidenceSpan, ...],
    second_spans: tuple[EvidenceSpan, ...],
    *,
    work_item_id: str,
    max_chars: int,
    fallback_window: AnnotationSourceWindow | None = None,
) -> AnnotationSourceWindow:
    if fallback_window is not None:
        return AnnotationSourceWindow(
            original_source=source,
            window_id=work_item_id,
            pages=fallback_window.pages,
        )
    referenced_pages = {
        span.page_number for span in (*first_spans, *second_spans)
    }
    selected = tuple(
        page for page in source.pages if page.page_number in referenced_pages
    )
    selected_chars = sum(len(" ".join(page.text.split())) for page in selected)
    if selected and selected_chars <= max_chars:
        return AnnotationSourceWindow(
            original_source=source,
            window_id=work_item_id,
            pages=selected,
        )
    if referenced_pages:
        source_by_page = {page.page_number: page for page in source.pages}
        spans = tuple(dict.fromkeys((*first_spans, *second_spans)))
        context_budget = max(0, max_chars - sum(len(span.quote) for span in spans))
        context_per_span = context_budget // len(spans)
        segments = []
        for span in spans:
            page = source_by_page[span.page_number]
            left_context = context_per_span // 2
            start = max(0, span.start_char - left_context)
            end = min(
                len(page.text),
                max(span.end_char, start + len(span.quote) + context_per_span),
            )
            segments.append(
                type(page)(
                    page_number=page.page_number,
                    text=page.text[start:end],
                    extraction_method=page.extraction_method,
                )
            )
        return AnnotationSourceWindow(
            original_source=source,
            window_id=work_item_id,
            pages=tuple(segments),
        )
    return build_annotation_source_window(
        source,
        slot_id=work_item_id,
        slot_index=0,
        slot_count=1,
        max_chars=max_chars,
    )


def _page_transport_segments(
    page: DocumentPage,
    max_chars: int,
) -> tuple[DocumentPage, ...]:
    if len(" ".join(page.text.split())) <= max_chars:
        return (page,)
    segments = []
    for start in range(0, len(page.text), max_chars):
        text = page.text[start : start + max_chars]
        if text:
            segments.append(
                type(page)(
                    page_number=page.page_number,
                    text=text,
                    extraction_method=page.extraction_method,
                )
            )
    return tuple(segments)


def _select_insurer_subset(
    source_count_by_insurer: dict[str, int],
    *,
    target_source_count: int,
    seed: str,
) -> frozenset[str]:
    insurers = sorted(
        source_count_by_insurer,
        key=lambda insurer: _stable_identifier(seed, insurer),
    )
    subsets_by_count: dict[int, tuple[str, ...]] = {0: ()}
    for insurer in insurers:
        insurer_count = source_count_by_insurer[insurer]
        for current_count, subset in tuple(subsets_by_count.items()):
            candidate_count = current_count + insurer_count
            subsets_by_count.setdefault(candidate_count, (*subset, insurer))

    total_sources = sum(source_count_by_insurer.values())
    eligible_counts = [
        count for count in subsets_by_count if 0 < count < total_sources
    ]
    selected_count = min(
        eligible_counts,
        key=lambda count: (
            abs(count - target_source_count),
            abs(count / total_sources - target_source_count / total_sources),
            count,
        ),
    )
    return frozenset(subsets_by_count[selected_count])


def _normalized_relative_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return unicodedata.normalize("NFC", relative).strip("/")


def _stable_identifier(namespace: str, value: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()
    prefix = namespace.replace("/", "-")
    return f"{prefix}-{digest}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
