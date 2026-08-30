from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Protocol

from insurance_rag.models import DocumentPage
from insurance_rag.silver_benchmark import (
    AnnotationDraft,
    AnnotationMetadata,
    AnnotationResponseMetadata,
    BenchmarkSource,
    EvidenceSpan,
)
from insurance_rag.silver_normalization import normalize_page


@dataclass(frozen=True)
class EvidenceCaseRequest:
    slot_id: str
    stratum: str
    additional_strata: tuple[str, ...] = ()
    hard_negative_category: str | None = None
    question: str | None = None


@dataclass(frozen=True)
class GroqPassConfig:
    annotator_id: str
    model_id: str
    prompt_version: str
    schema_version: str
    reasoning_effort: str
    temperature: float
    top_p: float
    seed: int
    max_completion_tokens: int
    max_retries: int = 2
    normalization_version: str = "normalized-page-text/v1.0.0"
    input_char_limit: int = 100_000
    transport_window_char_limit: int = 0


@dataclass(frozen=True)
class GroqCompletion:
    response_id: str
    system_fingerprint: str | None
    content: str
    prompt_tokens: int
    completion_tokens: int
    request_timestamp: str | None = None
    retry_count: int = 0


class GroqChatClient(Protocol):
    def create_completion(self, **request: object) -> GroqCompletion: ...


class OpenAIGroqChatClient:
    def __init__(self, sdk_client: object) -> None:
        self._sdk_client = sdk_client

    def create_completion(self, **request: object) -> GroqCompletion:
        groq_options = {}
        for option_name in ("reasoning_format", "service_tier"):
            if option_name in request:
                groq_options[option_name] = request.pop(option_name)
        if groq_options:
            existing_extra = request.pop("extra_body", {})
            request["extra_body"] = {**existing_extra, **groq_options}
        response = self._sdk_client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Groq completion did not return text content.")
        return GroqCompletion(
            response_id=response.id,
            system_fingerprint=getattr(response, "system_fingerprint", None),
            content=content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )


CaseRequestFactory = Callable[[BenchmarkSource], tuple[EvidenceCaseRequest, ...]]


class JsonAnnotationCheckpointStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def load(
        self,
        *,
        annotator_id: str,
        source_id: str,
        request_sha256: str,
    ) -> GroqCompletion | None:
        path = self._path(annotator_id, source_id)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("request_sha256") != request_sha256:
            return None
        return GroqCompletion(**record["completion"])

    def save(
        self,
        *,
        annotator_id: str,
        source_id: str,
        request_sha256: str,
        completion: GroqCompletion,
    ) -> None:
        path = self._path(annotator_id, source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "request_sha256": request_sha256,
                    "completion": asdict(completion),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _path(self, annotator_id: str, source_id: str) -> Path:
        annotator_key = _sha256(annotator_id)[:16]
        source_key = _sha256(source_id)[:24]
        return self._root / annotator_key / f"{source_key}.json"


class GroqAnnotationPass:
    def __init__(
        self,
        *,
        client: GroqChatClient,
        config: GroqPassConfig,
        prompt_text: str,
        case_requests: CaseRequestFactory,
        checkpoint_store: JsonAnnotationCheckpointStore | None = None,
        response_schema: Mapping[str, object] | None = None,
        input_char_limit: int = 100_000,
    ) -> None:
        self._client = client
        self._config = config
        self._prompt_text = prompt_text
        self._case_requests = case_requests
        self._checkpoint_store = checkpoint_store
        self._response_schema = response_schema or _default_response_schema()
        self._input_char_limit = input_char_limit

    def __call__(self, source: BenchmarkSource) -> tuple[AnnotationDraft, ...]:
        requests = self._case_requests(source)
        if not requests:
            raise ValueError("Groq annotation requires at least one evidence case request.")
        _validate_case_requests(requests)
        messages = _annotation_messages(source, requests, self._prompt_text)
        _validate_input_size(messages, self._input_char_limit)
        completion_request = dict(
            model=self._config.model_id,
            messages=messages,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            seed=self._config.seed,
            max_completion_tokens=self._config.max_completion_tokens,
            reasoning_effort=self._config.reasoning_effort,
            reasoning_format="hidden",
            service_tier="on_demand",
            stream=False,
            n=1,
            response_format=_response_format(self._response_schema),
        )
        request_sha256 = _sha256(
            json.dumps(
                completion_request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        completion = (
            self._checkpoint_store.load(
                annotator_id=self._config.annotator_id,
                source_id=source.source_id,
                request_sha256=request_sha256,
            )
            if self._checkpoint_store is not None
            else None
        )
        if completion is None:
            completion = _create_with_retries(
                self._client, self._config, completion_request
            )
            if self._checkpoint_store is not None:
                self._checkpoint_store.save(
                    annotator_id=self._config.annotator_id,
                    source_id=source.source_id,
                    request_sha256=request_sha256,
                    completion=completion,
                )
        payload = json.loads(completion.content)
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("Groq annotation response requires a cases array.")
        case_by_slot = {
            str(case.get("slot_id")): case
            for case in raw_cases
            if isinstance(case, dict)
        }
        expected_slots = {request.slot_id for request in requests}
        if set(case_by_slot) != expected_slots:
            raise ValueError("Groq annotation response slots do not match the request.")

        stable_metadata = _annotation_metadata(
            self._config, self._prompt_text, self._response_schema
        )
        response_metadata = _response_metadata(completion)
        return tuple(
            _draft_from_case(
                source,
                request,
                case_by_slot[request.slot_id],
                stable_metadata,
                response_metadata,
            )
            for request in requests
        )

class GroqAdjudicationPass:
    def __init__(
        self,
        *,
        client: GroqChatClient,
        config: GroqPassConfig,
        prompt_text: str,
        checkpoint_store: JsonAnnotationCheckpointStore | None = None,
        response_schema: Mapping[str, object] | None = None,
        input_char_limit: int = 100_000,
    ) -> None:
        self._client = client
        self._config = config
        self._prompt_text = prompt_text
        self._checkpoint_store = checkpoint_store
        self._response_schema = response_schema or _default_response_schema()
        self._input_char_limit = input_char_limit

    def __call__(
        self,
        source: BenchmarkSource,
        first: AnnotationDraft,
        second: AnnotationDraft,
    ) -> AnnotationDraft:
        if first.strata != second.strata:
            raise ValueError("Adjudication drafts must use the same frozen strata.")
        hard_negative_category = (
            first.hard_negative_category or second.hard_negative_category
        )
        request = EvidenceCaseRequest(
            slot_id="adjudication",
            stratum=first.stratum,
            additional_strata=first.additional_strata,
            hard_negative_category=hard_negative_category,
        )
        ordered_drafts = (first, second)
        order_key = _sha256(f"{source.source_id}:{first.slot_id}")
        if int(order_key[-1], 16) % 2:
            ordered_drafts = (second, first)
        messages = _annotation_messages(source, (request,), self._prompt_text)
        messages[1]["content"] += "\n\nANONYMOUS_DRAFTS\n" + json.dumps(
            {
                "draft_a": _draft_for_adjudication(ordered_drafts[0]),
                "draft_b": _draft_for_adjudication(ordered_drafts[1]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        _validate_input_size(messages, self._input_char_limit)
        completion_request = dict(
            model=self._config.model_id,
            messages=messages,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            seed=self._config.seed,
            max_completion_tokens=self._config.max_completion_tokens,
            reasoning_effort=self._config.reasoning_effort,
            reasoning_format="hidden",
            service_tier="on_demand",
            stream=False,
            n=1,
            response_format=_response_format(self._response_schema),
        )
        request_sha256 = _sha256(
            json.dumps(
                completion_request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        work_item_id = f"{source.source_id}:{request_sha256}"
        completion = (
            self._checkpoint_store.load(
                annotator_id=self._config.annotator_id,
                source_id=work_item_id,
                request_sha256=request_sha256,
            )
            if self._checkpoint_store is not None
            else None
        )
        if completion is None:
            completion = _create_with_retries(
                self._client, self._config, completion_request
            )
            if self._checkpoint_store is not None:
                self._checkpoint_store.save(
                    annotator_id=self._config.annotator_id,
                    source_id=work_item_id,
                    request_sha256=request_sha256,
                    completion=completion,
                )
        payload = json.loads(completion.content)
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list) or len(raw_cases) != 1:
            raise ValueError("Groq adjudication response requires exactly one case.")
        raw_case = raw_cases[0]
        if (
            not isinstance(raw_case, dict)
            or raw_case.get("slot_id") != "adjudication"
        ):
            raise ValueError("Groq adjudication response has an invalid slot.")
        metadata = _annotation_metadata(
            self._config, self._prompt_text, self._response_schema
        )
        response_metadata = replace(
            _response_metadata(completion),
            draft_order=tuple(
                draft.metadata.annotator_id for draft in ordered_drafts
            ),
        )
        result = _draft_from_case(
            source,
            request,
            raw_case,
            metadata,
            response_metadata,
        )
        return replace(result, slot_id=first.slot_id)


def _create_with_retries(
    client: GroqChatClient,
    config: GroqPassConfig,
    completion_request: dict[str, object],
) -> GroqCompletion:
    for retry_count in range(config.max_retries + 1):
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            completion = client.create_completion(**completion_request)
            return replace(
                completion,
                request_timestamp=timestamp,
                retry_count=retry_count,
            )
        except Exception as error:
            if retry_count >= config.max_retries or not _is_retryable(error):
                raise
            time.sleep(_retry_delay_seconds(error, retry_count))
    raise RuntimeError("Groq retry loop exited unexpectedly.")


def _annotation_metadata(
    config: GroqPassConfig,
    prompt_text: str,
    response_schema: Mapping[str, object],
) -> AnnotationMetadata:
    return AnnotationMetadata(
        annotator_id=config.annotator_id,
        model_id=config.model_id,
        prompt_version=config.prompt_version,
        generation_parameters=(
            ("schema_version", config.schema_version),
            ("schema_sha256", _schema_sha256(response_schema)),
            ("prompt_sha256", _sha256(prompt_text)),
            ("reasoning_effort", config.reasoning_effort),
            ("temperature", config.temperature),
            ("top_p", config.top_p),
            ("seed", config.seed),
            ("max_completion_tokens", config.max_completion_tokens),
            ("normalization_version", config.normalization_version),
            ("input_char_limit", config.input_char_limit),
            ("transport_window_char_limit", config.transport_window_char_limit),
            ("reasoning_format", "hidden"),
            ("service_tier", "on_demand"),
            ("stream", False),
            ("n", 1),
            ("tools", "none"),
            ("response_format", "strict_json_schema"),
        ),
    )


def _response_metadata(
    completion: GroqCompletion,
) -> AnnotationResponseMetadata:
    return AnnotationResponseMetadata(
        response_id=completion.response_id,
        system_fingerprint=completion.system_fingerprint,
        request_timestamp=completion.request_timestamp or "",
        retry_count=completion.retry_count,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )

def _annotation_messages(
    source: BenchmarkSource,
    requests: tuple[EvidenceCaseRequest, ...],
    prompt_text: str,
) -> list[dict[str, str]]:
    normalized_pages = tuple(normalize_page(page) for page in source.pages)
    request_payload = []
    for request in requests:
        record = {
            "slot_id": request.slot_id,
            "stratum": request.stratum,
            "additional_strata": list(request.additional_strata),
            "hard_negative_category": request.hard_negative_category,
        }
        if request.question is not None:
            record["question"] = request.question
        request_payload.append(record)
    source_text = "\n\n".join(
        f"[PAGE {page.page_number}]\n{page.text}" for page in normalized_pages
    )
    return [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": (
                "CASE_REQUESTS\n"
                + json.dumps(request_payload, ensure_ascii=False, sort_keys=True)
                + "\n\nNORMALIZED_PAGE_TEXT\n"
                + source_text
            ),
        },
    ]


def _draft_from_case(
    source: object,
    request: EvidenceCaseRequest,
    raw_case: Mapping[str, object],
    metadata: AnnotationMetadata,
    response_metadata: AnnotationResponseMetadata,
) -> AnnotationDraft:
    model_question = str(raw_case.get("question", "")).strip()
    question = request.question.strip() if request.question else model_question
    model_uncertain = raw_case.get("annotation_uncertain") is True
    authoritative_pages = getattr(source, "authoritative_pages", source.pages)
    evidence_spans = _map_response_spans(
        authoritative_pages, raw_case.get("evidence_spans")
    )
    hard_negative_spans = _map_response_spans(
        authoritative_pages,
        raw_case.get("hard_negative_spans"),
    )
    invalid = (
        not question
        or not evidence_spans
        or (
            request.stratum == "cross_page_clause"
            and len({span.page_number for span in evidence_spans}) < 2
        )
        or (
            request.hard_negative_category is not None
            and not hard_negative_spans
        )
    )
    annotation_uncertain = model_uncertain or invalid
    if annotation_uncertain:
        evidence_spans = ()
        hard_negative_spans = ()
    return AnnotationDraft(
        question=question or f"annotation_uncertain:{request.slot_id}",
        evidence_spans=evidence_spans,
        stratum=request.stratum,
        hard_negative_category=(
            None if annotation_uncertain else request.hard_negative_category
        ),
        metadata=metadata,
        hard_negative_spans=hard_negative_spans,
        annotation_uncertain=annotation_uncertain,
        additional_strata=request.additional_strata,
        response_metadata=response_metadata,
        slot_id=request.slot_id,
    )


def _draft_for_adjudication(draft: AnnotationDraft) -> dict[str, object]:
    return {
        "question": draft.question,
        "evidence_spans": [
            {
                "page_number": span.page_number,
                "quote": " ".join(span.quote.split()),
            }
            for span in draft.evidence_spans
        ],
        "hard_negative_spans": [
            {
                "page_number": span.page_number,
                "quote": " ".join(span.quote.split()),
            }
            for span in draft.hard_negative_spans
        ],
        "annotation_uncertain": draft.annotation_uncertain,
    }


def _map_response_spans(
    pages: tuple[DocumentPage, ...],
    value: object,
) -> tuple[EvidenceSpan, ...]:
    if not isinstance(value, list):
        return ()
    normalized_by_page = {
        page.page_number: normalize_page(page) for page in pages
    }
    mapped: list[EvidenceSpan] = []
    for raw_span in value:
        if not isinstance(raw_span, dict):
            return ()
        try:
            page_number = int(raw_span["page_number"])
            quote = str(raw_span["quote"])
        except (KeyError, TypeError, ValueError):
            return ()
        page = normalized_by_page.get(page_number)
        if page is None or not quote or page.text.count(quote) != 1:
            return ()
        normalized_start = page.text.index(quote)
        normalized_end = normalized_start + len(quote)
        if (
            normalized_start not in page.normalized_to_raw
            or normalized_end - 1 not in page.normalized_to_raw
        ):
            return ()
        raw_start = page.normalized_to_raw[normalized_start]
        raw_end = page.normalized_to_raw[normalized_end - 1] + 1
        mapped.append(
            EvidenceSpan(
                page_number=page_number,
                start_char=raw_start,
                end_char=raw_end,
                quote=page.raw_text[raw_start:raw_end],
            )
        )
    return tuple(mapped)


def _validate_case_requests(
    requests: tuple[EvidenceCaseRequest, ...],
) -> None:
    slot_ids = [request.slot_id for request in requests]
    if any(not slot_id.strip() for slot_id in slot_ids):
        raise ValueError("Evidence case request slot IDs cannot be empty.")
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("Evidence case request slot IDs must be unique.")


def _default_response_schema() -> dict[str, object]:
    span_schema = {
        "type": "object",
        "properties": {
            "page_number": {"type": "integer"},
            "quote": {"type": "string"},
        },
        "required": ["page_number", "quote"],
        "additionalProperties": False,
    }
    case_schema = {
        "type": "object",
        "properties": {
            "slot_id": {"type": "string"},
            "question": {"type": "string"},
            "evidence_spans": {"type": "array", "items": span_schema},
            "hard_negative_spans": {"type": "array", "items": span_schema},
            "annotation_uncertain": {"type": "boolean"},
        },
        "required": [
            "slot_id",
            "question",
            "evidence_spans",
            "hard_negative_spans",
            "annotation_uncertain",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"cases": {"type": "array", "items": case_schema}},
        "required": ["cases"],
        "additionalProperties": False,
    }


def _response_format(schema: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "silver_evidence_annotation_v1",
            "strict": True,
            "schema": schema,
        },
    }


def _validate_input_size(messages: list[dict[str, str]], limit: int) -> None:
    if limit <= 0:
        raise ValueError("Groq input character limit must be positive.")
    character_count = sum(len(message["content"]) for message in messages)
    if character_count > limit:
        raise ValueError(
            "Groq annotation input character limit exceeded: "
            f"{character_count} > {limit}. Create deterministic page windows "
            "instead of truncating the policy."
        )


def _schema_sha256(schema: Mapping[str, object]) -> str:
    return _sha256(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _is_retryable(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        return True
    return status_code == 429 or status_code >= 500


def _retry_delay_seconds(error: Exception, retry_count: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") if headers is not None else None
    try:
        base_delay = max(0.0, float(retry_after))
    except (TypeError, ValueError):
        base_delay = 15.0 * (2**retry_count)
    return min(60.0, base_delay + random.uniform(0.0, 0.5))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
