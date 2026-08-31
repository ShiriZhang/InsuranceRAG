from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import insurance_rag.silver_annotation as annotation_module
from insurance_rag.models import DocumentPage
from insurance_rag.silver_annotation import (
    AdjudicationPass,
    AnnotationCompletion,
    AnnotationPass,
    AnnotationPassConfig,
    EvidenceCaseRequest,
    JsonAnnotationCheckpointStore,
    OpenAIDeepSeekResponsesClient,
)
from insurance_rag.silver_benchmark import (
    AnnotationDraft,
    AnnotationMetadata,
    BenchmarkSource,
    EvidenceSpan,
    SourceApproval,
)


ROOT = Path(__file__).parents[1]


class FakeAnnotationClient:
    def __init__(self, completion: AnnotationCompletion) -> None:
        self.completion = completion
        self.requests: list[dict[str, object]] = []

    def create_completion(self, **request):
        self.requests.append(request)
        return self.completion


class SequenceAnnotationClient:
    def __init__(self, completions: list[AnnotationCompletion]) -> None:
        self.completions = iter(completions)
        self.requests: list[dict[str, object]] = []

    def create_completion(self, **request):
        self.requests.append(request)
        return next(self.completions)


def _source() -> BenchmarkSource:
    return BenchmarkSource(
        source_id="fixture-source",
        source_name="fixture.pdf",
        approval=SourceApproval.PROJECT_OWNED,
        approval_reference="repository test fixture",
        insurer_family="fixture-insurer",
        product_family="fixture-product",
        pages=(
            DocumentPage(
                page_number=1,
                text="第一条 责任  本合同负责。",
                extraction_method="text",
            ),
        ),
    )


def test_annotation_maps_unique_normalized_quote_back_to_raw_span():
    payload = {
        "cases": [
            {
                "slot_id": "slot-1",
                "question": "本合同承担什么责任？",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [],
                "annotation_uncertain": False,
            }
        ]
    }
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="response-1",
            system_fingerprint="fingerprint-1",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=8192,
        ),
        prompt_text="Return exact normalized evidence for every requested slot.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
    )

    drafts = annotation_pass(_source())

    assert len(drafts) == 1
    span = drafts[0].evidence_spans[0]
    assert span.page_number == 1
    assert span.quote == "责任  本合同负责。"
    assert _source().pages[0].text[span.start_char : span.end_char] == span.quote
    assert drafts[0].metadata.model_id == "openai/gpt-oss-120b"
    assert drafts[0].slot_id == "slot-1"
    assert drafts[0].response_metadata.response_id == "response-1"
    assert drafts[0].response_metadata.system_fingerprint == "fingerprint-1"
    request = client.requests[0]
    assert "责任 本合同负责。" in request["input"][1]["content"]
    assert "责任  本合同负责。" not in request["input"][1]["content"]
    assert request["text"]["format"]["strict"] is True
    assert request["reasoning"] == {"effort": "low"}
    assert "temperature" not in request
    assert "seed" not in request
    generation_parameters = dict(drafts[0].metadata.generation_parameters)
    assert len(generation_parameters["schema_sha256"]) == 64


def test_annotation_retries_non_uncertain_response_with_unmappable_evidence(
    monkeypatch,
):
    monkeypatch.setattr(annotation_module.time, "sleep", lambda _seconds: None)

    def completion(response_id: str, quote: str) -> AnnotationCompletion:
        return AnnotationCompletion(
            response_id=response_id,
            system_fingerprint=None,
            content=json.dumps(
                {
                    "cases": [
                        {
                            "slot_id": "slot-1",
                            "question": "本合同承担什么责任？",
                            "evidence_spans": [
                                {"page_number": 1, "quote": quote}
                            ],
                            "hard_negative_spans": [],
                            "annotation_uncertain": False,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            prompt_tokens=100,
            completion_tokens=20,
        )

    client = SequenceAnnotationClient(
        [
            completion("invalid-response", "原文中不存在的证据"),
            completion("valid-response", "责任 本合同负责。"),
        ]
    )
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=8192,
            max_retries=1,
        ),
        prompt_text="Return exact normalized evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
    )

    result = annotation_pass(_source())

    assert len(client.requests) == 2
    assert result[0].annotation_uncertain is False
    assert result[0].response_metadata.response_id == "valid-response"


def test_annotation_reuses_checkpoint_after_semantic_retries_are_exhausted(
    monkeypatch,
):
    monkeypatch.setattr(annotation_module.time, "sleep", lambda _seconds: None)
    completion = AnnotationCompletion(
        response_id="unmappable-response",
        system_fingerprint=None,
        content=json.dumps(
            {
                "cases": [
                    {
                        "slot_id": "slot-1",
                        "question": "本合同承担什么责任？",
                        "evidence_spans": [
                            {"page_number": 1, "quote": "原文中不存在的证据"}
                        ],
                        "hard_negative_spans": [],
                        "annotation_uncertain": False,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        prompt_tokens=100,
        completion_tokens=20,
    )
    client = FakeAnnotationClient(completion)
    checkpoint_root = ROOT / "tmp" / "annotation-checkpoint-tests" / uuid4().hex
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=8192,
            max_retries=1,
        ),
        prompt_text="Return exact normalized evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
        checkpoint_store=JsonAnnotationCheckpointStore(checkpoint_root),
    )

    first = annotation_pass(_source())
    second = annotation_pass(_source())

    assert len(client.requests) == 2
    assert first[0].annotation_uncertain is True
    assert second[0].annotation_uncertain is True
    checkpoint = json.loads(
        next(checkpoint_root.rglob("*.json")).read_text(encoding="utf-8")
    )
    assert checkpoint["completion"]["semantic_validation_exhausted"] is True


def test_annotation_rejects_source_over_frozen_input_limit_before_api():
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="unused",
            system_fingerprint=None,
            content='{"cases": []}',
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=8192,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
        input_char_limit=5,
    )

    with pytest.raises(ValueError, match="input character limit"):
        annotation_pass(_source())

    assert client.requests == []


def test_annotation_does_not_retry_non_transient_bad_request():
    class BadRequestClient:
        def __init__(self):
            self.calls = 0

        def create_completion(self, **request):
            self.calls += 1
            error = RuntimeError("invalid structured output")
            error.status_code = 400
            raise error

    client = BadRequestClient()
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=4096,
            max_retries=0,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
    )

    with pytest.raises(RuntimeError, match="invalid structured output"):
        annotation_pass(_source())

    assert client.calls == 1


def test_annotation_waits_for_retry_after_on_429(monkeypatch):
    payload = {
        "cases": [
            {
                "slot_id": "slot-1",
                "question": "本合同承担什么责任？",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [],
                "annotation_uncertain": False,
            }
        ]
    }

    class RateLimitedClient:
        def __init__(self):
            self.calls = 0

        def create_completion(self, **request):
            self.calls += 1
            if self.calls == 1:
                error = RuntimeError("rate limited")
                error.status_code = 429
                error.response = SimpleNamespace(headers={"retry-after": "2.5"})
                raise error
            return AnnotationCompletion(
                response_id="retried",
                system_fingerprint=None,
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=100,
                completion_tokens=20,
            )

    sleeps = []
    monkeypatch.setattr(annotation_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(annotation_module.random, "uniform", lambda _start, _end: 0.25)
    client = RateLimitedClient()
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=4096,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
    )

    result = annotation_pass(_source())

    assert result[0].response_metadata.retry_count == 1
    assert sleeps == [2.75]


def test_second_annotation_uses_frozen_question_from_request():
    payload = {
        "cases": [
            {
                "slot_id": "slot-1",
                "question": "模型试图改写的问题",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [],
                "annotation_uncertain": False,
            }
        ]
    }
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="response-fixed-question",
            system_fingerprint="fingerprint",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    frozen_question = "本合同承担什么责任？"
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-b",
            model_id="openai/gpt-oss-20b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="none",
            max_output_tokens=4096,
        ),
        prompt_text="Use a supplied question exactly.",
        case_requests=lambda _source: (
            EvidenceCaseRequest(
                "slot-1", "single_sentence", question=frozen_question
            ),
        ),
    )

    draft = annotation_pass(_source())[0]

    assert draft.question == frozen_question
    request_text = client.requests[0]["input"][1]["content"]
    assert frozen_question in request_text


def test_cross_page_case_is_uncertain_when_evidence_uses_only_one_page():
    payload = {
        "cases": [
            {
                "slot_id": "slot-cross-page",
                "question": "跨页条款的条件是什么？",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [],
                "annotation_uncertain": False,
            }
        ]
    }
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="response-cross-page",
            system_fingerprint="fingerprint",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=4096,
            max_retries=0,
        ),
        prompt_text="Return cross-page evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-cross-page", "cross_page_clause"),
        ),
    )

    draft = annotation_pass(_source())[0]

    assert draft.annotation_uncertain is True
    assert draft.evidence_spans == ()


def test_annotation_discards_unrequested_hard_negative_spans():
    payload = {
        "cases": [
            {
                "slot_id": "slot-1",
                "question": "本合同承担什么责任？",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [
                    {"page_number": 1, "quote": "第一条"}
                ],
                "annotation_uncertain": False,
            }
        ]
    }
    annotation_pass = AnnotationPass(
        client=FakeAnnotationClient(
            AnnotationCompletion(
                response_id="extra-hard-negative",
                system_fingerprint=None,
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=100,
                completion_tokens=20,
            )
        ),
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=8192,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
    )

    draft = annotation_pass(_source())[0]

    assert draft.annotation_uncertain is False
    assert draft.hard_negative_category is None
    assert draft.hard_negative_spans == ()


def test_annotation_reuses_matching_local_checkpoint():
    payload = {
        "cases": [
            {
                "slot_id": "slot-1",
                "question": "本合同承担什么责任？",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [],
                "annotation_uncertain": False,
            }
        ]
    }
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="checkpointed-response",
            system_fingerprint="fingerprint-1",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    checkpoint_store = JsonAnnotationCheckpointStore(
        ROOT / "tmp" / "annotation-checkpoint-tests" / uuid4().hex
    )
    config = AnnotationPassConfig(
        annotator_id="annotator-a",
        model_id="openai/gpt-oss-120b",
        prompt_version="silver-evidence-annotator/v1.0.0",
        schema_version="silver-evidence-schema/v1.0.0",
        reasoning_effort="low",
        max_output_tokens=8192,
    )
    annotation_pass = AnnotationPass(
        client=client,
        config=config,
        prompt_text="Return exact normalized evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
        checkpoint_store=checkpoint_store,
    )

    first = annotation_pass(_source())
    second = annotation_pass(_source())

    assert first == second
    assert first[0].response_metadata.response_id == "checkpointed-response"
    assert len(client.requests) == 1


def test_annotation_binds_single_response_to_single_requested_slot():
    valid_case = {
        "slot_id": "slot-1",
        "question": "本合同承担什么责任？",
        "evidence_spans": [{"page_number": 1, "quote": "责任 本合同负责。"}],
        "hard_negative_spans": [],
        "annotation_uncertain": False,
    }
    completion = AnnotationCompletion(
        response_id="provider-slot",
        system_fingerprint=None,
        content=json.dumps({"cases": [{**valid_case, "slot_id": "wrong"}]}),
        prompt_tokens=100,
        completion_tokens=20,
    )

    class SequencedClient:
        def __init__(self):
            self.requests = []

        def create_completion(self, **request):
            self.requests.append(request)
            return completion

    client = SequencedClient()
    checkpoint_root = ROOT / "tmp" / "annotation-checkpoint-tests" / uuid4().hex
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-b",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="none",
            max_output_tokens=4096,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
        checkpoint_store=JsonAnnotationCheckpointStore(checkpoint_root),
    )

    draft = annotation_pass(_source())[0]

    assert len(client.requests) == 1
    assert draft.slot_id == "slot-1"
    assert draft.response_metadata.response_id == "provider-slot"
    records = list(checkpoint_root.rglob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["completion"][
        "response_id"
    ] == "provider-slot"


def test_openai_compatible_client_calls_deepseek_responses_api():
    requests: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                id="deepseek-response",
                system_fingerprint="deepseek-fingerprint",
                model="deepseek-v4-flash-0731",
                status="completed",
                incomplete_details=None,
                output_text='{"cases": []}',
                usage=SimpleNamespace(input_tokens=12, output_tokens=3),
            )

    sdk_client = SimpleNamespace(responses=FakeCompletions())
    client = OpenAIDeepSeekResponsesClient(sdk_client)

    completion = client.create_completion(
        model="deepseek-v4-flash",
        input=[{"role": "user", "content": "fixture"}],
        reasoning={"effort": "low"},
    )

    assert completion.response_id == "deepseek-response"
    assert completion.prompt_tokens == 12
    assert completion.returned_model == "deepseek-v4-flash-0731"
    assert completion.response_status == "completed"
    assert requests[0]["reasoning"] == {"effort": "low"}
    assert "extra_body" not in requests[0]


def test_annotation_exhausted_empty_output_becomes_uncertain_without_checkpoint(
    monkeypatch,
):
    monkeypatch.setattr(annotation_module.time, "sleep", lambda _seconds: None)
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="incomplete-response",
            system_fingerprint=None,
            content="",
            prompt_tokens=100,
            completion_tokens=4096,
            returned_model="deepseek-v4-flash-0731",
            response_status="incomplete",
            incomplete_reason="max_output_tokens",
        )
    )
    checkpoint_root = ROOT / "tmp" / "annotation-checkpoint-tests" / uuid4().hex
    checkpoint_store = JsonAnnotationCheckpointStore(checkpoint_root)
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="adjudicator-c",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-adjudicator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="high",
            max_output_tokens=4096,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
        checkpoint_store=checkpoint_store,
    )

    result = annotation_pass(_source())

    assert len(client.requests) == 3
    assert result[0].annotation_uncertain is True
    assert result[0].evidence_spans == ()
    assert list(checkpoint_root.rglob("*.json")) == []


def test_annotation_exhausted_invalid_json_becomes_uncertain_without_checkpoint(
    monkeypatch,
):
    monkeypatch.setattr(annotation_module.time, "sleep", lambda _seconds: None)
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="invalid-json",
            system_fingerprint=None,
            content="not-json",
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    checkpoint_root = ROOT / "tmp" / "annotation-checkpoint-tests" / uuid4().hex
    annotation_pass = AnnotationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="annotator-a",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="low",
            max_output_tokens=8192,
            max_retries=2,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
        checkpoint_store=JsonAnnotationCheckpointStore(checkpoint_root),
    )

    result = annotation_pass(_source())

    assert len(client.requests) == 3
    assert result[0].annotation_uncertain is True
    assert list(checkpoint_root.rglob("*.json")) == []


def test_adjudicator_uses_anonymous_drafts_and_returns_mapped_span():
    payload = {
        "cases": [
            {
                "slot_id": "provider-generated-slot",
                "question": "本合同承担什么责任？",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [],
                "annotation_uncertain": False,
            }
        ]
    }
    client = FakeAnnotationClient(
        AnnotationCompletion(
            response_id="adjudication-response",
            system_fingerprint="adjudication-fingerprint",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=150,
            completion_tokens=30,
        )
    )
    first = AnnotationDraft(
        question="第一份问题",
        evidence_spans=(),
        stratum="single_sentence",
        hard_negative_category=None,
        metadata=AnnotationMetadata(
            "annotator-a", "model-a", "prompt-v1", (("temperature", 0),)
        ),
        annotation_uncertain=True,
    )
    second = AnnotationDraft(
        question="第二份问题",
        evidence_spans=(),
        stratum="single_sentence",
        hard_negative_category=None,
        metadata=AnnotationMetadata(
            "annotator-b", "model-b", "prompt-v1", (("temperature", 0),)
        ),
        annotation_uncertain=True,
    )
    adjudicator = AdjudicationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="adjudicator-c",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-adjudicator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="high",
            max_output_tokens=16384,
        ),
        prompt_text="Resolve draft_a and draft_b against normalized source text.",
    )

    result = adjudicator(_source(), first, second)

    assert result.annotation_uncertain is False
    assert result.slot_id == first.slot_id
    assert result.evidence_spans[0].quote == "责任  本合同负责。"
    assert result.response_metadata.response_id == "adjudication-response"
    assert set(result.response_metadata.draft_order) == {
        "annotator-a",
        "annotator-b",
    }
    user_content = client.requests[0]["input"][1]["content"]
    assert "draft_a" in user_content
    assert "draft_b" in user_content
    assert "model-a" not in user_content
    assert "model-b" not in user_content


def test_adjudicator_accepts_uncertain_after_reviewing_source_backed_drafts():
    completion = AnnotationCompletion(
        response_id="adjudication-uncertain",
        system_fingerprint=None,
        content=json.dumps(
            {"cases": [{
            "slot_id": "adjudication",
            "question": "本合同承担什么责任？",
            "evidence_spans": [],
            "hard_negative_spans": [],
            "annotation_uncertain": True,
            }]},
            ensure_ascii=False,
        ),
        prompt_tokens=100,
        completion_tokens=20,
    )
    client = FakeAnnotationClient(completion)
    exact = AnnotationDraft(
        question="本合同承担什么责任？",
        evidence_spans=(EvidenceSpan(1, 3, 12, "责任  本合同负责。"),),
        stratum="single_sentence",
        hard_negative_category=None,
        metadata=AnnotationMetadata(
            "annotator-a", "model-a", "prompt-v1", (("temperature", 0),)
        ),
        slot_id="slot-1",
    )
    uncertain = replace(
        exact,
        evidence_spans=(),
        annotation_uncertain=True,
        metadata=replace(exact.metadata, annotator_id="annotator-b"),
    )
    adjudicator = AdjudicationPass(
        client=client,
        config=AnnotationPassConfig(
            annotator_id="adjudicator-c",
            model_id="deepseek-v4-flash",
            prompt_version="silver-evidence-adjudicator/v1.1.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="high",
            max_output_tokens=16384,
            max_retries=2,
        ),
        prompt_text="Use uncertain only when neither draft has exact evidence.",
    )

    result = adjudicator(_source(), exact, uncertain)

    assert len(client.requests) == 1
    assert result.annotation_uncertain is True
    assert result.response_metadata.response_id == "adjudication-uncertain"
