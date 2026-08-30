from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import insurance_rag.groq_silver_annotation as groq_module
from insurance_rag.models import DocumentPage
from insurance_rag.groq_silver_annotation import (
    EvidenceCaseRequest,
    GroqAdjudicationPass,
    GroqAnnotationPass,
    GroqCompletion,
    GroqPassConfig,
    JsonAnnotationCheckpointStore,
    OpenAIGroqChatClient,
)
from insurance_rag.silver_benchmark import (
    AnnotationDraft,
    AnnotationMetadata,
    BenchmarkSource,
    SourceApproval,
)


ROOT = Path(__file__).parents[1]


class FakeGroqClient:
    def __init__(self, completion: GroqCompletion) -> None:
        self.completion = completion
        self.requests: list[dict[str, object]] = []

    def create_completion(self, **request):
        self.requests.append(request)
        return self.completion


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


def test_groq_annotation_maps_unique_normalized_quote_back_to_raw_span():
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
    client = FakeGroqClient(
        GroqCompletion(
            response_id="response-1",
            system_fingerprint="fingerprint-1",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    annotation_pass = GroqAnnotationPass(
        client=client,
        config=GroqPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="medium",
            temperature=0,
            top_p=1,
            seed=16001,
            max_completion_tokens=8192,
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
    assert "责任 本合同负责。" in request["messages"][1]["content"]
    assert "责任  本合同负责。" not in request["messages"][1]["content"]
    assert request["response_format"]["json_schema"]["strict"] is True
    generation_parameters = dict(drafts[0].metadata.generation_parameters)
    assert len(generation_parameters["schema_sha256"]) == 64


def test_groq_annotation_rejects_source_over_frozen_input_limit_before_api():
    client = FakeGroqClient(
        GroqCompletion(
            response_id="unused",
            system_fingerprint=None,
            content='{"cases": []}',
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    annotation_pass = GroqAnnotationPass(
        client=client,
        config=GroqPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="medium",
            temperature=0,
            top_p=1,
            seed=16001,
            max_completion_tokens=8192,
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


def test_groq_annotation_does_not_retry_non_transient_bad_request():
    class BadRequestClient:
        def __init__(self):
            self.calls = 0

        def create_completion(self, **request):
            self.calls += 1
            error = RuntimeError("invalid structured output")
            error.status_code = 400
            raise error

    client = BadRequestClient()
    annotation_pass = GroqAnnotationPass(
        client=client,
        config=GroqPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="medium",
            temperature=0,
            top_p=1,
            seed=16001,
            max_completion_tokens=4096,
        ),
        prompt_text="Return exact evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-1", "single_sentence"),
        ),
    )

    with pytest.raises(RuntimeError, match="invalid structured output"):
        annotation_pass(_source())

    assert client.calls == 1


def test_groq_annotation_waits_for_retry_after_on_429(monkeypatch):
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
            return GroqCompletion(
                response_id="retried",
                system_fingerprint=None,
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=100,
                completion_tokens=20,
            )

    sleeps = []
    monkeypatch.setattr(groq_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(groq_module.random, "uniform", lambda _start, _end: 0.25)
    client = RateLimitedClient()
    annotation_pass = GroqAnnotationPass(
        client=client,
        config=GroqPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="medium",
            temperature=0,
            top_p=1,
            seed=16001,
            max_completion_tokens=4096,
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
    client = FakeGroqClient(
        GroqCompletion(
            response_id="response-fixed-question",
            system_fingerprint="fingerprint",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    frozen_question = "本合同承担什么责任？"
    annotation_pass = GroqAnnotationPass(
        client=client,
        config=GroqPassConfig(
            annotator_id="annotator-b",
            model_id="openai/gpt-oss-20b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="medium",
            temperature=0,
            top_p=1,
            seed=16002,
            max_completion_tokens=4096,
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
    request_text = client.requests[0]["messages"][1]["content"]
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
    client = FakeGroqClient(
        GroqCompletion(
            response_id="response-cross-page",
            system_fingerprint="fingerprint",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    annotation_pass = GroqAnnotationPass(
        client=client,
        config=GroqPassConfig(
            annotator_id="annotator-a",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-annotator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="medium",
            temperature=0,
            top_p=1,
            seed=16001,
            max_completion_tokens=4096,
        ),
        prompt_text="Return cross-page evidence.",
        case_requests=lambda _source: (
            EvidenceCaseRequest("slot-cross-page", "cross_page_clause"),
        ),
    )

    draft = annotation_pass(_source())[0]

    assert draft.annotation_uncertain is True
    assert draft.evidence_spans == ()


def test_groq_annotation_reuses_matching_local_checkpoint():
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
    client = FakeGroqClient(
        GroqCompletion(
            response_id="checkpointed-response",
            system_fingerprint="fingerprint-1",
            content=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    checkpoint_store = JsonAnnotationCheckpointStore(
        ROOT / "tmp" / "groq-checkpoint-tests" / uuid4().hex
    )
    config = GroqPassConfig(
        annotator_id="annotator-a",
        model_id="openai/gpt-oss-120b",
        prompt_version="silver-evidence-annotator/v1.0.0",
        schema_version="silver-evidence-schema/v1.0.0",
        reasoning_effort="medium",
        temperature=0,
        top_p=1,
        seed=16001,
        max_completion_tokens=8192,
    )
    annotation_pass = GroqAnnotationPass(
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


def test_openai_compatible_client_moves_groq_options_into_extra_body():
    requests: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                id="groq-response",
                system_fingerprint="groq-fingerprint",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"cases": []}')
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
            )

    sdk_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    client = OpenAIGroqChatClient(sdk_client)

    completion = client.create_completion(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "fixture"}],
        reasoning_format="hidden",
        service_tier="on_demand",
        temperature=0,
    )

    assert completion.response_id == "groq-response"
    assert completion.prompt_tokens == 12
    assert requests[0]["extra_body"] == {
        "reasoning_format": "hidden",
        "service_tier": "on_demand",
    }
    assert "reasoning_format" not in requests[0]


def test_groq_adjudicator_uses_anonymous_drafts_and_returns_mapped_span():
    payload = {
        "cases": [
            {
                "slot_id": "adjudication",
                "question": "本合同承担什么责任？",
                "evidence_spans": [
                    {"page_number": 1, "quote": "责任 本合同负责。"}
                ],
                "hard_negative_spans": [],
                "annotation_uncertain": False,
            }
        ]
    }
    client = FakeGroqClient(
        GroqCompletion(
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
    adjudicator = GroqAdjudicationPass(
        client=client,
        config=GroqPassConfig(
            annotator_id="adjudicator-c",
            model_id="openai/gpt-oss-120b",
            prompt_version="silver-evidence-adjudicator/v1.0.0",
            schema_version="silver-evidence-schema/v1.0.0",
            reasoning_effort="high",
            temperature=0,
            top_p=1,
            seed=16003,
            max_completion_tokens=16384,
        ),
        prompt_text="Resolve draft_a and draft_b against normalized source text.",
    )

    result = adjudicator(_source(), first, second)

    assert result.annotation_uncertain is False
    assert result.evidence_spans[0].quote == "责任  本合同负责。"
    assert result.response_metadata.response_id == "adjudication-response"
    assert set(result.response_metadata.draft_order) == {
        "annotator-a",
        "annotator-b",
    }
    user_content = client.requests[0]["messages"][1]["content"]
    assert "draft_a" in user_content
    assert "draft_b" in user_content
    assert "model-a" not in user_content
    assert "model-b" not in user_content
