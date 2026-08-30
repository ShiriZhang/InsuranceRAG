from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from insurance_rag.silver_generation import (
    load_silver_generation_config,
    write_json_atomic,
)


ROOT = Path(__file__).parents[1]


def test_committed_silver_generation_config_freezes_issue_16_decisions():
    config = load_silver_generation_config(
        ROOT / "configs" / "silver_dataset_v2.json"
    )

    assert config.document_split_version == "silver-document-split/v1.0.0"
    assert config.benchmark_version == "silver-evidence-benchmark/v2.0.0"
    assert config.release_version == "clause-v2-silver/v2.0.0"
    assert config.annotator_a.model_id == "deepseek-v4-flash"
    assert config.annotator_b.model_id == "deepseek-v4-flash"
    assert config.adjudicator.model_id == "deepseek-v4-flash"
    assert config.annotator_a.reasoning_effort == "low"
    assert config.annotator_b.reasoning_effort == "none"
    assert config.adjudicator.reasoning_effort == "high"
    assert config.annotator_a.max_output_tokens == 8192
    assert config.annotator_b.max_output_tokens == 4096
    assert config.adjudicator.max_output_tokens == 16384
    assert config.size_grid == ((600, 900), (900, 1200), (1200, 1600))
    assert config.context_token_budgets == (2000, 4000, 8000)
    assert config.primary_context_token_budget == 4000
    assert config.overlap_variants == (
        "zero_body_overlap",
        "preceding_semantic_unit",
    )
    assert config.annotation_input_char_limit == 100_000
    assert config.annotation_window_char_limit == 1_400


def test_generation_config_rejects_prompt_version_mismatch():
    payload = json.loads(
        (ROOT / "configs" / "silver_dataset_v2.json").read_text(
            encoding="utf-8"
        )
    )
    payload["passes"]["annotator_a"]["prompt_version"] = "wrong-version"
    path = ROOT / "tmp" / "silver-generation-tests" / uuid4().hex / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt_version"):
        load_silver_generation_config(path)


def test_generation_config_rejects_schema_version_mismatch():
    payload = json.loads(
        (ROOT / "configs" / "silver_dataset_v2.json").read_text(
            encoding="utf-8"
        )
    )
    payload["passes"]["adjudicator"]["schema_version"] = "wrong-schema"
    path = ROOT / "tmp" / "silver-generation-tests" / uuid4().hex / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_silver_generation_config(path)


def test_generation_config_rejects_normalization_implementation_mismatch():
    payload = json.loads(
        (ROOT / "configs" / "silver_dataset_v2.json").read_text(
            encoding="utf-8"
        )
    )
    payload["versions"]["normalization"] = "normalized-page-text/v2.0.0"
    path = ROOT / "tmp" / "silver-generation-tests" / uuid4().hex / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization version"):
        load_silver_generation_config(path)


def test_write_json_atomic_preserves_immutable_frozen_document():
    path = (
        ROOT
        / "tmp"
        / "silver-generation-tests"
        / uuid4().hex
        / "nested"
        / "manifest.json"
    )

    write_json_atomic(path, {"phase": "split_frozen"})

    with pytest.raises(ValueError, match="immutable frozen artifact"):
        write_json_atomic(path, {"phase": "complete", "count": 176})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "phase": "split_frozen"
    }
    assert not path.with_suffix(".tmp").exists()
