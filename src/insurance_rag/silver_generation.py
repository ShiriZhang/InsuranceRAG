from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from insurance_rag.silver_annotation import AnnotationPassConfig
from insurance_rag.silver_benchmark import BenchmarkConfig
from insurance_rag.silver_dataset import DatasetFreezeConfig
from insurance_rag.silver_normalization import NORMALIZATION_VERSION


@dataclass(frozen=True)
class SilverGenerationConfig:
    document_split_version: str
    benchmark_version: str
    release_version: str
    normalization_version: str
    schema_version: str
    annotator_prompt_version: str
    adjudicator_prompt_version: str
    approval_reference: str
    held_out_fraction: float
    split_seed: str
    development_cases_per_source: int
    held_out_cases_per_source: int
    annotation_input_char_limit: int
    annotation_window_char_limit: int
    annotator_a: AnnotationPassConfig
    annotator_b: AnnotationPassConfig
    adjudicator: AnnotationPassConfig
    size_grid: tuple[tuple[int, int], ...]
    context_token_budgets: tuple[int, ...]
    primary_context_token_budget: int
    overlap_variants: tuple[str, ...]
    retrieval_depth: int
    embedding_model_id: str
    tokenizer_id: str

    def benchmark_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            version=self.benchmark_version,
            judge_id="exact-span-v1",
            embedding_model_id=self.embedding_model_id,
            query_rewrite_version="production-v1",
            reranker_version="rules-v1",
            retrieval_depth=self.retrieval_depth,
            context_token_budget=self.primary_context_token_budget,
            tokenizer_id=self.tokenizer_id,
        )

    def freeze_config(self) -> DatasetFreezeConfig:
        return DatasetFreezeConfig(
            version=self.release_version,
            benchmark_version=self.benchmark_version,
            document_split_version=self.document_split_version,
            size_grid=self.size_grid,
            context_token_budgets=self.context_token_budgets,
            overlap_variants=self.overlap_variants,
        )


def load_silver_generation_config(path: Path) -> SilverGenerationConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    versions = _mapping(payload, "versions")
    prompts = _mapping(payload, "prompt_versions")
    passes = _mapping(payload, "passes")
    dataset = _mapping(payload, "dataset")
    benchmark = _mapping(payload, "benchmark")

    annotator_prompt_version = str(prompts["annotator"])
    adjudicator_prompt_version = str(prompts["adjudicator"])
    normalization_version = str(versions["normalization"])
    if normalization_version != NORMALIZATION_VERSION:
        raise ValueError(
            "Configured normalization version does not match the executable implementation."
        )
    input_char_limit = int(dataset["annotation_input_char_limit"])
    window_char_limit = int(dataset["annotation_window_char_limit"])
    annotator_a = _pass_config(
        _mapping(passes, "annotator_a"),
        annotator_prompt_version,
        normalization_version,
        input_char_limit,
        window_char_limit,
        str(versions["schema"]),
    )
    annotator_b = _pass_config(
        _mapping(passes, "annotator_b"), annotator_prompt_version,
        normalization_version, input_char_limit, window_char_limit,
        str(versions["schema"]),
    )
    adjudicator = _pass_config(
        _mapping(passes, "adjudicator"), adjudicator_prompt_version,
        normalization_version, input_char_limit, window_char_limit,
        str(versions["schema"]),
    )
    size_grid = tuple(
        (int(pair[0]), int(pair[1])) for pair in dataset["size_grid"]
    )
    budgets = tuple(int(value) for value in dataset["context_token_budgets"])
    primary_budget = int(dataset["primary_context_token_budget"])
    if primary_budget not in budgets:
        raise ValueError("primary_context_token_budget must be in the frozen grid.")

    return SilverGenerationConfig(
        document_split_version=str(versions["document_split"]),
        benchmark_version=str(versions["benchmark"]),
        release_version=str(versions["release"]),
        normalization_version=normalization_version,
        schema_version=str(versions["schema"]),
        annotator_prompt_version=annotator_prompt_version,
        adjudicator_prompt_version=adjudicator_prompt_version,
        approval_reference=str(payload["approval_reference"]),
        held_out_fraction=float(dataset["held_out_fraction"]),
        split_seed=str(dataset["split_seed"]),
        development_cases_per_source=int(
            dataset["development_cases_per_source"]
        ),
        held_out_cases_per_source=int(dataset["held_out_cases_per_source"]),
        annotation_input_char_limit=int(dataset["annotation_input_char_limit"]),
        annotation_window_char_limit=int(dataset["annotation_window_char_limit"]),
        annotator_a=annotator_a,
        annotator_b=annotator_b,
        adjudicator=adjudicator,
        size_grid=size_grid,
        context_token_budgets=budgets,
        primary_context_token_budget=primary_budget,
        overlap_variants=tuple(str(value) for value in dataset["overlap_variants"]),
        retrieval_depth=int(benchmark["retrieval_depth"]),
        embedding_model_id=str(benchmark["embedding_model_id"]),
        tokenizer_id=str(benchmark["tokenizer_id"]),
    )


def write_json_atomic(
    path: Path,
    payload: object,
    *,
    allow_replace: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == rendered:
            return
        if not allow_replace:
            raise ValueError(
                f"Refusing to overwrite immutable frozen artifact: {path}"
            )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    selected = value.get(key)
    if not isinstance(selected, dict):
        raise ValueError(f"Silver generation config requires object {key!r}.")
    return selected


def _pass_config(
    payload: Mapping[str, object],
    expected_prompt_version: str,
    normalization_version: str,
    input_char_limit: int,
    transport_window_char_limit: int,
    expected_schema_version: str,
) -> AnnotationPassConfig:
    prompt_version = str(payload["prompt_version"])
    if prompt_version != expected_prompt_version:
        raise ValueError(
            "Pass prompt_version does not match the frozen prompt_versions section."
        )
    schema_version = str(payload["schema_version"])
    if schema_version != expected_schema_version:
        raise ValueError(
            "Pass schema_version does not match the frozen versions section."
        )
    return AnnotationPassConfig(
        annotator_id=str(payload["annotator_id"]),
        model_id=str(payload["model_id"]),
        prompt_version=prompt_version,
        schema_version=schema_version,
        reasoning_effort=str(payload["reasoning_effort"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        max_retries=int(payload.get("max_retries", 2)),
        normalization_version=normalization_version,
        input_char_limit=input_char_limit,
        transport_window_char_limit=transport_window_char_limit,
    )
