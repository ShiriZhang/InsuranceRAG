from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
from openai import OpenAI

from insurance_rag.config import AppConfig
from insurance_rag.groq_silver_annotation import (
    GroqAdjudicationPass,
    GroqAnnotationPass,
    JsonAnnotationCheckpointStore,
    OpenAIGroqChatClient,
)
from insurance_rag.silver_benchmark import generate_frozen_benchmark
from insurance_rag.silver_corpus import (
    build_adjudication_source_window,
    build_annotation_source_window,
    build_approved_corpus_inventory,
    build_evidence_case_plan,
    freeze_insurer_document_split,
)
from insurance_rag.silver_dataset import freeze_silver_datasets
from insurance_rag.silver_generation import (
    load_silver_generation_config,
    write_json_atomic,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the approved local PDF corpus and generate Silver labels with Groq."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "silver_dataset_v1.json",
    )
    parser.add_argument("--documents", type=Path, default=ROOT / "documents")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "silver_dataset_manifests",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "silver_benchmark_data" / "groq_checkpoints_v1",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze and write the corpus split without making API calls.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        help="Generate a resumable pilot for the first N frozen sources; release gates are not run.",
    )
    args = parser.parse_args(argv)
    if args.source_limit is not None and args.source_limit <= 0:
        parser.error("--source-limit must be positive")

    config_path = _absolute(args.config)
    documents_dir = _absolute(args.documents)
    output_dir = _absolute(args.output_dir)
    checkpoint_dir = _absolute(args.checkpoint_dir)
    _require_git_ignored_output(output_dir)
    _require_git_ignored_output(checkpoint_dir)
    config = load_silver_generation_config(config_path)

    inventory = build_approved_corpus_inventory(
        documents_dir,
        approval_reference=config.approval_reference,
        parse_config=AppConfig(openai_api_key=None, ocr_enabled=False),
    )
    document_split = freeze_insurer_document_split(
        inventory,
        version=config.document_split_version,
        held_out_fraction=config.held_out_fraction,
        seed=config.split_seed,
    )
    split_path = output_dir / "document_split_v1.json"
    write_json_atomic(split_path, document_split.to_manifest())
    print(
        f"Frozen split before inference: {len(inventory.sources)} PDFs, "
        f"sha256={document_split.manifest_sha256}"
    )
    if args.prepare_only:
        return 0

    load_dotenv(ROOT / ".env", override=False)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required; OPENAI_API_KEY is not used.")

    annotator_prompt = (ROOT / "prompts" / "silver_evidence_annotator_v1.md").read_text(
        encoding="utf-8"
    )
    adjudicator_prompt = (
        ROOT / "prompts" / "silver_evidence_adjudicator_v1.md"
    ).read_text(encoding="utf-8")
    schema_record = json.loads(
        (ROOT / "schemas" / "silver_evidence_v1.json").read_text(encoding="utf-8")
    )
    if schema_record.pop("$id", None) != config.schema_version:
        raise ValueError("Schema $id does not match the frozen schema version.")

    sdk_client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=120.0,
        max_retries=0,
    )
    client = OpenAIGroqChatClient(sdk_client)
    checkpoint_store = JsonAnnotationCheckpointStore(checkpoint_dir)
    case_plan = build_evidence_case_plan(
        document_split,
        development_cases_per_source=config.development_cases_per_source,
        held_out_cases_per_source=config.held_out_cases_per_source,
    )
    frozen_questions: dict[str, tuple[str, ...]] = {}
    annotation_windows = {}
    first_pass = _windowed_annotation_pass(
        client=client,
        pass_config=config.annotator_a,
        prompt_text=annotator_prompt,
        case_plan=case_plan,
        checkpoint_store=checkpoint_store,
        response_schema=schema_record,
        input_char_limit=config.annotation_input_char_limit,
        window_char_limit=config.annotation_window_char_limit,
        output_questions=frozen_questions,
        output_windows=annotation_windows,
    )
    second_pass = _windowed_annotation_pass(
        client=client,
        pass_config=config.annotator_b,
        prompt_text=annotator_prompt,
        case_plan=case_plan,
        checkpoint_store=checkpoint_store,
        response_schema=schema_record,
        input_char_limit=config.annotation_input_char_limit,
        window_char_limit=config.annotation_window_char_limit,
        fixed_questions=frozen_questions,
    )
    adjudicator = _windowed_adjudication_pass(
        client=client,
        pass_config=config.adjudicator,
        prompt_text=adjudicator_prompt,
        checkpoint_store=checkpoint_store,
        response_schema=schema_record,
        input_char_limit=config.annotation_input_char_limit,
        window_char_limit=config.annotation_window_char_limit,
        fallback_windows=annotation_windows,
    )

    sources = document_split.sources
    if args.source_limit is not None:
        sources = sources[: args.source_limit]
    benchmark = generate_frozen_benchmark(
        sources=sources,
        first_pass=_with_progress("annotator_a", first_pass, len(sources)),
        second_pass=_with_progress("annotator_b", second_pass, len(sources)),
        adjudication_pass=adjudicator,
        config=config.benchmark_config(),
    )

    runtime = {
        "provider": "groq",
        "endpoint": "https://api.groq.com/openai/v1",
        "openai_sdk_version": package_version("openai"),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "document_split_manifest_sha256": document_split.manifest_sha256,
        "benchmark_manifest_sha256": benchmark.manifest_sha256,
        "source_count": len(sources),
        "case_count": len(benchmark.cases),
        "checkpoint_dir": checkpoint_dir.relative_to(ROOT).as_posix(),
    }
    if args.source_limit is not None:
        write_json_atomic(
            output_dir / "pilot_benchmark_v1.json",
            benchmark.to_manifest(),
            allow_replace=True,
        )
        write_json_atomic(
            output_dir / "pilot_run_summary_v1.json",
            runtime,
            allow_replace=True,
        )
        print(f"Pilot complete: {len(sources)} sources, {len(benchmark.cases)} cases")
        return 0

    frozen = freeze_silver_datasets(
        benchmark=benchmark,
        document_split=document_split,
        config=config.freeze_config(),
    )
    write_json_atomic(output_dir / "silver_release_v1.json", frozen.to_manifest())
    runtime["release_manifest_sha256"] = frozen.manifest_sha256
    runtime["freeze_report"] = asdict(frozen.report)
    write_json_atomic(output_dir / "formal_run_summary_v1.json", runtime)
    print(
        f"Formal Silver release complete: {len(benchmark.cases)} cases, "
        f"sha256={frozen.manifest_sha256}"
    )
    return 0


def _with_progress(name, annotation_pass, total):
    completed = 0

    def run(source):
        nonlocal completed
        result = annotation_pass(source)
        completed += 1
        print(f"{name}: {completed}/{total} sources ({source.source_name})", flush=True)
        return result

    return run


def _windowed_annotation_pass(
    *,
    client,
    pass_config,
    prompt_text,
    case_plan,
    checkpoint_store,
    response_schema,
    input_char_limit,
    window_char_limit,
    fixed_questions=None,
    output_questions=None,
    output_windows=None,
):
    def run(source):
        requests = case_plan(source)
        if fixed_questions is not None:
            questions = fixed_questions.get(source.source_id)
            if questions is None or len(questions) != len(requests):
                raise ValueError(
                    "Second annotation pass requires frozen first-pass questions."
                )
            requests = tuple(
                replace(request, question=questions[index])
                for index, request in enumerate(requests)
            )
        drafts = []
        for index, request in enumerate(requests):
            window = build_annotation_source_window(
                source,
                slot_id=request.slot_id,
                slot_index=index,
                slot_count=len(requests),
                max_chars=window_char_limit,
                stratum=request.stratum,
            )
            annotation_pass = GroqAnnotationPass(
                client=client,
                config=pass_config,
                prompt_text=prompt_text,
                case_requests=lambda _source, selected=request: (selected,),
                checkpoint_store=checkpoint_store,
                response_schema=response_schema,
                input_char_limit=input_char_limit,
            )
            drafts.extend(annotation_pass(window))
            if output_windows is not None:
                output_windows[(source.source_id, drafts[-1].slot_id)] = window
        result = tuple(drafts)
        if output_questions is not None:
            output_questions[source.source_id] = tuple(
                draft.question for draft in result
            )
        return result

    return run


def _windowed_adjudication_pass(
    *,
    client,
    pass_config,
    prompt_text,
    checkpoint_store,
    response_schema,
    input_char_limit,
    window_char_limit,
    fallback_windows,
):
    adjudicator = GroqAdjudicationPass(
        client=client,
        config=pass_config,
        prompt_text=prompt_text,
        checkpoint_store=checkpoint_store,
        response_schema=response_schema,
        input_char_limit=input_char_limit,
    )

    def run(source, first, second):
        work_item_id = first.metadata.annotator_id + "-" + second.metadata.annotator_id
        window = build_adjudication_source_window(
            source,
            first.evidence_spans,
            second.evidence_spans,
            work_item_id=work_item_id,
            max_chars=window_char_limit,
            fallback_window=fallback_windows.get(
                (source.source_id, first.slot_id)
            ),
        )
        return adjudicator(window, first, second)

    return run


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _require_git_ignored_output(path: Path) -> None:
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            f"Source-bearing Silver output must stay inside the repository: {path}"
        ) from error
    probe = relative / ".source-bearing-output-check"
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", probe.as_posix()],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            "Source-bearing Silver output directory must be covered by .gitignore: "
            f"{relative.as_posix()}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
