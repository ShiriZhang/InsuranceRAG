from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
import tiktoken

from insurance_rag.clause_v2_selection import (
    ClauseV2SelectionManifest,
    render_development_selection_markdown,
    run_development_selection,
)
from insurance_rag.config import AppConfig
from insurance_rag.retriever import OpenAIEmbedder
from insurance_rag.silver_benchmark import Embedder, load_frozen_benchmark_manifest
from insurance_rag.silver_corpus import (
    build_approved_corpus_inventory,
    freeze_insurer_document_split,
)
from insurance_rag.silver_dataset import DatasetSplit
from insurance_rag.silver_generation import (
    load_silver_generation_config,
    write_json_atomic,
)


class TiktokenCounter:
    def __init__(self, tokenizer_id: str) -> None:
        self.tokenizer_id = tokenizer_id
        self._encoding = tiktoken.get_encoding(tokenizer_id)

    def __call__(self, text: str) -> int:
        return len(self._encoding.encode(text))


class CachingEmbedder:
    def __init__(
        self,
        embedder: Embedder,
        model_id: str,
        cache_path: Path | None = None,
    ) -> None:
        self._embedder = embedder
        self.model_id = model_id
        self._cache_path = cache_path
        self._vectors_by_text: dict[str, list[float]] = {}
        self._vectors_by_key: dict[str, list[float]] = {}
        if cache_path is not None and cache_path.exists():
            with cache_path.open(encoding="utf-8") as cache_file:
                for line in cache_file:
                    record = json.loads(line)
                    self._vectors_by_key[str(record["key"])] = list(
                        record["vector"]
                    )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            key = self._key(text)
            if text not in self._vectors_by_text and key in self._vectors_by_key:
                self._vectors_by_text[text] = self._vectors_by_key[key]
        missing = list(
            dict.fromkeys(
                text for text in texts if text not in self._vectors_by_text
            )
        )
        if missing:
            vectors = self._embedder.embed_texts(missing)
            self._vectors_by_text.update(zip(missing, vectors, strict=True))
            if self._cache_path is not None:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_path.open("a", encoding="utf-8") as cache_file:
                    for text, vector in zip(missing, vectors, strict=True):
                        key = self._key(text)
                        self._vectors_by_key[key] = vector
                        cache_file.write(
                            json.dumps(
                                {"key": key, "vector": vector},
                                ensure_ascii=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
        return [self._vectors_by_text[text] for text in texts]

    def _key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.model_id}\0{text}".encode("utf-8")
        ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the three preregistered chunking families on development "
            "labels only and freeze the selected clause_v2 configuration."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "silver_dataset_v2.json",
    )
    parser.add_argument("--documents", type=Path, default=ROOT / "documents")
    parser.add_argument(
        "--development-benchmark",
        type=Path,
        default=(
            ROOT
            / "silver_dataset_manifests"
            / "development_benchmark_v2.json"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "configs" / "clause_v2_selection_v1.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "eval_reports" / "clause_v2_development_selection_v1.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    config = load_silver_generation_config(args.config.resolve())
    inventory = build_approved_corpus_inventory(
        args.documents.resolve(),
        approval_reference=config.approval_reference,
        parse_config=AppConfig(openai_api_key=None, ocr_enabled=False),
    )
    document_split = freeze_insurer_document_split(
        inventory,
        version=config.document_split_version,
        held_out_fraction=config.held_out_fraction,
        seed=config.split_seed,
    )
    development_payload = json.loads(
        args.development_benchmark.resolve().read_text(encoding="utf-8")
    )
    development_benchmark = load_frozen_benchmark_manifest(
        development_payload,
        sources=document_split.sources_for(DatasetSplit.DEVELOPMENT),
    )

    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the frozen embedding model.")
    embedder = CachingEmbedder(
        OpenAIEmbedder(api_key, config.embedding_model_id),
        config.embedding_model_id,
        ROOT / ".rag_eval_cache" / "clause_v2_embeddings_v1.jsonl",
    )
    token_counter = TiktokenCounter(config.tokenizer_id)
    selection = run_development_selection(
        development_benchmark,
        dataset_split=DatasetSplit.DEVELOPMENT,
        size_grid=config.size_grid,
        context_token_budgets=config.context_token_budgets,
        document_split_manifest_sha256=document_split.manifest_sha256,
        embedder=embedder,
        token_counter=token_counter,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_json_atomic(
        args.manifest_output.resolve(), selection.manifest.to_manifest()
    )
    args.report_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_output.resolve().write_text(
        render_development_selection_markdown(selection), encoding="utf-8"
    )
    if isinstance(selection.manifest, ClauseV2SelectionManifest):
        print(
            "Frozen clause_v2 selection: "
            f"{selection.manifest.body_overlap_mode}, "
            f"{selection.manifest.target_chars}/{selection.manifest.hard_max_chars}, "
            f"budget={selection.manifest.context_token_budget}"
        )
    else:
        print("No clause_v2 candidate passed every promotion guardrail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
