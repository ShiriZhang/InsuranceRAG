from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_rag.clause_parser import parse_clause_metadata
from insurance_rag.config import AppConfig
from insurance_rag.silver_corpus import (
    build_approved_corpus_inventory,
    freeze_insurer_document_split,
)
from insurance_rag.silver_dataset import DatasetSplit
from insurance_rag.silver_generation import load_silver_generation_config


def main() -> int:
    config = load_silver_generation_config(
        ROOT / "configs" / "silver_dataset_v2.json"
    )
    inventory = build_approved_corpus_inventory(
        ROOT / "documents",
        approval_reference=config.approval_reference,
        parse_config=AppConfig(openai_api_key=None, ocr_enabled=False),
    )
    split = freeze_insurer_document_split(
        inventory,
        version=config.document_split_version,
        held_out_fraction=config.held_out_fraction,
        seed=config.split_seed,
    )
    semantic_lengths: list[int] = []
    clause_lengths: list[int] = []
    for source in split.sources_for(DatasetSplit.DEVELOPMENT):
        current_clause: list[str] = []
        for page in source.pages:
            normalized = " ".join(page.text.split())
            semantic_lengths.extend(
                len(unit.strip())
                for unit in re.split(r"(?<=[。！？；])|\n+", normalized)
                if unit.strip()
            )
            for line in (line.strip() for line in page.text.splitlines()):
                if not line:
                    continue
                metadata = parse_clause_metadata(line)
                if (
                    metadata.heading_confidence in {"high", "medium"}
                    and current_clause
                ):
                    clause_lengths.append(len(" ".join(current_clause)))
                    current_clause = []
                current_clause.append(line)
        if current_clause:
            clause_lengths.append(len(" ".join(current_clause)))

    payload = {
        "analysis_version": "silver-size-grid-analysis/v1.0.0",
        "document_split_manifest_sha256": split.manifest_sha256,
        "split": DatasetSplit.DEVELOPMENT.value,
        "source_count": len(split.sources_for(DatasetSplit.DEVELOPMENT)),
        "clause_recognizer": "insurance_rag.clause_parser.parse_clause_metadata",
        "semantic_unit_rule": "split-after-Chinese-terminal-punctuation/v1",
        "semantic_units": _distribution(semantic_lengths),
        "approximate_policy_clauses": _distribution(clause_lengths),
        "selected_size_grid": [list(pair) for pair in config.size_grid],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _distribution(values: list[int]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "percentiles": {
            str(percentile): ordered[round((len(ordered) - 1) * percentile / 100)]
            for percentile in (50, 75, 90, 95, 99)
        },
        "over_threshold": {
            str(threshold): sum(value > threshold for value in ordered)
            for threshold in (900, 1200, 1600)
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
