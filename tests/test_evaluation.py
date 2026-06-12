import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from insurance_rag.evaluation import (
    DeterministicEvalEmbedder,
    evaluate_synthetic_cases,
    render_markdown_report,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals" / "synthetic_cases.json"


def test_deterministic_embedder_returns_stable_eight_dimensional_vectors():
    embedder = DeterministicEvalEmbedder()

    first = embedder.embed_texts(["等待期", "责任免除"])
    second = embedder.embed_texts(["等待期", "责任免除"])

    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 8 for vector in first)
    assert first[0] != first[1]


def test_synthetic_evaluation_reports_expected_sections_and_passes_cases():
    report = evaluate_synthetic_cases(CASES_PATH, max_expected_rank=1)

    assert report.total_cases == 2
    assert report.passed_cases >= 1
    assert report.passed_cases == report.total_cases
    results_by_id = {result.case_id: result for result in report.results}
    assert results_by_id["synthetic_waiting_period"].expected_section == "等待期"
    assert "等待期" in results_by_id["synthetic_waiting_period"].retrieved_sections
    assert results_by_id["synthetic_exclusion"].expected_section == "责任免除"
    assert "责任免除" in results_by_id["synthetic_exclusion"].retrieved_sections
    assert all(result.retrieved_chunk_ids for result in report.results)
    assert all(result.top_fusion_score > 0 for result in report.results)
    assert all(result.expected_rank == 1 for result in report.results)


def test_synthetic_evaluation_fails_when_expected_evidence_is_after_max_rank():
    cases_path = _write_cases(
        "rank-sensitive",
        [
            {
                "case_id": "rank_sensitive_case",
                "question": "alpha alpha alpha beta",
                "expected_section": "Alpha section",
                "expected_terms": ["alpha"],
                "chunks": [
                    {
                        "chunk_id": "alpha-main",
                        "section_title": "Alpha section",
                        "text": "alpha alpha alpha alpha",
                    },
                    {
                        "chunk_id": "beta-main",
                        "section_title": "Beta section",
                        "text": "beta",
                    },
                ],
            }
        ],
    )

    report = evaluate_synthetic_cases(
        cases_path,
        top_k=2,
        max_expected_rank=1,
    )

    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.results[0].passed is False
    assert report.results[0].expected_rank == 2


def test_synthetic_evaluation_rejects_empty_case_file():
    cases_path = _write_cases("empty", [])

    with pytest.raises(ValueError, match="at least one"):
        evaluate_synthetic_cases(cases_path)


def test_synthetic_evaluation_rejects_malformed_case_with_case_id_and_field():
    cases_path = _write_cases(
        "malformed",
        [
            {
                "case_id": "missing_chunks_case",
                "question": "What is covered?",
                "expected_section": "Coverage",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing_chunks_case.*chunks"):
        evaluate_synthetic_cases(cases_path)


def test_markdown_report_contains_required_summary_and_status_columns():
    report = evaluate_synthetic_cases(CASES_PATH)

    markdown = render_markdown_report(report)

    assert "# InsuranceRAG Evaluation Report" in markdown
    assert "Passed 2 / 2" in markdown
    assert "Fusion Score" in markdown
    assert "PASS" in markdown
    assert "FAIL" in markdown


def test_cli_synthetic_writes_report_to_selected_report_dir():
    report_dir = _repo_tmp_dir("reports")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rag.py",
            "--synthetic",
            "--report-dir",
            str(report_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report_path = report_dir / "synthetic_eval_report.md"
    assert report_path.exists()
    assert "# InsuranceRAG Evaluation Report" in report_path.read_text(encoding="utf-8")
    assert "PASS" in completed.stdout


def test_cli_synthetic_returns_one_for_invalid_cases_file():
    cases_path = _write_cases("cli-empty", [])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rag.py",
            "--synthetic",
            "--cases",
            str(cases_path),
            "--report-dir",
            str(_repo_tmp_dir("cli-empty-report")),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "at least one" in completed.stderr


def test_cli_missing_local_documents_prints_skip_message():
    workspace_tmp = _repo_tmp_dir("missing-local-documents")
    missing_path = workspace_tmp / "missing_docs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rag.py",
            "--synthetic",
            "--report-dir",
            str(workspace_tmp / "reports"),
            "--local-documents",
            str(missing_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Skipping local document evaluation" in completed.stdout


def _repo_tmp_dir(name: str) -> Path:
    path = ROOT / "tmp" / "eval-tests" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_cases(name: str, cases: list[dict[str, object]]) -> Path:
    path = _repo_tmp_dir(name) / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path
