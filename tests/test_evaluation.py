import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import fitz
import pytest

from insurance_rag.config import AppConfig
from insurance_rag.evaluation import (
    DeterministicEvalEmbedder,
    evaluate_hard_negative_cases,
    evaluate_local_hard_negative_documents,
    evaluate_local_documents,
    evaluate_synthetic_cases,
    render_hard_negative_markdown_report,
    render_local_markdown_report,
    render_markdown_report,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals" / "synthetic_cases.json"
HARD_NEGATIVE_CASES_PATH = ROOT / "evals" / "hard_negative_cases.json"


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


def test_hard_negative_evaluation_passes_repo_cases():
    report = evaluate_hard_negative_cases(HARD_NEGATIVE_CASES_PATH)

    assert report.total_cases == 4
    assert report.passed_cases == report.total_cases
    assert all(result.positive_rank is not None for result in report.results)
    assert all(
        result.verifier_status in {"pass", "warn", "block"}
        for result in report.results
    )


def test_hard_negative_report_contains_rerank_and_verifier_details():
    report = evaluate_hard_negative_cases(HARD_NEGATIVE_CASES_PATH)
    markdown = render_hard_negative_markdown_report(report)

    assert "# InsuranceRAG Hard Negative Evaluation Report" in markdown
    assert "Positive Rank" in markdown
    assert "Verifier" in markdown
    assert "hard_negative_waiting_period_number" in markdown


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


def test_local_document_evaluation_scores_real_pdf_terms():
    docs_dir = _repo_tmp_dir("local-docs")
    _write_pdf(
        docs_dir / "sample.pdf",
        "第六条 等待期\n等待期为九十日。\n第七条 保险责任\n本合同承担重大疾病保险责任。",
    )

    report = evaluate_local_documents(docs_dir, sample_limit=1, top_k=3)

    assert report.total_documents == 1
    assert report.parsed_documents == 1
    assert report.total_cases >= 2
    assert report.top3_cases == report.total_cases
    assert report.unknown_chunk_rate < 1.0
    markdown = render_local_markdown_report(report)
    assert "# InsuranceRAG Local Document Evaluation Report" in markdown
    assert "Top3" in markdown
    assert "等待期" in markdown


def test_local_document_evaluation_uses_selected_chunking_strategy():
    docs_dir = _repo_tmp_dir("local-docs-clause-v2")
    _write_pdf(
        docs_dir / "sample.pdf",
        "第六条 等待期\n等待期为九十日。\n第七条 保险责任\n本合同承担重大疾病保险责任。",
    )
    config = AppConfig(
        openai_api_key=None,
        ocr_enabled=False,
        chunking_strategy="clause_v2",
    )

    report = evaluate_local_documents(
        docs_dir,
        sample_limit=1,
        top_k=3,
        config=config,
    )

    assert report.total_chunks == 2


def test_local_hard_negative_evaluation_builds_cases_from_pdf():
    docs_dir = _repo_tmp_dir("local-hard-negative-docs")
    _write_pdf(
        docs_dir / "sample.pdf",
        "第六条 等待期\n等待期为九十日。\n第七条 保险期间\n保险期间为一年。\n第八条 责任免除\n酒后驾驶属于责任免除。",
    )

    report = evaluate_local_hard_negative_documents(docs_dir, sample_limit=1)

    assert report.total_documents == 1
    assert report.parsed_documents == 1
    assert report.total_cases >= 1


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


def test_cli_explicit_missing_local_documents_returns_one_without_synthetic():
    workspace_tmp = _repo_tmp_dir("missing-local-documents-only")
    missing_path = workspace_tmp / "missing_docs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rag.py",
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

    assert completed.returncode == 1
    assert "does not exist" in completed.stdout


def test_cli_existing_local_documents_writes_local_report():
    workspace_tmp = _repo_tmp_dir("existing-local-documents")
    docs_dir = workspace_tmp / "docs"
    docs_dir.mkdir()
    _write_pdf(
        docs_dir / "sample.pdf",
        "第六条 等待期\n等待期为九十日。\n第七条 保险责任\n本合同承担重大疾病保险责任。",
    )
    report_dir = workspace_tmp / "reports"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rag.py",
            "--synthetic",
            "--report-dir",
            str(report_dir),
            "--local-documents",
            str(docs_dir),
            "--local-sample-limit",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report_path = report_dir / "local_document_eval_report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# InsuranceRAG Local Document Evaluation Report" in report_text
    assert "sample.pdf" in report_text


def test_cli_existing_local_hard_negative_documents_writes_report():
    workspace_tmp = _repo_tmp_dir("existing-local-hard-negative-documents")
    docs_dir = workspace_tmp / "docs"
    docs_dir.mkdir()
    _write_pdf(
        docs_dir / "sample.pdf",
        "第六条 等待期\n等待期为九十日。\n第七条 保险期间\n保险期间为一年。\n第八条 责任免除\n酒后驾驶属于责任免除。",
    )
    report_dir = workspace_tmp / "reports"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rag.py",
            "--report-dir",
            str(report_dir),
            "--local-hard-negative",
            str(docs_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report_path = report_dir / "local_hard_negative_eval_report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# InsuranceRAG Local Document Evaluation Report" in report_text
    assert "sample.pdf" in report_text


def _repo_tmp_dir(name: str) -> Path:
    path = ROOT / "tmp" / "eval-tests" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_cases(name: str, cases: list[dict[str, object]]) -> Path:
    path = _repo_tmp_dir(name) / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def _write_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontname="china-s", fontsize=11)
    path.write_bytes(document.tobytes())
    document.close()
