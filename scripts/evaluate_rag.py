from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_rag.config import AppConfig
from insurance_rag.evaluation import (
    evaluate_hard_negative_cases,
    evaluate_local_documents,
    evaluate_synthetic_cases,
    render_hard_negative_markdown_report,
    render_local_markdown_report,
    render_markdown_report,
)


def main(argv: list[str] | None = None) -> int:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Run offline InsuranceRAG evaluations.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "evals" / "synthetic_cases.json",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(config.eval_report_dir),
    )
    parser.add_argument("--local-documents", type=Path)
    parser.add_argument("--local-sample-limit", type=int, default=20)
    parser.add_argument("--hard-negative", action="store_true")
    parser.add_argument(
        "--hard-negative-cases",
        type=Path,
        default=ROOT / "evals" / "hard_negative_cases.json",
    )
    args = parser.parse_args(argv)

    if not args.synthetic and not args.hard_negative and args.local_documents is None:
        print(
            "No evaluation selected. Use --synthetic, --hard-negative, or --local-documents."
        )
        return 2

    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    local_failed = False
    if args.local_documents is not None:
        if not args.local_documents.exists():
            print(
                f"Skipping local document evaluation: {args.local_documents} does not exist."
            )
            local_failed = not args.synthetic
        else:
            local_report = evaluate_local_documents(
                args.local_documents,
                sample_limit=args.local_sample_limit,
            )
            local_markdown = render_local_markdown_report(local_report)
            (report_dir / "local_document_eval_report.md").write_text(
                local_markdown,
                encoding="utf-8",
            )
            print(local_markdown)
            local_failed = (
                local_report.total_cases == 0
                or local_report.top3_cases != local_report.total_cases
            )

    synthetic_failed = False
    if args.synthetic:
        cases_path = args.cases
        if not cases_path.is_absolute():
            cases_path = ROOT / cases_path
        try:
            report = evaluate_synthetic_cases(cases_path)
        except ValueError as exc:
            print(f"Evaluation failed: {exc}", file=sys.stderr)
            return 1
        markdown = render_markdown_report(report)
        (report_dir / "synthetic_eval_report.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        synthetic_failed = report.passed_cases != report.total_cases

    hard_negative_failed = False
    if args.hard_negative:
        cases_path = args.hard_negative_cases
        if not cases_path.is_absolute():
            cases_path = ROOT / cases_path
        try:
            hard_report = evaluate_hard_negative_cases(cases_path)
        except ValueError as exc:
            print(f"Hard negative evaluation failed: {exc}", file=sys.stderr)
            return 1
        hard_markdown = render_hard_negative_markdown_report(hard_report)
        (report_dir / "hard_negative_eval_report.md").write_text(
            hard_markdown,
            encoding="utf-8",
        )
        print(hard_markdown)
        hard_negative_failed = hard_report.passed_cases != hard_report.total_cases

    return 1 if synthetic_failed or local_failed or hard_negative_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
