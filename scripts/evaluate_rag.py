from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_rag.config import AppConfig
from insurance_rag.evaluation import evaluate_synthetic_cases, render_markdown_report


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
    args = parser.parse_args(argv)

    if not args.synthetic:
        print("No evaluation selected. Use --synthetic.")
        return 2

    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.local_documents is not None:
        if not args.local_documents.exists():
            print(
                f"Skipping local document evaluation: {args.local_documents} does not exist."
            )
        else:
            print(
                "Local document evaluation is optional; synthetic evaluation remains the CI target."
            )

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
    return 0 if report.passed_cases == report.total_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
