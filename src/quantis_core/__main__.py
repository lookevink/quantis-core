"""Command-line entry point for reproducible Quantis experiments."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .evaluation import (
    EvaluationConfig,
    run_evaluation,
    write_evaluation_artifacts,
)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quantis_core")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="run the synthetic thesis test")
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="directory for reports and fitted artifacts",
    )
    evaluate.add_argument(
        "--quick",
        action="store_true",
        help="use four held-out scenarios for a fast CI evaluation",
    )
    parsed = parser.parse_args(arguments)

    if parsed.command == "evaluate":
        config = (
            EvaluationConfig(
                train_seeds=(11, 23, 37),
                test_seeds=(101, 103, 107, 109),
                scenario_length=360,
            )
            if parsed.quick
            else EvaluationConfig()
        )
        report = run_evaluation(config)
        paths = write_evaluation_artifacts(report, parsed.output)
        status = "PASS" if report.acceptance["all_passed"] else "FAIL"
        print(f"Acceptance: {status}")
        print(f"Report: {paths['report']}")
        return 0 if report.acceptance["all_passed"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
