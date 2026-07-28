"""Train and score the frozen action-dynamics development-v1 matrix."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)
from quantis_core.action_dynamics_real_corpus import (
    RealCorpusRun,
    RealCorpusStudyConfig,
    build_development_validation_queries,
    train_and_evaluate_real_corpus,
    write_real_corpus_study_artifacts,
)


def run_development_training(
    *,
    corpus_directory: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Load qualified evidence, fit training only, and score validation."""

    corpus = load_action_dynamics_development_corpus(
        corpus_directory
    )
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite development training: "
            f"{output_directory}"
        )
    graph = corpus.runs[0].graph
    queries = build_development_validation_queries(
        corpus.validation_runs, graph
    )
    admitted = tuple(
        RealCorpusRun(run=run, corpus_role="development")
        for run in corpus.runs
    )
    result = train_and_evaluate_real_corpus(
        runs=admitted,
        graph=graph,
        queries=queries,
        config=RealCorpusStudyConfig(),
    )

    output_directory.mkdir(parents=True)
    study = write_real_corpus_study_artifacts(
        result, output_directory / "study"
    )
    run_manifest: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "action_dynamics_development_training_run",
        "evidence_boundary": (
            "open development only; not confirmation or a "
            "world-model claim"
        ),
        "source_corpus": {
            "directory": str(corpus_directory),
            "identity": corpus.identity.to_dict(),
            "summary": corpus.summary.to_dict(),
        },
        "frozen_config": {
            "context_length": 20,
            "rollout_horizon": 10,
            "ridge": 0.001,
            "variance_floor": 0.0001,
        },
        "query_count": len(queries),
        "query_declaration_sha256": (
            result.query_declaration_sha256
        ),
        "study_artifact_manifest_sha256": (
            study.manifest_sha256
        ),
        "assessment": result.assessment.to_dict(),
    }
    run_manifest_path = output_directory / "run-manifest.json"
    run_manifest_path.write_text(_pretty_json(run_manifest))
    report_path = output_directory / "report.md"
    report_path.write_text(_report(run_manifest))
    artifact_manifest_path = (
        output_directory / "artifact-manifest.json"
    )
    artifact_hashes = {
        path.relative_to(output_directory).as_posix(): _file_sha256(
            path
        )
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path != artifact_manifest_path
    }
    artifact_manifest = {
        "schema_version": 1,
        "kind": "action_dynamics_development_training_artifacts",
        "sha256": artifact_hashes,
    }
    artifact_manifest_path.write_text(
        _pretty_json(artifact_manifest)
    )
    return run_manifest


def _report(run: Mapping[str, Any]) -> str:
    assessment = run["assessment"]
    if not isinstance(assessment, dict):
        raise ValueError("training assessment is invalid")
    measurements = assessment["measurements"]
    gates = assessment["gates"]
    if not isinstance(measurements, dict) or not isinstance(
        gates, dict
    ):
        raise ValueError("training gate evidence is invalid")
    lines = [
        "# Action-dynamics development v1 result",
        "",
        f"Decision: `{assessment['decision']}`",
        "",
        "This is open development evidence, not confirmation or a "
        "world-model claim.",
        "",
        "## Measurements",
        "",
    ]
    lines.extend(
        f"- `{name}`: {float(value):.6f}"
        for name, value in measurements.items()
    )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- `{name}`: {'PASS' if value else 'FAIL'}"
        for name, value in gates.items()
    )
    lines.append("")
    return "\n".join(lines)


def _pretty_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/development-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/development-training-v1"
        ),
    )
    parsed = parser.parse_args(arguments)
    run = run_development_training(
        corpus_directory=parsed.corpus,
        output_directory=parsed.output,
    )
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
