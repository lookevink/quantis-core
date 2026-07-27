import subprocess
import sys
import json


def test_evaluate_command_writes_evidence_and_reports_gate_status(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantis_core",
            "evaluate",
            "--quick",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Acceptance: PASS" in completed.stdout
    assert (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "report.md").exists()


def test_replay_otlp_command_writes_compiled_values_and_quality_evidence(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantis_core",
            "replay-otlp",
            "--capture",
            "tests/fixtures/otlp/semantic-metrics.jsonl",
            "--feature-spec",
            "tests/fixtures/otlp/semantic-feature-spec.json",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "OTLP replay: PASS" in completed.stdout
    summary = json.loads((tmp_path / "replay.json").read_text())
    assert summary["data_quality"] == {
        "flagged_points": 1,
        "missing_cells": 11,
        "reset_points": 2,
    }
    assert (tmp_path / "compiled-telemetry.json").exists()
