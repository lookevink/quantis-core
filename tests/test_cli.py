import subprocess
import sys
import json

from tests.corpus_test_support import (
    FRESH_CASE_IDS,
    write_fresh_development_runs,
)
from tests.multimodal_test_support import (
    write_normal_log_captures,
)


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


def test_train_jepa_command_writes_development_artifacts(tmp_path):
    (
        captures_directory,
        manifests_directory,
        feature_spec_path,
    ) = write_fresh_development_runs(tmp_path / "fresh-corpus")
    split_path = tmp_path / "split.json"
    output_path = tmp_path / "output"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_case_ids": list(FRESH_CASE_IDS[:2]),
                "validation_case_ids": [FRESH_CASE_IDS[2]],
                "reserved_case_ids": [],
                "lookback": 6,
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantis_core",
            "train-jepa-world-model",
            "--captures-directory",
            str(captures_directory),
            "--manifests-directory",
            str(manifests_directory),
            "--feature-spec",
            str(feature_spec_path),
            "--split-spec",
            str(split_path),
            "--epochs",
            "20",
            "--latent-dimension",
            "3",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "JEPA development training: PASS" in completed.stdout
    assert (output_path / "corpus.json").exists()
    assert (output_path / "model.json").exists()
    assert (output_path / "development.json").exists()
    assert (output_path / "report.md").exists()


def test_train_multimodal_jepa_command_uses_application_logs(tmp_path):
    (
        captures_directory,
        manifests_directory,
        feature_spec_path,
    ) = write_fresh_development_runs(
        tmp_path / "fresh-multimodal-corpus"
    )
    write_normal_log_captures(
        captures_directory,
        manifests_directory,
    )
    split_path = tmp_path / "multimodal-split.json"
    output_path = tmp_path / "multimodal-output"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_case_ids": list(FRESH_CASE_IDS[:2]),
                "validation_case_ids": [FRESH_CASE_IDS[2]],
                "reserved_case_ids": [],
                "lookback": 6,
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantis_core",
            "train-multimodal-jepa-world-model",
            "--captures-directory",
            str(captures_directory),
            "--manifests-directory",
            str(manifests_directory),
            "--metric-feature-spec",
            str(feature_spec_path),
            "--log-feature-spec",
            "lab/fault_matrix/log-feature-spec.json",
            "--split-spec",
            str(split_path),
            "--epochs",
            "20",
            "--metric-latent-dimension",
            "3",
            "--log-latent-dimension",
            "2",
            "--maximum-validation-alert-rate",
            "0.10",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Multimodal JEPA development training: PASS" in (
        completed.stdout
    )
    assert (output_path / "corpus.json").exists()
    assert (output_path / "model.json").exists()
    assert (output_path / "metrics-only-model.json").exists()
    assert (
        output_path
        / "capacity-matched-metrics-only-model.json"
    ).exists()
    assert (output_path / "shuffled-log-model.json").exists()
    assert (output_path / "development.json").exists()
    assert (output_path / "report.md").exists()
