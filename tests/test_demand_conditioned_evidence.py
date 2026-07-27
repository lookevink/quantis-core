import hashlib
import json
import subprocess
from pathlib import Path

from quantis_core.demand_conditioning import (
    canonical_request_schedule,
    train_demand_conditioned_model,
)
from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
    evaluate_demand_conditioned_fault_matrix,
)
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec


def test_v2_model_and_regression_recompute_from_raw_captures() -> None:
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    artifacts = repository / "artifacts"
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab / "feature-spec.json").read_text())
    )
    development_runs = _runs(
        lab / "experiments",
        artifacts / "fault-matrix" / "cases",
    )

    recomputed_model = train_demand_conditioned_model(
        development_runs, feature_spec
    )
    model_path = artifacts / "demand-conditioned-v2" / "model.json"
    checked_model_bytes = model_path.read_bytes()

    assert recomputed_model.to_bytes() == checked_model_bytes
    assert recomputed_model.protocol["training_structural_points"] == 0
    assert recomputed_model.protocol["training_window_count"] == 90
    assert recomputed_model.detector_artifact["threshold"] > 0.0
    model_sha256 = hashlib.sha256(checked_model_bytes).hexdigest()
    training_evidence = json.loads(
        (
            artifacts / "demand-conditioned-v2" / "training.json"
        ).read_text()
    )
    assert training_evidence["model_file_sha256"] == model_sha256
    assert training_evidence["protocol"] == recomputed_model.protocol

    regression = evaluate_demand_conditioned_fault_matrix(
        development_runs, feature_spec, checked_model_bytes
    )
    checked_regression = json.loads(
        (
            artifacts
            / "demand-conditioned-v2"
            / "regression"
            / "verification.json"
        ).read_text()
    )
    assert regression.to_dict() == checked_regression
    assert regression.acceptance["all_passed"] is True
    assert regression.protocol["confirmation_status"] == (
        "development_regression"
    )
    assert len(regression.protocol["training_case_overlap"]) == 3
    assert len(regression.protocol["training_schedule_overlap"]) == 3
    assert len(
        regression.protocol["training_fault_timing_overlap"]
    ) == 3
    assert regression.aggregate["pre_noise_alerts"] == 0
    assert regression.aggregate["pre_noise_points"] == 108
    assert regression.aggregate["routine_noise_alerts"] == 0
    assert regression.aggregate["routine_noise_points"] == 21

    confirmation_runs = _runs(
        lab / "experiments_v2_confirmation",
        artifacts
        / "demand-conditioned-v2"
        / "confirmation"
        / "cases",
    )
    protocol_path = lab / "v2-confirmation-protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    checked_confirmation = json.loads(
        (
            artifacts
            / "demand-conditioned-v2"
            / "confirmation"
            / "verification.json"
        ).read_text()
    )
    preregistered_commit = checked_confirmation["protocol"][
        "preregistered_git_commit"
    ]
    confirmation = evaluate_demand_conditioned_fault_matrix(
        confirmation_runs,
        feature_spec,
        checked_model_bytes,
        protocol_bytes,
        preregistered_commit,
    )

    assert confirmation.to_dict() == checked_confirmation
    assert confirmation.acceptance["all_passed"] is True
    assert confirmation.protocol["confirmation_status"] == (
        "preregistered_held_out_confirmation"
    )
    assert confirmation.protocol["training_case_overlap"] == []
    assert confirmation.protocol["training_schedule_overlap"] == []
    assert confirmation.protocol["training_fault_timing_overlap"] == []
    assert confirmation.protocol["artifact_file_sha256"]["model"] == (
        model_sha256
    )
    assert confirmation.aggregate["structural_events_detected"] == 3
    assert confirmation.aggregate["attribution_hits_at_3"] == 3
    assert confirmation.aggregate["maximum_detection_delay_windows"] == 0
    assert confirmation.aggregate["pre_noise_alerts"] == 22
    assert confirmation.aggregate["pre_noise_points"] == 148
    assert confirmation.aggregate["routine_noise_alerts"] == 3
    assert confirmation.aggregate["routine_noise_points"] == 21

    protocol = json.loads(protocol_bytes)
    assert _git_bytes(
        repository,
        preregistered_commit,
        "lab/fault_matrix/v2-confirmation-protocol.json",
    ) == protocol_bytes
    for relative_path, expected_sha256 in protocol[
        "frozen_files"
    ].items():
        committed_bytes = _git_bytes(
            repository, preregistered_commit, relative_path
        )
        assert hashlib.sha256(committed_bytes).hexdigest() == (
            expected_sha256
        )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            preregistered_commit,
            "HEAD",
        ],
        cwd=repository,
        check=True,
    )

    training_schedules = {
        canonical_request_schedule(
            run.manifest.requests_per_window,
            run.manifest.load_pattern_offsets,
        )
        for run in development_runs
    }
    confirmation_schedules = {
        canonical_request_schedule(
            run.manifest.requests_per_window,
            run.manifest.load_pattern_offsets,
        )
        for run in confirmation_runs
    }
    assert training_schedules.isdisjoint(confirmation_schedules)
    training_fault_timings = {
        (
            run.manifest.fault_kind,
            run.manifest.structural_interval,
        )
        for run in development_runs
    }
    confirmation_fault_timings = {
        (
            run.manifest.fault_kind,
            run.manifest.structural_interval,
        )
        for run in confirmation_runs
    }
    assert training_fault_timings.isdisjoint(
        confirmation_fault_timings
    )

    build_context_hash = hashlib.sha256()
    for name in (
        "Dockerfile",
        "requirements.txt",
        "run_experiment.py",
        "service.py",
    ):
        build_context_hash.update(name.encode("utf-8"))
        build_context_hash.update(b"\0")
        build_context_hash.update(
            _git_bytes(
                repository,
                preregistered_commit,
                f"lab/fault_matrix/{name}",
            )
        )
        build_context_hash.update(b"\0")
    expected_build_hash = build_context_hash.hexdigest()
    assert all(
        case["capture"]["application_build_context_sha256"]
        == expected_build_hash
        for case in confirmation.cases.values()
    )


def _runs(
    manifests_directory: Path,
    captures_directory: Path,
) -> list[FaultMatrixRun]:
    runs = []
    for manifest_path in sorted(manifests_directory.glob("*.json")):
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(manifest_path.read_text())
        )
        runs.append(
            FaultMatrixRun(
                manifest,
                read_otlp_capture(
                    captures_directory
                    / manifest.case_id
                    / "collector-output.jsonl"
                ),
            )
        )
    return runs


def _git_bytes(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
