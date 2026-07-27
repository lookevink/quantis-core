import hashlib
import json
import subprocess
from pathlib import Path

from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
    evaluate_fault_matrix,
)
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec


def test_checked_in_fault_matrix_evidence_recomputes_and_preserves_failure():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    artifacts = repository / "artifacts" / "fault-matrix"
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab / "feature-spec.json").read_text())
    )
    runs = []
    for manifest_path in sorted((lab / "experiments").glob("*.json")):
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(manifest_path.read_text())
        )
        capture_path = (
            artifacts
            / "cases"
            / manifest.case_id
            / "collector-output.jsonl"
        )
        runs.append(
            FaultMatrixRun(
                manifest=manifest,
                capture=read_otlp_capture(capture_path),
            )
        )
    compiler_path = (
        repository
        / "artifacts"
        / "fault-lab"
        / "window-compiler.json"
    )
    detector_path = (
        repository / "artifacts" / "fault-lab" / "detector.json"
    )
    compiler_bytes = compiler_path.read_bytes()
    detector_bytes = detector_path.read_bytes()
    recomputed = evaluate_fault_matrix(
        runs,
        feature_spec,
        compiler_bytes,
        detector_bytes,
    )
    checked_in = json.loads(
        (artifacts / "verification.json").read_text()
    )

    assert recomputed.to_dict() == checked_in
    assert recomputed.acceptance["all_passed"] is False
    assert recomputed.aggregate["structural_events_detected"] == 3
    assert recomputed.aggregate["attribution_hits_at_3"] == 3
    assert recomputed.aggregate["maximum_detection_delay_windows"] == 0
    assert recomputed.aggregate["routine_noise_alerts"] == 18
    assert recomputed.aggregate["routine_noise_points"] == 21
    assert recomputed.aggregate["pre_noise_alerts"] == 87
    assert recomputed.aggregate["pre_noise_points"] == 108
    assert (
        recomputed.acceptance["gates"][
            "aggregate_routine_noise_alert_rate_within_limit"
        ]
        is False
    )
    assert (
        recomputed.acceptance["gates"][
            "aggregate_pre_noise_alert_rate_within_limit"
        ]
        is False
    )
    assert all(
        case["acceptance"]["raw_effects_observed"]
        for case in recomputed.cases.values()
    )

    build_context_hash = hashlib.sha256()
    evidence_commit = "610ecf8"
    for name in (
        "Dockerfile",
        "requirements.txt",
        "run_experiment.py",
        "service.py",
    ):
        build_context_hash.update(name.encode("utf-8"))
        build_context_hash.update(b"\0")
        build_context_hash.update(
            subprocess.run(
                [
                    "git",
                    "show",
                    f"{evidence_commit}:lab/fault_matrix/{name}",
                ],
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        )
        build_context_hash.update(b"\0")
    expected_build_hash = build_context_hash.hexdigest()
    assert all(
        case["capture"]["application_build_context_sha256"]
        == expected_build_hash
        for case in recomputed.cases.values()
    )

    pinned_sources = (
        (lab / "Dockerfile").read_text()
        + (lab / "compose.yaml").read_text()
    )
    assert all(
        image in pinned_sources
        for run in runs
        for image in run.manifest.images.values()
    )
