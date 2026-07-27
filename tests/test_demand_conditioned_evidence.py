import hashlib
import json
from pathlib import Path

from quantis_core.demand_conditioning import (
    train_demand_conditioned_model,
)
from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
    evaluate_demand_conditioned_fault_matrix,
)
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec


def test_v2_model_and_regression_recompute_from_raw_captures():
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
