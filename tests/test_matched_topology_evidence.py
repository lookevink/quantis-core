import hashlib
import json
import subprocess
from pathlib import Path

from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
    evaluate_demand_conditioned_fault_matrix,
)
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec


def test_matched_topology_diagnostic_recomputes_no_material_effect():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    artifact_root = (
        repository
        / "artifacts"
        / "demand-conditioned-v2"
        / "matched-topology-diagnostic"
    )
    runs = _runs(
        lab / "experiments_v2_matched",
        artifact_root / "cases",
    )
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab / "feature-spec.json").read_text())
    )
    model_bytes = (
        repository
        / "artifacts"
        / "demand-conditioned-v2"
        / "model.json"
    ).read_bytes()
    protocol_path = lab / "v2-matched-topology-protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    checked = json.loads(
        (artifact_root / "verification.json").read_text()
    )
    preregistered_commit = checked["protocol"][
        "preregistered_git_commit"
    ]

    recomputed = evaluate_demand_conditioned_fault_matrix(
        runs,
        feature_spec,
        model_bytes,
        protocol_bytes,
        preregistered_commit,
    )

    assert recomputed.to_dict() == checked
    assert recomputed.acceptance["all_passed"] is False
    assert recomputed.aggregate["structural_events_detected"] == 9
    assert recomputed.aggregate["attribution_hits_at_3"] == 9
    assert recomputed.aggregate["maximum_detection_delay_windows"] == 0
    assert recomputed.aggregate["pre_noise_alerts"] == 133
    assert recomputed.aggregate["pre_noise_points"] == 393
    assert recomputed.aggregate["routine_noise_alerts"] == 27
    assert recomputed.aggregate["routine_noise_points"] == 63
    assert recomputed.aggregate["topology_strata"] == {
        "workers-1": {
            "structural_events": 3,
            "structural_events_detected": 3,
            "structural_event_recall": 1.0,
            "attribution_hits_at_3": 3,
            "attribution_hit_rate_at_3": 1.0,
            "maximum_detection_delay_windows": 0,
            "pre_noise_alerts": 46,
            "pre_noise_points": 131,
            "pre_noise_alert_rate": 46 / 131,
            "routine_noise_alerts": 7,
            "routine_noise_points": 21,
            "routine_noise_alert_rate": 7 / 21,
        },
        "workers-2": {
            "structural_events": 3,
            "structural_events_detected": 3,
            "structural_event_recall": 1.0,
            "attribution_hits_at_3": 3,
            "attribution_hit_rate_at_3": 1.0,
            "maximum_detection_delay_windows": 0,
            "pre_noise_alerts": 44,
            "pre_noise_points": 131,
            "pre_noise_alert_rate": 44 / 131,
            "routine_noise_alerts": 9,
            "routine_noise_points": 21,
            "routine_noise_alert_rate": 9 / 21,
        },
        "workers-3": {
            "structural_events": 3,
            "structural_events_detected": 3,
            "structural_event_recall": 1.0,
            "attribution_hits_at_3": 3,
            "attribution_hit_rate_at_3": 1.0,
            "maximum_detection_delay_windows": 0,
            "pre_noise_alerts": 43,
            "pre_noise_points": 131,
            "pre_noise_alert_rate": 43 / 131,
            "routine_noise_alerts": 11,
            "routine_noise_points": 21,
            "routine_noise_alert_rate": 11 / 21,
        },
    }
    diagnostic = recomputed.aggregate[
        "matched_topology_diagnostic"
    ]
    assert diagnostic["classification"] == (
        "no_material_topology_effect"
    )
    assert diagnostic["paired_effects"]["workers-2"][
        "risk_differences"
    ] == [
        9 / 42 - 13 / 42,
        18 / 44 - 24 / 44,
        17 / 45 - 9 / 45,
    ]
    assert diagnostic["paired_effects"]["workers-3"][
        "risk_differences"
    ] == [
        11 / 42 - 13 / 42,
        18 / 44 - 24 / 44,
        14 / 45 - 9 / 45,
    ]
    assert recomputed.acceptance["gates"][
        "matched_topology_design_complete"
    ] is True
    assert (
        "better explained by its schedule confound"
        in (artifact_root / "report.md").read_text()
    )

    protocol = json.loads(protocol_bytes)
    assert _git_bytes(
        repository,
        preregistered_commit,
        "lab/fault_matrix/v2-matched-topology-protocol.json",
    ) == protocol_bytes
    for relative_path, expected_sha256 in protocol[
        "frozen_files"
    ].items():
        assert hashlib.sha256(
            _git_bytes(
                repository,
                preregistered_commit,
                relative_path,
            )
        ).hexdigest() == expected_sha256
    capture_tree = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            preregistered_commit,
            "--",
            "artifacts/demand-conditioned-v2/"
            "matched-topology-diagnostic",
        ],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert capture_tree == ""
    for run in runs:
        observed_counts = {
            point.resource_attributes[
                "quantis.experiment.worker.replicas.observed"
            ]
            for point in run.capture.points
        }
        assert observed_counts == {run.manifest.worker_replicas}


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
