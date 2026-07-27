import hashlib
import json
import subprocess
from pathlib import Path

from quantis_core.demand_conditioning import canonical_request_schedule
from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
    evaluate_demand_conditioned_fault_matrix,
)
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec


def test_expanded_confirmation_recomputes_and_preserves_failure() -> None:
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    artifact_root = (
        repository
        / "artifacts"
        / "demand-conditioned-v2"
        / "expanded-confirmation"
    )
    runs = _runs(
        lab / "experiments_v2_expanded", artifact_root / "cases"
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
    protocol_path = lab / "v2-expanded-confirmation-protocol.json"
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
    assert recomputed.aggregate["pre_noise_alerts"] == 216
    assert recomputed.aggregate["pre_noise_points"] == 377
    assert recomputed.aggregate["routine_noise_alerts"] == 34
    assert recomputed.aggregate["routine_noise_points"] == 63
    assert recomputed.aggregate["topology_strata"] == {
        "workers-1": {
            "structural_events": 3,
            "structural_events_detected": 3,
            "structural_event_recall": 1.0,
            "attribution_hits_at_3": 3,
            "attribution_hit_rate_at_3": 1.0,
            "maximum_detection_delay_windows": 0,
            "pre_noise_alerts": 12,
            "pre_noise_points": 120,
            "pre_noise_alert_rate": 12 / 120,
            "routine_noise_alerts": 0,
            "routine_noise_points": 21,
            "routine_noise_alert_rate": 0 / 21,
        },
        "workers-2": {
            "structural_events": 3,
            "structural_events_detected": 3,
            "structural_event_recall": 1.0,
            "attribution_hits_at_3": 3,
            "attribution_hit_rate_at_3": 1.0,
            "maximum_detection_delay_windows": 0,
            "pre_noise_alerts": 86,
            "pre_noise_points": 122,
            "pre_noise_alert_rate": 86 / 122,
            "routine_noise_alerts": 13,
            "routine_noise_points": 21,
            "routine_noise_alert_rate": 13 / 21,
        },
        "workers-3": {
            "structural_events": 3,
            "structural_events_detected": 3,
            "structural_event_recall": 1.0,
            "attribution_hits_at_3": 3,
            "attribution_hit_rate_at_3": 1.0,
            "maximum_detection_delay_windows": 0,
            "pre_noise_alerts": 118,
            "pre_noise_points": 135,
            "pre_noise_alert_rate": 118 / 135,
            "routine_noise_alerts": 21,
            "routine_noise_points": 21,
            "routine_noise_alert_rate": 21 / 21,
        },
    }
    assert recomputed.acceptance["gates"][
        "complete_fault_topology_coverage"
    ] is True
    assert recomputed.acceptance["gates"][
        "aggregate_pre_noise_alert_rate_within_limit"
    ] is False
    assert recomputed.acceptance["gates"][
        "aggregate_routine_noise_alert_rate_within_limit"
    ] is False
    assert recomputed.acceptance["gates"][
        "all_topology_strata_within_limits"
    ] is False
    assert (
        "association with multi-worker operation"
        in (artifact_root / "report.md").read_text()
    )

    protocol = json.loads(protocol_bytes)
    assert protocol["required_topologies"] == {
        "workers-1": 1,
        "workers-2": 2,
        "workers-3": 3,
    }
    assert _git_bytes(
        repository,
        preregistered_commit,
        "lab/fault_matrix/v2-expanded-confirmation-protocol.json",
    ) == protocol_bytes
    for relative_path, expected_sha256 in protocol[
        "frozen_files"
    ].items():
        assert hashlib.sha256(
            _git_bytes(
                repository, preregistered_commit, relative_path
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
            "artifacts/demand-conditioned-v2/expanded-confirmation",
        ],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert capture_tree == ""

    prior_runs = _runs(
        lab / "experiments",
        repository / "artifacts" / "fault-matrix" / "cases",
    ) + _runs(
        lab / "experiments_v2_confirmation",
        repository
        / "artifacts"
        / "demand-conditioned-v2"
        / "confirmation"
        / "cases",
    )
    current_manifests = [run.manifest for run in runs]
    current_case_ids = {
        manifest.case_id for manifest in current_manifests
    }
    prior_manifests = [run.manifest for run in prior_runs]
    for commit in (
        "82ca21c",
        "f55e7a1",
        "da6ee89",
        "5d11f18",
        "d4dd4a3",
        "af9a4aa",
        "8d1e616",
        "e0bb4b9",
        "39bcbf7",
        "84850c6",
    ):
        prior_manifests.extend(
            manifest
            for manifest in _git_expanded_manifests(
                repository, commit
            )
            if manifest.case_id not in current_case_ids
        )
    assert _schedule_signatures(current_manifests).isdisjoint(
        _schedule_signatures(prior_manifests)
    )
    assert _fault_timings(current_manifests).isdisjoint(
        _fault_timings(prior_manifests)
    )
    assert len(_schedule_signatures(current_manifests)) == len(runs)
    assert len(_fault_timing_intervals(current_manifests)) == len(runs)

    for run in runs:
        observed_counts = {
            point.resource_attributes[
                "quantis.experiment.worker.replicas.observed"
            ]
            for point in run.capture.points
        }
        assert observed_counts == {run.manifest.worker_replicas}

    expected_build_hash = hashlib.sha256()
    for name in (
        "Dockerfile",
        "requirements.txt",
        "run_experiment.py",
        "service.py",
    ):
        expected_build_hash.update(name.encode("utf-8"))
        expected_build_hash.update(b"\0")
        expected_build_hash.update(
            _git_bytes(
                repository,
                preregistered_commit,
                f"lab/fault_matrix/{name}",
            )
        )
        expected_build_hash.update(b"\0")
    assert all(
        case["capture"]["application_build_context_sha256"]
        == expected_build_hash.hexdigest()
        for case in recomputed.cases.values()
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


def _schedule_signatures(
    manifests: list[FaultMatrixCaseManifest],
) -> set[tuple[int, ...]]:
    return {
        canonical_request_schedule(
            manifest.requests_per_window,
            manifest.load_pattern_offsets,
        )
        for manifest in manifests
    }


def _fault_timings(
    manifests: list[FaultMatrixCaseManifest],
) -> set[tuple[str, tuple[int, int]]]:
    return {
        (manifest.fault_kind, manifest.structural_interval)
        for manifest in manifests
    }


def _fault_timing_intervals(
    manifests: list[FaultMatrixCaseManifest],
) -> set[tuple[int, int]]:
    return {
        manifest.structural_interval for manifest in manifests
    }


def _git_expanded_manifests(
    repository: Path,
    commit: str,
) -> list[FaultMatrixCaseManifest]:
    paths = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            "lab/fault_matrix/experiments_v2_expanded",
        ],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    return [
        FaultMatrixCaseManifest.from_dict(
            json.loads(_git_bytes(repository, commit, path))
        )
        for path in paths
        if path.endswith(".json")
    ]


def _git_bytes(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
