import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from quantis_core.demand_conditioning import canonical_request_schedule
from quantis_core.fault_matrix import (
    MATCHED_TOPOLOGY_CONTROLLED_FIELDS,
    FaultMatrixCaseManifest,
    validate_matched_topology_design,
)


REQUIRED_TOPOLOGIES = {
    "workers-1": 1,
    "workers-2": 2,
    "workers-3": 3,
}


def test_matched_topology_design_holds_non_topology_fields_fixed():
    manifests = _matched_manifests()

    design = validate_matched_topology_design(
        manifests,
        REQUIRED_TOPOLOGIES,
        "workers-1",
    )

    assert design["kind"] == "matched_topology_diagnostic"
    assert design["controlled_fields"] == list(
        MATCHED_TOPOLOGY_CONTROLLED_FIELDS
    )
    assert set(design["blocks"]) == {
        "cache_outage",
        "database_lock",
        "worker_crash",
    }
    schedules = {
        canonical_request_schedule(
            manifest.requests_per_window,
            manifest.load_pattern_offsets,
        )
        for manifest in manifests
    }
    assert len(schedules) == 3


def test_matched_topology_design_rejects_a_schedule_confounded_block():
    manifests = _matched_manifests()
    target = next(
        manifest
        for manifest in manifests
        if (
            manifest.fault_kind == "worker_crash"
            and manifest.topology_id == "workers-2"
        )
    )
    confounded = [
        (
            replace(
                manifest,
                requests_per_window=manifest.requests_per_window + 1,
            )
            if manifest is target
            else manifest
        )
        for manifest in manifests
    ]

    with pytest.raises(
        ValueError,
        match="changes fields other than topology",
    ):
        validate_matched_topology_design(
            confounded,
            REQUIRED_TOPOLOGIES,
            "workers-1",
        )


def test_matched_topology_design_rejects_incomplete_treatments():
    with pytest.raises(
        ValueError,
        match="one case per fault and topology",
    ):
        validate_matched_topology_design(
            _matched_manifests()[:-1],
            REQUIRED_TOPOLOGIES,
            "workers-1",
        )


def test_matched_topology_protocol_freezes_disjoint_inputs():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    manifests = _matched_manifests()
    protocol = json.loads(
        (lab / "v2-matched-topology-protocol.json").read_text()
    )

    assert protocol["diagnostic_design"] == {
        "kind": "matched_topology_diagnostic",
        "reference_topology_id": "workers-1",
        "primary_outcome": "pre_noise_alert_rate",
        "minimum_material_risk_difference": 0.2,
    }
    assert protocol["required_topologies"] == REQUIRED_TOPOLOGIES
    assert protocol["confirmation_manifest_sha256"] == {
        manifest.case_id: _canonical_sha256(manifest.to_dict())
        for manifest in manifests
    }
    for relative_path, expected_sha256 in protocol[
        "frozen_files"
    ].items():
        assert hashlib.sha256(
            (repository / relative_path).read_bytes()
        ).hexdigest() == expected_sha256

    prior_manifests = []
    for directory in (
        "experiments",
        "experiments_v2_confirmation",
        "experiments_v2_expanded",
    ):
        prior_manifests.extend(
            FaultMatrixCaseManifest.from_dict(
                json.loads(path.read_text())
            )
            for path in (lab / directory).glob("*.json")
        )
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
            _git_expanded_manifests(repository, commit)
        )

    matched_schedules = {
        canonical_request_schedule(
            manifest.requests_per_window,
            manifest.load_pattern_offsets,
        )
        for manifest in manifests
    }
    prior_schedules = {
        canonical_request_schedule(
            manifest.requests_per_window,
            manifest.load_pattern_offsets,
        )
        for manifest in prior_manifests
    }
    matched_fault_timings = {
        (manifest.fault_kind, manifest.structural_interval)
        for manifest in manifests
    }
    prior_fault_timings = {
        (manifest.fault_kind, manifest.structural_interval)
        for manifest in prior_manifests
    }
    assert matched_schedules.isdisjoint(prior_schedules)
    assert matched_fault_timings.isdisjoint(prior_fault_timings)


def _matched_manifests() -> list[FaultMatrixCaseManifest]:
    repository = Path(__file__).resolve().parents[1]
    return [
        FaultMatrixCaseManifest.from_dict(
            json.loads(path.read_text())
        )
        for path in sorted(
            (
                repository
                / "lab"
                / "fault_matrix"
                / "experiments_v2_matched"
            ).glob("*.json")
        )
    ]


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
            json.loads(
                subprocess.run(
                    ["git", "show", f"{commit}:{path}"],
                    cwd=repository,
                    check=True,
                    stdout=subprocess.PIPE,
                ).stdout
            )
        )
        for path in paths
        if path.endswith(".json")
    ]


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
