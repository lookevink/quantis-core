import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quantis_core.demand_conditioning import canonical_request_schedule
from quantis_core.fault_matrix import FaultMatrixCaseManifest
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import (
    OtlpFeatureSpec,
    OtlpWindowCompiler,
)
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec


def test_normal_telemetry_manifest_declares_every_point_fault_free():
    payload = {
        "schema_version": 2,
        "case_id": "jepa-normal-f01-w1-13",
        "fault_kind": "none",
        "point_count": 340,
        "sample_period_seconds": 0.05,
        "logical_window_period_nano": 1_000_000_000,
        "baseline_interval": [0, 340],
        "routine_noise_interval": [340, 340],
        "structural_interval": [340, 340],
        "affected_features": [],
        "requests_per_window": 5,
        "load_pattern_offsets": [0, 1, -1],
        "routine_noise_delay_ms": 0,
        "topology_id": "workers-1",
        "worker_replicas": 1,
        "images": {},
    }

    manifest = FaultMatrixCaseManifest.from_dict(payload)

    assert manifest.baseline_slice == slice(0, 340)
    assert manifest.fault_kind == "none"
    assert manifest.affected_features == ()

    payload["baseline_interval"] = [0, 339]
    with pytest.raises(ValueError, match="entire run"):
        FaultMatrixCaseManifest.from_dict(payload)


def test_normal_otlp_fixture_preserves_no_fault_identity_and_values():
    repository = Path(__file__).resolve().parents[1]
    capture = read_otlp_capture(
        repository
        / "tests"
        / "fixtures"
        / "otlp"
        / "normal-run-metrics.jsonl"
    )
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "fault_matrix"
                / "feature-spec.json"
            ).read_text()
        )
    )

    compiled = OtlpWindowCompiler(feature_spec).compile(capture)

    assert capture.sha256 == (
        "d5705583adf7691316407d91c51c46b956f91be9009f3fc4828fd5c0a1ce1855"
    )
    assert {
        point.resource_attributes[
            "quantis.experiment.fault.kind"
        ]
        for point in capture.points
    } == {"none"}
    assert {
        point.resource_attributes[
            "quantis.experiment.worker.replicas.observed"
        ]
        for point in capture.points
    } == {2}
    np.testing.assert_allclose(
        compiled.values,
        np.asarray(
            [
                [100.0, 1.5, 0.0, 0.0, 100.0, 0.02, 100.0],
                [120.0, 1.7, 0.0, 0.0, 120.0, 0.01, 120.0],
            ]
        ),
    )
    assert compiled.data_quality["missing_cells"] == 0


def test_normal_corpus_preparation_holds_out_entire_schedule_families(
    tmp_path,
):
    completed = subprocess.run(
        [
            sys.executable,
            "lab/fault_matrix/prepare_jepa_normal_corpus.py",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifests = [
        FaultMatrixCaseManifest.from_dict(
            json.loads(path.read_text())
        )
        for path in sorted((tmp_path / "manifests").glob("*.json"))
    ]
    split = TelemetryCorpusSplitSpec.from_dict(
        json.loads((tmp_path / "split.json").read_text())
    )
    schedules = {
        manifest.case_id: canonical_request_schedule(
            manifest.requests_per_window,
            manifest.load_pattern_offsets,
        )
        for manifest in manifests
    }

    assert len(manifests) == 30
    assert len(split.training_case_ids) == 24
    assert len(split.validation_case_ids) == 6
    assert len(set(schedules.values())) == 10
    assert {
        schedules[case_id] for case_id in split.training_case_ids
    }.isdisjoint(
        {
            schedules[case_id]
            for case_id in split.validation_case_ids
        }
    )
    assert {manifest.worker_replicas for manifest in manifests} == {
        1,
        2,
        3,
    }
    assert all(
        {
            manifest.worker_replicas
            for manifest in manifests
            if schedules[manifest.case_id] == schedule
        }
        == {1, 2, 3}
        for schedule in set(schedules.values())
    )
    assert all(
        manifest.fault_kind == "none"
        and manifest.baseline_interval
        == (0, manifest.point_count)
        for manifest in manifests
    )
    assert sum(
        manifest.point_count - split.lookback
        for manifest in manifests
    ) == 10_020
