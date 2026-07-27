import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quantis_core.demand_conditioning import (
    DemandConditioner,
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


RAW_FEATURES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "queue_depth",
    "worker_rate",
    "worker_heartbeat_age_s",
    "db_write_rate",
)


def test_request_schedule_is_canonicalized_from_realized_demand():
    assert canonical_request_schedule(6, (-1, 1, -1, 1)) == (5, 7)
    assert canonical_request_schedule(5, (0, 2)) == (5, 7)


def test_demand_conditioner_replaces_throughput_with_completion_ratios():
    raw = np.asarray(
        [
            [10.0, 4.0, 0.0, 2.0, 8.0, 0.1, 6.0],
            [20.0, 5.0, 0.1, 3.0, 10.0, 0.2, 5.0],
        ]
    )

    conditioned = DemandConditioner().transform(raw, RAW_FEATURES)

    assert conditioned.feature_names == (
        "request_latency_ms",
        "error_rate",
        "queue_depth",
        "worker_completion_ratio",
        "worker_heartbeat_age_s",
        "db_write_completion_ratio",
    )
    np.testing.assert_allclose(
        conditioned.values,
        [
            [4.0, 0.0, 2.0, 0.8, 0.1, 0.6],
            [5.0, 0.1, 3.0, 0.5, 0.2, 0.25],
        ],
    )
    assert DemandConditioner.from_dict(
        DemandConditioner().to_dict()
    ).to_dict() == {
        "schema_version": 1,
        "kind": "request_demand_ratios",
    }

    with pytest.raises(ValueError, match="positive request demand"):
        DemandConditioner().transform(
            np.asarray([[0.0, 4.0, 0.0, 0.0, 0.0, 0.1, 0.0]]),
            RAW_FEATURES,
        )


def test_v2_training_combines_fault_free_schedules_without_crossing_runs():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    artifacts = repository / "artifacts" / "fault-matrix" / "cases"
    runs = []
    for manifest_path in sorted((lab / "experiments").glob("*.json")):
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(manifest_path.read_text())
        )
        runs.append(
            FaultMatrixRun(
                manifest,
                read_otlp_capture(
                    artifacts
                    / manifest.case_id
                    / "collector-output.jsonl"
                ),
            )
        )
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab / "feature-spec.json").read_text())
    )

    with pytest.raises(ValueError, match="capture does not match manifest"):
        train_demand_conditioned_model(
            [
                FaultMatrixRun(runs[0].manifest, runs[1].capture),
                runs[1],
                runs[2],
            ],
            feature_spec,
        )
    equivalent_schedule_runs = [
        FaultMatrixRun(
            replace(
                runs[0].manifest,
                requests_per_window=6,
                load_pattern_offsets=(0, 1),
            ),
            runs[0].capture,
        ),
        FaultMatrixRun(
            replace(
                runs[1].manifest,
                requests_per_window=5,
                load_pattern_offsets=(1, 2, 1, 2),
            ),
            runs[1].capture,
        ),
        FaultMatrixRun(
            replace(
                runs[2].manifest,
                requests_per_window=7,
                load_pattern_offsets=(-1, 0),
            ),
            runs[2].capture,
        ),
    ]
    with pytest.raises(ValueError, match="three distinct schedules"):
        train_demand_conditioned_model(
            equivalent_schedule_runs, feature_spec
        )

    model = train_demand_conditioned_model(runs, feature_spec)

    assert model.protocol["training_run_count"] == 3
    assert model.protocol["distinct_load_schedule_count"] == 3
    assert model.protocol["training_structural_points"] == 0
    assert model.protocol["training_point_count"] == 108
    assert model.protocol["training_window_count"] == 90
    assert model.detector_artifact["kind"] == (
        "demand_conditioned_coherent_predictive"
    )
    assert model.detector_artifact["threshold"] > 0.0
    assert model.to_dict()["conditioner"] == (
        DemandConditioner().to_dict()
    )

    regression = evaluate_demand_conditioned_fault_matrix(
        runs, feature_spec, model.to_bytes()
    )

    assert regression.acceptance["all_passed"] is True
    assert regression.aggregate["structural_events_detected"] == 3
    assert regression.aggregate["attribution_hits_at_3"] == 3
    assert regression.aggregate["pre_noise_alerts"] == 0
    assert regression.aggregate["pre_noise_points"] == 108
    assert regression.aggregate["routine_noise_alerts"] == 0
    assert regression.aggregate["routine_noise_points"] == 21
    assert regression.protocol["model_fit_calls"] == 0
