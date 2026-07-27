import json
from pathlib import Path

from quantis_core.demand_conditioning import canonical_request_schedule
from quantis_core.fault_matrix import FaultMatrixCaseManifest
from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec


def test_multimodal_pilot_is_normal_only_and_schedule_disjoint() -> None:
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    manifests = [
        FaultMatrixCaseManifest.from_dict(
            json.loads(path.read_text())
        )
        for path in sorted(
            (lab / "multimodal_pilot_manifests").glob("*.json")
        )
    ]
    split = TelemetryCorpusSplitSpec.from_dict(
        json.loads(
            (lab / "multimodal-pilot-split.json").read_text()
        )
    )
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads((lab / "log-feature-spec.json").read_text())
    )
    schedules = {
        manifest.case_id: canonical_request_schedule(
            manifest.requests_per_window,
            manifest.load_pattern_offsets,
        )
        for manifest in manifests
    }

    assert len(manifests) == 3
    assert set(schedules) == (
        set(split.training_case_ids)
        | set(split.validation_case_ids)
    )
    assert {
        schedules[case_id] for case_id in split.training_case_ids
    }.isdisjoint(
        {
            schedules[case_id]
            for case_id in split.validation_case_ids
        }
    )
    assert all(
        manifest.fault_kind == "none"
        and manifest.baseline_interval == (0, 100)
        for manifest in manifests
    )
    assert tuple(
        feature.name for feature in log_spec.features
    ) == (
        "checkout_accepted_count",
        "checkout_rejected_count",
        "checkout_completed_count",
        "error_event_count",
    )
