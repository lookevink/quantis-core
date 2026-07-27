import json
from pathlib import Path

from quantis_core.demand_conditioning import canonical_request_schedule
from quantis_core.fault_matrix import FaultMatrixCaseManifest
from quantis_core.telemetry_corpus import (
    RESERVED_EVIDENCE_CASE_IDS,
    TelemetryCorpusSplitSpec,
)


def test_jepa_pilot_declares_fresh_schedule_disjoint_runs():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    manifests = [
        FaultMatrixCaseManifest.from_dict(
            json.loads(path.read_text())
        )
        for path in sorted(
            (lab / "jepa_pilot_manifests").glob("*.json")
        )
    ]
    split = TelemetryCorpusSplitSpec.from_dict(
        json.loads(
            (lab / "jepa-pilot-split.json").read_text()
        )
    )

    assert len(manifests) == 3
    assert {manifest.case_id for manifest in manifests} == (
        set(split.training_case_ids)
        | set(split.validation_case_ids)
    )
    assert not (
        {manifest.case_id for manifest in manifests}
        & RESERVED_EVIDENCE_CASE_IDS
    )
    schedules = {
        manifest.case_id: canonical_request_schedule(
            manifest.requests_per_window,
            manifest.load_pattern_offsets,
        )
        for manifest in manifests
    }
    assert len(set(schedules.values())) == 3
    assert {
        schedules[case_id] for case_id in split.training_case_ids
    }.isdisjoint(
        {
            schedules[case_id]
            for case_id in split.validation_case_ids
        }
    )
    assert all(
        manifest.baseline_interval == (0, 60)
        for manifest in manifests
    )
