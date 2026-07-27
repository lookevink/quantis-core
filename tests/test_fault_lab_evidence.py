import hashlib
import json
from pathlib import Path

from quantis_core.fault_lab import FaultLabManifest, evaluate_fault_lab
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec


def test_checked_in_fault_lab_evidence_recomputes_from_raw_collector_capture():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault"
    artifacts = repository / "artifacts" / "fault-lab"
    capture = read_otlp_capture(artifacts / "collector-output.jsonl")
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab / "feature-spec.json").read_text())
    )
    manifest = FaultLabManifest.from_dict(
        json.loads((lab / "experiment.json").read_text())
    )

    recomputed = evaluate_fault_lab(capture, feature_spec, manifest)
    checked_in = json.loads((artifacts / "verification.json").read_text())

    assert recomputed.to_dict() == checked_in
    assert recomputed.acceptance["all_passed"] is True
    assert recomputed.capture["sha256"] == capture.sha256
    assert recomputed.compiled["data_quality"]["missing_cells"] == 0
    assert recomputed.raw_effects["queue_depth_growth"] >= 20.0
    assert recomputed.detection["validation_alert_rate"] <= 0.2
    assert recomputed.detection["routine_noise_alert_rate"] <= 0.2
    assert recomputed.attribution["hit_at_3"] is True
    assert recomputed.protocol["application_image_id"].startswith(
        "sha256:"
    )
    build_context_hash = hashlib.sha256()
    for name in (
        "Dockerfile",
        "experiment.json",
        "requirements.txt",
        "run_experiment.py",
        "service.py",
    ):
        build_context_hash.update(name.encode("utf-8"))
        build_context_hash.update(b"\0")
        build_context_hash.update((lab / name).read_bytes())
        build_context_hash.update(b"\0")
    assert recomputed.protocol[
        "application_build_context_sha256"
    ] == build_context_hash.hexdigest()

    pinned_sources = (
        (lab / "Dockerfile").read_text()
        + (lab / "compose.yaml").read_text()
    )
    assert all(
        image in pinned_sources for image in manifest.images.values()
    )
