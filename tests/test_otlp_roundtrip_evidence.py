import hashlib
import json
from pathlib import Path

import numpy as np

from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler
from quantis_core.scenarios import ScenarioSpec, generate_scenario


def test_checked_in_collector_roundtrip_matches_golden_transport_and_tensors():
    repository = Path(__file__).resolve().parents[1]
    artifact_dir = repository / "artifacts" / "otlp-replay"
    lab_dir = repository / "lab" / "otel"
    manifest = json.loads((lab_dir / "roundtrip-manifest.json").read_text())
    verification = json.loads(
        (artifact_dir / "verification.json").read_text()
    )
    capture = read_otlp_capture(artifact_dir / "collector-output.jsonl")
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab_dir / "scenario-feature-spec.json").read_text())
    )
    compiled = OtlpWindowCompiler(feature_spec).compile(capture)
    compiled_encoded = (
        json.dumps(
            compiled.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    compiled_sha256 = hashlib.sha256(compiled_encoded.encode()).hexdigest()
    checked_in_compiled = json.loads(
        (artifact_dir / "compiled-telemetry.json").read_text()
    )
    scenario = generate_scenario(
        ScenarioSpec(
            seed=int(manifest["scenario_seed"]),
            length=int(manifest["scenario_length"]),
        )
    )

    assert capture.sha256 == manifest["capture_sha256"]
    assert compiled_sha256 == manifest["compiled_sha256"]
    assert compiled.to_dict() == checked_in_compiled
    np.testing.assert_allclose(compiled.values, scenario.telemetry)
    assert verification["capture"]["sha256"] == capture.sha256
    assert verification["compiled"]["sha256"] == compiled_sha256
    assert verification["collector"]["config_sha256"] == hashlib.sha256(
        (lab_dir / "collector.yaml").read_bytes()
    ).hexdigest()
    assert verification["acceptance"]["all_passed"] is True
    assert verification["acceptance"]["gates"]["capture_matches_golden"] is True
    assert verification["acceptance"]["gates"]["compiled_matches_golden"] is True
    assert verification["parity"]["max_value_absolute_difference"] == 0.0
    assert verification["parity"]["max_score_absolute_difference"] == 0.0
    assert verification["compiled"]["data_quality"]["missing_cells"] == 0
    assert verification["acceptance"]["gates"][
        "runtime_format_action_counts"
    ] is True
    assert verification["capture"]["runtime_format_action_counts"] == {
        "quantis.experiment.error_count": 3.0,
        "quantis.experiment.request_count": 12.0,
    }
