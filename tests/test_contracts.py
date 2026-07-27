import json

from quantis_core.contracts import DetectionEvent, FeatureEvidence


def test_detection_event_is_versioned_json_and_round_trips_exactly():
    event = DetectionEvent(
        model_kind="coherent_latent_predictive",
        model_version="a" * 64,
        feature_schema_id="b" * 64,
        capture_sha256="c" * 64,
        window_end_unix_nano=12_000_000_000,
        score=4.25,
        threshold=3.5,
        alert=True,
        top_features=(
            FeatureEvidence("request_latency_ms", 5.0, 1),
            FeatureEvidence("active_connections", 4.0, -1),
        ),
        data_quality={"missing_cells": 0, "imputed_cells": 0},
    )

    encoded = json.dumps(event.to_dict(), allow_nan=False)
    restored = DetectionEvent.from_dict(json.loads(encoded))

    assert event.to_dict()["schema_version"] == 1
    assert restored == event
    assert restored.top_features[0].direction == 1
