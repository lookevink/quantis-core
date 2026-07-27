"""Verify Collector OTLP replay parity against the direct scenario path."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from quantis_core.contracts import DetectionEvent, FeatureEvidence
from quantis_core.detectors import detector_from_dict
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_windowing import (
    ForwardFillPolicy,
    OtlpFeatureSpec,
    OtlpWindowCompiler,
    materialize_compiled_telemetry,
)
from quantis_core.scenarios import Phase, ScenarioSpec, generate_scenario
from quantis_core.windowing import WindowCompiler


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    output = repository / "artifacts" / "otlp-replay"
    capture_path = output / "collector-output.jsonl"
    spec_path = repository / "lab" / "otel" / "scenario-feature-spec.json"
    config_path = repository / "lab" / "otel" / "collector.yaml"
    manifest_path = repository / "lab" / "otel" / "roundtrip-manifest.json"
    evaluation = repository / "artifacts" / "evaluation"
    manifest = json.loads(manifest_path.read_text())
    scenario_seed = int(manifest["scenario_seed"])
    scenario_length = int(manifest["scenario_length"])

    capture = read_otlp_capture(capture_path)
    feature_spec = OtlpFeatureSpec.from_dict(json.loads(spec_path.read_text()))
    compiled = OtlpWindowCompiler(feature_spec).compile(capture)
    materialized = materialize_compiled_telemetry(
        compiled, ForwardFillPolicy(max_gap_windows=0)
    )

    scenario = generate_scenario(
        ScenarioSpec(seed=scenario_seed, length=scenario_length)
    )
    value_difference = np.abs(materialized.values - scenario.telemetry)
    max_value_difference = float(np.max(value_difference))

    model_compiler = WindowCompiler.from_dict(
        json.loads((evaluation / "window-compiler.json").read_text())
    )
    detector_path = (
        evaluation / "detectors" / "coherent_latent_predictive.json"
    )
    detector_payload = json.loads(detector_path.read_text())
    detector = detector_from_dict(detector_payload)
    direct_windows = model_compiler.transform(
        scenario.telemetry, scenario.feature_names
    )
    replay_windows = model_compiler.transform(
        materialized.values, materialized.feature_names
    )
    direct_scores = detector.score(direct_windows)
    replay_scores = detector.score(replay_windows)
    max_score_difference = float(
        np.max(np.abs(direct_scores.scores - replay_scores.scores))
    )
    alerts_identical = bool(
        np.array_equal(direct_scores.alerts, replay_scores.alerts)
    )
    phases = scenario.phases[replay_windows.point_indices]
    structural_mask = phases == Phase.STRUCTURAL.value
    noise_mask = phases == Phase.ROUTINE_NOISE.value
    structural_event_detected = bool(
        np.any(replay_scores.alerts & structural_mask)
    )
    noise_points = int(np.count_nonzero(noise_mask))
    noise_alerts = int(np.count_nonzero(replay_scores.alerts & noise_mask))
    noise_alert_rate = noise_alerts / noise_points if noise_points else 0.0
    model_version = _sha256(detector_path)
    detection_events = []
    for score_index in np.flatnonzero(replay_scores.alerts):
        evidence = replay_scores.feature_evidence[score_index]
        top_indices = np.argsort(evidence)[-3:][::-1]
        signed = replay_scores.signed_feature_evidence
        features = []
        for feature_index in top_indices:
            direction = 0
            if signed is not None:
                direction = int(np.sign(signed[score_index, feature_index]))
            features.append(
                FeatureEvidence(
                    name=replay_windows.feature_names[feature_index],
                    magnitude=float(evidence[feature_index]),
                    direction=direction,
                )
            )
        point_index = int(replay_windows.point_indices[score_index])
        detection_events.append(
            DetectionEvent(
                model_kind=str(detector_payload["kind"]),
                model_version=model_version,
                feature_schema_id=compiled.feature_schema_id,
                capture_sha256=capture.sha256,
                window_end_unix_nano=int(
                    materialized.window_end_unix_nano[point_index]
                ),
                score=float(replay_scores.scores[score_index]),
                threshold=float(replay_scores.threshold),
                alert=True,
                top_features=tuple(features),
                data_quality={
                    **dict(compiled.data_quality),
                    "imputed_cells": int(
                        np.count_nonzero(materialized.imputed_mask)
                    ),
                },
            )
        )

    gates = {
        "capture_matches_golden": (
            capture.sha256 == manifest["capture_sha256"]
        ),
        "no_missing_cells": compiled.data_quality["missing_cells"] == 0,
        "values_match_direct_path": max_value_difference == 0.0,
        "scores_match_direct_path": max_score_difference <= 1e-12,
        "alerts_match_direct_path": alerts_identical,
        "structural_event_detected": structural_event_detected,
    }
    compiled_encoded = (
        json.dumps(
            compiled.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    compiled_sha256 = hashlib.sha256(compiled_encoded.encode("utf-8")).hexdigest()
    gates["compiled_matches_golden"] = (
        compiled_sha256 == manifest["compiled_sha256"]
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "acceptance": {"all_passed": all(gates.values()), "gates": gates},
        "collector": {
            "image": str(manifest["collector_image"]),
            "config_sha256": _sha256(config_path),
        },
        "capture": {
            "path": str(capture_path.relative_to(repository)),
            "sha256": capture.sha256,
            "json_message_count": capture.json_message_count,
            "metric_point_count": len(capture.points),
        },
        "compiled": {
            "feature_schema_id": compiled.feature_schema_id,
            "sha256": compiled_sha256,
            "window_count": len(compiled.window_end_unix_nano),
            "feature_count": len(compiled.feature_names),
            "data_quality": dict(compiled.data_quality),
        },
        "parity": {
            "max_value_absolute_difference": max_value_difference,
            "max_score_absolute_difference": max_score_difference,
            "alerts_identical": alerts_identical,
        },
        "detection": {
            "detector_kind": str(detector_payload["kind"]),
            "model_version": model_version,
            "detection_event_count": len(detection_events),
            "structural_event_detected": structural_event_detected,
            "routine_noise_points": noise_points,
            "routine_noise_alerts": noise_alerts,
            "routine_noise_alert_rate": noise_alert_rate,
        },
        "scenario": {
            "seed": scenario_seed,
            "length": scenario_length,
            "generator_version": scenario.manifest["generator_version"],
        },
        "limitations": [
            "The Collector round trip carries deterministic synthetic gauges, "
            "not telemetry from a production workload.",
            "The parity result validates transport and replay semantics, not "
            "real-world anomaly detection.",
            "The checked-in capture covers gauges; worked fixtures separately "
            "cover sums, histograms, resets, flags, and missingness.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "compiled-telemetry.json").write_text(
        compiled_encoded
    )
    (output / "detection-events.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [event.to_dict() for event in detection_events],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    (output / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "report.md").write_text(_markdown_report(report))
    status = "PASS" if report["acceptance"]["all_passed"] else "FAIL"
    print(f"OTLP round-trip acceptance: {status}")
    if not report["acceptance"]["all_passed"]:
        raise SystemExit(1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_report(report: Dict[str, Any]) -> str:
    acceptance = report["acceptance"]
    parity = report["parity"]
    compiled = report["compiled"]
    detection = report["detection"]
    lines = [
        "# Quantis OTLP Collector round-trip verification",
        "",
        f"Overall acceptance: **{'PASS' if acceptance['all_passed'] else 'FAIL'}**",
        "",
        f"- Collector: `{report['collector']['image']}`",
        f"- Capture SHA-256: `{report['capture']['sha256']}`",
        f"- Feature schema: `{compiled['feature_schema_id']}`",
        f"- Windows × features: {compiled['window_count']} × "
        f"{compiled['feature_count']}",
        f"- Missing cells: {compiled['data_quality']['missing_cells']}",
        f"- Maximum value difference: "
        f"{parity['max_value_absolute_difference']:.3g}",
        f"- Maximum score difference: "
        f"{parity['max_score_absolute_difference']:.3g}",
        f"- Alerts identical: {parity['alerts_identical']}",
        f"- Structural event detected: "
        f"{detection['structural_event_detected']}",
        f"- Routine-noise alert rate: "
        f"{detection['routine_noise_alert_rate']:.3f}",
        "",
        "## Gates",
        "",
    ]
    for gate, passed in acceptance["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{gate}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
