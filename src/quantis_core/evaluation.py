"""Reproducible held-out evaluation for the synthetic Quantis claim."""

import json
import platform
import sys
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .detectors import (
    CoherentLatentPredictiveDetector,
    DetectionScores,
    LatentPredictiveDetector,
    PersistenceDetector,
    RobustFeatureDetector,
)
from .scenarios import Phase, Scenario, ScenarioSpec, generate_scenario
from .windowing import ModelWindows, WindowCompiler


@dataclass(frozen=True)
class EvaluationConfig:
    """All experiment choices that affect generated evidence."""

    train_seeds: Tuple[int, ...] = (11, 23, 37, 41, 53)
    test_seeds: Tuple[int, ...] = (101, 103, 107, 109, 113, 127, 131, 137)
    scenario_length: int = 600
    lookback: int = 12
    noise_events: int = 14
    calibration_quantile: float = 0.99
    latent_dimension: int = 1
    ridge: float = 1e-2
    consensus_rank: int = 3

    def __post_init__(self) -> None:
        if not self.train_seeds or not self.test_seeds:
            raise ValueError("evaluation requires training and test seeds")
        if set(self.train_seeds) & set(self.test_seeds):
            raise ValueError("training and test seeds must be disjoint")


@dataclass(frozen=True)
class DetectorEvaluation:
    """Operational and attribution measurements for one detector."""

    threshold: float
    routine_noise_points: int
    routine_noise_alerts: int
    routine_noise_alert_rate: float
    normal_points: int
    normal_alerts: int
    normal_alert_rate: float
    structural_events: int
    structural_events_detected: int
    structural_event_recall: float
    structural_point_alert_rate: float
    mean_detection_delay_points: Optional[float]
    attribution_hit_at_3: float
    attribution_recall_at_3: float
    median_noise_score: float
    median_structural_score: float
    mean_streaming_scoring_ms_per_point: float


class _ScoringDetector(Protocol):
    def score(self, windows: ModelWindows) -> DetectionScores:
        ...


@dataclass(frozen=True)
class EvaluationReport:
    """Results and every artifact needed to audit the experiment."""

    config: EvaluationConfig
    protocol: Mapping[str, Any]
    detectors: Mapping[str, DetectorEvaluation]
    acceptance: Mapping[str, Any]
    scenario_manifests: Mapping[str, Sequence[Mapping[str, Any]]]
    window_compiler_artifact: Mapping[str, Any]
    detector_artifacts: Mapping[str, Mapping[str, Any]]
    limitations: Tuple[str, ...]
    environment: Mapping[str, str]

    def to_dict(self, include_runtime: bool = True) -> Dict[str, Any]:
        detector_payload = {
            name: asdict(result) for name, result in self.detectors.items()
        }
        if not include_runtime:
            for result in detector_payload.values():
                result.pop("mean_streaming_scoring_ms_per_point")
        return {
            "schema_version": 1,
            "config": _config_payload(self.config),
            "protocol": dict(self.protocol),
            "detectors": detector_payload,
            "acceptance": dict(self.acceptance),
            "scenario_manifests": {
                key: list(value) for key, value in self.scenario_manifests.items()
            },
            "window_compiler_artifact": dict(self.window_compiler_artifact),
            "detector_artifacts": {
                key: dict(value) for key, value in self.detector_artifacts.items()
            },
            "limitations": list(self.limitations),
            "environment": dict(self.environment),
        }


def run_evaluation(config: EvaluationConfig = EvaluationConfig()) -> EvaluationReport:
    """Train on normal/noisy scenarios and evaluate on held-out structural drift."""

    training_scenarios = [
        generate_scenario(
            ScenarioSpec(
                seed=seed,
                length=config.scenario_length,
                noise_events=config.noise_events,
                include_structural=False,
            )
        )
        for seed in config.train_seeds
    ]
    test_scenarios = [
        generate_scenario(
            ScenarioSpec(
                seed=seed,
                length=config.scenario_length,
                noise_events=config.noise_events,
                include_structural=True,
            )
        )
        for seed in config.test_seeds
    ]

    compiler = WindowCompiler(config.lookback).fit(
        np.concatenate([scenario.telemetry for scenario in training_scenarios], axis=0)
    )
    training_windows = _combine_windows(
        [
            compiler.transform(scenario.telemetry, scenario.feature_names)
            for scenario in training_scenarios
        ]
    )
    compiled_tests = [
        compiler.transform(scenario.telemetry, scenario.feature_names)
        for scenario in test_scenarios
    ]

    detectors = (
        PersistenceDetector(config.calibration_quantile),
        RobustFeatureDetector(config.calibration_quantile),
        LatentPredictiveDetector(
            latent_dimension=config.latent_dimension,
            ridge=config.ridge,
            calibration_quantile=config.calibration_quantile,
        ),
        CoherentLatentPredictiveDetector(
            latent_dimension=config.latent_dimension,
            ridge=config.ridge,
            calibration_quantile=config.calibration_quantile,
            consensus_rank=config.consensus_rank,
        ),
    )
    results: Dict[str, DetectorEvaluation] = {}
    detector_artifacts: Dict[str, Mapping[str, Any]] = {}
    for detector in detectors:
        detector.fit(training_windows)
        scored_tests = [detector.score(windows) for windows in compiled_tests]
        results[detector.kind] = _summarize_detector(
            test_scenarios,
            compiled_tests,
            scored_tests,
            _measure_streaming_scoring(detector, compiled_tests),
        )
        detector_artifacts[detector.kind] = detector.to_dict()

    latent = results[CoherentLatentPredictiveDetector.kind]
    persistence = results[PersistenceDetector.kind]
    gates = {
        "structural_event_recall_at_least_0_8": (
            latent.structural_event_recall >= 0.8
        ),
        "routine_noise_alert_rate_at_most_0_1": (
            latent.routine_noise_alert_rate <= 0.1
        ),
        "attribution_hit_at_3_at_least_0_8": (
            latent.attribution_hit_at_3 >= 0.8
        ),
        "attribution_hit_at_3_above_random_chance": (
            latent.attribution_hit_at_3
            > _random_hit_at_k(
                feature_count=len(test_scenarios[0].feature_names),
                affected_count=3,
                k=3,
            )
        ),
        "fewer_noise_alerts_than_persistence": (
            latent.routine_noise_alert_rate < persistence.routine_noise_alert_rate
        ),
        "mean_scoring_below_1_ms_per_point": (
            latent.mean_streaming_scoring_ms_per_point < 1.0
        ),
    }
    acceptance: Dict[str, Any] = {
        "all_passed": all(gates.values()),
        "gates": gates,
    }
    protocol = {
        "training_scenario_count": len(training_scenarios),
        "test_scenario_count": len(test_scenarios),
        "training_points": sum(len(item.telemetry) for item in training_scenarios),
        "test_points": sum(len(item.telemetry) for item in test_scenarios),
        "training_structural_points": int(
            sum(
                np.count_nonzero(item.phases == Phase.STRUCTURAL.value)
                for item in training_scenarios
            )
        ),
        "calibration_source": "training scores only",
        "test_seed_overlap": False,
        "attribution_hit_at_3": (
            "binary any-affected-feature hit over detected events"
        ),
        "attribution_recall_at_3": (
            "affected-feature recall over all events; missed events score zero"
        ),
        "attribution_random_hit_at_3": _random_hit_at_k(
            feature_count=len(test_scenarios[0].feature_names),
            affected_count=3,
            k=3,
        ),
    }
    return EvaluationReport(
        config=config,
        protocol=protocol,
        detectors=results,
        acceptance=acceptance,
        scenario_manifests={
            "training": [item.manifest for item in training_scenarios],
            "test": [item.manifest for item in test_scenarios],
        },
        window_compiler_artifact=compiler.to_dict(),
        detector_artifacts=detector_artifacts,
        limitations=(
            "Synthetic scenarios share one generator family and are not evidence of "
            "real-world zero-day detection.",
            "The linear latent target encoder is fitted with PCA; this is not a "
            "learned JEPA encoder or evidence for JEPA-specific advantages.",
            "Injected affected-feature labels support associative attribution "
            "evaluation, not causal root-cause identification.",
            "The runtime measurement scores one window at a time in Python; it "
            "is not a production OpenTelemetry throughput benchmark.",
        ),
        environment={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )


def write_evaluation_artifacts(
    report: EvaluationReport, output_directory: Path
) -> Dict[str, Path]:
    """Write auditable machine- and human-readable experiment evidence."""

    output = Path(output_directory)
    detector_directory = output / "detectors"
    detector_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output / "evaluation.json",
        "report": output / "report.md",
        "scenario_manifest": output / "scenario-manifest.json",
        "window_compiler": output / "window-compiler.json",
    }
    paths["evaluation"].write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    paths["report"].write_text(_markdown_report(report))
    paths["scenario_manifest"].write_text(
        json.dumps(
            report.scenario_manifests,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    paths["window_compiler"].write_text(
        json.dumps(
            report.window_compiler_artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    for name, artifact in report.detector_artifacts.items():
        key = f"detector_{name}"
        paths[key] = detector_directory / f"{name}.json"
        paths[key].write_text(
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return paths


def _combine_windows(windows: Sequence[ModelWindows]) -> ModelWindows:
    if not windows:
        raise ValueError("at least one ModelWindows value is required")
    return ModelWindows(
        contexts=np.concatenate([item.contexts for item in windows], axis=0),
        targets=np.concatenate([item.targets for item in windows], axis=0),
        point_indices=np.arange(
            sum(len(item.targets) for item in windows), dtype=np.int64
        ),
        feature_names=windows[0].feature_names,
    )


def _summarize_detector(
    scenarios: Sequence[Scenario],
    windows: Sequence[ModelWindows],
    scores: Sequence[DetectionScores],
    mean_streaming_scoring_ms_per_point: float,
) -> DetectorEvaluation:
    noise_points = 0
    noise_alerts = 0
    normal_points = 0
    normal_alerts = 0
    structural_points = 0
    structural_alerts = 0
    event_detections = 0
    delays: List[float] = []
    attribution_hits: List[float] = []
    attribution_scores: List[float] = []
    noise_scores: List[NDArray[np.float64]] = []
    structural_scores: List[NDArray[np.float64]] = []

    for scenario, compiled, detection in zip(scenarios, windows, scores):
        phases = scenario.phases[compiled.point_indices]
        noise_mask = phases == Phase.ROUTINE_NOISE.value
        normal_mask = phases == Phase.NORMAL.value
        structural_mask = phases == Phase.STRUCTURAL.value

        noise_points += int(np.count_nonzero(noise_mask))
        noise_alerts += int(np.count_nonzero(detection.alerts & noise_mask))
        normal_points += int(np.count_nonzero(normal_mask))
        normal_alerts += int(np.count_nonzero(detection.alerts & normal_mask))
        structural_points += int(np.count_nonzero(structural_mask))
        structural_alerts += int(np.count_nonzero(detection.alerts & structural_mask))
        noise_scores.append(detection.scores[noise_mask])
        structural_scores.append(detection.scores[structural_mask])

        event_positions = np.flatnonzero(structural_mask)
        detected_positions = np.flatnonzero(detection.alerts & structural_mask)
        if len(detected_positions) == 0:
            attribution_scores.append(0.0)
            continue
        event_detections += 1
        first_detection = int(detected_positions[0])
        delays.append(float(first_detection - int(event_positions[0])))
        original_index = int(compiled.point_indices[first_detection])
        expected = scenario.affected_features[original_index]
        evidence_start = max(0, first_detection - compiled.contexts.shape[1] + 1)
        if detection.signed_feature_evidence is not None:
            buffered_evidence = np.abs(
                np.mean(
                    detection.signed_feature_evidence[
                        evidence_start : first_detection + 1
                    ],
                    axis=0,
                )
            )
        else:
            buffered_evidence = np.mean(
                detection.feature_evidence[evidence_start : first_detection + 1],
                axis=0,
            )
        top_count = min(3, detection.feature_evidence.shape[1])
        top_features = np.argsort(buffered_evidence)[-top_count:]
        expected_count = min(top_count, int(np.count_nonzero(expected)))
        hits = int(np.count_nonzero(expected[top_features]))
        attribution_hits.append(float(hits > 0))
        attribution_scores.append(hits / expected_count if expected_count else 0.0)

    event_count = len(scenarios)
    return DetectorEvaluation(
        threshold=float(scores[0].threshold),
        routine_noise_points=noise_points,
        routine_noise_alerts=noise_alerts,
        routine_noise_alert_rate=_rate(noise_alerts, noise_points),
        normal_points=normal_points,
        normal_alerts=normal_alerts,
        normal_alert_rate=_rate(normal_alerts, normal_points),
        structural_events=event_count,
        structural_events_detected=event_detections,
        structural_event_recall=_rate(event_detections, event_count),
        structural_point_alert_rate=_rate(structural_alerts, structural_points),
        mean_detection_delay_points=float(np.mean(delays)) if delays else None,
        attribution_hit_at_3=(
            float(np.mean(attribution_hits)) if attribution_hits else 0.0
        ),
        attribution_recall_at_3=float(np.mean(attribution_scores)),
        median_noise_score=_concatenated_median(noise_scores),
        median_structural_score=_concatenated_median(structural_scores),
        mean_streaming_scoring_ms_per_point=(
            mean_streaming_scoring_ms_per_point
        ),
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _random_hit_at_k(feature_count: int, affected_count: int, k: int) -> float:
    if not 0 < affected_count <= feature_count:
        raise ValueError("affected_count must be within the feature space")
    if not 0 < k <= feature_count:
        raise ValueError("k must be within the feature space")
    unaffected = feature_count - affected_count
    misses = comb(unaffected, k) if unaffected >= k else 0
    return 1.0 - misses / comb(feature_count, k)


def _measure_streaming_scoring(
    detector: _ScoringDetector, windows: Sequence[ModelWindows]
) -> float:
    point_count = sum(len(item.targets) for item in windows)
    if point_count == 0:
        raise ValueError("streaming benchmark requires at least one point")

    first = windows[0]
    detector.score(_one_window(first, 0))
    start_ns = perf_counter_ns()
    for item in windows:
        for index in range(len(item.targets)):
            detector.score(_one_window(item, index))
    elapsed_ns = perf_counter_ns() - start_ns
    return elapsed_ns / 1_000_000.0 / point_count


def _one_window(windows: ModelWindows, index: int) -> ModelWindows:
    return ModelWindows(
        contexts=windows.contexts[index : index + 1],
        targets=windows.targets[index : index + 1],
        point_indices=windows.point_indices[index : index + 1],
        feature_names=windows.feature_names,
    )


def _concatenated_median(values: Sequence[NDArray[np.float64]]) -> float:
    nonempty = [value for value in values if len(value)]
    return float(np.median(np.concatenate(nonempty))) if nonempty else float("nan")


def _config_payload(config: EvaluationConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["train_seeds"] = list(config.train_seeds)
    payload["test_seeds"] = list(config.test_seeds)
    return payload


def _markdown_report(report: EvaluationReport) -> str:
    status = "PASS" if report.acceptance["all_passed"] else "FAIL"
    lines = [
        "# Quantis synthetic evaluation",
        "",
        f"Overall acceptance: **{status}**",
        "",
        "This report evaluates a linear latent predictive detector and a "
        "coherence-aware variant. It does not establish the behavior of a "
        "learned JEPA.",
        "",
        "## Results",
        "",
        "| Detector | Noise alert rate | Structural event recall | "
        "Attribution hit@3 | Attribution recall@3 | Mean delay | ms / point |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report.detectors.items():
        delay = (
            f"{result.mean_detection_delay_points:.2f}"
            if result.mean_detection_delay_points is not None
            else "n/a"
        )
        lines.append(
            f"| {name} | {result.routine_noise_alert_rate:.3f} | "
            f"{result.structural_event_recall:.3f} | "
            f"{result.attribution_hit_at_3:.3f} | "
            f"{result.attribution_recall_at_3:.3f} | "
            f"{delay} | "
            f"{result.mean_streaming_scoring_ms_per_point:.6f} |"
        )
    lines.extend(["", "## Acceptance gates", ""])
    for gate, passed in report.acceptance["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{gate}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in report.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)
