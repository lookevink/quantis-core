"""Training, comparison, and artifacts for application-log JEPA development."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from .detectors import JepaWorldModelDetector
from .multimodal_corpus import (
    MultimodalModelWindows,
    MultimodalTelemetryCorpus,
)
from .multimodal_world_model import (
    MultimodalJepaWorldModelDetector,
)
from .windowing import ModelWindows


@dataclass(frozen=True)
class MultimodalJepaTrainingConfig:
    """Deterministic choices shared by multimodal and baseline training."""

    metric_latent_dimension: int = 3
    log_latent_dimension: int = 2
    epochs: int = 200
    learning_rate: float = 2e-2
    ema_decay: float = 0.98
    weight_decay: float = 1e-4
    calibration_quantile: float = 0.98
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_latent_dimension": (
                self.metric_latent_dimension
            ),
            "log_latent_dimension": self.log_latent_dimension,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "ema_decay": self.ema_decay,
            "weight_decay": self.weight_decay,
            "calibration_quantile": self.calibration_quantile,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MultimodalJepaDevelopmentResult:
    """Comparable multimodal and metric-only models with one corpus."""

    config: MultimodalJepaTrainingConfig
    corpus_metadata: Mapping[str, Any]
    model_artifact: Mapping[str, Any]
    metrics_only_model_artifact: Mapping[str, Any]
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]
    protocol: Mapping[str, Any]
    limitations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "multimodal_jepa_world_model_development",
            "config": self.config.to_dict(),
            "corpus": dict(self.corpus_metadata),
            "model": dict(self.model_artifact),
            "metrics_only_model": dict(
                self.metrics_only_model_artifact
            ),
            "metrics": {
                model_name: {
                    split_name: dict(split_metrics)
                    for split_name, split_metrics in splits.items()
                }
                for model_name, splits in self.metrics.items()
            },
            "protocol": dict(self.protocol),
            "limitations": list(self.limitations),
        }


def train_multimodal_jepa_world_model(
    corpus: MultimodalTelemetryCorpus,
    config: MultimodalJepaTrainingConfig = (
        MultimodalJepaTrainingConfig()
    ),
) -> MultimodalJepaDevelopmentResult:
    """Fit the fused model and a same-run metrics-only baseline."""

    multimodal = MultimodalJepaWorldModelDetector(
        metric_latent_dimension=config.metric_latent_dimension,
        log_latent_dimension=config.log_latent_dimension,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ema_decay=config.ema_decay,
        weight_decay=config.weight_decay,
        calibration_quantile=config.calibration_quantile,
        seed=config.seed,
    ).fit(corpus.training.windows)
    metrics_only = JepaWorldModelDetector(
        latent_dimension=config.metric_latent_dimension,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ema_decay=config.ema_decay,
        weight_decay=config.weight_decay,
        calibration_quantile=config.calibration_quantile,
        seed=config.seed,
    ).fit(corpus.training.windows.metric)
    model_artifact = multimodal.to_dict()
    metrics_only_artifact = metrics_only.to_dict()
    corpus_metadata = corpus.metadata_dict()
    return MultimodalJepaDevelopmentResult(
        config=config,
        corpus_metadata=corpus_metadata,
        model_artifact=model_artifact,
        metrics_only_model_artifact=metrics_only_artifact,
        metrics={
            "multimodal": {
                "training": _multimodal_metrics(
                    multimodal,
                    corpus.training.windows,
                ),
                "validation": _multimodal_metrics(
                    multimodal,
                    corpus.validation.windows,
                ),
            },
            "metrics_only": {
                "training": _metric_metrics(
                    metrics_only,
                    corpus.training.windows.metric,
                ),
                "validation": _metric_metrics(
                    metrics_only,
                    corpus.validation.windows.metric,
                ),
            },
        },
        protocol={
            "model_selection_status": "development_only",
            "training_case_ids": list(
                corpus.training.case_ids
            ),
            "validation_case_ids": list(
                corpus.validation.case_ids
            ),
            "training_uses_validation_windows": False,
            "target_encoder_update": "ema_only",
            "prediction_horizon_points": 1,
            "corpus_metadata_sha256": _canonical_sha256(
                corpus_metadata
            ),
            "model_artifact_sha256": _canonical_sha256(
                model_artifact
            ),
            "metrics_only_model_artifact_sha256": (
                _canonical_sha256(metrics_only_artifact)
            ),
        },
        limitations=(
            "This is development evidence, not confirmation evidence.",
            "The application-log vocabulary is limited to declared "
            "structured event counts.",
            "Raw message text, identifiers, payloads, and stack traces are "
            "not model features.",
            "Synchronous lab export overhead is part of the observed system.",
            "The target is one future telemetry point, not a future block.",
            "A local lab corpus does not establish production "
            "generalization.",
        ),
    )


def write_multimodal_jepa_development_artifacts(
    result: MultimodalJepaDevelopmentResult,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Write models, corpus provenance, comparison metrics, and report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "corpus": output / "corpus.json",
        "model": output / "model.json",
        "metrics_only_model": output / "metrics-only-model.json",
        "development": output / "development.json",
        "report": output / "report.md",
    }
    _write_json(paths["corpus"], result.corpus_metadata)
    _write_json(paths["model"], result.model_artifact)
    _write_json(
        paths["metrics_only_model"],
        result.metrics_only_model_artifact,
    )
    _write_json(paths["development"], result.to_dict())
    paths["report"].write_text(_markdown_report(result))
    return paths


def _multimodal_metrics(
    detector: MultimodalJepaWorldModelDetector,
    windows: MultimodalModelWindows,
) -> Mapping[str, Any]:
    scores = detector.score(windows)
    return _score_metrics(scores.scores, scores.alerts)


def _metric_metrics(
    detector: JepaWorldModelDetector,
    windows: ModelWindows,
) -> Mapping[str, Any]:
    scores = detector.score(windows)
    return _score_metrics(scores.scores, scores.alerts)


def _score_metrics(
    scores: NDArray[np.float64],
    alerts: NDArray[np.bool_],
) -> Mapping[str, Any]:
    return {
        "window_count": len(scores),
        "latent_loss_mean": float(np.mean(np.square(scores))),
        "score_median": float(np.median(scores)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "alerts": int(np.count_nonzero(alerts)),
        "alert_rate": float(np.mean(alerts)),
    }


def _markdown_report(
    result: MultimodalJepaDevelopmentResult,
) -> str:
    multimodal = result.metrics["multimodal"]
    baseline = result.metrics["metrics_only"]
    lines = [
        "# Quantis application-log JEPA development",
        "",
        "Status: **development only**",
        "",
        "This report is not confirmation evidence.",
        "",
        "## Application-log JEPA",
        "",
        _metric_line("Training", multimodal["training"]),
        _metric_line("Validation", multimodal["validation"]),
        "",
        "## Metrics-only baseline",
        "",
        _metric_line("Training", baseline["training"]),
        _metric_line("Validation", baseline["validation"]),
        "",
        "## Limitations",
        "",
    ]
    lines.extend(
        f"- {limitation}" for limitation in result.limitations
    )
    lines.append("")
    return "\n".join(lines)


def _metric_line(
    label: str,
    metrics: Mapping[str, Any],
) -> str:
    return (
        f"- {label}: {int(metrics['window_count'])} windows, "
        f"mean loss {float(metrics['latent_loss_mean']):.6f}, "
        f"alert rate {float(metrics['alert_rate']):.1%}"
    )


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
