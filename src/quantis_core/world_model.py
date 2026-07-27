"""Development training and evidence for the telemetry JEPA world model."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .detectors import JepaWorldModelDetector
from .telemetry_corpus import TelemetryCorpus, TelemetryCorpusSplit


@dataclass(frozen=True)
class JepaTrainingConfig:
    """Choices that affect deterministic JEPA development training."""

    latent_dimension: int = 4
    epochs: int = 200
    learning_rate: float = 2e-2
    ema_decay: float = 0.98
    weight_decay: float = 1e-4
    calibration_quantile: float = 0.98
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_dimension": self.latent_dimension,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "ema_decay": self.ema_decay,
            "weight_decay": self.weight_decay,
            "calibration_quantile": self.calibration_quantile,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class JepaDevelopmentResult:
    """Serializable model, corpus provenance, and split metrics."""

    config: JepaTrainingConfig
    corpus_metadata: Mapping[str, Any]
    model_artifact: Mapping[str, Any]
    metrics: Mapping[str, Mapping[str, Any]]
    protocol: Mapping[str, Any]
    limitations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "jepa_world_model_development",
            "config": self.config.to_dict(),
            "corpus": dict(self.corpus_metadata),
            "model": dict(self.model_artifact),
            "metrics": {
                name: dict(values)
                for name, values in self.metrics.items()
            },
            "protocol": dict(self.protocol),
            "limitations": list(self.limitations),
        }


def train_jepa_world_model(
    corpus: TelemetryCorpus,
    config: JepaTrainingConfig = JepaTrainingConfig(),
) -> JepaDevelopmentResult:
    """Fit one deterministic JEPA model and score held-out schedules."""

    detector = JepaWorldModelDetector(
        latent_dimension=config.latent_dimension,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ema_decay=config.ema_decay,
        weight_decay=config.weight_decay,
        calibration_quantile=config.calibration_quantile,
        seed=config.seed,
    ).fit(corpus.training.windows)
    model_artifact = detector.to_dict()
    corpus_metadata = corpus.metadata_dict()
    metrics = {
        "training": _split_metrics(detector, corpus.training),
        "validation": _split_metrics(detector, corpus.validation),
    }
    return JepaDevelopmentResult(
        config=config,
        corpus_metadata=corpus_metadata,
        model_artifact=model_artifact,
        metrics=metrics,
        protocol={
            "model_selection_status": "development_only",
            "training_case_ids": list(
                corpus.training.case_ids
            ),
            "validation_case_ids": list(
                corpus.validation.case_ids
            ),
            "corpus_metadata_sha256": _canonical_sha256(
                corpus_metadata
            ),
            "model_artifact_sha256": _canonical_sha256(
                model_artifact
            ),
            "training_uses_validation_windows": False,
            "target_encoder_update": "ema_only",
            "prediction_horizon_points": 1,
        },
        limitations=(
            "This is development evidence, not confirmation evidence.",
            "The target is one future telemetry point, not a future block.",
            "The NumPy v0 is a small training-path tracer bullet, not a "
            "production neural architecture.",
            "Request demand is handled by deterministic preprocessing rather "
            "than learned as an explicit control variable.",
            "Feature evidence is target-encoder sensitivity, not causal "
            "attribution.",
            "The six demand-conditioned metrics are a small state "
            "vocabulary.",
            "A local lab corpus does not establish production "
            "generalization.",
            "A learned joint embedding is not by itself a complete world "
            "model.",
        ),
    )


def write_jepa_development_artifacts(
    result: JepaDevelopmentResult,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Write corpus metadata, model state, and development evidence."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "corpus": output / "corpus.json",
        "model": output / "model.json",
        "development": output / "development.json",
        "report": output / "report.md",
    }
    _write_json(paths["corpus"], result.corpus_metadata)
    _write_json(paths["model"], result.model_artifact)
    _write_json(paths["development"], result.to_dict())
    paths["report"].write_text(_markdown_report(result))
    return paths


def _split_metrics(
    detector: JepaWorldModelDetector,
    split: TelemetryCorpusSplit,
) -> Mapping[str, Any]:
    scores = detector.score(split.windows)
    return {
        "case_ids": list(split.case_ids),
        "window_count": len(scores.scores),
        "latent_loss_mean": float(
            np.mean(np.square(scores.scores))
        ),
        "score_median": float(np.median(scores.scores)),
        "score_p95": float(np.quantile(scores.scores, 0.95)),
        "alerts": int(np.count_nonzero(scores.alerts)),
        "alert_rate": float(np.mean(scores.alerts)),
    }


def _markdown_report(result: JepaDevelopmentResult) -> str:
    training = result.metrics["training"]
    validation = result.metrics["validation"]
    lines = [
        "# Quantis JEPA world-model v0 development",
        "",
        "Status: **development only**",
        "",
        "This report is not confirmation evidence.",
        "",
        "## Corpus",
        "",
        f"- Training runs: {len(result.protocol['training_case_ids'])}",
        f"- Validation runs: "
        f"{len(result.protocol['validation_case_ids'])}",
        f"- Training windows: {training['window_count']}",
        f"- Validation windows: {validation['window_count']}",
        "",
        "## Latent prediction",
        "",
        f"- Training mean loss: "
        f"{float(training['latent_loss_mean']):.6f}",
        f"- Validation mean loss: "
        f"{float(validation['latent_loss_mean']):.6f}",
        f"- Training alert rate: "
        f"{float(training['alert_rate']):.1%}",
        f"- Validation alert rate: "
        f"{float(validation['alert_rate']):.1%}",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(
        f"- {limitation}" for limitation in result.limitations
    )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
