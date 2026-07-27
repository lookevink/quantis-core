"""Training controls and evidence for the conditioned contextual JEPA."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
    ContextualMultimodalTelemetryCorpus,
    subset_contextual_windows,
)
from .contextual_multimodal_world_model import (
    ContextualMultimodalJepaWorldModelDetector,
)
from .windowing import MAD_NORMAL_SCALE


JEPA_REFERENCES = (
    {
        "name": "I-JEPA",
        "url": "https://arxiv.org/abs/2301.08243",
        "applied_to": "large contextual target blocks",
    },
    {
        "name": "V-JEPA",
        "url": "https://arxiv.org/abs/2404.08471",
        "applied_to": (
            "continuous temporal masks, EMA targets, and robust loss"
        ),
    },
    {
        "name": "V-JEPA 2",
        "url": "https://arxiv.org/abs/2506.09985",
        "applied_to": (
            "frozen-encoder conditioned dynamics and short rollout"
        ),
    },
    {
        "name": "MJEPA",
        "url": "https://arxiv.org/abs/2606.25225",
        "applied_to": (
            "separate modality stems and explicit cross-modal objectives"
        ),
    },
)


@dataclass(frozen=True)
class ContextualMultimodalJepaTrainingConfig:
    """Deterministic v1 training and development-selection choices."""

    metric_latent_dimension: int = 3
    log_latent_dimension: int = 1
    pretraining_epochs: int = 200
    predictor_refinement_epochs: int = 100
    cross_validation_epochs: int = 40
    learning_rate: float = 2e-2
    ema_decay: float = 0.98
    weight_decay: float = 1e-4
    loss: str = "huber"
    huber_delta: float = 1.0
    auxiliary_loss_weight: float = 0.2
    rollout_loss_weight: float = 0.2
    calibration_quantile: float = 0.98
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cross_validation_epochs < 0:
            raise ValueError(
                "cross_validation_epochs cannot be negative"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_latent_dimension": (
                self.metric_latent_dimension
            ),
            "log_latent_dimension": self.log_latent_dimension,
            "pretraining_epochs": self.pretraining_epochs,
            "predictor_refinement_epochs": (
                self.predictor_refinement_epochs
            ),
            "cross_validation_epochs": (
                self.cross_validation_epochs
            ),
            "learning_rate": self.learning_rate,
            "ema_decay": self.ema_decay,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "auxiliary_loss_weight": self.auxiliary_loss_weight,
            "rollout_loss_weight": self.rollout_loss_weight,
            "calibration_quantile": self.calibration_quantile,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ContextualMultimodalJepaDevelopmentResult:
    """A v1 candidate, same-corpus controls, and development evidence."""

    config: ContextualMultimodalJepaTrainingConfig
    corpus_metadata: Mapping[str, Any]
    model_artifact: Mapping[str, Any]
    metrics_only_model_artifact: Mapping[str, Any]
    capacity_matched_model_artifact: Mapping[str, Any]
    shuffled_log_model_artifact: Mapping[str, Any]
    log_only_model_artifact: Mapping[str, Any]
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]
    cross_validation: Mapping[str, Any]
    protocol: Mapping[str, Any]
    selection: Mapping[str, Any]
    limitations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": (
                "contextual_multimodal_jepa_world_model_development"
            ),
            "config": self.config.to_dict(),
            "corpus": dict(self.corpus_metadata),
            "model": dict(self.model_artifact),
            "metrics_only_model": dict(
                self.metrics_only_model_artifact
            ),
            "capacity_matched_metrics_only_model": dict(
                self.capacity_matched_model_artifact
            ),
            "shuffled_log_model": dict(
                self.shuffled_log_model_artifact
            ),
            "log_only_model": dict(self.log_only_model_artifact),
            "metrics": {
                model_name: {
                    split_name: dict(split_metrics)
                    for split_name, split_metrics in splits.items()
                }
                for model_name, splits in self.metrics.items()
            },
            "cross_validation": dict(self.cross_validation),
            "protocol": dict(self.protocol),
            "selection": dict(self.selection),
            "limitations": list(self.limitations),
            "design_references": list(JEPA_REFERENCES),
        }


def train_contextual_multimodal_jepa_world_model(
    corpus: ContextualMultimodalTelemetryCorpus,
    config: ContextualMultimodalJepaTrainingConfig = (
        ContextualMultimodalJepaTrainingConfig()
    ),
) -> ContextualMultimodalJepaDevelopmentResult:
    """Fit the v1 candidate and controls without tuning on exposed data."""

    training = corpus.training.windows
    validation = corpus.validation.windows
    detector = _new_detector(config).fit(training)
    shuffled_training_seed = config.seed + 1001
    shuffled_validation_seed = config.seed + 2001
    shuffled_training = _shuffle_logs(
        training,
        shuffled_training_seed,
    )
    shuffled_validation = _shuffle_logs(
        validation,
        shuffled_validation_seed,
    )
    shuffled_detector = _new_detector(config).fit(
        shuffled_training
    )
    metrics_only = _new_detector(
        config,
        metric_latent_dimension=config.metric_latent_dimension,
        log_latent_dimension=0,
    ).fit(training)
    capacity_matched = _new_detector(
        config,
        metric_latent_dimension=(
            config.metric_latent_dimension
            + config.log_latent_dimension
        ),
        log_latent_dimension=0,
    ).fit(training)
    log_only = _new_detector(
        config,
        metric_latent_dimension=0,
        log_latent_dimension=config.log_latent_dimension,
    ).fit(training)

    model_artifact = detector.to_dict()
    model_artifact["preprocessing"] = dict(
        corpus.preprocessing
    )
    model_artifact["design_references"] = list(JEPA_REFERENCES)
    shuffled_artifact = shuffled_detector.to_dict()
    shuffled_artifact["preprocessing"] = dict(
        corpus.preprocessing
    )
    shuffled_artifact["control_protocol"] = {
        "kind": "shuffled_demand_residual_log_alignment",
        "training_seed": shuffled_training_seed,
        "validation_seed": shuffled_validation_seed,
        "preserves_log_context_target_blocks": True,
        "breaks_metric_log_alignment": True,
        "keeps_exogenous_controls_metric_aligned": True,
    }
    metrics_only_artifact = metrics_only.to_dict()
    capacity_matched_artifact = capacity_matched.to_dict()
    log_only_artifact = log_only.to_dict()
    for control_artifact in (
        metrics_only_artifact,
        capacity_matched_artifact,
        log_only_artifact,
    ):
        control_artifact["preprocessing"] = dict(
            corpus.preprocessing
        )
    metrics: Dict[str, Dict[str, Mapping[str, Any]]] = {
        "contextual_multimodal": {
            "training": _contextual_metrics(detector, training),
            "validation": _contextual_metrics(
                detector,
                validation,
            ),
        },
        "metrics_only": {
            "training": _contextual_metrics(
                metrics_only,
                training,
            ),
            "validation": _contextual_metrics(
                metrics_only,
                validation,
            ),
        },
        "capacity_matched_metrics_only": {
            "training": _contextual_metrics(
                capacity_matched,
                training,
            ),
            "validation": _contextual_metrics(
                capacity_matched,
                validation,
            ),
        },
        "shuffled_logs": {
            "training": _contextual_metrics(
                shuffled_detector,
                shuffled_training,
            ),
            "validation": _contextual_metrics(
                shuffled_detector,
                shuffled_validation,
            ),
        },
        "log_only": {
            "training": _contextual_metrics(
                log_only,
                training,
            ),
            "validation": _contextual_metrics(
                log_only,
                validation,
            ),
        },
        "modality_dropout": {
            "metric_context_only": _contextual_metrics(
                detector,
                validation,
                include_metric_context=True,
                include_log_context=False,
            ),
            "log_context_only": _contextual_metrics(
                detector,
                validation,
                include_metric_context=False,
                include_log_context=True,
            ),
        },
    }
    cross_validation = _cross_validate(corpus, config)
    selection = _selection_assessment(
        detector,
        cross_validation,
    )
    corpus_metadata = corpus.metadata_dict()
    protocol = {
        "model_selection_status": "development_only",
        "training_case_ids": list(corpus.training.case_ids),
        "validation_case_ids": list(corpus.validation.case_ids),
        "training_uses_validation_windows": False,
        "exposed_validation_use": "diagnostic_only",
        "cross_validation": {
            "status": cross_validation["status"],
            "grouping": (
                "canonical_request_schedule_family"
            ),
            "uses_only_training_cases": True,
            "pretraining_epochs_per_fold": (
                config.cross_validation_epochs
            ),
            "predictor_refinement_epochs_per_fold": max(
                1,
                config.cross_validation_epochs // 2,
            ),
        },
        "controls": [
            "metrics_only",
            "capacity_matched_metrics_only",
            "shuffled_logs",
            "log_only",
            "metric_context_only",
            "log_context_only",
        ],
        "corpus_metadata_sha256": _canonical_sha256(
            corpus_metadata
        ),
        "model_artifact_sha256": _canonical_sha256(
            model_artifact
        ),
        "design_references": list(JEPA_REFERENCES),
    }
    return ContextualMultimodalJepaDevelopmentResult(
        config=config,
        corpus_metadata=corpus_metadata,
        model_artifact=model_artifact,
        metrics_only_model_artifact=metrics_only_artifact,
        capacity_matched_model_artifact=(
            capacity_matched_artifact
        ),
        shuffled_log_model_artifact=shuffled_artifact,
        log_only_model_artifact=log_only_artifact,
        metrics=metrics,
        cross_validation=cross_validation,
        protocol=protocol,
        selection=selection,
        limitations=(
            "This is development evidence, not confirmation evidence.",
            "The original families 9 and 10 validation runs have already "
            "been inspected and are diagnostic only.",
            "Cross-validation uses only original training families.",
            "Future-block scoring has one-window latency because each "
            "target contains two contiguous points.",
            "Request demand and worker topology must be observable at "
            "scoring time; the lab schedule itself is never a feature.",
            "The application-log vocabulary remains limited to bounded "
            "structured events.",
            "The NumPy architecture is a small-data experiment, not a "
            "claim that video-scale JEPA recipes transfer unchanged.",
            "A new untouched corpus is required for publication evidence.",
        ),
    )


def write_contextual_multimodal_jepa_artifacts(
    result: ContextualMultimodalJepaDevelopmentResult,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Write the v1 candidate, controls, provenance, and report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "corpus": output / "corpus.json",
        "model": output / "model.json",
        "metrics_only_model": output / "metrics-only-model.json",
        "capacity_matched_model": (
            output / "capacity-matched-metrics-only-model.json"
        ),
        "shuffled_log_model": output / "shuffled-log-model.json",
        "log_only_model": output / "log-only-model.json",
        "development": output / "development.json",
        "report": output / "report.md",
    }
    _write_json(paths["corpus"], result.corpus_metadata)
    _write_json(paths["model"], result.model_artifact)
    _write_json(
        paths["metrics_only_model"],
        result.metrics_only_model_artifact,
    )
    _write_json(
        paths["capacity_matched_model"],
        result.capacity_matched_model_artifact,
    )
    _write_json(
        paths["shuffled_log_model"],
        result.shuffled_log_model_artifact,
    )
    _write_json(
        paths["log_only_model"],
        result.log_only_model_artifact,
    )
    _write_json(paths["development"], result.to_dict())
    paths["report"].write_text(_markdown_report(result))
    return paths


def _new_detector(
    config: ContextualMultimodalJepaTrainingConfig,
    *,
    pretraining_epochs: int = -1,
    predictor_refinement_epochs: int = -1,
    metric_latent_dimension: int = -1,
    log_latent_dimension: int = -1,
) -> ContextualMultimodalJepaWorldModelDetector:
    return ContextualMultimodalJepaWorldModelDetector(
        metric_latent_dimension=(
            config.metric_latent_dimension
            if metric_latent_dimension < 0
            else metric_latent_dimension
        ),
        log_latent_dimension=(
            config.log_latent_dimension
            if log_latent_dimension < 0
            else log_latent_dimension
        ),
        pretraining_epochs=(
            config.pretraining_epochs
            if pretraining_epochs < 0
            else pretraining_epochs
        ),
        predictor_refinement_epochs=(
            config.predictor_refinement_epochs
            if predictor_refinement_epochs < 0
            else predictor_refinement_epochs
        ),
        learning_rate=config.learning_rate,
        ema_decay=config.ema_decay,
        weight_decay=config.weight_decay,
        loss=config.loss,
        huber_delta=config.huber_delta,
        auxiliary_loss_weight=config.auxiliary_loss_weight,
        rollout_loss_weight=config.rollout_loss_weight,
        calibration_quantile=config.calibration_quantile,
        seed=config.seed,
    )


def _shuffle_logs(
    windows: ContextualMultimodalModelWindows,
    seed: int,
) -> ContextualMultimodalModelWindows:
    permutation = np.random.default_rng(seed).permutation(
        len(windows.log_contexts)
    )
    return ContextualMultimodalModelWindows(
        metric_contexts=windows.metric_contexts,
        log_contexts=windows.log_contexts[permutation],
        metric_target_blocks=windows.metric_target_blocks,
        log_target_blocks=windows.log_target_blocks[permutation],
        target_controls=windows.target_controls,
        point_indices=windows.point_indices,
        metric_feature_names=windows.metric_feature_names,
        log_feature_names=windows.log_feature_names,
        control_feature_names=windows.control_feature_names,
        horizons=windows.horizons,
        target_block_size=windows.target_block_size,
    )


def _contextual_metrics(
    detector: ContextualMultimodalJepaWorldModelDetector,
    windows: ContextualMultimodalModelWindows,
    *,
    include_metric_context: bool = True,
    include_log_context: bool = True,
) -> Mapping[str, Any]:
    scores = detector.score_with_context(
        windows,
        include_metric_context=include_metric_context,
        include_log_context=include_log_context,
    )
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


def refit_contextual_fold_preprocessing(
    training: ContextualMultimodalModelWindows,
    held_out: ContextualMultimodalModelWindows,
    preprocessing: Mapping[str, Any],
) -> Tuple[
    ContextualMultimodalModelWindows,
    ContextualMultimodalModelWindows,
    Mapping[str, Any],
]:
    """Refit every normalized channel without held-out family values."""

    metric_source = dict(
        dict(preprocessing["metrics"])["normalizer"]
    )
    log_source = dict(
        dict(preprocessing["logs"])["normalizer"]
    )
    control_source = dict(
        dict(preprocessing["controls"])["normalizer"]
    )
    raw_training = _unnormalized_windows(
        training,
        metric_source,
        log_source,
        control_source,
    )
    raw_held_out = _unnormalized_windows(
        held_out,
        metric_source,
        log_source,
        control_source,
    )
    metric_state = _fit_fold_normalizer(
        raw_training.metric_contexts,
        raw_training.metric_target_blocks,
    )
    log_state = _fit_fold_normalizer(
        raw_training.log_contexts,
        raw_training.log_target_blocks,
    )
    control_state = _fit_fold_normalizer(
        None,
        raw_training.target_controls,
    )
    artifact = {
        "schema_version": 1,
        "kind": "fold_training_only_normalization",
        "metrics": metric_state,
        "logs": log_state,
        "controls": control_state,
        "held_out_values_used": False,
    }
    return (
        _normalize_fold_windows(
            raw_training,
            metric_state,
            log_state,
            control_state,
        ),
        _normalize_fold_windows(
            raw_held_out,
            metric_state,
            log_state,
            control_state,
        ),
        artifact,
    )


def _unnormalized_windows(
    windows: ContextualMultimodalModelWindows,
    metric_state: Mapping[str, Any],
    log_state: Mapping[str, Any],
    control_state: Mapping[str, Any],
) -> ContextualMultimodalModelWindows:
    return ContextualMultimodalModelWindows(
        metric_contexts=_unscale(
            windows.metric_contexts,
            metric_state,
        ),
        log_contexts=_unscale(
            windows.log_contexts,
            log_state,
        ),
        metric_target_blocks=_unscale(
            windows.metric_target_blocks,
            metric_state,
        ),
        log_target_blocks=_unscale(
            windows.log_target_blocks,
            log_state,
        ),
        target_controls=_unscale(
            windows.target_controls,
            control_state,
        ),
        point_indices=windows.point_indices,
        metric_feature_names=windows.metric_feature_names,
        log_feature_names=windows.log_feature_names,
        control_feature_names=windows.control_feature_names,
        horizons=windows.horizons,
        target_block_size=windows.target_block_size,
    )


def _normalize_fold_windows(
    windows: ContextualMultimodalModelWindows,
    metric_state: Mapping[str, Any],
    log_state: Mapping[str, Any],
    control_state: Mapping[str, Any],
) -> ContextualMultimodalModelWindows:
    return ContextualMultimodalModelWindows(
        metric_contexts=_scale(
            windows.metric_contexts,
            metric_state,
        ),
        log_contexts=_scale(
            windows.log_contexts,
            log_state,
        ),
        metric_target_blocks=_scale(
            windows.metric_target_blocks,
            metric_state,
        ),
        log_target_blocks=_scale(
            windows.log_target_blocks,
            log_state,
        ),
        target_controls=_scale(
            windows.target_controls,
            control_state,
        ),
        point_indices=windows.point_indices,
        metric_feature_names=windows.metric_feature_names,
        log_feature_names=windows.log_feature_names,
        control_feature_names=windows.control_feature_names,
        horizons=windows.horizons,
        target_block_size=windows.target_block_size,
    )


def _fit_fold_normalizer(
    contexts: Optional[NDArray[np.float64]],
    targets: NDArray[np.float64],
) -> Mapping[str, Any]:
    groups = [targets.reshape(-1, targets.shape[-1])]
    if contexts is not None:
        groups.insert(
            0,
            contexts.reshape(-1, contexts.shape[-1]),
        )
    values = np.concatenate(groups, axis=0)
    location = np.median(values, axis=0)
    scale = (
        MAD_NORMAL_SCALE
        * np.median(np.abs(values - location), axis=0)
    )
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {
        "schema_version": 1,
        "kind": "robust_location_scale",
        "location": location.tolist(),
        "scale": scale.tolist(),
        "fit_value_count": len(values),
    }


def _unscale(
    values: NDArray[np.float64],
    state: Mapping[str, Any],
) -> NDArray[np.float64]:
    location = np.asarray(state["location"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    return values * scale + location


def _scale(
    values: NDArray[np.float64],
    state: Mapping[str, Any],
) -> NDArray[np.float64]:
    location = np.asarray(state["location"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    return (values - location) / scale


def _cross_validate(
    corpus: ContextualMultimodalTelemetryCorpus,
    config: ContextualMultimodalJepaTrainingConfig,
) -> Mapping[str, Any]:
    if config.cross_validation_epochs == 0:
        return {
            "status": "disabled",
            "folds": [],
            "uses_exposed_validation": False,
        }
    groups = _training_schedule_groups(corpus)
    if len(groups) < 3:
        return {
            "status": "insufficient_groups",
            "group_count": len(groups),
            "folds": [],
            "uses_exposed_validation": False,
        }
    case_ids = np.asarray(
        corpus.training.window_case_ids,
        dtype=object,
    )
    folds: List[Dict[str, Any]] = []
    for fold_index, group in enumerate(groups):
        held_out = np.isin(case_ids, np.asarray(group, dtype=object))
        training_windows = subset_contextual_windows(
            corpus.training.windows,
            ~held_out,
        )
        held_out_windows = subset_contextual_windows(
            corpus.training.windows,
            held_out,
        )
        (
            training_windows,
            held_out_windows,
            fold_preprocessing,
        ) = refit_contextual_fold_preprocessing(
            training_windows,
            held_out_windows,
            corpus.preprocessing,
        )
        refinement_epochs = max(
            1,
            config.cross_validation_epochs // 2,
        )
        fold_detector = _new_detector(
            config,
            pretraining_epochs=config.cross_validation_epochs,
            predictor_refinement_epochs=refinement_epochs,
        ).fit(training_windows)
        metric_detector = _new_detector(
            config,
            pretraining_epochs=config.cross_validation_epochs,
            predictor_refinement_epochs=refinement_epochs,
            metric_latent_dimension=(
                config.metric_latent_dimension
            ),
            log_latent_dimension=0,
        ).fit(training_windows)
        folds.append(
            {
                "fold": fold_index + 1,
                "held_out_case_ids": list(group),
                "training_case_ids": sorted(
                    set(corpus.training.case_ids) - set(group)
                ),
                "preprocessing": fold_preprocessing,
                "predictor_refinement_epochs": (
                    refinement_epochs
                ),
                "contextual_multimodal": _contextual_metrics(
                    fold_detector,
                    held_out_windows,
                ),
                "metrics_only": _contextual_metrics(
                    metric_detector,
                    held_out_windows,
                ),
            }
        )
    contextual_rates = np.asarray(
        [
            fold["contextual_multimodal"]["alert_rate"]
            for fold in folds
        ],
        dtype=np.float64,
    )
    metric_rates = np.asarray(
        [
            fold["metrics_only"]["alert_rate"]
            for fold in folds
        ],
        dtype=np.float64,
    )
    return {
        "status": "completed",
        "folds": folds,
        "uses_exposed_validation": False,
        "summary": {
            "fold_count": len(folds),
            "contextual_mean_alert_rate": float(
                np.mean(contextual_rates)
            ),
            "metrics_only_mean_alert_rate": float(
                np.mean(metric_rates)
            ),
            "improved_fold_fraction": float(
                np.mean(contextual_rates < metric_rates)
            ),
            "no_worse_fold_fraction": float(
                np.mean(contextual_rates <= metric_rates)
            ),
        },
    }


def _training_schedule_groups(
    corpus: ContextualMultimodalTelemetryCorpus,
) -> Tuple[Tuple[str, ...], ...]:
    base = corpus.base_corpus_metadata
    metric_corpus = dict(base["metric_corpus"])
    protocol = dict(metric_corpus["protocol"])
    runs = dict(protocol["runs"])
    grouped: Dict[Tuple[int, ...], list[str]] = {}
    for case_id in corpus.training.case_ids:
        run = dict(runs[case_id])
        schedule = tuple(
            int(value)
            for value in run["canonical_request_schedule"]
        )
        grouped.setdefault(schedule, []).append(case_id)
    return tuple(
        tuple(sorted(grouped[schedule]))
        for schedule in sorted(grouped)
    )


def _selection_assessment(
    detector: ContextualMultimodalJepaWorldModelDetector,
    cross_validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if cross_validation["status"] != "completed":
        return {
            "status": "not_assessed",
            "reason": (
                "family-held-out cross-validation was not completed"
            ),
            "publication_eligible": False,
        }
    summary = dict(cross_validation["summary"])
    diagnostic = detector.diagnostics
    gates = {
        "conditional_mean_improvement_over_metrics_only": {
            "observed": summary[
                "contextual_mean_alert_rate"
            ],
            "comparator": summary[
                "metrics_only_mean_alert_rate"
            ],
            "passed": (
                summary["contextual_mean_alert_rate"]
                < summary["metrics_only_mean_alert_rate"]
            ),
        },
        "no_worse_on_at_least_half_of_folds": {
            "observed": summary["no_worse_fold_fraction"],
            "minimum": 0.5,
            "passed": summary["no_worse_fold_fraction"] >= 0.5,
        },
        "metric_active_latent_rank": {
            "observed": diagnostic["metric_effective_rank"],
            "minimum": 0.5 * detector.metric_latent_dimension,
            "passed": (
                diagnostic["metric_effective_rank"]
                >= 0.5 * detector.metric_latent_dimension
            ),
        },
        "log_active_latent_rank": {
            "observed": diagnostic["log_effective_rank"],
            "minimum": 0.5 * detector.log_latent_dimension,
            "passed": (
                diagnostic["log_effective_rank"]
                >= 0.5 * detector.log_latent_dimension
            ),
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "status": "passed" if passed else "failed",
        "all_passed": passed,
        "gates": gates,
        "publication_eligible": False,
        "publication_blocker": (
            "a new untouched corpus is required regardless of "
            "development selection"
        ),
    }


def _markdown_report(
    result: ContextualMultimodalJepaDevelopmentResult,
) -> str:
    contextual = result.metrics["contextual_multimodal"]
    metrics_only = result.metrics["metrics_only"]
    shuffled = result.metrics["shuffled_logs"]
    lines = [
        "# Quantis contextual metrics + logs JEPA development",
        "",
        "Status: **development only**",
        "",
        "The original validation families were previously inspected; "
        "their results below are diagnostic and cannot support "
        "publication.",
        "",
        "## Contextual conditioned JEPA",
        "",
        _metric_line("Training", contextual["training"]),
        _metric_line(
            "Previously exposed validation",
            contextual["validation"],
        ),
        "",
        "## Same-corpus controls",
        "",
        _metric_line(
            "Metrics-only validation",
            metrics_only["validation"],
        ),
        _metric_line(
            "Shuffled-log validation",
            shuffled["validation"],
        ),
        _metric_line(
            "Log-only validation",
            result.metrics["log_only"]["validation"],
        ),
        "",
        "## Family-held-out development selection",
        "",
        f"- Cross-validation: {result.cross_validation['status']}",
        f"- Selection: {result.selection['status']}",
        "- Publication eligible: no; a new untouched corpus is required",
        "",
        "## Primary JEPA references",
        "",
    ]
    lines.extend(
        (
            f"- [{reference['name']}]({reference['url']}): "
            f"{reference['applied_to']}"
        )
        for reference in JEPA_REFERENCES
    )
    lines.extend(["", "## Limitations", ""])
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
        f"mean calibrated loss "
        f"{float(metrics['latent_loss_mean']):.6f}, "
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
