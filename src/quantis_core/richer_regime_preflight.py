"""Fit-only mechanism preflight for richer-regime alerting retries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from .richer_regime_retry import WORKLOAD_FAMILIES


@dataclass(frozen=True)
class RicherRegimeFitEvidence:
    """Transition rows from fit replicates zero and one only."""

    metric_context: NDArray[np.float64]
    event_context: NDArray[np.float64]
    targets: NDArray[np.float64]
    demand: NDArray[np.float64]
    topology: NDArray[np.float64]
    workload_families: Tuple[str, ...]
    replicates: NDArray[np.int64]
    regime_classification_accuracy: float

    def __post_init__(self) -> None:
        row_count = len(self.metric_context)
        if (
            row_count < 4
            or self.metric_context.ndim != 2
            or self.event_context.ndim != 2
            or self.targets.ndim != 2
            or self.event_context.shape[0] != row_count
            or self.targets.shape[0] != row_count
            or self.demand.shape != (row_count,)
            or self.topology.shape != (row_count,)
            or self.replicates.shape != (row_count,)
            or len(self.workload_families) != row_count
            or not all(
                family in WORKLOAD_FAMILIES
                for family in self.workload_families
            )
            or not 0.0 <= self.regime_classification_accuracy <= 1.0
            or any(
                not np.all(np.isfinite(values))
                for values in (
                    self.metric_context,
                    self.event_context,
                    self.targets,
                    self.demand,
                    self.topology,
                )
            )
        ):
            raise ValueError("richer-regime fit evidence is invalid")


def assess_richer_regime_fit_preflight(
    evidence: RicherRegimeFitEvidence,
) -> Mapping[str, Any]:
    """Measure whether fit-only mechanisms justify each expensive retry."""

    unique_replicates = set(
        int(value) for value in evidence.replicates.tolist()
    )
    if unique_replicates != {0, 1}:
        raise ValueError(
            "preflight requires fit replicates 0 and 1 only"
        )
    training = evidence.replicates == 0
    probe = evidence.replicates == 1
    families = np.asarray(evidence.workload_families, dtype=object)
    if any(
        not np.any(training & (families == family))
        or not np.any(probe & (families == family))
        for family in WORKLOAD_FAMILIES
    ):
        raise ValueError(
            "preflight requires every workload family in both replicas"
        )
    operating_context = np.column_stack(
        (evidence.demand, evidence.topology)
    )
    base_context = np.column_stack(
        (evidence.metric_context, operating_context)
    )
    family_context = _family_one_hot(evidence.workload_families)
    regime_context = np.column_stack(
        (base_context, family_context)
    )
    contextual = np.column_stack(
        (regime_context, evidence.event_context)
    )
    base_prediction = _ridge_probe_prediction(
        base_context, evidence.targets, training, probe
    )
    regime_prediction = _ridge_probe_prediction(
        regime_context, evidence.targets, training, probe
    )
    contextual_prediction = _ridge_probe_prediction(
        contextual, evidence.targets, training, probe
    )
    observed = evidence.targets[probe]
    base_mse = _mse(observed, base_prediction)
    regime_mse = _mse(observed, regime_prediction)
    contextual_mse = _mse(observed, contextual_prediction)
    residuals = observed - base_prediction
    probe_families = families[probe]
    variance_by_family = {
        family: float(
            np.mean(
                np.square(
                    residuals[probe_families == family]
                )
            )
        )
        for family in WORKLOAD_FAMILIES
    }
    positive_variances = [
        max(value, 1e-12) for value in variance_by_family.values()
    ]
    heteroscedastic_ratio = max(positive_variances) / min(
        positive_variances
    )
    multimodal_reduction = _two_cluster_sse_reduction(residuals)
    contextual_ratio = contextual_mse / max(base_mse, 1e-12)
    regime_ratio = regime_mse / max(base_mse, 1e-12)
    event_ratio = contextual_mse / max(regime_mse, 1e-12)
    gates = {
        "fit_probe_role_isolation": bool(
            np.any(training) and np.any(probe)
        ),
        "all_workload_families_observed": set(
            evidence.workload_families
        )
        == set(WORKLOAD_FAMILIES),
        "operating_regime_observable": (
            evidence.regime_classification_accuracy >= 0.9
        ),
        "finite_mechanism_measurements": all(
            math.isfinite(value)
            for value in (
                base_mse,
                regime_mse,
                contextual_mse,
                heteroscedastic_ratio,
                multimodal_reduction,
            )
        ),
    }
    recommendations: Dict[str, str] = {
        "contextual_multimodal_jepa": (
            "retry"
            if contextual_ratio <= 0.95
            and evidence.regime_classification_accuracy >= 0.9
            else "do_not_retry"
        ),
        "hepa": "retry" if event_ratio <= 0.95 else "do_not_retry",
        "error_certificate_jepa": (
            "retry"
            if heteroscedastic_ratio >= 1.5
            else "do_not_retry"
        ),
        "multi_hypothesis_jepa": (
            "retry"
            if multimodal_reduction >= 0.1
            else "do_not_retry"
        ),
    }
    return {
        "schema_version": 1,
        "kind": "richer_regime_fit_preflight",
        "status": "qualified" if all(gates.values()) else "failed",
        "evidence_boundary": (
            "fit replicas 0 and 1 only; no selection, calibration, "
            "or evaluation evidence"
        ),
        "measurements": {
            "base_metrics_mse": base_mse,
            "regime_context_mse": regime_mse,
            "contextual_metrics_events_mse": contextual_mse,
            "regime_context_mse_ratio": regime_ratio,
            "event_context_mse_ratio": event_ratio,
            "contextual_mse_ratio": contextual_ratio,
            "heteroscedastic_variance_ratio": heteroscedastic_ratio,
            "two_cluster_residual_sse_reduction": (
                multimodal_reduction
            ),
            "regime_classification_accuracy": (
                evidence.regime_classification_accuracy
            ),
        },
        "residual_variance_by_workload_family": variance_by_family,
        "gates": gates,
        "recommendations": recommendations,
    }


def _ridge_probe_prediction(
    context: NDArray[np.float64],
    targets: NDArray[np.float64],
    training: NDArray[np.bool_],
    probe: NDArray[np.bool_],
) -> NDArray[np.float64]:
    train_x = np.asarray(context[training], dtype=np.float64)
    probe_x = np.asarray(context[probe], dtype=np.float64)
    train_y = np.asarray(targets[training], dtype=np.float64)
    x_mean = np.mean(train_x, axis=0)
    x_scale = np.std(train_x, axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = np.mean(train_y, axis=0)
    y_scale = np.std(train_y, axis=0)
    y_scale[y_scale < 1e-8] = 1.0
    normalized_x = (train_x - x_mean) / x_scale
    normalized_y = (train_y - y_mean) / y_scale
    design = np.column_stack(
        (np.ones(len(normalized_x)), normalized_x)
    )
    ridge = np.eye(design.shape[1], dtype=np.float64) * 1e-3
    ridge[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + ridge,
        design.T @ normalized_y,
    )
    probe_design = np.column_stack(
        (
            np.ones(len(probe_x)),
            (probe_x - x_mean) / x_scale,
        )
    )
    return np.asarray(
        (probe_design @ weights) * y_scale + y_mean,
        dtype=np.float64,
    )


def _family_one_hot(families: Tuple[str, ...]) -> NDArray[np.float64]:
    return np.asarray(
        [
            [1.0 if family == expected else 0.0 for expected in WORKLOAD_FAMILIES]
            for family in families
        ],
        dtype=np.float64,
    )


def _mse(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> float:
    return float(np.mean(np.square(observed - predicted)))


def _two_cluster_sse_reduction(
    residuals: NDArray[np.float64],
) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    center = np.mean(values, axis=0)
    one_sse = float(np.sum(np.square(values - center)))
    if one_sse <= 1e-12:
        return 0.0
    norms = np.linalg.norm(values - center, axis=1)
    first = values[int(np.argmin(norms))].copy()
    second = values[int(np.argmax(norms))].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(30):
        distances = np.column_stack(
            (
                np.sum(np.square(values - first), axis=1),
                np.sum(np.square(values - second), axis=1),
            )
        )
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels) and np.any(labels == 1):
            break
        labels = updated
        if not np.any(labels == 0) or not np.any(labels == 1):
            return 0.0
        first = np.mean(values[labels == 0], axis=0)
        second = np.mean(values[labels == 1], axis=0)
    two_sse = float(
        np.sum(np.square(values[labels == 0] - first))
        + np.sum(np.square(values[labels == 1] - second))
    )
    return max(0.0, min(1.0, 1.0 - two_sse / one_sse))
