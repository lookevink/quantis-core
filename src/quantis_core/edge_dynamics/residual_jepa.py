"""Frozen low-rank dynamics with auxiliary JEPA residual correction."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
)
from .action_conditioned_jepa import ActionConditionedJepaDynamics
from .models import ContractiveLowRankDynamics


class FrozenBaselineResidualDynamics:
    """Add a learned residual without mutating the fitted raw-state model."""

    kind = "frozen_low_rank_residual_jepa_dynamics_v1"

    def __init__(
        self,
        *,
        baseline: ContractiveLowRankDynamics,
        correction: ActionConditionedJepaDynamics,
        correction_gain: float = 1.0,
    ) -> None:
        self.baseline = baseline
        self.correction = correction
        self._correction_gain = self._validate_gain(correction_gain)
        self._selection_curve: Tuple[Mapping[str, float], ...] = ()
        self._fitted = False
        self._baseline_sha256 = ""

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "FrozenBaselineResidualDynamics":
        """Fit only the correction branch to frozen baseline errors."""

        baseline_artifact = self.baseline.to_dict()
        baseline_bytes = _canonical_json_bytes(baseline_artifact)
        baseline_prediction = self.baseline.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        residual_targets = (
            np.asarray(windows.future_states, dtype=np.float64)
            - baseline_prediction
        )
        self.correction.fit_with_decoded_targets(
            windows, residual_targets
        )
        if _canonical_json_bytes(self.baseline.to_dict()) != baseline_bytes:
            raise RuntimeError("residual fitting mutated the frozen baseline")
        self._baseline_sha256 = hashlib.sha256(
            baseline_bytes
        ).hexdigest()
        self._fitted = True
        return self

    def set_correction_gain(
        self, gain: float
    ) -> "FrozenBaselineResidualDynamics":
        """Set a bounded correction gain."""

        self._correction_gain = self._validate_gain(gain)
        return self

    def select_correction_gain(
        self,
        windows: ActionConditionedWindows,
        candidates: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> float:
        """Choose gain by selection action-overlap MSE."""

        self._require_fitted()
        gains = tuple(self._validate_gain(value) for value in candidates)
        if not gains or len(set(gains)) != len(gains):
            raise ValueError("correction gain candidates must be unique")
        baseline = self.baseline.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        correction = self.correction.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        observed = np.asarray(windows.future_states, dtype=np.float64)
        try:
            applicable = windows.action_feature_names.index("applicable")
        except ValueError as error:
            raise ValueError(
                "gain selection requires the applicable action feature"
            ) from error
        action_overlap = np.any(
            windows.future_actions[..., applicable] > 0.5, axis=2
        )
        if not np.any(action_overlap):
            raise ValueError("gain selection has no action-overlap windows")
        rows = []
        for gain in sorted(gains):
            prediction = baseline + gain * correction
            squared_error = np.square(prediction - observed)
            rows.append(
                {
                    "gain": gain,
                    "normalized_mse_overall": float(
                        np.mean(squared_error)
                    ),
                    "normalized_mse_action_overlap": float(
                        np.mean(squared_error[action_overlap])
                    ),
                }
            )
        selected = min(
            rows,
            key=lambda row: (
                row["normalized_mse_action_overlap"],
                row["gain"],
            ),
        )
        self._selection_curve = tuple(rows)
        self._correction_gain = selected["gain"]
        return self._correction_gain

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: Any,
    ) -> TrajectoryDistribution:
        """Return frozen prediction plus the selected residual correction."""

        self._require_fitted()
        baseline = self.baseline.rollout(
            histories, future_controls, future_actions, graph
        )
        if self._correction_gain == 0.0:
            return baseline
        correction = self.correction.rollout(
            histories, future_controls, future_actions, graph
        )
        return TrajectoryDistribution(
            mean=(
                baseline.mean
                + self._correction_gain * correction.mean
            ),
            variance=(
                baseline.variance
                + self._correction_gain**2 * correction.variance
            ),
        )

    def latent_divergence(
        self, windows: ActionConditionedWindows
    ) -> NDArray[np.float64]:
        """Return JEPA future-token divergence without decoded correction."""

        self._require_fitted()
        return self.correction.latent_prediction_errors(windows)

    @property
    def selected_gain(self) -> float:
        self._require_fitted()
        return self._correction_gain

    @property
    def selection_curve(self) -> Tuple[Mapping[str, float], ...]:
        self._require_fitted()
        return self._selection_curve

    @property
    def parameter_count(self) -> int:
        self._require_fitted()
        return (
            self.baseline.parameter_count
            + self.correction.parameter_count
        )

    def to_dict(self) -> Mapping[str, Any]:
        """Serialize the frozen baseline and correction together."""

        self._require_fitted()
        return {
            "schema_version": 1,
            "kind": self.kind,
            "baseline_sha256": self._baseline_sha256,
            "correction_gain": self._correction_gain,
            "selection_curve": [
                dict(row) for row in self._selection_curve
            ],
            "parameter_count": self.parameter_count,
            "baseline": self.baseline.to_dict(),
            "correction": self.correction.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "FrozenBaselineResidualDynamics":
        """Restore one composed residual model artifact."""

        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported residual JEPA artifact")
        baseline_payload = payload.get("baseline")
        correction_payload = payload.get("correction")
        if not isinstance(baseline_payload, Mapping) or not isinstance(
            correction_payload, Mapping
        ):
            raise ValueError("residual JEPA artifact is malformed")
        model = cls(
            baseline=ContractiveLowRankDynamics.from_dict(
                baseline_payload
            ),
            correction=ActionConditionedJepaDynamics.from_dict(
                correction_payload
            ),
            correction_gain=float(payload["correction_gain"]),
        )
        raw_curve = payload.get("selection_curve", ())
        if not isinstance(raw_curve, (list, tuple)):
            raise ValueError("residual JEPA selection curve is malformed")
        model._selection_curve = tuple(
            {
                str(key): float(value)
                for key, value in row.items()
            }
            for row in raw_curve
            if isinstance(row, Mapping)
        )
        model._baseline_sha256 = str(payload["baseline_sha256"])
        expected_hash = hashlib.sha256(
            _canonical_json_bytes(model.baseline.to_dict())
        ).hexdigest()
        if model._baseline_sha256 != expected_hash:
            raise ValueError("residual JEPA baseline hash does not match")
        model._fitted = True
        return model

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("residual JEPA model is not fitted")

    @staticmethod
    def _validate_gain(value: float) -> float:
        gain = float(value)
        if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("correction gain must be in [0, 1]")
        return gain


def latent_divergence_detection(
    *,
    model: FrozenBaselineResidualDynamics,
    calibration: ActionConditionedWindows,
    evaluation: ActionConditionedWindows,
    alpha: float = 0.05,
) -> Mapping[str, Any]:
    """Calibrate JEPA point divergence on control trajectories."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("latent divergence alpha must be in (0, 1)")
    calibration_scores = model.latent_divergence(calibration)[:, 0]
    evaluation_scores = model.latent_divergence(evaluation)[:, 0]
    calibration_groups = _trajectory_groups(calibration)
    control_maxima = np.asarray(
        [
            np.max(
                calibration_scores[
                    np.asarray(group["positions"], dtype=np.int64)
                ]
            )
            for group in calibration_groups.values()
            if group["onset"] is None
        ],
        dtype=np.float64,
    )
    if len(control_maxima) < 2:
        raise ValueError("latent divergence needs control calibration")
    threshold = float(
        np.quantile(control_maxima, 1.0 - alpha, method="higher")
    )
    calibration_rows = _latent_detection_rows(
        windows=calibration,
        scores=calibration_scores,
        threshold=threshold,
    )
    rows = _latent_detection_rows(
        windows=evaluation,
        scores=evaluation_scores,
        threshold=threshold,
    )
    controls = [row for row in rows if not row["is_treatment"]]
    treatments = [row for row in rows if row["is_treatment"]]
    detected = [
        row
        for row in treatments
        if row["post_onset_detection_transition"] is not None
    ]
    delays = [
        int(row["post_onset_detection_transition"])
        - int(row["onset_transition"])
        for row in detected
    ]
    if not controls or not treatments:
        raise ValueError("latent divergence evaluation needs matched roles")
    return {
        "schema_version": 1,
        "kind": (
            "residual_jepa_trajectory_calibrated_"
            "latent_divergence_detection_v1"
        ),
        "alpha": alpha,
        "calibration_unit": "control_trajectory_maximum",
        "calibration_control_trajectory_count": len(control_maxima),
        "point_threshold": threshold,
        "evaluation_control_trajectory_false_alarm_rate": float(
            np.mean([bool(row["any_alarm"]) for row in controls])
        ),
        "evaluation_treatment_trajectory_detection_rate": float(
            len(detected) / len(treatments)
        ),
        "median_post_onset_detection_delay_transitions": (
            float(np.median(delays)) if delays else None
        ),
        "calibration_trajectory_rows": calibration_rows,
        "trajectory_rows": rows,
        "limitations": [
            "open development evaluation, not sealed confirmation",
            "overlapping scores are reduced to one maximum per trajectory",
            "few calibration control trajectories limit alpha resolution",
        ],
    }


def _latent_detection_rows(
    *,
    windows: ActionConditionedWindows,
    scores: NDArray[np.float64],
    threshold: float,
) -> list[Mapping[str, Any]]:
    groups = _trajectory_groups(windows)
    rows: list[Mapping[str, Any]] = []
    for trajectory_id, group in groups.items():
        positions = group["positions"]
        transitions = np.asarray(
            [windows.transition_indices[index] for index in positions],
            dtype=np.int64,
        )
        trajectory_scores = scores[
            np.asarray(positions, dtype=np.int64)
        ]
        alarms = trajectory_scores > threshold
        onset = group["onset"]
        if onset is None:
            detection_transition: Optional[int] = None
        else:
            eligible = np.flatnonzero(alarms & (transitions >= onset))
            detection_transition = (
                int(transitions[eligible[0]]) if len(eligible) else None
            )
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "is_treatment": onset is not None,
                "onset_transition": onset,
                "any_alarm": bool(np.any(alarms)),
                "post_onset_detection_transition": detection_transition,
            }
        )
    return rows


def assess_residual_jepa_development(
    *,
    baseline_transfer: Mapping[str, Any],
    supervised_transfer: Mapping[str, Any],
    jepa_transfer: Mapping[str, Any],
    selected_gain: float,
    action_sanity: Mapping[str, Any],
    latent_detection: Mapping[str, Any],
    seed_robustness: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply preregistered residual-JEPA development gates."""

    baseline_effect = _positive(
        baseline_transfer, "downstream_effect_mse"
    )
    baseline_action = _positive(
        baseline_transfer, "normalized_mse_action_overlap"
    )
    baseline_overall = _positive(
        baseline_transfer, "normalized_mse_overall"
    )
    supervised_effect = _nonnegative(
        supervised_transfer, "downstream_effect_mse"
    )
    jepa_effect = _nonnegative(
        jepa_transfer, "downstream_effect_mse"
    )
    jepa_action = _nonnegative(
        jepa_transfer, "normalized_mse_action_overlap"
    )
    jepa_overall = _nonnegative(
        jepa_transfer, "normalized_mse_overall"
    )
    effect_improvement = 1.0 - jepa_effect / baseline_effect
    action_ratio = jepa_action / baseline_action
    overall_ratio = jepa_overall / baseline_overall
    predictive_gates = {
        "downstream_effect_improvement_at_least_10_percent": (
            effect_improvement >= 0.10
        ),
        "action_overlap_mse_within_5_percent": action_ratio <= 1.05,
        "overall_mse_within_5_percent": overall_ratio <= 1.05,
        "action_and_target_hit_at_1_at_least_95_percent": (
            float(jepa_transfer["action_and_target_hit_at_1"]) >= 0.95
        ),
        "no_action_specificity_is_100_percent": (
            float(jepa_transfer["no_action_specificity"]) == 1.0
        ),
        "correct_action_beats_both_on_80_percent_of_pairs": (
            float(
                action_sanity[
                    "correct_action_beats_both_fraction"
                ]
            )
            >= 0.80
        ),
        "selected_nonzero_correction_gain": selected_gain > 0.0,
        "jepa_downstream_no_worse_than_supervised": (
            jepa_effect <= supervised_effect
        ),
    }
    raw_delay = latent_detection[
        "median_post_onset_detection_delay_transitions"
    ]
    delay = float(raw_delay) if raw_delay is not None else float("inf")
    investigation_gates = {
        "control_trajectory_false_alarm_at_most_5_percent": (
            float(
                latent_detection[
                    "evaluation_control_trajectory_false_alarm_rate"
                ]
            )
            <= 0.05
        ),
        "treatment_trajectory_detection_at_least_80_percent": (
            float(
                latent_detection[
                    "evaluation_treatment_trajectory_detection_rate"
                ]
            )
            >= 0.80
        ),
        "median_post_onset_delay_at_most_10": delay <= 10.0,
    }
    tracer_passed = all(predictive_gates.values())
    seed_count = int(seed_robustness["seed_count"])
    required_seed_count = int(seed_robustness["required_seed_count"])
    seeds_passed = (
        seed_count >= required_seed_count
        and bool(seed_robustness["passed"])
    )
    return {
        "schema_version": 1,
        "kind": "residual_jepa_development_assessment_v1",
        "evidence_boundary": (
            "open development only; fresh sealed matched pairs are "
            "required for confirmation"
        ),
        "observed": {
            "downstream_effect_relative_improvement": (
                effect_improvement
            ),
            "action_overlap_mse_ratio": action_ratio,
            "overall_mse_ratio": overall_ratio,
            "selected_correction_gain": selected_gain,
            "jepa_to_supervised_downstream_effect_ratio": (
                jepa_effect / supervised_effect
                if supervised_effect > 0.0
                else None
            ),
            "seed_count": seed_count,
            "required_seed_count": required_seed_count,
        },
        "predictive_gates": predictive_gates,
        "predictive_tracer_gates_passed": tracer_passed,
        "seed_robustness_passed": seeds_passed,
        "investigation_gates": investigation_gates,
        "investigation_gates_passed": all(
            investigation_gates.values()
        ),
        "decision": (
            "advance_to_sealed_confirmation"
            if tracer_passed and seeds_passed
            else (
                "run_seed_robustness"
                if tracer_passed
                else "reject_this_configuration"
            )
        ),
        "sealed_confirmation": False,
    }


def write_residual_jepa_artifacts(
    *,
    output_directory: Path,
    report: Mapping[str, Any],
    model_artifacts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Write one immutable residual-JEPA evidence bundle."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite residual JEPA results: {output}"
        )
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    (output / "results.json").write_text(_pretty_json(report))
    (output / "report.md").write_text(_markdown_report(report))
    for name, artifact in model_artifacts.items():
        (models / f"{name}.json").write_text(_pretty_json(artifact))
    hashes = {
        path.relative_to(output).as_posix(): _file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "kind": "residual_jepa_development_manifest_v1",
        "sha256": hashes,
    }
    (output / "artifact-manifest.json").write_text(
        _pretty_json(manifest)
    )
    return manifest


def _trajectory_groups(
    windows: ActionConditionedWindows,
) -> Mapping[str, Mapping[str, Any]]:
    try:
        applicable = windows.action_feature_names.index("applicable")
    except ValueError as error:
        raise ValueError(
            "latent divergence needs the applicable action feature"
        ) from error
    positions: Dict[str, list[int]] = {}
    onsets: Dict[str, list[int]] = {}
    for index, trajectory_id in enumerate(windows.trajectory_ids):
        positions.setdefault(trajectory_id, []).append(index)
        if np.any(
            windows.future_actions[index, 0, :, applicable] > 0.5
        ):
            onsets.setdefault(trajectory_id, []).append(
                int(windows.transition_indices[index])
            )
    return {
        trajectory_id: {
            "positions": tuple(indices),
            "onset": (
                min(onsets[trajectory_id])
                if trajectory_id in onsets
                else None
            ),
        }
        for trajectory_id, indices in positions.items()
    }


def _positive(values: Mapping[str, Any], name: str) -> float:
    value = _nonnegative(values, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative(values: Mapping[str, Any], name: str) -> float:
    value = float(values[name])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_report(report: Mapping[str, Any]) -> str:
    assessment = report.get("assessment", {})
    decision = (
        assessment.get("decision", "unavailable")
        if isinstance(assessment, Mapping)
        else "unavailable"
    )
    transfer = report.get("transfer_scores", {})
    rows = []
    if isinstance(transfer, Mapping):
        for name, score in transfer.items():
            if not isinstance(score, Mapping):
                continue
            rows.append(
                "| {name} | {action:.4f} | {overall:.4f} | "
                "{effect:.4f} | {hit:.1%} |".format(
                    name=name,
                    action=float(
                        score.get(
                            "normalized_mse_action_overlap", float("nan")
                        )
                    ),
                    overall=float(
                        score.get(
                            "normalized_mse_overall", float("nan")
                        )
                    ),
                    effect=float(
                        score.get("downstream_effect_mse", float("nan"))
                    ),
                    hit=float(
                        score.get(
                            "action_and_target_hit_at_1", float("nan")
                        )
                    ),
                )
            )
    table = "\n".join(rows) if rows else "| no scores | - | - | - | - |"
    return (
        "# Residual JEPA correction development v1\n\n"
        f"Decision: **{decision}**.\n\n"
        "This is open development evidence, not sealed confirmation and "
        "not a general world-model claim.\n\n"
        "## Held-out-topology scores\n\n"
        "| Model | Action MSE | Overall MSE | Downstream effect MSE | "
        "Attribution hit@1 |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{table}\n"
    )
