"""Exact Jacobian-volume scoring for frozen complete-LeJEPA projectors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from quantis_core.edge_dynamics.complete_lejepa import (
    CompleteLejepaConfig,
    CompleteLejepaRepresentation,
    TelemetryViewSchedule,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


JEPA_SCORE_EPSILON = 1e-6
JEPA_SCORE_VIEW_NAME = "global_a"
JEPA_SCORE_VIEW_STEP = 1600


@dataclass(frozen=True)
class TorchJepaScore:
    """Torch tensors produced by the literal Appendix-B score reduction."""

    jepa_score: Any
    anomaly_score: Any
    singular_values: Any
    clipped_count: Any


@dataclass(frozen=True)
class JepaScoreBatch:
    """Materialized exact scores and diagnostics for a history batch."""

    jepa_score: NDArray[np.float64]
    anomaly_score: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    clipped_count: NDArray[np.int64]
    projector_embeddings: NDArray[np.float64]
    unowned_jacobian_max_abs: NDArray[np.float64]


def jepa_score_from_jacobian(
    jacobian: Any, *, epsilon: float = JEPA_SCORE_EPSILON
) -> TorchJepaScore:
    """Apply the source paper's clipped full-SVD score to a batch Jacobian."""

    torch = _require_torch()
    if (
        not isinstance(epsilon, float)
        or not np.isfinite(epsilon)
        or epsilon <= 0.0
        or not isinstance(jacobian, torch.Tensor)
        or jacobian.ndim < 3
        or jacobian.shape[0] < 1
        or jacobian.shape[1] < 1
        or not bool(torch.isfinite(jacobian).all())
    ):
        raise ValueError("JEPA-SCORE Jacobian inputs are invalid")
    matrices = jacobian.flatten(start_dim=2).permute(1, 0, 2)
    singular_values = torch.linalg.svdvals(matrices)
    clipped_count = (singular_values < epsilon).sum(dim=1)
    scores = singular_values.clamp_min(epsilon).log().sum(dim=1)
    return TorchJepaScore(
        jepa_score=scores,
        anomaly_score=-scores,
        singular_values=singular_values,
        clipped_count=clipped_count,
    )


class ExactJepaScorer:
    """Restore and exactly score a frozen complete-LeJEPA projector."""

    kind = "exact_complete_lejepa_projector_jepa_score"
    schema_version = 1

    def __init__(
        self,
        *,
        strict_model_payload: Mapping[str, Any],
        source_model_file_sha256: str,
        source_model_payload_sha256: str,
        model: CompleteLejepaRepresentation,
        visible_tokens: NDArray[np.bool_],
        present_tokens: NDArray[np.bool_],
        epsilon: float = JEPA_SCORE_EPSILON,
    ) -> None:
        if (
            not isinstance(epsilon, float)
            or epsilon != JEPA_SCORE_EPSILON
        ):
            raise ValueError("JEPA-SCORE epsilon is invalid")
        graph, feature_names, ownership, network = model._fitted_values()
        projector = getattr(model, "_projector", None)
        visible = np.asarray(visible_tokens, dtype=np.bool_)
        present = np.asarray(present_tokens, dtype=np.bool_)
        if (
            projector is None
            or visible.shape != (20, len(graph.entities))
            or present.shape != visible.shape
            or np.any(visible & ~present)
            or not _is_sha256(source_model_file_sha256)
            or not _is_sha256(source_model_payload_sha256)
        ):
            raise ValueError("JEPA-SCORE source model has no projector")
        for parameter in tuple(network.parameters()) + tuple(
            projector.parameters()
        ):
            parameter.requires_grad_(False)
        network.eval()
        projector.eval()
        self._strict_model_payload = _json_copy(strict_model_payload)
        self._source_model_file_sha256 = source_model_file_sha256
        self._source_model_payload_sha256 = source_model_payload_sha256
        self._model = model
        self._graph = graph
        self._feature_names = feature_names
        self._ownership_mask = np.asarray(
            ownership, dtype=np.bool_
        ).copy()
        self._network = network
        self._projector = projector
        self._visible_tokens = visible.copy()
        self._present_tokens = present.copy()
        self.epsilon = epsilon

    @classmethod
    def from_model_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        source_model_file_sha256: str,
        epsilon: float = JEPA_SCORE_EPSILON,
    ) -> "ExactJepaScorer":
        """Restore the exact retained backbone/projector payload."""

        if (
            payload.get("kind")
            != CompleteLejepaRepresentation.kind
            or payload.get("schema_version")
            != CompleteLejepaRepresentation.schema_version
        ):
            raise ValueError("JEPA-SCORE source model payload is invalid")
        model = _restore_complete_model(payload)
        graph, _, ownership, _ = model._fitted_values()
        view = TelemetryViewSchedule(
            graph=graph,
            ownership_mask=ownership,
            varying_entity_mask=np.any(ownership, axis=1),
            seed=model.config.view_seed,
        ).batch(
            np.zeros(
                (
                    1,
                    20,
                    len(graph.entities),
                    ownership.shape[1],
                ),
                dtype=np.float64,
            ),
            step=JEPA_SCORE_VIEW_STEP,
        )
        canonical_bytes = _canonical_json_bytes(payload)
        return cls(
            strict_model_payload=_strict_model_payload(payload),
            source_model_file_sha256=source_model_file_sha256,
            source_model_payload_sha256=hashlib.sha256(
                canonical_bytes
            ).hexdigest(),
            model=model,
            visible_tokens=view.visible_tokens[0, 0],
            present_tokens=view.present_tokens[0, 0],
            epsilon=epsilon,
        )

    @classmethod
    def from_model_json_bytes(
        cls,
        raw_json: bytes,
        *,
        epsilon: float = JEPA_SCORE_EPSILON,
    ) -> "ExactJepaScorer":
        """Restore a source file while retaining its raw byte identity."""

        try:
            payload = json.loads(raw_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "JEPA-SCORE source model JSON is invalid"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError("JEPA-SCORE source model JSON is invalid")
        return cls.from_model_payload(
            cast(Dict[str, Any], payload),
            source_model_file_sha256=hashlib.sha256(
                raw_json
            ).hexdigest(),
            epsilon=epsilon,
        )

    @property
    def source_model_file_sha256(self) -> str:
        return self._source_model_file_sha256

    @property
    def source_model_payload_sha256(self) -> str:
        return self._source_model_payload_sha256

    @property
    def source_model_sha256(self) -> str:
        """Backward-compatible name for the canonical parsed-payload hash."""

        return self.source_model_payload_sha256

    @property
    def parameter_count(self) -> int:
        return int(
            sum(
                parameter.numel()
                for module in (self._network, self._projector)
                for parameter in module.parameters()
            )
        )

    def score(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> JepaScoreBatch:
        """Compute every exact input-to-projector singular value."""

        torch = _require_torch()
        source = self._validate_inputs(histories, graph)
        inputs = torch.as_tensor(source, dtype=torch.float32).requires_grad_(
            True
        )
        with torch.no_grad():
            embeddings = self._embed(inputs).detach()
        jacobian = torch.autograd.functional.jacobian(
            lambda values: self._embed(values).sum(dim=0),
            inputs,
            create_graph=False,
            strict=False,
            vectorize=False,
        )
        result = jepa_score_from_jacobian(
            jacobian, epsilon=self.epsilon
        )
        ownership = torch.as_tensor(
            self._ownership_mask,
            dtype=torch.bool,
            device=jacobian.device,
        )
        unowned = jacobian[
            :, :, :, ~ownership
        ].reshape(jacobian.shape[0], jacobian.shape[1], -1)
        unowned_max_abs = unowned.abs().amax(dim=(0, 2))
        return JepaScoreBatch(
            jepa_score=_to_float64(result.jepa_score),
            anomaly_score=_to_float64(result.anomaly_score),
            singular_values=_to_float64(result.singular_values),
            clipped_count=(
                result.clipped_count.detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            ),
            projector_embeddings=_to_float64(embeddings),
            unowned_jacobian_max_abs=_to_float64(
                unowned_max_abs
            ),
        )

    def projector_embeddings(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return the exact full-view projector route without gradients."""

        torch = _require_torch()
        source = self._validate_inputs(histories, graph)
        with torch.no_grad():
            values = self._embed(
                torch.as_tensor(source, dtype=torch.float32)
            )
        return _to_float64(values)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scorer and its complete frozen source model."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "epsilon": self.epsilon,
            "source_model_file_sha256": (
                self.source_model_file_sha256
            ),
            "source_model_payload_sha256": (
                self.source_model_payload_sha256
            ),
            "parameter_count": self.parameter_count,
            "strict_model_payload_sha256": _canonical_sha256(
                self._strict_model_payload
            ),
            "view_name": JEPA_SCORE_VIEW_NAME,
            "view_step": JEPA_SCORE_VIEW_STEP,
            "visible_tokens": self._visible_tokens.astype(int).tolist(),
            "present_tokens": self._present_tokens.astype(int).tolist(),
            "view_sha256": _view_sha256(
                self._visible_tokens, self._present_tokens
            ),
            "strict_model_payload": _json_copy(
                self._strict_model_payload
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactJepaScorer":
        """Restore a scorer while checking its declared identities."""

        expected = {
            "schema_version",
            "kind",
            "epsilon",
            "source_model_file_sha256",
            "source_model_payload_sha256",
            "parameter_count",
            "strict_model_payload_sha256",
            "view_name",
            "view_step",
            "visible_tokens",
            "present_tokens",
            "view_sha256",
            "strict_model_payload",
        }
        if (
            set(payload) != expected
            or payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
            or payload.get("view_name") != JEPA_SCORE_VIEW_NAME
            or payload.get("view_step") != JEPA_SCORE_VIEW_STEP
        ):
            raise ValueError("unsupported JEPA-SCORE artifact")
        if (
            not isinstance(payload["epsilon"], float)
            or payload["epsilon"] != JEPA_SCORE_EPSILON
        ):
            raise ValueError("unsupported JEPA-SCORE epsilon")
        strict_payload = dict(payload["strict_model_payload"])
        if _canonical_sha256(strict_payload) != str(
            payload["strict_model_payload_sha256"]
        ):
            raise ValueError("JEPA-SCORE strict model identity mismatch")
        model = _restore_complete_model(
            _expand_strict_model_payload(
                strict_payload,
                parameter_count=int(payload["parameter_count"]),
            )
        )
        graph, _, ownership, _ = model._fitted_values()
        regenerated = TelemetryViewSchedule(
            graph=graph,
            ownership_mask=ownership,
            varying_entity_mask=np.any(ownership, axis=1),
            seed=model.config.view_seed,
        ).batch(
            np.zeros(
                (
                    1,
                    20,
                    len(graph.entities),
                    ownership.shape[1],
                ),
                dtype=np.float64,
            ),
            step=JEPA_SCORE_VIEW_STEP,
        )
        visible = np.asarray(payload["visible_tokens"], dtype=np.bool_)
        present = np.asarray(payload["present_tokens"], dtype=np.bool_)
        if (
            not np.array_equal(visible, regenerated.visible_tokens[0, 0])
            or not np.array_equal(
                present, regenerated.present_tokens[0, 0]
            )
            or _view_sha256(visible, present)
            != payload["view_sha256"]
        ):
            raise ValueError("JEPA-SCORE frozen view identity mismatch")
        scorer = cls(
            strict_model_payload=strict_payload,
            source_model_file_sha256=str(
                payload["source_model_file_sha256"]
            ),
            source_model_payload_sha256=str(
                payload["source_model_payload_sha256"]
            ),
            model=model,
            visible_tokens=visible,
            present_tokens=present,
            epsilon=payload["epsilon"],
        )
        if (
            scorer.parameter_count != int(payload["parameter_count"])
        ):
            raise ValueError("JEPA-SCORE artifact identity mismatch")
        return scorer

    def _validate_inputs(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        source = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != self._graph.to_dict()
            or source.ndim != 4
            or source.shape[1:]
            != (
                20,
                len(self._graph.entities),
                len(self._feature_names),
            )
            or len(source) < 1
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("JEPA-SCORE input schema or values are invalid")
        return source

    def _embed(self, inputs: Any) -> Any:
        torch = _require_torch()
        ownership = torch.as_tensor(
            self._ownership_mask,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        values = inputs * ownership[None, None]
        visible = torch.as_tensor(
            self._visible_tokens,
            dtype=torch.bool,
            device=values.device,
        )[None].expand(len(values), -1, -1)
        present = torch.as_tensor(
            self._present_tokens,
            dtype=torch.bool,
            device=values.device,
        )[None].expand(len(values), -1, -1)
        positions = np.arange(
            20 * len(self._graph.entities), dtype=np.int64
        )
        hidden = self._network(
            values, visible, present, positions
        )
        selected_visible = visible.reshape(len(values), -1)
        pooled = (
            hidden
            * selected_visible.to(hidden.dtype).unsqueeze(-1)
        ).sum(dim=1) / selected_visible.sum(
            dim=1
        ).clamp_min(1).unsqueeze(-1)
        return self._projector(pooled)


def assess_jepa_score_gates(
    *,
    interpretable: bool,
    protocol_checks: Mapping[str, bool],
    candidate_metrics: Mapping[str, Mapping[str, Any]],
    raw_metrics: Mapping[str, Mapping[str, Any]],
    selection_pair_win_fraction: float,
    median_latency_ms: float,
    p95_latency_ms: float,
    bundle_bytes: int,
    parameter_count: int,
) -> Dict[str, Any]:
    """Purely recompute the frozen JEPA-SCORE promotion decision."""

    required_protocols = (
        "source_identities_recompute",
        "role_contract_recomputes",
        "fixed_anchors_recompute",
        "action_blind_sampling_recomputes",
        "model_restoration_recomputes",
        "exact_score_recomputes",
        "batch_and_literal_parity_recompute",
        "latency_contract_recomputes",
        "evidence_arrays_are_finite",
        "calibration_isolation_recomputes",
        "alert_metrics_recompute",
        "evaluation_has_no_selection_authority",
        "source_snapshots_and_manifest_verify",
    )
    protocol = {
        name: bool(protocol_checks.get(name, False))
        for name in required_protocols
    }
    iid = candidate_metrics["iid_evaluation"]
    transfer = candidate_metrics["transfer_evaluation"]
    raw_transfer = raw_metrics["transfer_evaluation"]
    _validate_alert_metrics(iid)
    _validate_alert_metrics(transfer)
    _validate_alert_metrics(raw_metrics["iid_evaluation"])
    _validate_alert_metrics(raw_transfer)
    if (
        not np.isfinite(selection_pair_win_fraction)
        or not 0.0 <= selection_pair_win_fraction <= 1.0
    ):
        raise ValueError("JEPA-SCORE selection metric is invalid")
    edge_safety = {
        "median_latency_at_most_100_ms": (
            np.isfinite(median_latency_ms)
            and 0.0 < median_latency_ms <= 100.0
        ),
        "p95_latency_at_most_125_ms": (
            np.isfinite(p95_latency_ms)
            and 0.0 < p95_latency_ms <= 125.0
        ),
        "bundle_at_most_8_mib": (
            isinstance(bundle_bytes, int)
            and not isinstance(bundle_bytes, bool)
            and 0 < bundle_bytes <= 8 * 1024 * 1024
        ),
        "parameters_at_most_120000": (
            isinstance(parameter_count, int)
            and not isinstance(parameter_count, bool)
            and 0 < parameter_count <= 120_000
        ),
        "iid_control_false_alarm_at_most_0_05": (
            float(iid["control_trajectory_false_alarm_rate"]) <= 0.05
        ),
        "transfer_control_false_alarm_at_most_0_05": (
            float(transfer["control_trajectory_false_alarm_rate"]) <= 0.05
        ),
        "iid_pre_onset_alert_at_most_0_05": (
            float(iid["treatment_pre_onset_alert_rate"]) <= 0.05
        ),
        "transfer_pre_onset_alert_at_most_0_05": (
            float(transfer["treatment_pre_onset_alert_rate"]) <= 0.05
        ),
    }
    pareto_no_worse, material_improvement = _transfer_pareto(
        transfer, raw_transfer
    )
    value = {
        "selection_pair_win_fraction_at_least_0_60": (
            np.isfinite(selection_pair_win_fraction)
            and selection_pair_win_fraction >= 0.60
        ),
        "iid_detection_at_least_0_80": (
            float(iid["treatment_detection_rate"]) >= 0.80
        ),
        "transfer_detection_at_least_0_80": (
            float(transfer["treatment_detection_rate"]) >= 0.80
        ),
        "transfer_pareto_no_worse_than_raw": pareto_no_worse,
        "transfer_materially_improves_raw": material_improvement,
    }
    if not isinstance(interpretable, bool):
        raise ValueError("JEPA-SCORE interpretability flag is invalid")
    protocol_passed = all(protocol.values())
    edge_safety_passed = all(edge_safety.values())
    value_passed = all(value.values())
    scientific_gates_passed = (
        protocol_passed and edge_safety_passed and value_passed
    )
    passed = interpretable and scientific_gates_passed
    return {
        "protocol_gates": protocol,
        "edge_safety_gates": edge_safety,
        "value_gates": value,
        "protocol_passed": protocol_passed,
        "edge_safety_passed": edge_safety_passed,
        "value_passed": value_passed,
        "interpretable": interpretable,
        "scientific_gates_passed": scientific_gates_passed,
        "passed": passed,
        "selection_pair_win_fraction": float(
            selection_pair_win_fraction
        ),
        "decision": (
            "non_interpretable_jepa_score_smoke"
            if not interpretable
            else (
                "advance_exact_jepa_score_to_fixed_seed_robustness"
                if passed
                else "reject_exact_jepa_score_edge_alert_recipe"
            )
        ),
    }


def _transfer_pareto(
    candidate: Mapping[str, Any], raw: Mapping[str, Any]
) -> tuple[bool, bool]:
    candidate_false_alarm = float(
        candidate["control_trajectory_false_alarm_rate"]
    )
    raw_false_alarm = float(raw["control_trajectory_false_alarm_rate"])
    candidate_detection = float(candidate["treatment_detection_rate"])
    raw_detection = float(raw["treatment_detection_rate"])
    candidate_delay = candidate["median_post_onset_delay_transitions"]
    raw_delay = raw["median_post_onset_delay_transitions"]
    if candidate_delay is None and raw_delay is None:
        delay_no_worse = True
        delay_improvement = False
    elif candidate_delay is None:
        delay_no_worse = False
        delay_improvement = False
    elif raw_delay is None:
        delay_no_worse = True
        delay_improvement = False
    else:
        candidate_delay_value = float(candidate_delay)
        raw_delay_value = float(raw_delay)
        delay_no_worse = candidate_delay_value <= raw_delay_value
        delay_improvement = (
            raw_delay_value - candidate_delay_value >= 20.0
        )
    no_worse = (
        candidate_false_alarm <= raw_false_alarm
        and candidate_detection >= raw_detection
        and delay_no_worse
    )
    material = no_worse and (
        raw_false_alarm - candidate_false_alarm >= 0.05
        or candidate_detection - raw_detection >= 0.05
        or delay_improvement
    )
    return no_worse, material


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _strict_model_payload(
    payload: Mapping[str, Any]
) -> Dict[str, Any]:
    keys = (
        "graph",
        "feature_names",
        "ownership_mask",
        "network_state",
        "projector_state",
    )
    try:
        result = {key: payload[key] for key in keys}
        config = dict(payload["config"])
    except KeyError as error:
        raise ValueError(
            "JEPA-SCORE source model payload is invalid"
        ) from error
    result["inference_config"] = {
        key: config[key]
        for key in (
            "width",
            "block_count",
            "head_count",
            "feedforward_width",
            "projector_width",
            "preprocessing_protocol",
            "view_seed",
        )
    }
    return _json_copy(result)


def _expand_strict_model_payload(
    payload: Mapping[str, Any], *, parameter_count: int
) -> Dict[str, Any]:
    expected = {
        "graph",
        "feature_names",
        "ownership_mask",
        "network_state",
        "projector_state",
        "inference_config",
    }
    if (
        set(payload) != expected
        or isinstance(parameter_count, bool)
        or parameter_count < 1
    ):
        raise ValueError("JEPA-SCORE strict model payload is invalid")
    inference = dict(payload["inference_config"])
    if set(inference) != {
        "width",
        "block_count",
        "head_count",
        "feedforward_width",
        "projector_width",
        "preprocessing_protocol",
        "view_seed",
    }:
        raise ValueError("JEPA-SCORE inference config is invalid")
    config = CompleteLejepaConfig(
        width=int(inference["width"]),
        block_count=int(inference["block_count"]),
        head_count=int(inference["head_count"]),
        feedforward_width=int(inference["feedforward_width"]),
        projector_width=int(inference["projector_width"]),
        preprocessing_protocol=str(
            inference["preprocessing_protocol"]
        ),
        view_seed=int(inference["view_seed"]),
    )
    projector_parameters = (
        config.width * config.projector_width
        + config.projector_width
        + config.projector_width * config.width
        + config.width
    )
    inference_parameters = parameter_count - projector_parameters
    if inference_parameters < 1:
        raise ValueError("JEPA-SCORE parameter count is invalid")
    result = {
        "schema_version": CompleteLejepaRepresentation.schema_version,
        "kind": CompleteLejepaRepresentation.kind,
        "config": config.to_dict(),
        "graph": payload["graph"],
        "feature_names": payload["feature_names"],
        "ownership_mask": payload["ownership_mask"],
        "network_state": payload["network_state"],
        "projector_state": payload["projector_state"],
        "inference_parameter_count": inference_parameters,
        "training_only_parameter_count": projector_parameters,
    }
    result.update(
        {
            "decoder_state": None,
            "sigreg_generator_state": [],
            "training_metrics": [],
        }
    )
    return result


def _view_sha256(
    visible: NDArray[np.bool_], present: NDArray[np.bool_]
) -> str:
    return _canonical_sha256(
        {
            "visible_tokens": visible.astype(int).tolist(),
            "present_tokens": present.astype(int).tolist(),
        }
    )


def _validate_alert_metrics(metrics: Mapping[str, Any]) -> None:
    for name in (
        "control_trajectory_false_alarm_rate",
        "treatment_detection_rate",
        "treatment_pre_onset_alert_rate",
    ):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("JEPA-SCORE alert rate is invalid")
    delay = metrics.get("median_post_onset_delay_transitions")
    if delay is not None and (
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not np.isfinite(delay)
        or float(delay) < 0.0
    ):
        raise ValueError("JEPA-SCORE alert delay is invalid")


def _restore_complete_model(
    payload: Mapping[str, Any]
) -> CompleteLejepaRepresentation:
    torch = _require_torch()
    cpu_state = torch.random.get_rng_state()
    mps_state = (
        torch.mps.get_rng_state()
        if torch.backends.mps.is_available()
        else None
    )
    try:
        return CompleteLejepaRepresentation.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "JEPA-SCORE source model payload is invalid"
        ) from error
    finally:
        torch.random.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_copy(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return cast(
        Dict[str, Any],
        json.loads(
            json.dumps(payload, allow_nan=False, separators=(",", ":"))
        ),
    )


def _to_float64(values: Any) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        values.detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False),
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("JEPA-SCORE requires torch") from error
    return torch
