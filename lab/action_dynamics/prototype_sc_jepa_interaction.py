#!/usr/bin/env python3
"""Retained runner for the frozen ticket 013 SC-JEPA factorial."""

import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.hepa_jepa import (
    EntityStateRidgeProbe,
    HepaEntityPcaBaseline,
    HepaEventDefinition,
    calibrate_probability_surface,
    fit_logit_calibrator,
    trajectory_action_onsets,
    trajectory_alert_threshold,
)
from quantis_core.edge_dynamics.sc_jepa import (
    SC_JEPA_ASSESSMENT_MODEL_NAMES,
    SC_JEPA_ASSESSMENT_ROLE_NAMES,
    SC_JEPA_CELL_NAMES,
    ScJepaConfig,
    ScJepaModel,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-sc-jepa-interaction-v1"
)
FROZEN_PRETRAIN_STEPS = 300
FROZEN_ALERT_STEPS = 200
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_sc_jepa_interaction.py",
    "lab/action_dynamics/prototype_sc_jepa_interaction_assessor.py",
    "src/quantis_core/edge_dynamics/sc_jepa.py",
    "tests/test_sc_jepa.py",
    "docs/specs/sc-jepa-interaction-v1.md",
    "docs/specs/jepa-experiment-ladder-v1.md",
    "docs/research/sc-jepa-primary-source-notes.md",
    "docs/research/jepa-frontier-technique-audit-2026.md",
    "docs/wayfinding/jepa-implementation-program/013-test-complete-sc-jepa-interaction.md",
    "src/quantis_core/edge_dynamics/hepa_jepa.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/graph_telemetry.py",
)
IMPLEMENTATION_SYMBOL_SOURCES = {
    "src/quantis_core/action_conditioned_dynamics.py": (
        "ActionConditionedWindows",
        "_semantic_schema_sha256",
    ),
}


class RawLowRankRiskModel:
    """Restorable fitting-only rank-32 raw patch risk classifier."""

    kind = "sc_jepa_raw_low_rank_risk"
    schema_version = 1

    def __init__(
        self,
        *,
        width: int = 32,
        hidden_width: int = 64,
        alert_steps: int = 200,
        checkpoint_interval: int = 25,
        batch_size: int = 128,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        seed: int = 13013,
    ) -> None:
        self.width = width
        self.hidden_width = hidden_width
        self.alert_steps = alert_steps
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self._graph: Any = None
        self._feature_names: Tuple[str, ...] = ()
        self._ownership: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._components: Optional[NDArray[np.float64]] = None
        self._head: Any = None
        self._checkpoints: Tuple[
            Tuple[int, Mapping[str, Any]], ...
        ] = ()
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selected_step: Optional[int] = None
        self._calibration: Optional[Mapping[str, float]] = None

    @property
    def calibration(self) -> Optional[Mapping[str, float]]:
        return (
            None
            if self._calibration is None
            else dict(self._calibration)
        )

    @property
    def inference_parameter_count(self) -> int:
        _, _, _, _, head = self._selected_values()
        return int(sum(parameter.numel() for parameter in head.parameters()))

    def fit(
        self,
        windows: Any,
        event_definition: HepaEventDefinition,
    ) -> "RawLowRankRiskModel":
        torch = _require_torch()
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        ownership = _fit_owned_feature_mask(windows)
        patches = _raw_patch_matrix(windows.histories, ownership)
        flattened = patches.reshape(-1, patches.shape[-1])
        center = np.mean(flattened, axis=0)
        _, _, right = np.linalg.svd(
            flattened - center, full_matrices=False
        )
        if len(right) < self.width:
            raise ValueError("raw low-rank fitting rank is insufficient")
        components = right[: self.width].copy()
        _orient_components(components)
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership = ownership
        self._center = center
        self._components = components
        self._head = _build_risk_head(
            torch,
            input_width=5 * self.width,
            hidden_width=self.hidden_width,
        )
        labels = event_definition.labels(windows)[:, -1]
        features = self._features(windows.histories, windows.graph)
        positives = int(np.sum(labels))
        negatives = int(len(labels) - positives)
        if positives < 1 or negatives < 1:
            raise ValueError("raw low-rank alert fitting needs both classes")
        optimizer = torch.optim.AdamW(
            self._head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        generator = np.random.default_rng(self.seed + 41)
        positive_weight = negatives / float(positives)
        checkpoints = []
        metrics = []
        self._head.train()
        for step in range(self.alert_steps):
            indices = generator.integers(
                0,
                len(features),
                size=min(self.batch_size, len(features)),
            )
            x = torch.as_tensor(features[indices], dtype=torch.float32)
            truth = torch.as_tensor(
                labels[indices], dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            logits = self._head(x).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                truth,
                pos_weight=torch.as_tensor(
                    positive_weight, dtype=torch.float32
                ),
            )
            loss.backward()
            optimizer.step()
            metrics.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach()),
                    "positive_weight": positive_weight,
                }
            )
            if (
                (step + 1) % self.checkpoint_interval == 0
                or step + 1 == self.alert_steps
            ):
                checkpoints.append(
                    (
                        step + 1,
                        copy.deepcopy(self._head.state_dict()),
                    )
                )
        self._training_metrics = tuple(metrics)
        self._checkpoints = tuple(checkpoints)
        return self

    def select(
        self,
        windows: Any,
        event_definition: HepaEventDefinition,
    ) -> "RawLowRankRiskModel":
        torch = _require_torch()
        _, _, _, _, head = self._fitted_values()
        labels = event_definition.labels(windows)[:, -1]
        features = torch.as_tensor(
            self._features(windows.histories, windows.graph),
            dtype=torch.float32,
        )
        scores = []
        best_score = float("inf")
        best_step = -1
        best_state = None
        for step, state in self._checkpoints:
            head.load_state_dict(state)
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    head(features).squeeze(-1)
                ).cpu().numpy()
            brier = float(
                np.mean(
                    np.square(
                        probabilities - labels.astype(np.float64)
                    )
                )
            )
            scores.append({"step": float(step), "brier": brier})
            if brier < best_score - 1e-12:
                best_score = brier
                best_step = step
                best_state = state
        if best_state is None:
            raise RuntimeError("raw low-rank selected no checkpoint")
        head.load_state_dict(best_state)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        self._selection_metrics = tuple(scores)
        self._selected_step = best_step
        self._checkpoints = ()
        return self

    def fit_calibration(
        self,
        windows: Any,
        event_definition: HepaEventDefinition,
    ) -> "RawLowRankRiskModel":
        risks = self.predict_risk(windows.histories, windows.graph)
        labels = event_definition.labels(windows)[:, -1]
        slope, intercept, brier = fit_logit_calibrator(
            risks[:, None], labels[:, None]
        )
        calibrated = calibrate_probability_surface(
            risks[:, None], slope=slope, intercept=intercept
        )
        threshold = trajectory_alert_threshold(
            calibrated,
            windows.trajectory_ids,
            _control_trajectory_ids(windows),
        )
        self._calibration = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": brier,
            "alert_threshold": threshold,
        }
        return self

    def predict_risk(
        self, histories: NDArray[np.float64], graph: Any
    ) -> NDArray[np.float64]:
        torch = _require_torch()
        _, _, _, _, head = self._selected_values()
        with torch.no_grad():
            values = torch.sigmoid(
                head(
                    torch.as_tensor(
                        self._features(histories, graph),
                        dtype=torch.float32,
                    )
                ).squeeze(-1)
            ).cpu().numpy()
        return np.asarray(values, dtype=np.float64)

    def calibrated_risk(
        self, histories: NDArray[np.float64], graph: Any
    ) -> NDArray[np.float64]:
        if self._calibration is None:
            raise ValueError("raw low-rank calibration is missing")
        values = self.predict_risk(histories, graph)
        return calibrate_probability_surface(
            values[:, None],
            slope=float(self._calibration["slope"]),
            intercept=float(self._calibration["intercept"]),
        )[:, 0]

    def alert_decisions(
        self, histories: NDArray[np.float64], graph: Any
    ) -> NDArray[np.bool_]:
        if self._calibration is None:
            raise ValueError("raw low-rank calibration is missing")
        return self.calibrated_risk(histories, graph) > float(
            self._calibration["alert_threshold"]
        )

    def to_dict(self) -> Mapping[str, Any]:
        graph, features, ownership, components, head = (
            self._selected_values()
        )
        if self._center is None or self._calibration is None:
            raise ValueError("raw low-rank model is incomplete")
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": {
                "width": self.width,
                "hidden_width": self.hidden_width,
                "alert_steps": self.alert_steps,
                "checkpoint_interval": self.checkpoint_interval,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
            },
            "graph": graph.to_dict(),
            "feature_names": list(features),
            "ownership_mask": ownership.astype(int).tolist(),
            "center": self._center.tolist(),
            "components": components.tolist(),
            "head_state_dict": _state_dict_to_payload(
                head.state_dict()
            ),
            "training_metrics": [
                dict(value) for value in self._training_metrics
            ],
            "selection_metrics": [
                dict(value) for value in self._selection_metrics
            ],
            "selected_step": self._selected_step,
            "calibration": dict(self._calibration),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RawLowRankRiskModel":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported raw low-rank artifact")
        torch = _require_torch()
        model = cls(**dict(payload["config"]))
        from quantis_core.graph_telemetry import DeclaredTelemetryGraph

        model._graph = DeclaredTelemetryGraph.from_dict(
            payload["graph"]
        )
        model._feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        model._ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        model._center = np.asarray(
            payload["center"], dtype=np.float64
        )
        model._components = np.asarray(
            payload["components"], dtype=np.float64
        )
        model._head = _build_risk_head(
            torch,
            input_width=5 * model.width,
            hidden_width=model.hidden_width,
        )
        model._head.load_state_dict(
            _state_dict_from_payload(
                torch, payload["head_state_dict"]
            )
        )
        model._head.eval()
        for parameter in model._head.parameters():
            parameter.requires_grad_(False)
        model._training_metrics = _metric_rows(
            payload["training_metrics"]
        )
        model._selection_metrics = _metric_rows(
            payload["selection_metrics"]
        )
        model._selected_step = int(payload["selected_step"])
        model._calibration = {
            str(key): float(value)
            for key, value in dict(payload["calibration"]).items()
        }
        return model

    def _features(
        self, histories: NDArray[np.float64], graph: Any
    ) -> NDArray[np.float64]:
        graph_, features, ownership, components, _ = (
            self._fitted_values()
        )
        values = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != graph_.to_dict()
            or values.shape[1:] != (
                20,
                len(graph_.entities),
                len(features),
            )
            or self._center is None
        ):
            raise ValueError("raw low-rank histories are invalid")
        patches = _raw_patch_matrix(values, ownership)
        projected = (patches - self._center) @ components.T
        return projected.reshape(len(values), -1)

    def _fitted_values(
        self,
    ) -> Tuple[Any, Tuple[str, ...], NDArray[np.bool_], Any, Any]:
        if (
            self._graph is None
            or not self._feature_names
            or self._ownership is None
            or self._components is None
            or self._head is None
        ):
            raise ValueError("raw low-rank model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership,
            self._components,
            self._head,
        )

    def _selected_values(
        self,
    ) -> Tuple[Any, Tuple[str, ...], NDArray[np.bool_], Any, Any]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("raw low-rank head is not selected")
        return values


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    pretrain_steps: int,
    alert_steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
    expected_pair_count: int = 40,
) -> Path:
    """Run the non-overwriting factorial and atomically publish evidence."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "SC-JEPA refuses an existing output or staging directory"
        )
    frozen_cache = (Path.cwd() / FROZEN_CACHE).resolve()
    frozen_output = (Path.cwd() / FROZEN_OUTPUT).resolve()
    interpretable = (
        cache == frozen_cache
        and pretrain_steps == FROZEN_PRETRAIN_STEPS
        and alert_steps == FROZEN_ALERT_STEPS
        and latency_repetitions == 100
        and expected_pair_count == 40
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen SC-JEPA runs require "
            "--allow-noninterpretable-smoke"
        )
    if not interpretable and output == frozen_output:
        raise ValueError(
            "a smoke run cannot use the frozen SC-JEPA result path"
        )
    implementation_commit = _git_head()
    implementation_sources = implementation_source_identity(
        commit=implementation_commit,
        require_head_match=interpretable,
    )
    building.mkdir(parents=True)
    started = time.time()
    try:
        data = load_edge_dynamics_cache(cache)
        partitions = {
            role: partition_worker_topology(windows)
            for role, windows in data.windows.items()
        }
        fit_windows = partitions["fit"].in_distribution
        selection_windows = partitions["selection"].in_distribution
        role_windows = {
            "calibration": partitions[
                "calibration"
            ].in_distribution,
            "evaluation_iid": partitions[
                "evaluation"
            ].in_distribution,
            "evaluation_transfer": partitions[
                "evaluation"
            ].held_out,
        }
        role_input_identities = {
            "fit": _role_input_identity(fit_windows),
            "selection": _role_input_identity(selection_windows),
            **{
                role: _role_input_identity(windows)
                for role, windows in role_windows.items()
            },
        }
        phases: Dict[str, float] = {
            "pretraining_started_unix_seconds": time.time()
        }
        event_definition = HepaEventDefinition.fit(fit_windows)
        event_fit_evidence = _event_fit_evidence(fit_windows)
        models: Dict[str, ScJepaModel] = {}
        fit_seconds: Dict[str, float] = {}
        for name, use_codebook, multi_resolution in (
            ("continuous_single", False, False),
            ("continuous_multi", False, True),
            ("codebook_single", True, False),
            ("codebook_multi", True, True),
        ):
            fit_started = time.perf_counter()
            model = ScJepaModel(
                ScJepaConfig(
                    use_codebook=use_codebook,
                    multi_resolution=multi_resolution,
                    pretrain_steps=pretrain_steps,
                    alert_steps=alert_steps,
                    expected_pair_count=expected_pair_count,
                )
            )
            model.fit(fit_windows).select(selection_windows)
            model.fit_alert_head(
                fit_windows, event_definition
            ).select_alert_head(
                selection_windows, event_definition
            ).fit_calibration(
                role_windows["calibration"], event_definition
            )
            fit_seconds[name] = time.perf_counter() - fit_started
            models[name] = model
            _print_progress(
                "fitted",
                {
                    "model": name,
                    "seconds": fit_seconds[name],
                    "selection": model.selection_metrics,
                    "alert_selection": model.alert_selection_metrics,
                    "calibration": model.calibration,
                },
            )
        raw_started = time.perf_counter()
        raw_model = RawLowRankRiskModel(
            alert_steps=alert_steps
        ).fit(
            fit_windows, event_definition
        ).select(
            selection_windows, event_definition
        ).fit_calibration(
            role_windows["calibration"], event_definition
        )
        fit_seconds["raw_low_rank"] = (
            time.perf_counter() - raw_started
        )
        phases["calibration_completed_unix_seconds"] = time.time()
        model_payloads = {
            name: model.to_dict() for name, model in models.items()
        }
        raw_payload = raw_model.to_dict()
        training_bindings = {
            **{
                name: _training_binding(
                    model_payloads[name], role_input_identities
                )
                for name in SC_JEPA_CELL_NAMES
            },
            "raw_low_rank": _training_binding(
                raw_payload, role_input_identities
            ),
        }
        restored_models = {
            name: ScJepaModel.from_dict(payload)
            for name, payload in model_payloads.items()
        }
        restored_raw = RawLowRankRiskModel.from_dict(raw_payload)
        restored_model_payloads = {
            name: model.to_dict()
            for name, model in restored_models.items()
        }
        restored_raw_payload = restored_raw.to_dict()
        risks: Dict[str, Dict[str, NDArray[np.float64]]] = {}
        restored_risks: Dict[
            str, Dict[str, NDArray[np.float64]]
        ] = {}
        calibrated_risks: Dict[
            str, Dict[str, NDArray[np.float64]]
        ] = {}
        restored_calibrated_risks: Dict[
            str, Dict[str, NDArray[np.float64]]
        ] = {}
        alert_decisions: Dict[
            str, Dict[str, NDArray[np.bool_]]
        ] = {}
        restored_alert_decisions: Dict[
            str, Dict[str, NDArray[np.bool_]]
        ] = {}
        labels = {}
        trajectory_ids = {}
        transition_indices = {}
        trajectory_onsets = {}
        phases["evaluation_started_unix_seconds"] = time.time()
        for role, windows in role_windows.items():
            labels[role] = event_definition.labels(windows)[:, -1]
            trajectory_ids[role] = windows.trajectory_ids
            transition_indices[role] = windows.transition_indices.copy()
            trajectory_onsets[role] = trajectory_action_onsets(windows)
            risks[role] = {}
            restored_risks[role] = {}
            calibrated_risks[role] = {}
            restored_calibrated_risks[role] = {}
            alert_decisions[role] = {}
            restored_alert_decisions[role] = {}
            for name in SC_JEPA_CELL_NAMES:
                risks[role][name] = models[name].predict_risk(
                    windows.histories, windows.graph
                )
                restored_risks[role][name] = restored_models[
                    name
                ].predict_risk(windows.histories, windows.graph)
                calibrated_risks[role][name] = models[
                    name
                ].calibrated_risk(windows.histories, windows.graph)
                restored_calibrated_risks[role][name] = (
                    restored_models[name].calibrated_risk(
                        windows.histories, windows.graph
                    )
                )
                alert_decisions[role][name] = models[
                    name
                ].alert_decisions(windows.histories, windows.graph)
                restored_alert_decisions[role][name] = (
                    restored_models[name].alert_decisions(
                        windows.histories, windows.graph
                    )
                )
            risks[role]["raw_low_rank"] = raw_model.predict_risk(
                windows.histories, windows.graph
            )
            restored_risks[role][
                "raw_low_rank"
            ] = restored_raw.predict_risk(
                windows.histories, windows.graph
            )
            calibrated_risks[role][
                "raw_low_rank"
            ] = raw_model.calibrated_risk(
                windows.histories, windows.graph
            )
            restored_calibrated_risks[role][
                "raw_low_rank"
            ] = restored_raw.calibrated_risk(
                windows.histories, windows.graph
            )
            alert_decisions[role][
                "raw_low_rank"
            ] = raw_model.alert_decisions(
                windows.histories, windows.graph
            )
            restored_alert_decisions[role][
                "raw_low_rank"
            ] = restored_raw.alert_decisions(
                windows.histories, windows.graph
            )
        phases["evaluation_completed_unix_seconds"] = time.time()
        candidate_fit = models["codebook_multi"].encode(
            fit_windows.histories, fit_windows.graph
        )
        transfer_windows = role_windows["evaluation_transfer"]
        representations = {
            name: models[name].encode(
                transfer_windows.histories, transfer_windows.graph
            )
            for name in SC_JEPA_CELL_NAMES
        }
        restored_representations = {
            name: restored_models[name].encode(
                transfer_windows.histories, transfer_windows.graph
            )
            for name in SC_JEPA_CELL_NAMES
        }
        candidate_transfer = representations["codebook_multi"]
        restored_candidate_transfer = restored_representations[
            "codebook_multi"
        ]
        if candidate_transfer.code_probabilities is None:
            raise RuntimeError("SC-JEPA candidate emitted no codes")
        if restored_candidate_transfer.code_probabilities is None:
            raise RuntimeError(
                "restored SC-JEPA candidate emitted no codes"
            )
        pca = HepaEntityPcaBaseline(width=32).fit(fit_windows)
        pca_fit_tokens = pca.encode(
            fit_windows.histories, fit_windows.graph
        ).tokens
        pca_transfer_tokens = pca.encode(
            transfer_windows.histories, transfer_windows.graph
        ).tokens
        current_fit = fit_windows.histories[:, -1]
        current_transfer = transfer_windows.histories[:, -1]
        candidate_probe = EntityStateRidgeProbe().fit(
            candidate_fit.tokens,
            current_fit,
            candidate_fit.ownership_mask,
        )
        pca_probe = EntityStateRidgeProbe().fit(
            pca_fit_tokens,
            current_fit,
            pca.encode(
                fit_windows.histories[:1], fit_windows.graph
            ).ownership_mask,
        )
        if (
            not np.array_equal(
                candidate_probe.target_scale, pca_probe.target_scale
            )
            or not np.array_equal(
                candidate_probe.target_varying_mask,
                pca_probe.target_varying_mask,
            )
        ):
            raise RuntimeError("SC-JEPA state probe targets diverged")
        state_predictions = {
            "codebook_multi": candidate_probe.predict(
                candidate_transfer.tokens
            ),
            "matched_pca": pca_probe.predict(pca_transfer_tokens),
        }
        all_models: Dict[str, Any] = {**models, "raw_low_rank": raw_model}
        latency_samples, peak_rss_bytes = _measure_latency(
            models=all_models,
            graph=fit_windows.graph,
            example=transfer_windows.histories[:1],
            repetitions=latency_repetitions,
        )
        final_implementation_sources = implementation_source_identity(
            commit=implementation_commit,
            require_head_match=interpretable,
        )
        if (
            final_implementation_sources != implementation_sources
            or (
                interpretable
                and _git_head() != implementation_commit
            )
        ):
            raise RuntimeError(
                "SC-JEPA implementation identity changed during run"
            )
        audit_histories = transfer_windows.histories[:2].copy()
        audit_forbidden = np.concatenate(
            (
                transfer_windows.future_states[:2].reshape(2, -1),
                transfer_windows.future_controls[:2].reshape(2, -1),
                transfer_windows.future_actions[:2].reshape(2, -1),
            ),
            axis=1,
        )
        audit_counterfactual_forbidden = audit_forbidden.copy()
        audit_counterfactual_forbidden[:, 0] += 1.0
        audit_counterfactual_forbidden[:, -1] += 1.0
        audit_original_outputs = np.concatenate(
            (
                models["codebook_multi"]
                .encode(audit_histories, transfer_windows.graph)
                .tokens.reshape(2, -1),
                models["codebook_multi"]
                .predict_risk(audit_histories, transfer_windows.graph)[
                    :, None
                ],
            ),
            axis=1,
        )
        audit_counterfactual_outputs = np.concatenate(
            (
                models["codebook_multi"]
                .encode(audit_histories.copy(), transfer_windows.graph)
                .tokens.reshape(2, -1),
                models["codebook_multi"]
                .predict_risk(
                    audit_histories.copy(), transfer_windows.graph
                )[:, None],
            ),
            axis=1,
        )
        audit_forbidden_keyword_rejections = np.asarray(
            [
                _rejects_forbidden_keyword(
                    models["codebook_multi"].encode,
                    audit_histories,
                    transfer_windows.graph,
                    "future_states",
                    transfer_windows.future_states[:2],
                ),
                _rejects_forbidden_keyword(
                    models["codebook_multi"].predict_risk,
                    audit_histories,
                    transfer_windows.graph,
                    "future_controls",
                    transfer_windows.future_controls[:2],
                ),
                _rejects_forbidden_keyword(
                    models["codebook_multi"].predict_risk,
                    audit_histories,
                    transfer_windows.graph,
                    "future_actions",
                    transfer_windows.future_actions[:2],
                ),
            ],
            dtype=np.bool_,
        )
        protocol = {
            "schema_version": 1,
            "kind": "sc_jepa_interaction_protocol",
            "contract": "sc-jepa-interaction-v1",
            "interpretable": interpretable,
            "smoke_only": not interpretable,
            "pretrain_steps": pretrain_steps,
            "alert_steps": alert_steps,
            "frozen_pretrain_steps": FROZEN_PRETRAIN_STEPS,
            "frozen_alert_steps": FROZEN_ALERT_STEPS,
            "seed": 13013,
            "expected_pair_count": expected_pair_count,
            "started_unix_seconds": started,
            "completed_unix_seconds": time.time(),
            "fit_seconds": fit_seconds,
            "runtime": _runtime_identity(),
        }
        data_identity = {
            "schema_version": 1,
            "kind": "sc_jepa_data_identity",
            "cache_directory": str(cache),
            "cache_manifest_sha256": _file_sha256(
                cache / "artifact-manifest.json"
            ),
            "source_corpus_sha256": data.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                data.source_artifact_manifest_sha256
            ),
            "preprocessing_protocol": data.preprocessing_protocol,
            "semantic_schema_sha256": (
                fit_windows.semantic_schema_sha256
            ),
            "implementation_commit": implementation_commit,
            "implementation_sources": implementation_sources,
            "git_status": _git_status(),
        }
        metadata = {
            "schema_version": 1,
            "kind": "sc_jepa_stored_assessment_inputs",
            "model_names": list(SC_JEPA_ASSESSMENT_MODEL_NAMES),
            "role_names": list(SC_JEPA_ASSESSMENT_ROLE_NAMES),
            "trajectory_ids": {
                role: list(values)
                for role, values in trajectory_ids.items()
            },
            "trajectory_onsets": {
                role: dict(values)
                for role, values in trajectory_onsets.items()
            },
            "event_fit_control_trajectory_ids": list(
                event_fit_evidence["control_trajectory_ids"]
            ),
            "source_role_pair_ids": {
                role: list(data.roles.pair_ids(role))
                for role in (
                    "fit",
                    "selection",
                    "calibration",
                    "evaluation",
                )
            },
            "used_pair_ids": {
                "fit": sorted(set(fit_windows.matched_pair_ids)),
                "selection": sorted(
                    set(selection_windows.matched_pair_ids)
                ),
                **{
                    role: sorted(set(windows.matched_pair_ids))
                    for role, windows in role_windows.items()
                },
            },
            "role_input_identities": role_input_identities,
            "inference_signatures": {
                "sc_jepa_encode": list(
                    inspect.signature(ScJepaModel.encode).parameters
                ),
                "sc_jepa_predict_risk": list(
                    inspect.signature(
                        ScJepaModel.predict_risk
                    ).parameters
                ),
                "raw_low_rank_predict_risk": list(
                    inspect.signature(
                        RawLowRankRiskModel.predict_risk
                    ).parameters
                ),
            },
            "phases": phases,
            "peak_rss_bytes": peak_rss_bytes,
        }
        _write_json(building / "protocol.json", protocol)
        _write_json(building / "data-identity.json", data_identity)
        _write_json(
            building / "event-definition.json",
            event_definition.to_dict(),
        )
        _write_json(
            building / "models.json",
            {
                "schema_version": 1,
                "kind": "sc_jepa_fitted_models",
                "models": model_payloads,
                "restored_models": restored_model_payloads,
                "raw_low_rank": raw_payload,
                "restored_raw_low_rank": restored_raw_payload,
                "training_bindings": training_bindings,
                "event_fit_binding": _event_fit_binding(
                    event_definition.to_dict(),
                    event_fit_evidence,
                    role_input_identities["fit"],
                ),
                "entity_pca": pca.to_dict(),
                "state_probes": {
                    "codebook_multi": candidate_probe.to_dict(),
                    "matched_pca": pca_probe.to_dict(),
                },
                "reference_bindings": _reference_bindings(
                    entity_pca=pca.to_dict(),
                    candidate_probe=candidate_probe.to_dict(),
                    pca_probe=pca_probe.to_dict(),
                    fit_identity=role_input_identities["fit"],
                ),
            },
        )
        _write_json(building / "assessment-metadata.json", metadata)
        _write_evidence(
            building / "sc-jepa-evidence.npz",
            risks=risks,
            restored_risks=restored_risks,
            calibrated_risks=calibrated_risks,
            restored_calibrated_risks=restored_calibrated_risks,
            alert_decisions=alert_decisions,
            restored_alert_decisions=restored_alert_decisions,
            labels=labels,
            transition_indices=transition_indices,
            representations=representations,
            restored_representations=restored_representations,
            state_truth=current_transfer,
            state_fit_truth=current_fit,
            state_ownership_mask=candidate_fit.ownership_mask,
            state_scale=candidate_probe.target_scale,
            state_varying_mask=(
                candidate_probe.target_varying_mask
            ),
            state_predictions=state_predictions,
            latency_samples=latency_samples,
            event_fit_deltas=event_fit_evidence["deltas"],
            event_fit_offsets=event_fit_evidence["offsets"],
            event_fit_ownership=event_fit_evidence["ownership"],
            event_fit_future_actions=event_fit_evidence[
                "future_actions"
            ],
            audit_histories=audit_histories,
            audit_counterfactual_histories=audit_histories.copy(),
            audit_forbidden=audit_forbidden,
            audit_counterfactual_forbidden=(
                audit_counterfactual_forbidden
            ),
            audit_original_outputs=audit_original_outputs,
            audit_counterfactual_outputs=(
                audit_counterfactual_outputs
            ),
            audit_forbidden_keyword_rejections=(
                audit_forbidden_keyword_rejections
            ),
        )
        _copy_reproduction_sources(building)
        if __package__:
            from lab.action_dynamics.prototype_sc_jepa_interaction_assessor import (
                assess_stored_bundle,
                verify_stored_assessment,
            )
        else:
            from prototype_sc_jepa_interaction_assessor import (
                assess_stored_bundle,
                verify_stored_assessment,
            )

        assessment = assess_stored_bundle(
            building, verify_manifest=False
        )
        _write_json(building / "assessment.json", assessment)
        (building / "report.md").write_text(
            _render_report(assessment, interpretable=interpretable)
        )
        write_artifact_manifest(building)
        verify_stored_assessment(building)
        os.replace(building, output)
        return output
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "kind": "sc_jepa_staging_failure",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        (building / "failure.json").write_text(_pretty_json(failure))
        raise


def _measure_latency(
    *,
    models: Mapping[str, Any],
    graph: Any,
    example: NDArray[np.float64],
    repetitions: int,
) -> Tuple[Mapping[str, NDArray[np.float64]], float]:
    if repetitions < 1:
        raise ValueError("SC-JEPA latency repetitions must be positive")
    result = {}
    for name, model in models.items():
        for _ in range(10):
            model.predict_risk(example, graph)
        samples = np.empty(repetitions, dtype=np.float64)
        for repetition in range(repetitions):
            started = time.perf_counter()
            model.predict_risk(example, graph)
            samples[repetition] = (
                time.perf_counter() - started
            ) * 1000.0
        result[name] = samples
    peak_rss_bytes = float(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    if platform.system() != "Darwin":
        peak_rss_bytes *= 1024.0
    return result, peak_rss_bytes


def _rejects_forbidden_keyword(
    function: Any,
    histories: NDArray[np.float64],
    graph: Any,
    keyword: str,
    value: NDArray[np.float64],
) -> bool:
    try:
        function(histories, graph, **{keyword: value})
    except TypeError:
        return True
    return False


def _write_evidence(
    path: Path,
    *,
    risks: Mapping[str, Mapping[str, NDArray[np.float64]]],
    restored_risks: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    calibrated_risks: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    restored_calibrated_risks: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    alert_decisions: Mapping[
        str, Mapping[str, NDArray[np.bool_]]
    ],
    restored_alert_decisions: Mapping[
        str, Mapping[str, NDArray[np.bool_]]
    ],
    labels: Mapping[str, NDArray[np.bool_]],
    transition_indices: Mapping[str, NDArray[np.int64]],
    representations: Mapping[str, Any],
    restored_representations: Mapping[str, Any],
    state_truth: NDArray[np.float64],
    state_fit_truth: NDArray[np.float64],
    state_ownership_mask: NDArray[np.bool_],
    state_scale: NDArray[np.float64],
    state_varying_mask: NDArray[np.bool_],
    state_predictions: Mapping[str, NDArray[np.float64]],
    latency_samples: Mapping[str, NDArray[np.float64]],
    event_fit_deltas: NDArray[np.float64],
    event_fit_offsets: NDArray[np.int64],
    event_fit_ownership: NDArray[np.bool_],
    event_fit_future_actions: NDArray[np.float64],
    audit_histories: NDArray[np.float64],
    audit_counterfactual_histories: NDArray[np.float64],
    audit_forbidden: NDArray[np.float64],
    audit_counterfactual_forbidden: NDArray[np.float64],
    audit_original_outputs: NDArray[np.float64],
    audit_counterfactual_outputs: NDArray[np.float64],
    audit_forbidden_keyword_rejections: NDArray[np.bool_],
) -> None:
    arrays: Dict[str, NDArray[Any]] = {}
    for prefix, roles in (
        ("risk", risks),
        ("restored_risk", restored_risks),
        ("calibrated", calibrated_risks),
        ("restored_calibrated", restored_calibrated_risks),
        ("alert_decision", alert_decisions),
        ("restored_alert_decision", restored_alert_decisions),
    ):
        arrays.update(
            {
                f"{prefix}__{role}__{model}": values
                for role, models in roles.items()
                for model, values in models.items()
            }
        )
    arrays.update(
        {f"labels__{role}": values for role, values in labels.items()}
    )
    arrays.update(
        {
            f"transition_indices__{role}": values
            for role, values in transition_indices.items()
        }
    )
    for name in SC_JEPA_CELL_NAMES:
        encoded = representations[name]
        restored = restored_representations[name]
        arrays[f"representation_tokens__{name}"] = encoded.tokens
        arrays[f"restored_representation_tokens__{name}"] = (
            restored.tokens
        )
        arrays[f"representation_patch_values__{name}"] = (
            encoded.patch_values
        )
        arrays[f"restored_representation_patch_values__{name}"] = (
            restored.patch_values
        )
        has_codes = encoded.code_probabilities is not None
        restored_has_codes = restored.code_probabilities is not None
        arrays[f"representation_has_codes__{name}"] = np.asarray(
            has_codes, dtype=np.bool_
        )
        arrays[f"restored_representation_has_codes__{name}"] = (
            np.asarray(restored_has_codes, dtype=np.bool_)
        )
        if has_codes:
            arrays[f"representation_code_probabilities__{name}"] = (
                encoded.code_probabilities
            )
        if restored_has_codes:
            arrays[
                f"restored_representation_code_probabilities__{name}"
            ] = restored.code_probabilities
    arrays["state_truth"] = state_truth
    arrays["state_fit_truth"] = state_fit_truth
    arrays["state_ownership_mask"] = state_ownership_mask
    arrays["state_scale"] = state_scale
    arrays["state_varying_mask"] = state_varying_mask
    arrays.update(
        {
            f"state_prediction__{name}": values
            for name, values in state_predictions.items()
        }
    )
    arrays.update(
        {
            f"latency_samples__{name}": values
            for name, values in latency_samples.items()
        }
    )
    arrays["event_fit_deltas"] = event_fit_deltas
    arrays["event_fit_offsets"] = event_fit_offsets
    arrays["event_fit_ownership"] = event_fit_ownership
    arrays["event_fit_future_actions"] = event_fit_future_actions
    arrays["audit_histories"] = audit_histories
    arrays[
        "audit_counterfactual_histories"
    ] = audit_counterfactual_histories
    arrays["audit_forbidden"] = audit_forbidden
    arrays[
        "audit_counterfactual_forbidden"
    ] = audit_counterfactual_forbidden
    arrays["audit_original_outputs"] = audit_original_outputs
    arrays[
        "audit_counterfactual_outputs"
    ] = audit_counterfactual_outputs
    arrays["audit_forbidden_keyword_rejections"] = (
        audit_forbidden_keyword_rejections
    )
    np.savez_compressed(path, **arrays)


def _raw_patch_matrix(
    histories: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> NDArray[np.float64]:
    values = np.asarray(histories, dtype=np.float64)
    batch, _, entities, features = values.shape
    patches = (
        values[:, -10:]
        .reshape(batch, 5, 2, entities, features)
        .transpose(0, 1, 3, 2, 4)
    )
    owned = np.where(
        ownership[None, None, :, None],
        patches,
        0.0,
    )
    return owned.reshape(batch, 5, -1)


def _fit_owned_feature_mask(windows: Any) -> NDArray[np.bool_]:
    entity_positions = {
        entity_id: position
        for position, entity_id in enumerate(windows.entity_names)
    }
    feature_positions = {
        name: position
        for position, name in enumerate(windows.state_feature_names)
    }
    mask = np.zeros(
        (len(windows.entity_names), len(windows.state_feature_names)),
        dtype=np.bool_,
    )
    for feature_key, entity_id in windows.graph.binding_map().items():
        feature_name = feature_key.split(".", 1)[-1]
        if (
            entity_id in entity_positions
            and feature_name in feature_positions
        ):
            mask[
                entity_positions[entity_id],
                feature_positions[feature_name],
            ] = True
    mask |= np.ptp(windows.histories, axis=(0, 1)) > 1e-9
    return mask


def _control_trajectory_ids(windows: Any) -> Tuple[str, ...]:
    applicable = windows.action_feature_names.index("applicable")
    treatments = {
        windows.trajectory_ids[index]
        for index in range(len(windows.histories))
        if np.any(windows.future_actions[index, ..., applicable] > 0.5)
    }
    return tuple(sorted(set(windows.trajectory_ids) - treatments))


def _event_fit_evidence(windows: Any) -> Mapping[str, Any]:
    controls = _control_trajectory_ids(windows)
    trajectories = _trajectory_state_values(windows)
    deltas = [
        np.diff(trajectories[trajectory_id], axis=0)
        for trajectory_id in controls
    ]
    offsets = np.zeros(len(deltas) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        np.asarray([len(values) for values in deltas], dtype=np.int64)
    )
    return {
        "control_trajectory_ids": controls,
        "deltas": np.concatenate(deltas, axis=0),
        "offsets": offsets,
        "ownership": _fit_owned_feature_mask(windows),
        "future_actions": windows.future_actions.copy(),
    }


def _trajectory_state_values(
    windows: Any,
) -> Mapping[str, NDArray[np.float64]]:
    rows: Dict[str, list[int]] = {}
    for index, trajectory_id in enumerate(windows.trajectory_ids):
        rows.setdefault(trajectory_id, []).append(index)
    result = {}
    for trajectory_id, positions in rows.items():
        points: Dict[int, NDArray[np.float64]] = {}
        for row in positions:
            transition = int(windows.transition_indices[row])
            start = transition - windows.histories.shape[1] + 1
            for offset, value in enumerate(windows.histories[row]):
                _merge_trajectory_point(points, start + offset, value)
            for offset, value in enumerate(
                windows.future_states[row], start=1
            ):
                _merge_trajectory_point(
                    points, transition + offset, value
                )
        indices = np.asarray(sorted(points), dtype=np.int64)
        if len(indices) < 2 or np.any(np.diff(indices) != 1):
            raise ValueError("SC-JEPA trajectory evidence has gaps")
        result[trajectory_id] = np.stack(
            [points[int(index)] for index in indices], axis=0
        )
    return result


def _merge_trajectory_point(
    points: Dict[int, NDArray[np.float64]],
    index: int,
    value: NDArray[np.float64],
) -> None:
    current = np.asarray(value, dtype=np.float64)
    existing = points.get(index)
    if existing is not None and not np.allclose(
        existing, current, atol=1e-6, rtol=0.0
    ):
        raise ValueError("SC-JEPA trajectory evidence overlaps differ")
    points[index] = current


def _orient_components(components: NDArray[np.float64]) -> None:
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0


def _build_risk_head(
    torch: Any, *, input_width: int, hidden_width: int
) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(input_width, hidden_width),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_width, 1),
    )


def _state_dict_to_payload(
    state_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "values": value.detach().cpu().numpy().tolist(),
        }
        for name, value in state_dict.items()
    }


def _state_dict_from_payload(
    torch: Any, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = {}
    for name, raw in payload.items():
        value = dict(raw)
        array = np.asarray(value["values"])
        shape = tuple(int(item) for item in value["shape"])
        if array.shape != shape:
            raise ValueError("raw low-rank state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("raw low-rank state tensor is non-finite")
        result[str(name)] = torch.as_tensor(
            array, dtype=torch.float32
        )
    return result


def _metric_rows(values: Any) -> Tuple[Mapping[str, float], ...]:
    return tuple(
        {
            str(key): float(value)
            for key, value in dict(row).items()
        }
        for row in values
    )


def _copy_reproduction_sources(building: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    reproduction = building / "reproduction"
    reproduction.mkdir()
    for relative in (
        *IMPLEMENTATION_SOURCE_PATHS,
        *IMPLEMENTATION_SYMBOL_SOURCES,
    ):
        source = root / relative
        shutil.copy2(source, reproduction / source.name)


def implementation_source_identity(
    *, commit: str, require_head_match: bool
) -> Mapping[str, Mapping[str, Any]]:
    """Bind every retained implementation source to the current commit."""

    root = Path(__file__).resolve().parents[2]
    result: Dict[str, Mapping[str, Any]] = {}
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        source = root / relative
        local_bytes = source.read_bytes()
        completed = subprocess.run(
            ("git", "show", f"{commit}:{relative}"),
            cwd=root,
            check=False,
            capture_output=True,
        )
        head_bytes = completed.stdout if completed.returncode == 0 else None
        matches_head = (
            head_bytes is not None and local_bytes == head_bytes
        )
        if require_head_match and not matches_head:
            raise ValueError(
                "frozen SC-JEPA source does not match HEAD: "
                f"{relative}"
            )
        result[relative] = {
            "path": relative,
            "scope": "file",
            "bytes": len(local_bytes),
            "sha256": hashlib.sha256(local_bytes).hexdigest(),
            "matches_head": matches_head,
            "head_sha256": (
                None
                if head_bytes is None
                else hashlib.sha256(head_bytes).hexdigest()
            ),
        }
    for relative, symbols in IMPLEMENTATION_SYMBOL_SOURCES.items():
        source = root / relative
        local_file = source.read_bytes()
        completed = subprocess.run(
            ("git", "show", f"{commit}:{relative}"),
            cwd=root,
            check=False,
            capture_output=True,
        )
        head_file = (
            completed.stdout if completed.returncode == 0 else None
        )
        for symbol in symbols:
            local_bytes = _source_symbol_bytes(local_file, symbol)
            head_bytes = (
                None
                if head_file is None
                else _source_symbol_bytes(head_file, symbol)
            )
            matches_head = (
                head_bytes is not None and local_bytes == head_bytes
            )
            if require_head_match and not matches_head:
                raise ValueError(
                    "frozen SC-JEPA dependency symbol does not match "
                    f"{commit}: {relative}::{symbol}"
                )
            result[f"{relative}::{symbol}"] = {
                "path": relative,
                "scope": "symbol",
                "symbol": symbol,
                "bytes": len(local_bytes),
                "sha256": hashlib.sha256(local_bytes).hexdigest(),
                "matches_head": matches_head,
                "head_sha256": (
                    None
                    if head_bytes is None
                    else hashlib.sha256(head_bytes).hexdigest()
                ),
            }
    return result


def _source_symbol_bytes(source: bytes, symbol: str) -> bytes:
    text = source.decode("utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == symbol:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            return segment.encode("utf-8")
    raise ValueError(f"SC-JEPA dependency symbol is missing: {symbol}")


def write_artifact_manifest(building: Path) -> None:
    """Write the content-identity manifest for an artifact directory."""

    files = {
        str(path.relative_to(building)): {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(building.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    _write_json(
        building / "artifact-manifest.json",
        {
            "schema_version": 1,
            "kind": "sc_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(
    assessment: Mapping[str, Any], *, interpretable: bool
) -> str:
    transfer = assessment["risk_metrics"]["evaluation_transfer"]
    alerts = assessment["alert_metrics"]["evaluation_transfer"]
    rows = []
    for name in SC_JEPA_ASSESSMENT_MODEL_NAMES:
        rows.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{transfer[name]['brier']:.6f}",
                    f"{alerts[name]['control_trajectory_false_alarm_rate']:.3f}",
                    f"{alerts[name]['treatment_detection_rate']:.3f}",
                    str(
                        alerts[name][
                            "median_post_onset_delay_transitions"
                        ]
                    ),
                )
            )
            + " |"
        )
    return "\n".join(
        (
            "# SC-JEPA interaction report",
            "",
            (
                "Status: **frozen interpretable tracer**."
                if interpretable
                else "Status: **NON-INTERPRETABLE SMOKE RUN**."
            ),
            "",
            "| model | Brier | control FPR | detection | median delay |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            (
                "Brier interaction: "
                f"`{assessment['interactions']['held_transfer_brier']:.6f}`."
            ),
            (
                "Detection interaction: "
                f"`{assessment['interactions']['held_transfer_detection']:.3f}`."
            ),
            f"Decision: `{assessment['decision']}`.",
            "",
        )
    )


def _runtime_identity() -> Mapping[str, Any]:
    torch = _require_torch()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> Sequence[str]:
    return subprocess.run(
        ("git", "status", "--short"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _role_input_identity(windows: Any) -> Mapping[str, Any]:
    fields: Dict[str, Any] = {
        "row_count": len(windows.histories),
        "matched_pair_ids": list(windows.matched_pair_ids),
        "trajectory_ids": list(windows.trajectory_ids),
        "transition_indices": [
            int(value) for value in windows.transition_indices
        ],
        "semantic_schema_sha256": windows.semantic_schema_sha256,
        "entity_names": list(windows.entity_names),
        "state_feature_names": list(windows.state_feature_names),
        "control_feature_names": list(windows.control_feature_names),
        "action_feature_names": list(windows.action_feature_names),
        "graph_sha256": hashlib.sha256(
            _canonical_json_bytes(windows.graph.to_dict())
        ).hexdigest(),
        "arrays": {
            "histories": _array_identity(windows.histories),
            "future_states": _array_identity(windows.future_states),
            "future_controls": _array_identity(
                windows.future_controls
            ),
            "future_actions": _array_identity(windows.future_actions),
        },
    }
    return {
        **fields,
        "identity_sha256": hashlib.sha256(
            _canonical_json_bytes(fields)
        ).hexdigest(),
    }


def _array_identity(values: NDArray[Any]) -> Mapping[str, Any]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _training_binding(
    model_payload: Mapping[str, Any],
    role_identities: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "model_payload_sha256": hashlib.sha256(
            _canonical_json_bytes(model_payload)
        ).hexdigest(),
        "fit_input_sha256": role_identities["fit"]["identity_sha256"],
        "selection_input_sha256": role_identities["selection"][
            "identity_sha256"
        ],
        "calibration_input_sha256": role_identities["calibration"][
            "identity_sha256"
        ],
    }


def _event_fit_binding(
    event_definition: Mapping[str, Any],
    evidence: Mapping[str, Any],
    fit_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "fit_input_sha256": fit_identity["identity_sha256"],
        "event_definition_sha256": hashlib.sha256(
            _canonical_json_bytes(event_definition)
        ).hexdigest(),
        "deltas": _array_identity(evidence["deltas"]),
        "offsets": _array_identity(evidence["offsets"]),
        "ownership": _array_identity(evidence["ownership"]),
        "future_actions": _array_identity(evidence["future_actions"]),
    }


def _reference_bindings(
    *,
    entity_pca: Mapping[str, Any],
    candidate_probe: Mapping[str, Any],
    pca_probe: Mapping[str, Any],
    fit_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    fit_digest = fit_identity["identity_sha256"]
    return {
        name: {
            "fit_input_sha256": fit_digest,
            "payload_sha256": hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest(),
        }
        for name, payload in (
            ("entity_pca", entity_pca),
            ("codebook_multi_probe", candidate_probe),
            ("matched_pca_probe", pca_probe),
        )
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(_pretty_json(value))


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _print_progress(name: str, value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {"stage": name, **dict(value)},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _require_torch() -> Any:
    import torch

    return torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=FROZEN_PRETRAIN_STEPS,
    )
    parser.add_argument(
        "--alert-steps", type=int, default=FROZEN_ALERT_STEPS
    )
    parser.add_argument(
        "--latency-repetitions", type=int, default=100
    )
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    arguments = parser.parse_args()
    result = run_experiment(
        cache_directory=arguments.cache,
        output_directory=arguments.output,
        pretrain_steps=arguments.pretrain_steps,
        alert_steps=arguments.alert_steps,
        latency_repetitions=arguments.latency_repetitions,
        allow_noninterpretable_smoke=(
            arguments.allow_noninterpretable_smoke
        ),
    )
    print(result)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
