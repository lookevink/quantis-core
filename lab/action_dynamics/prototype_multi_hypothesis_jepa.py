"""Retained reproduction runner for the four-hypothesis trajectory JEPA.

This remains non-production experiment code. Keep it with the immutable v1
artifact and use a fresh ``--output`` directory for every rerun.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.action_conditioned_dynamics import (
    MixtureTrajectoryDistribution,
)
from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)
from quantis_core.edge_dynamics.data import (
    ActionConditionedWindows,
    load_edge_dynamics_cache,
    partition_worker_topology,
    prepare_worker_topology_transfer_data,
    source_artifact_manifest_sha256,
    topology_transfer_cache_address,
    validate_topology_transfer_cache,
    write_edge_dynamics_cache,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)


MODEL_NAMES = (
    "multi_hypothesis_jepa",
    "one_component_jepa",
    "capacity_matched_single_gaussian",
    "supervised_four_component_mixture",
    "raw_low_rank",
)
SEED = 307


@dataclass(frozen=True)
class PrototypeConfig:
    """Frozen compact latent-mixture recipe."""

    component_count: int
    objective: str
    state_latent_width: int = 12
    context_width: int = 16
    predictor_width: int = 128
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.996
    latent_weight: float = 0.20
    target_reconstruction_weight: float = 0.10
    context_reconstruction_weight: float = 0.05
    variance_floor: float = 1e-4
    seed: int = SEED

    def __post_init__(self) -> None:
        if (
            self.component_count not in (1, 4)
            or self.objective not in ("jepa", "supervised")
            or self.state_latent_width < 1
            or self.context_width < 1
            or self.predictor_width < 1
            or self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or self.variance_floor <= 0.0
        ):
            raise ValueError("multi-hypothesis prototype config is invalid")


class MultiHypothesisJepaPrototype:
    """Non-production entity-preserving latent trajectory mixture."""

    kind = "prototype_multi_hypothesis_trajectory_jepa_v1"

    def __init__(self, config: PrototypeConfig) -> None:
        self.config = config
        self.training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._network: Any = None
        self._shape: Optional[Tuple[int, int, int, int, int]] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "MultiHypothesisJepaPrototype":
        """Fit the frozen prototype on one fitting role."""

        import torch

        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        shape = (
            windows.histories.shape[2],
            windows.histories.shape[3],
            windows.future_states.shape[1],
            windows.future_controls.shape[2],
            windows.future_actions.shape[3],
        )
        self._shape = shape
        self._network = _build_network(torch, self.config, shape)
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self._network.parameters()
                if parameter.requires_grad
            ],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed)
        sample_count = len(windows.histories)
        metrics = []
        for epoch in range(self.config.epochs):
            order = generator.permutation(sample_count)
            totals = {
                "total": 0.0,
                "negative_log_likelihood": 0.0,
                "latent": 0.0,
                "target_reconstruction": 0.0,
                "context_reconstruction": 0.0,
            }
            batches = 0
            self._network.train()
            for start in range(0, sample_count, self.config.batch_size):
                selection = order[start : start + self.config.batch_size]
                histories = torch.as_tensor(
                    windows.histories[selection],
                    dtype=torch.float32,
                )
                controls = torch.as_tensor(
                    windows.future_controls[selection],
                    dtype=torch.float32,
                )
                actions = torch.as_tensor(
                    windows.future_actions[selection],
                    dtype=torch.float32,
                )
                future = torch.as_tensor(
                    windows.future_states[selection],
                    dtype=torch.float32,
                )
                optimizer.zero_grad(set_to_none=True)
                output = self._network(
                    histories,
                    controls,
                    actions,
                    future,
                )
                losses = _training_losses(
                    torch,
                    output,
                    histories,
                    future,
                    self.config,
                )
                losses["total"].backward()
                optimizer.step()
                if self.config.objective == "jepa":
                    self._network.update_target(self.config.ema_decay)
                for name in totals:
                    totals[name] += float(losses[name].detach())
                batches += 1
            row = {
                name: value / float(batches)
                for name, value in totals.items()
            }
            row["epoch"] = float(epoch + 1)
            metrics.append(row)
            if epoch in (0, 9, 19, 29, 39):
                print(
                    f"{self.config.objective}/K={self.config.component_count} "
                    f"epoch={epoch + 1} total={row['total']:.6f} "
                    f"nll={row['negative_log_likelihood']:.6f}",
                    flush=True,
                )
        self.training_metrics = tuple(metrics)
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
    ) -> MixtureTrajectoryDistribution:
        """Return one complete-trajectory predictive mixture."""

        import torch

        network, shape = self._fitted()
        values = np.asarray(histories, dtype=np.float32)
        controls = np.asarray(future_controls, dtype=np.float32)
        actions = np.asarray(future_actions, dtype=np.float32)
        if (
            values.ndim != 4
            or values.shape[2:] != shape[:2]
            or controls.shape != (len(values), shape[2], shape[3])
            or actions.shape
            != (len(values), shape[2], shape[0], shape[4])
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(controls))
            or not np.all(np.isfinite(actions))
        ):
            raise ValueError("prototype rollout inputs are invalid")
        means = []
        variances = []
        weights = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                end = start + self.config.batch_size
                output = network.predict(
                    torch.as_tensor(values[start:end]),
                    torch.as_tensor(controls[start:end]),
                    torch.as_tensor(actions[start:end]),
                )
                means.append(
                    output["component_mean"].detach().numpy()
                )
                variances.append(
                    output["component_variance"].detach().numpy()
                )
                weights.append(output["weight"].detach().numpy())
        return MixtureTrajectoryDistribution(
            component_mean=np.concatenate(means).astype(
                np.float64, copy=False
            ),
            component_variance=np.concatenate(variances).astype(
                np.float64, copy=False
            ),
            weight=np.concatenate(weights).astype(np.float64, copy=False),
        )

    def encode(
        self, histories: NDArray[Any]
    ) -> NDArray[np.float64]:
        """Return entity-preserving context tokens in graph order."""

        import torch

        network, shape = self._fitted()
        values = np.asarray(histories, dtype=np.float32)
        if values.ndim != 4 or values.shape[2:] != shape[:2]:
            raise ValueError("prototype histories do not match schema")
        parts = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                encoded = network.encode_context(
                    torch.as_tensor(
                        values[start : start + self.config.batch_size]
                    )
                )
                parts.append(encoded.detach().numpy())
        return np.concatenate(parts).astype(np.float64, copy=False)

    @property
    def parameter_count(self) -> int:
        """Return inference-time scalar parameters."""

        network, _ = self._fitted()
        return int(
            sum(
                parameter.numel()
                for name, parameter in network.named_parameters()
                if not name.startswith("target_encoder.")
            )
        )

    def save(self, directory: Path, name: str) -> int:
        """Store JSON configuration and a compressed numeric sidecar."""

        network, shape = self._fitted()
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / f"{name}.json"
        sidecar_path = directory / f"{name}.npz"
        metadata_path.write_text(
            _pretty_json(
                {
                    "schema_version": 1,
                    "kind": self.kind,
                    "config": asdict(self.config),
                    "shape": list(shape),
                    "parameter_count": self.parameter_count,
                    "training_metrics": [
                        dict(row) for row in self.training_metrics
                    ],
                }
            )
        )
        np.savez_compressed(
            sidecar_path,
            **{
                key: value.detach().numpy()
                for key, value in network.state_dict().items()
            },
        )
        return metadata_path.stat().st_size + sidecar_path.stat().st_size

    @classmethod
    def load(
        cls, directory: Path, name: str
    ) -> "MultiHypothesisJepaPrototype":
        """Restore a prototype model for parity checking."""

        import torch

        metadata = json.loads(
            (directory / f"{name}.json").read_text()
        )
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != cls.kind
        ):
            raise ValueError("unsupported prototype model artifact")
        model = cls(PrototypeConfig(**metadata["config"]))
        model._shape = tuple(int(value) for value in metadata["shape"])
        model._network = _build_network(
            torch, model.config, model._shape
        )
        with np.load(
            directory / f"{name}.npz", allow_pickle=False
        ) as arrays:
            state = {
                key: torch.as_tensor(arrays[key])
                for key in model._network.state_dict()
            }
        model._network.load_state_dict(state)
        model.training_metrics = tuple(metadata["training_metrics"])
        return model

    def _fitted(self) -> Tuple[Any, Tuple[int, int, int, int, int]]:
        if self._network is None or self._shape is None:
            raise ValueError("multi-hypothesis prototype is not fitted")
        return self._network, self._shape


def _build_network(
    torch: Any,
    config: PrototypeConfig,
    shape: Tuple[int, int, int, int, int],
) -> Any:
    entity_count, feature_count, horizon, control_count, action_count = shape
    latent_width = config.state_latent_width
    context_width = config.context_width
    exogenous_width = horizon * (
        control_count + entity_count * action_count
    )
    predictor_input = entity_count * context_width + exogenous_width
    latent_output = (
        config.component_count
        * horizon
        * entity_count
        * latent_width
    )

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Linear(
                feature_count, latent_width
            )
            self.target_encoder = torch.nn.Linear(
                feature_count, latent_width
            )
            self.target_encoder.load_state_dict(
                self.state_encoder.state_dict()
            )
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.context_projector = torch.nn.Linear(
                3 * latent_width, context_width
            )
            self.predictor_hidden = torch.nn.Linear(
                predictor_input, config.predictor_width
            )
            self.predictor_output = torch.nn.Linear(
                config.predictor_width, latent_output
            )
            self.weight_head = torch.nn.Linear(
                config.predictor_width, config.component_count
            )
            self.decoder = torch.nn.Linear(
                latent_width, feature_count
            )
            initial_variance = 0.25
            initial_raw = math.log(math.expm1(initial_variance))
            self.raw_variance = torch.nn.Parameter(
                torch.full(
                    (
                        config.component_count,
                        entity_count,
                        feature_count,
                    ),
                    initial_raw,
                )
            )

        def encode_states(self, values: Any) -> Any:
            return torch.nn.functional.gelu(
                self.state_encoder(values)
            )

        def encode_context(self, histories: Any) -> Any:
            encoded = self.encode_states(histories)
            summary = torch.cat(
                (
                    encoded[:, -1],
                    torch.mean(encoded, dim=1),
                    encoded[:, -1] - encoded[:, 0],
                ),
                dim=-1,
            )
            return torch.nn.functional.gelu(
                self.context_projector(summary)
            )

        def predict(
            self, histories: Any, controls: Any, actions: Any
        ) -> Mapping[str, Any]:
            context = self.encode_context(histories)
            combined = torch.cat(
                (
                    context.flatten(1),
                    controls.flatten(1),
                    actions.flatten(1),
                ),
                dim=1,
            )
            hidden = torch.nn.functional.gelu(
                self.predictor_hidden(combined)
            )
            latent = self.predictor_output(hidden).reshape(
                len(histories),
                config.component_count,
                horizon,
                entity_count,
                latent_width,
            )
            mean = self.decoder(latent)
            weight = torch.softmax(self.weight_head(hidden), dim=1)
            weight = torch.clamp(weight, min=1e-12)
            weight = weight / torch.sum(weight, dim=1, keepdim=True)
            variance = (
                torch.nn.functional.softplus(self.raw_variance)
                + config.variance_floor
            )[None, :, None].expand(
                len(histories),
                config.component_count,
                horizon,
                entity_count,
                feature_count,
            )
            return {
                "context": context,
                "component_latent": latent,
                "component_mean": mean,
                "component_variance": variance,
                "weight": weight,
            }

        def forward(
            self,
            histories: Any,
            controls: Any,
            actions: Any,
            future: Any,
        ) -> Mapping[str, Any]:
            output = dict(self.predict(histories, controls, actions))
            with torch.no_grad():
                target = torch.nn.functional.gelu(
                    self.target_encoder(future)
                )
            output["target_latent"] = target
            output["target_reconstruction"] = self.decoder(target)
            output["context_reconstruction"] = self.decoder(
                self.encode_states(histories)[:, -1]
            )
            return output

        def update_target(self, decay: float) -> None:
            with torch.no_grad():
                for online, target in zip(
                    self.state_encoder.parameters(),
                    self.target_encoder.parameters(),
                ):
                    target.mul_(decay).add_(
                        online, alpha=1.0 - decay
                    )

    return Network()


def _training_losses(
    torch: Any,
    output: Mapping[str, Any],
    histories: Any,
    future: Any,
    config: PrototypeConfig,
) -> Mapping[str, Any]:
    mean = output["component_mean"]
    variance = output["component_variance"]
    weight = output["weight"]
    terms = -0.5 * (
        torch.square(future[:, None] - mean) / variance
        + torch.log(variance)
        + math.log(2.0 * math.pi)
    )
    log_density = torch.sum(terms, dim=(2, 3, 4))
    log_joint = torch.log(weight) + log_density
    coordinate_count = int(np.prod(future.shape[1:]))
    nll = torch.mean(
        -torch.logsumexp(log_joint, dim=1) / coordinate_count
    )
    responsibilities = torch.softmax(log_joint, dim=1).detach()
    latent_error = torch.mean(
        torch.abs(
            output["component_latent"]
            - output["target_latent"][:, None]
        ),
        dim=(2, 3, 4),
    )
    latent = torch.mean(
        torch.sum(responsibilities * latent_error, dim=1)
    )
    target_reconstruction = torch.mean(
        torch.square(output["target_reconstruction"] - future)
    )
    context_reconstruction = torch.mean(
        torch.square(
            output["context_reconstruction"] - histories[:, -1]
        )
    )
    if config.objective == "jepa":
        total = (
            nll
            + config.latent_weight * latent
            + config.target_reconstruction_weight
            * target_reconstruction
            + config.context_reconstruction_weight
            * context_reconstruction
        )
    else:
        total = nll
    return {
        "total": total,
        "negative_log_likelihood": nll,
        "latent": latent,
        "target_reconstruction": target_reconstruction,
        "context_reconstruction": context_reconstruction,
    }


def run_prototype(
    *,
    corpus_directory: Path,
    cache_root: Path,
    output_directory: Path,
    epochs: int = 40,
) -> Mapping[str, Any]:
    """Fit the frozen models and stop conclusively at the safe-null gate."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite prototype result: {output_directory}"
        )
    staging = output_directory.with_name(
        output_directory.name + ".staging"
    )
    if staging.exists():
        raise FileExistsError(
            f"prototype staging directory already exists: {staging}"
        )
    prepared, cache_directory, cache_reused = _load_data(
        corpus_directory, cache_root
    )
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    fit = partitions["fit"].in_distribution
    selection = partitions["selection"].in_distribution
    transfer = partitions["evaluation"].held_out
    held_out_values = {
        partition.held_out_normalized_value
        for partition in partitions.values()
    }
    if len(held_out_values) != 1:
        raise ValueError("held-out topology drifted across roles")

    candidate_config = PrototypeConfig(
        component_count=4,
        objective="jepa",
        epochs=epochs,
    )
    one_jepa_config = PrototypeConfig(
        component_count=1,
        objective="jepa",
        epochs=epochs,
    )
    supervised_four_config = PrototypeConfig(
        component_count=4,
        objective="supervised",
        epochs=epochs,
    )
    candidate_probe = MultiHypothesisJepaPrototype(candidate_config)
    candidate_probe._shape = (
        fit.histories.shape[2],
        fit.histories.shape[3],
        fit.future_states.shape[1],
        fit.future_controls.shape[2],
        fit.future_actions.shape[3],
    )
    import torch

    candidate_probe._network = _build_network(
        torch, candidate_config, candidate_probe._shape
    )
    target_parameters = candidate_probe.parameter_count
    matched_width = _capacity_matched_width(
        torch,
        fit,
        target_parameters,
        epochs,
    )
    capacity_config = PrototypeConfig(
        component_count=1,
        objective="supervised",
        predictor_width=matched_width,
        epochs=epochs,
    )
    del candidate_probe

    models: Dict[str, Any] = {
        "multi_hypothesis_jepa": MultiHypothesisJepaPrototype(
            candidate_config
        ),
        "one_component_jepa": MultiHypothesisJepaPrototype(
            one_jepa_config
        ),
        "capacity_matched_single_gaussian": (
            MultiHypothesisJepaPrototype(capacity_config)
        ),
        "supervised_four_component_mixture": (
            MultiHypothesisJepaPrototype(supervised_four_config)
        ),
        "raw_low_rank": ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ),
    }
    training_seconds: Dict[str, float] = {}
    for name, model in models.items():
        print(f"fitting {name}", flush=True)
        started = time.perf_counter()
        model.fit(fit)
        training_seconds[name] = time.perf_counter() - started
        print(
            f"fitted {name} in {training_seconds[name]:.2f}s",
            flush=True,
        )

    staging.mkdir(parents=True)
    (staging / "models").mkdir()
    (staging / "predictions").mkdir()
    protocol = {
        "schema_version": 1,
        "kind": "multi_hypothesis_jepa_prototype_v1",
        "seed": SEED,
        "epochs": epochs,
        "model_configs": {
            "multi_hypothesis_jepa": asdict(candidate_config),
            "one_component_jepa": asdict(one_jepa_config),
            "capacity_matched_single_gaussian": asdict(
                capacity_config
            ),
            "supervised_four_component_mixture": asdict(
                supervised_four_config
            ),
            "raw_low_rank": asdict(LowRankConfig(rank=32)),
        },
        "scoring_contract": (
            "docs/specs/multi-hypothesis-jepa-scoring-contract-v1.md"
        ),
    }
    (staging / "protocol.json").write_text(_pretty_json(protocol))
    data_identity = {
        "schema_version": 1,
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_cache_address": cache_directory.name,
        "preprocessing_protocol": prepared.preprocessing_protocol,
        "cache_reused": cache_reused,
        "held_out_worker_topology_normalized_value": next(
            iter(held_out_values)
        ),
        "window_counts": {
            "fit": len(fit.histories),
            "selection": len(selection.histories),
            "transfer": len(transfer.histories),
        },
        "pair_counts": {
            "fit": len(set(fit.matched_pair_ids)),
            "selection": len(set(selection.matched_pair_ids)),
            "transfer": len(set(transfer.matched_pair_ids)),
        },
    }
    (staging / "data-identity.json").write_text(
        _pretty_json(data_identity)
    )

    model_evidence = {}
    for name, model in models.items():
        if isinstance(model, MultiHypothesisJepaPrototype):
            serialized_bytes = model.save(staging / "models", name)
            restored = MultiHypothesisJepaPrototype.load(
                staging / "models", name
            )
            expected = model.rollout(
                selection.histories[:8],
                selection.future_controls[:8],
                selection.future_actions[:8],
            )
            actual = restored.rollout(
                selection.histories[:8],
                selection.future_controls[:8],
                selection.future_actions[:8],
            )
            parity = bool(
                np.allclose(
                    expected.component_mean,
                    actual.component_mean,
                    atol=1e-6,
                    rtol=1e-6,
                )
                and np.allclose(
                    expected.component_variance,
                    actual.component_variance,
                    atol=1e-7,
                    rtol=1e-7,
                )
                and np.allclose(
                    expected.weight,
                    actual.weight,
                    atol=1e-7,
                    rtol=1e-7,
                )
            )
            parameter_count = model.parameter_count
            training_metrics = [
                dict(row) for row in model.training_metrics
            ]
        else:
            artifact = model.to_dict()
            model_path = staging / "models" / f"{name}.json"
            model_path.write_text(_pretty_json(artifact))
            serialized_bytes = model_path.stat().st_size
            restored = ContractiveLowRankDynamics.from_dict(artifact)
            expected_raw = _raw_mixture(model, selection, 8)
            actual_raw = _raw_mixture(restored, selection, 8)
            parity = bool(
                np.allclose(
                    expected_raw.component_mean,
                    actual_raw.component_mean,
                )
                and np.allclose(
                    expected_raw.component_variance,
                    actual_raw.component_variance,
                )
            )
            parameter_count = model.parameter_count
            training_metrics = []
        latency = _batch_one_latency(model, selection)
        model_evidence[name] = {
            "parameter_count": parameter_count,
            "serialized_bytes": serialized_bytes,
            "restoration_parity": parity,
            "batch_one_latency_ms": latency,
            "training_seconds": training_seconds[name],
            "training_metrics": training_metrics,
        }

    for role, windows in (
        ("selection", selection),
        ("transfer", transfer),
    ):
        _write_role_inputs(staging, role, windows)
        for name, model in models.items():
            print(f"predicting {role}/{name}", flush=True)
            distribution = (
                model.rollout(
                    windows.histories,
                    windows.future_controls,
                    windows.future_actions,
                )
                if isinstance(model, MultiHypothesisJepaPrototype)
                else _raw_mixture(model, windows)
            )
            _write_distribution(staging, role, name, distribution)

    assessment = assess_stored_prototype(staging)
    result = {
        "schema_version": 1,
        "kind": "multi_hypothesis_jepa_prototype_result_v1",
        "evidence_boundary": (
            "open development tracer; not sealed confirmation"
        ),
        "protocol": protocol,
        "data_identity": data_identity,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "torch_threads": int(torch.get_num_threads()),
        },
        "models": model_evidence,
        "assessment": assessment,
    }
    (staging / "prototype-result.json").write_text(
        _pretty_json(result)
    )
    (staging / "report.md").write_text(_report_markdown(result))
    manifest = {
        "schema_version": 1,
        "kind": "multi_hypothesis_jepa_prototype_manifest_v1",
        "sha256": {
            path.relative_to(staging).as_posix(): _file_sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        },
    }
    (staging / "artifact-manifest.json").write_text(
        _pretty_json(manifest)
    )
    staging.rename(output_directory)
    print(json.dumps(assessment, indent=2, sort_keys=True), flush=True)
    return result


def assess_stored_prototype(directory: Path) -> Mapping[str, Any]:
    """Purely recompute the safe-null decision from stored arrays."""

    role_metrics: Dict[str, Dict[str, Any]] = {}
    for role in ("selection", "transfer"):
        with np.load(
            directory / "predictions" / f"{role}-inputs.npz",
            allow_pickle=False,
        ) as arrays:
            observed = np.asarray(
                arrays["observed"], dtype=np.float64
            )
            action_active = np.asarray(
                arrays["action_active"], dtype=np.bool_
            )
            trajectory_ids = tuple(
                str(value) for value in arrays["trajectory_ids"]
            )
        metrics = {}
        for name in MODEL_NAMES:
            distribution = _read_distribution(directory, role, name)
            nll = distribution.negative_log_likelihood(observed)
            compatible = distribution.as_trajectory_distribution()
            squared = np.square(compatible.mean - observed)
            support = _supported_pair_rate(
                distribution,
                np.any(action_active, axis=1),
            )
            metrics[name] = {
                "trajectory_balanced_log_score": (
                    _trajectory_balanced_mean(nll, trajectory_ids)
                ),
                "normalized_mse_overall": float(np.mean(squared)),
                "normalized_mse_action_overlap": float(
                    np.mean(squared[action_active])
                ),
                "supported_pair_rate_action_overlap": support,
                "effective_hypothesis_count": float(
                    np.mean(
                        np.exp(
                            -np.sum(
                                distribution.weight
                                * np.log(distribution.weight),
                                axis=1,
                            )
                        )
                    )
                ),
                "finite": bool(
                    np.all(np.isfinite(nll))
                    and np.all(np.isfinite(compatible.mean))
                    and np.all(np.isfinite(compatible.variance))
                ),
            }
        role_metrics[role] = metrics
    selection = role_metrics["selection"]
    candidate = selection["multi_hypothesis_jepa"]
    raw = selection["raw_low_rank"]
    beats_one_jepa = (
        candidate["trajectory_balanced_log_score"]
        <= selection["one_component_jepa"][
            "trajectory_balanced_log_score"
        ]
        - 0.01
    )
    beats_supervised_mixture = (
        candidate["trajectory_balanced_log_score"]
        <= selection["supervised_four_component_mixture"][
            "trajectory_balanced_log_score"
        ]
        - 0.01
    )
    overall_safe = (
        candidate["normalized_mse_overall"]
        <= 1.05 * raw["normalized_mse_overall"]
    )
    overlap_safe = (
        candidate["normalized_mse_action_overlap"]
        <= 1.05 * raw["normalized_mse_action_overlap"]
    )
    supported = (
        candidate["supported_pair_rate_action_overlap"] >= 0.20
    )
    gates = {
        "selection_log_score_beats_one_component_jepa_by_0_01": (
            beats_one_jepa
        ),
        "selection_log_score_beats_supervised_mixture_by_0_01": (
            beats_supervised_mixture
        ),
        "selection_overall_mse_within_5_percent_of_raw": overall_safe,
        "selection_action_overlap_mse_within_5_percent_of_raw": (
            overlap_safe
        ),
        "selection_supported_pair_rate_at_least_20_percent": supported,
        "all_stored_outputs_finite": all(
            row["finite"]
            for role in role_metrics.values()
            for row in role.values()
        ),
    }
    selected = all(gates.values())
    return {
        "schema_version": 1,
        "kind": "multi_hypothesis_jepa_safe_null_assessment_v1",
        "selection_metrics": role_metrics["selection"],
        "diagnostic_transfer_metrics": role_metrics["transfer"],
        "gates": gates,
        "safe_null_passed": selected,
        "decision": (
            "continue_to_calibration_and_full_value_assessment"
            if selected
            else "reject_recipe_at_safe_null_selection"
        ),
        "assessment_scope": (
            "full calibration, energy, alert, and investigation assessment "
            "is required only if the preregistered safe-null gate passes"
        ),
    }


def _capacity_matched_width(
    torch: Any,
    fit: ActionConditionedWindows,
    target_parameters: int,
    epochs: int,
) -> int:
    shape = (
        fit.histories.shape[2],
        fit.histories.shape[3],
        fit.future_states.shape[1],
        fit.future_controls.shape[2],
        fit.future_actions.shape[3],
    )
    for width in range(1, 1025):
        config = PrototypeConfig(
            component_count=1,
            objective="supervised",
            predictor_width=width,
            epochs=epochs,
        )
        network = _build_network(torch, config, shape)
        count = int(
            sum(
                parameter.numel()
                for name, parameter in network.named_parameters()
                if not name.startswith("target_encoder.")
            )
        )
        if count >= 0.95 * target_parameters:
            if count > 1.05 * target_parameters:
                raise ValueError("cannot capacity-match single Gaussian")
            return width
    raise ValueError("capacity-matched width exceeds search boundary")


def _load_data(
    corpus_directory: Path,
    cache_root: Path,
) -> Tuple[Any, Path, bool]:
    source_manifest = source_artifact_manifest_sha256(
        corpus_directory
    )
    cache_directory = cache_root / topology_transfer_cache_address(
        source_manifest
    )
    if cache_directory.exists():
        prepared = load_edge_dynamics_cache(cache_directory)
        reused = True
    else:
        corpus = load_action_dynamics_development_corpus(
            corpus_directory
        )
        prepared = prepare_worker_topology_transfer_data(corpus)
        write_edge_dynamics_cache(prepared, cache_directory)
        reused = False
    validate_topology_transfer_cache(prepared, corpus_directory)
    return prepared, cache_directory, reused


def _raw_mixture(
    model: ContractiveLowRankDynamics,
    windows: ActionConditionedWindows,
    count: Optional[int] = None,
) -> MixtureTrajectoryDistribution:
    end = len(windows.histories) if count is None else count
    distribution = model.rollout(
        windows.histories[:end],
        windows.future_controls[:end],
        windows.future_actions[:end],
        windows.graph,
    )
    return MixtureTrajectoryDistribution(
        component_mean=distribution.mean[:, None],
        component_variance=distribution.variance[:, None],
        weight=np.ones((end, 1), dtype=np.float64),
    )


def _batch_one_latency(model: Any, windows: ActionConditionedWindows) -> float:
    timings = []
    for _ in range(5):
        if isinstance(model, MultiHypothesisJepaPrototype):
            model.rollout(
                windows.histories[:1],
                windows.future_controls[:1],
                windows.future_actions[:1],
            )
        else:
            _raw_mixture(model, windows, 1)
    for _ in range(20):
        started = time.perf_counter_ns()
        if isinstance(model, MultiHypothesisJepaPrototype):
            model.rollout(
                windows.histories[:1],
                windows.future_controls[:1],
                windows.future_actions[:1],
            )
        else:
            _raw_mixture(model, windows, 1)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return float(np.median(timings))


def _write_role_inputs(
    directory: Path,
    role: str,
    windows: ActionConditionedWindows,
) -> None:
    np.savez_compressed(
        directory / "predictions" / f"{role}-inputs.npz",
        observed=windows.future_states,
        action_active=np.any(
            windows.future_actions[..., 1] > 0.5, axis=2
        ),
        trajectory_ids=np.asarray(windows.trajectory_ids),
        matched_pair_ids=np.asarray(windows.matched_pair_ids),
        transition_indices=windows.transition_indices,
    )


def _write_distribution(
    directory: Path,
    role: str,
    name: str,
    distribution: MixtureTrajectoryDistribution,
) -> None:
    np.savez_compressed(
        directory / "predictions" / f"{role}-{name}.npz",
        component_mean=distribution.component_mean.astype(np.float32),
        component_variance=distribution.component_variance.astype(
            np.float32
        ),
        weight=distribution.weight.astype(np.float32),
    )


def _read_distribution(
    directory: Path,
    role: str,
    name: str,
) -> MixtureTrajectoryDistribution:
    with np.load(
        directory / "predictions" / f"{role}-{name}.npz",
        allow_pickle=False,
    ) as arrays:
        return MixtureTrajectoryDistribution(
            component_mean=np.asarray(
                arrays["component_mean"], dtype=np.float64
            ),
            component_variance=np.asarray(
                arrays["component_variance"], dtype=np.float64
            ),
            weight=np.asarray(arrays["weight"], dtype=np.float64),
        )


def _trajectory_balanced_mean(
    values: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
) -> float:
    grouped = {}
    for value, trajectory_id in zip(values, trajectory_ids):
        grouped.setdefault(trajectory_id, []).append(float(value))
    return float(
        np.mean(
            [
                np.mean(grouped[trajectory_id])
                for trajectory_id in sorted(grouped)
            ]
        )
    )


def _supported_pair_rate(
    distribution: MixtureTrajectoryDistribution,
    sample_mask: NDArray[np.bool_],
) -> float:
    if distribution.component_mean.shape[1] < 2:
        return 0.0
    supported = np.zeros(len(distribution.weight), dtype=np.bool_)
    for left in range(distribution.component_mean.shape[1]):
        for right in range(left + 1, distribution.component_mean.shape[1]):
            variance = 0.5 * (
                distribution.component_variance[:, left]
                + distribution.component_variance[:, right]
            )
            distance = np.sqrt(
                np.mean(
                    np.square(
                        distribution.component_mean[:, left]
                        - distribution.component_mean[:, right]
                    )
                    / variance,
                    axis=(1, 2, 3),
                )
            )
            supported |= (
                (distribution.weight[:, left] >= 0.10)
                & (distribution.weight[:, right] >= 0.10)
                & (distance >= 1.0)
            )
    return float(np.mean(supported[sample_mask]))


def _report_markdown(result: Mapping[str, Any]) -> str:
    assessment = result["assessment"]
    lines = [
        "# Multi-hypothesis trajectory JEPA prototype v1 result",
        "",
        f"Decision: **{assessment['decision']}**.",
        "",
        "## Selection gates",
        "",
    ]
    for name, passed in assessment["gates"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Open development, one deterministic seed, and one held-out "
            "worker topology. A failed safe-null gate ends the recipe before "
            "calibration or value-lane assessment.",
            "",
        ]
    )
    return "\n".join(lines)


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/action-dynamics/development-v1"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-preprocessing-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "prototype-multi-hypothesis-jepa-v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parsed = parser.parse_args(arguments)
    run_prototype(
        corpus_directory=parsed.corpus,
        cache_root=parsed.cache_root,
        output_directory=parsed.output,
        epochs=parsed.epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
