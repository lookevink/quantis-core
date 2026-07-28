"""Optional-PyTorch temporal graph JEPA for hybrid telemetry tokens.

This module deliberately imports PyTorch only when a model is fitted, restored,
or used. NumPy-only ingestion therefore remains usable without training extras.
"""

from dataclasses import asdict, dataclass
import copy
import importlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .hybrid_graph_tokens import (
    HybridGraphTokens,
    MultiMaskBatch,
    MultiMaskConfig,
    sample_multi_masks,
)


@dataclass(frozen=True)
class HybridJepaConfig:
    """Optimization and architecture controls for the hybrid graph JEPA."""

    latent_dimension: int = 64
    attention_heads: int = 4
    transformer_layers: int = 2
    feedforward_multiplier: int = 2
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.996
    mask_count: int = 2
    target_coverage: float = 0.75
    reconstruction_weight: float = 0.2
    context_reconstruction_weight: float = 0.2
    variance_weight: float = 0.05
    covariance_weight: float = 0.01
    device: str = "auto"
    seed: int = 89

    def __post_init__(self) -> None:
        positive_integers = (
            self.latent_dimension,
            self.attention_heads,
            self.transformer_layers,
            self.feedforward_multiplier,
            self.epochs,
            self.batch_size,
            self.mask_count,
        )
        if any(
            isinstance(value, bool) or value < 1
            for value in positive_integers
        ):
            raise ValueError("hybrid JEPA integer controls must be positive")
        if self.latent_dimension % self.attention_heads:
            raise ValueError(
                "latent dimension must be divisible by attention heads"
            )
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or not 0.0 < self.target_coverage < 1.0
            or min(
                self.reconstruction_weight,
                self.context_reconstruction_weight,
                self.variance_weight,
                self.covariance_weight,
            )
            < 0.0
        ):
            raise ValueError("hybrid JEPA numeric controls are invalid")
        if self.device not in ("auto", "cpu", "mps"):
            raise ValueError("hybrid JEPA device must be auto, cpu, or mps")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible configuration."""

        return dict(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HybridJepaConfig":
        """Restore a configuration from serialized values."""

        return cls(
            latent_dimension=int(payload["latent_dimension"]),
            attention_heads=int(payload["attention_heads"]),
            transformer_layers=int(payload["transformer_layers"]),
            feedforward_multiplier=int(
                payload["feedforward_multiplier"]
            ),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            ema_decay=float(payload["ema_decay"]),
            mask_count=int(payload["mask_count"]),
            target_coverage=float(payload["target_coverage"]),
            reconstruction_weight=float(
                payload["reconstruction_weight"]
            ),
            context_reconstruction_weight=float(
                payload.get(
                    "context_reconstruction_weight",
                    payload["reconstruction_weight"],
                )
            ),
            variance_weight=float(payload["variance_weight"]),
            covariance_weight=float(payload["covariance_weight"]),
            device=str(payload["device"]),
            seed=int(payload["seed"]),
        )


@dataclass(frozen=True)
class HybridJepaPrediction:
    """Future latents, exact-shape state recovery, and collapse evidence."""

    predicted_latents: NDArray[np.float64]
    target_latents: NDArray[np.float64]
    reconstructed_targets: NDArray[np.float64]
    reconstructed_contexts: NDArray[np.float64]
    validation_embeddings: NDArray[np.float64]
    diagnostics: Mapping[str, Any]


class HybridGraphJepa:
    """Train a masked temporal, relational, EMA-target JEPA."""

    kind = "hybrid_temporal_graph_jepa_v1"

    def __init__(self, config: HybridJepaConfig) -> None:
        self.config = config
        self.training_losses: Tuple[float, ...] = ()
        self.epoch_metrics: Tuple[Mapping[str, float], ...] = ()
        self.device = "uninitialized"
        self._network: Any = None
        self._schema: Optional[Dict[str, int]] = None
        self._semantic_schema: Optional[Dict[str, Any]] = None
        self._training_topology: Optional[NDArray[np.bool_]] = None

    def fit(
        self,
        tokens: HybridGraphTokens,
        masks: Optional[MultiMaskBatch] = None,
    ) -> "HybridGraphJepa":
        """Fit from NumPy tokens while transferring only minibatches to device."""

        torch = _require_torch()
        self.device = _select_device(torch, self.config.device)
        _seed_torch(torch, self.config.seed)
        schema = _schema(tokens)
        self._schema = schema
        self._semantic_schema = _semantic_schema(tokens)
        self._training_topology = tokens.typed_adjacency.copy()
        self._network = _build_network(torch, self.config, schema)
        self._network.to(self.device)
        optimizer = torch.optim.AdamW(
            self._network.trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if masks is None:
            masks = sample_multi_masks(
                tokens,
                MultiMaskConfig(
                    mask_count=self.config.mask_count,
                    target_coverage=self.config.target_coverage,
                    seed=self.config.seed,
                ),
            )
        _validate_masks(tokens, masks, self.config.mask_count)
        generator = np.random.default_rng(self.config.seed)
        losses: List[float] = []
        metrics: List[Mapping[str, float]] = []
        sample_count = len(tokens.fine_context)
        for _ in range(self.config.epochs):
            order = generator.permutation(sample_count)
            sums = {
                "total": 0.0,
                "latent": 0.0,
                "reconstruction": 0.0,
                "context_reconstruction": 0.0,
                "variance": 0.0,
                "covariance": 0.0,
            }
            batches = 0
            self._network.train()
            for start in range(0, sample_count, self.config.batch_size):
                selection = order[start : start + self.config.batch_size]
                batch = _batch_to_torch(
                    torch,
                    tokens,
                    selection,
                    self.device,
                )
                optimizer.zero_grad(set_to_none=True)
                mask_losses = []
                components = []
                for mask_index in range(self.config.mask_count):
                    visible = torch.as_tensor(
                        masks.context_masks[mask_index, selection],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    target_mask = torch.as_tensor(
                        masks.target_masks[mask_index, selection],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    output = self._network.forward_training(
                        batch,
                        visible,
                        target_mask,
                    )
                    component = _loss_components(
                        torch,
                        output,
                        batch,
                        self.config,
                    )
                    mask_losses.append(component["total"])
                    components.append(component)
                total = torch.stack(mask_losses).mean()
                total.backward()
                optimizer.step()
                self._network.update_target(self.config.ema_decay)
                for name in sums:
                    sums[name] += float(
                        torch.stack(
                            [item[name].detach() for item in components]
                        ).mean().cpu()
                    )
                batches += 1
            epoch = {
                name: value / float(max(batches, 1))
                for name, value in sums.items()
            }
            losses.append(epoch["total"])
            metrics.append(epoch)
        self.training_losses = tuple(losses)
        self.epoch_metrics = tuple(metrics)
        return self

    def predict(
        self,
        tokens: HybridGraphTokens,
        *,
        allow_topology_ablation: bool = False,
    ) -> HybridJepaPrediction:
        """Predict in bounded minibatches and return CPU NumPy evidence."""

        torch = _require_torch()
        self._ensure_compatible(
            tokens,
            allow_topology_ablation=allow_topology_ablation,
        )
        self._network.eval()
        predicted_parts = []
        target_parts = []
        reconstruction_parts = []
        context_reconstruction_parts = []
        embedding_parts = []
        with torch.no_grad():
            for start in range(
                0, len(tokens.fine_context), self.config.batch_size
            ):
                selection = np.arange(
                    start,
                    min(
                        start + self.config.batch_size,
                        len(tokens.fine_context),
                    ),
                    dtype=np.int64,
                )
                batch = _batch_to_torch(
                    torch,
                    tokens,
                    selection,
                    self.device,
                )
                output = self._network.forward_prediction(batch)
                predicted_parts.append(
                    output["predicted"].detach().cpu().numpy()
                )
                target_parts.append(
                    output["target"].detach().cpu().numpy()
                )
                reconstruction_parts.append(
                    output["reconstructed"].detach().cpu().numpy()
                )
                context_reconstruction_parts.append(
                    output["reconstructed_context"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                embedding_parts.append(
                    output["embedding"].detach().cpu().numpy()
                )
        predicted = np.concatenate(predicted_parts).astype(
            np.float64, copy=False
        )
        target = np.concatenate(target_parts).astype(
            np.float64, copy=False
        )
        reconstructed = np.concatenate(reconstruction_parts).astype(
            np.float64, copy=False
        )
        reconstructed_contexts = np.concatenate(
            context_reconstruction_parts
        ).astype(np.float64, copy=False)
        embeddings = np.concatenate(embedding_parts).astype(
            np.float64, copy=False
        )
        diagnostics = latent_diagnostics(embeddings)
        future_diagnostics = latent_diagnostics(
            predicted.reshape(-1, predicted.shape[-1])
        )
        diagnostics["future_effective_rank"] = future_diagnostics[
            "effective_rank"
        ]
        diagnostics["future_per_dimension_variance"] = (
            future_diagnostics["per_dimension_variance"]
        )
        return HybridJepaPrediction(
            predicted_latents=predicted,
            target_latents=target,
            reconstructed_targets=reconstructed,
            reconstructed_contexts=reconstructed_contexts,
            validation_embeddings=embeddings,
            diagnostics=diagnostics,
        )

    def decode_context_embeddings(
        self,
        embeddings: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Decode local histories exclusively from exported probe embeddings."""

        torch = _require_torch()
        if self._network is None or self._schema is None:
            raise RuntimeError("hybrid JEPA must be fitted before decoding")
        values = np.asarray(embeddings, dtype=np.float64)
        expected_width = (
            self._schema["entity_count"]
            * self.config.latent_dimension
        )
        if (
            values.ndim != 2
            or values.shape[1] != expected_width
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("context embeddings do not match model")
        parts: List[NDArray[np.float64]] = []
        self._network.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                batch = torch.as_tensor(
                    values[start : start + self.config.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(
                    -1,
                    self._schema["entity_count"],
                    self.config.latent_dimension,
                )
                parts.append(
                    np.asarray(
                        self._network.decode_context_base(batch)
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float64,
                    )
                )
        return np.concatenate(parts).astype(np.float64, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration, learning evidence, and CPU parameters."""

        if (
            self._network is None
            or self._schema is None
            or self._semantic_schema is None
            or self._training_topology is None
        ):
            raise RuntimeError("hybrid JEPA must be fitted before serialization")
        state = {
            name: value.detach().cpu().tolist()
            for name, value in self._network.state_dict().items()
        }
        return {
            "kind": self.kind,
            "config": self.config.to_dict(),
            "schema": dict(self._schema),
            "semantic_schema": copy.deepcopy(self._semantic_schema),
            "training_topology": self._training_topology.tolist(),
            "training_losses": list(self.training_losses),
            "epoch_metrics": [
                dict(values) for values in self.epoch_metrics
            ],
            "state": state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HybridGraphJepa":
        """Restore a fitted model from a JSON-compatible mapping."""

        if payload.get("kind") != cls.kind:
            raise ValueError("unsupported hybrid JEPA serialization")
        torch = _require_torch()
        config_payload = payload["config"]
        schema_payload = payload["schema"]
        if not isinstance(config_payload, Mapping) or not isinstance(
            schema_payload, Mapping
        ):
            raise ValueError("hybrid JEPA serialization is malformed")
        model = cls(HybridJepaConfig.from_dict(config_payload))
        model.device = _select_device(torch, model.config.device)
        model._schema = {
            str(key): int(value)
            for key, value in schema_payload.items()
        }
        semantic_payload = payload.get("semantic_schema")
        if not isinstance(semantic_payload, Mapping):
            raise ValueError("hybrid JEPA semantic schema is malformed")
        model._semantic_schema = {
            str(key): copy.deepcopy(value)
            for key, value in semantic_payload.items()
        }
        raw_topology = np.asarray(payload.get("training_topology"))
        expected_topology_shape = (
            model._schema["relation_count"],
            model._schema["entity_count"],
            model._schema["entity_count"],
        )
        if (
            raw_topology.shape != expected_topology_shape
            or not np.issubdtype(raw_topology.dtype, np.bool_)
        ):
            raise ValueError("hybrid JEPA training topology is malformed")
        model._training_topology = raw_topology.astype(
            np.bool_, copy=True
        )
        model._network = _build_network(
            torch, model.config, model._schema
        )
        state_payload = payload["state"]
        if not isinstance(state_payload, Mapping):
            raise ValueError("hybrid JEPA state is malformed")
        expected = model._network.state_dict()
        state = {
            name: torch.as_tensor(
                state_payload[name],
                dtype=tensor.dtype,
            )
            for name, tensor in expected.items()
        }
        model._network.load_state_dict(state)
        model._network.to(model.device)
        model.training_losses = tuple(
            float(value) for value in payload["training_losses"]
        )
        raw_metrics = payload.get("epoch_metrics", ())
        if not isinstance(raw_metrics, (list, tuple)):
            raise ValueError("hybrid JEPA epoch metrics are malformed")
        model.epoch_metrics = tuple(
            {
                str(key): float(value)
                for key, value in item.items()
            }
            for item in raw_metrics
            if isinstance(item, Mapping)
        )
        return model

    def _ensure_compatible(
        self,
        tokens: HybridGraphTokens,
        *,
        allow_topology_ablation: bool,
    ) -> None:
        if (
            self._network is None
            or self._schema is None
            or self._semantic_schema is None
            or self._training_topology is None
        ):
            raise RuntimeError("hybrid JEPA must be fitted before prediction")
        if _schema(tokens) != self._schema:
            raise ValueError("hybrid graph token schema does not match model")
        if _semantic_schema(tokens) != self._semantic_schema:
            raise ValueError(
                "hybrid graph token semantic schema does not match model"
            )
        if (
            not allow_topology_ablation
            and not np.array_equal(
                tokens.typed_adjacency,
                self._training_topology,
            )
        ):
            raise ValueError(
                "runtime topology differs from training topology; "
                "set allow_topology_ablation=True for an explicit ablation"
            )


def latent_diagnostics(
    embeddings: NDArray[np.float64],
) -> Dict[str, Any]:
    """Measure rank, per-dimension spread, and covariance deterministically."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("latent diagnostics require a rank-2 matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("latent diagnostics require finite values")
    centered = values - np.mean(values, axis=0, keepdims=True)
    variances = np.mean(centered * centered, axis=0)
    if len(values) < 2 or not np.any(variances > 1e-12):
        effective_rank = 0.0
        covariance = np.zeros(
            (values.shape[1], values.shape[1]), dtype=np.float64
        )
    else:
        covariance = centered.T @ centered / float(len(values) - 1)
        eigenvalues = np.maximum(
            np.linalg.eigvalsh(covariance), 0.0
        )
        total = float(np.sum(eigenvalues))
        if total <= 1e-12:
            effective_rank = 0.0
        else:
            probabilities = eigenvalues[eigenvalues > 1e-12] / total
            effective_rank = float(
                np.exp(-np.sum(probabilities * np.log(probabilities)))
            )
    off_diagonal = covariance - np.diag(np.diag(covariance))
    return {
        "effective_rank": effective_rank,
        "per_dimension_variance": variances.tolist(),
        "minimum_dimension_variance": float(np.min(variances)),
        "mean_dimension_variance": float(np.mean(variances)),
        "mean_absolute_off_diagonal_covariance": float(
            np.mean(np.abs(off_diagonal))
        ),
    }


def pool_context_mask(
    fine_mask: NDArray[np.bool_],
    coarse_factor: int,
) -> NDArray[np.bool_]:
    """Pool visibility without leaking any partially masked fine token."""

    values = np.asarray(fine_mask, dtype=np.bool_)
    if values.ndim != 3:
        raise ValueError("fine context mask must have shape [sample,time,entity]")
    if isinstance(coarse_factor, bool) or coarse_factor < 1:
        raise ValueError("coarse factor must be positive")
    chunks = [
        np.all(values[:, start : start + coarse_factor], axis=1)
        for start in range(0, values.shape[1], coarse_factor)
    ]
    return np.stack(chunks, axis=1)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError(
            "HybridGraphJepa requires the optional PyTorch training extra"
        ) from error


def _select_device(torch: Any, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if requested == "mps" and not available:
        raise RuntimeError("MPS was requested but is unavailable")
    return "mps" if available else "cpu"


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _schema(tokens: HybridGraphTokens) -> Dict[str, int]:
    return {
        "feature_count": int(tokens.fine_context.shape[-1]),
        "entity_count": int(tokens.fine_context.shape[-2]),
        "kind_count": int(np.max(tokens.kind_ids)) + 1,
        "relation_count": len(tokens.relation_names),
        "lookback": int(tokens.fine_context.shape[1]),
        "coarse_lookback": int(tokens.coarse_context.shape[1]),
        "horizon_count": int(tokens.fine_targets.shape[1]),
        "target_block_size": int(tokens.fine_targets.shape[2]),
        "control_count": int(tokens.target_controls.shape[-1]),
        "coarse_factor": int(tokens.coarse_factor),
    }


def _semantic_schema(tokens: HybridGraphTokens) -> Dict[str, Any]:
    return {
        "feature_names": list(tokens.feature_names),
        "feature_mask": tokens.feature_mask.tolist(),
        "entity_names": list(tokens.entity_names),
        "entity_ids": tokens.entity_ids.tolist(),
        "kind_names": list(tokens.kind_names),
        "kind_ids": tokens.kind_ids.tolist(),
        "entity_type_names": list(tokens.entity_type_names),
        "entity_type_ids": tokens.entity_type_ids.tolist(),
        "relation_names": list(tokens.relation_names),
        "relation_ids": tokens.relation_ids.tolist(),
        "local_feature_keys": [
            list(values) for values in tokens.local_feature_keys
        ],
        "horizons": list(tokens.horizons),
        "control_feature_names": list(tokens.control_feature_names),
    }


def _validate_masks(
    tokens: HybridGraphTokens,
    masks: MultiMaskBatch,
    mask_count: int,
) -> None:
    expected = (
        mask_count,
        tokens.fine_context.shape[0],
        tokens.fine_context.shape[1],
        tokens.fine_context.shape[2],
    )
    if (
        masks.context_masks.shape != expected
        or masks.target_masks.shape != expected
    ):
        raise ValueError(
            "multi-mask tensors do not align with hybrid graph tokens"
        )
    if not np.all(np.any(masks.target_masks, axis=(2, 3))):
        raise ValueError("each hybrid JEPA mask must hold out a target")


def _build_network(
    torch: Any,
    config: HybridJepaConfig,
    schema: Mapping[str, int],
) -> Any:
    nn = torch.nn
    latent = config.latent_dimension
    max_time = max(
        schema["lookback"],
        schema["coarse_lookback"],
        schema["horizon_count"],
    )

    class RelationalTemporalEncoder(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(
                schema["feature_count"], latent
            )
            self.entity_embedding = nn.Embedding(
                schema["entity_count"], latent
            )
            self.kind_embedding = nn.Embedding(
                schema["kind_count"], latent
            )
            self.time_embedding = nn.Embedding(max_time, latent)
            self.relation_projections = nn.ModuleList(
                [
                    nn.Linear(latent, latent, bias=False)
                    for _ in range(schema["relation_count"])
                ]
            )
            layer = nn.TransformerEncoderLayer(
                d_model=latent,
                nhead=config.attention_heads,
                dim_feedforward=latent
                * config.feedforward_multiplier,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                layer,
                num_layers=config.transformer_layers,
                enable_nested_tensor=False,
            )
            self.output_norm = nn.LayerNorm(latent)

        def forward(
            self,
            values: Any,
            entity_ids: Any,
            kind_ids: Any,
            adjacency: Any,
        ) -> Any:
            batch, time, entities, _ = values.shape
            time_ids = torch.arange(
                time, device=values.device, dtype=torch.long
            )
            result = (
                self.input_projection(values)
                + self.entity_embedding(entity_ids)[None, None, :, :]
                + self.kind_embedding(kind_ids)[None, None, :, :]
                + self.time_embedding(time_ids)[None, :, None, :]
            )
            graph = torch.zeros_like(result)
            for relation, projection in enumerate(
                self.relation_projections
            ):
                projected = projection(result)
                relation_adjacency = adjacency[relation].to(
                    dtype=result.dtype
                )
                degree = relation_adjacency.sum(dim=0).clamp(min=1.0)
                graph = graph + torch.einsum(
                    "ij,btid->btjd",
                    relation_adjacency,
                    projected,
                ) / degree[None, None, :, None]
            result = result + graph
            temporal = result.permute(0, 2, 1, 3).reshape(
                batch * entities, time, latent
            )
            temporal = self.temporal(temporal)
            return self.output_norm(
                temporal.reshape(
                    batch, entities, time, latent
                ).permute(0, 2, 1, 3)
            )

    class Network(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.online = RelationalTemporalEncoder()
            self.target = copy.deepcopy(self.online)
            for parameter in self.target.parameters():
                parameter.requires_grad_(False)
            self.horizon_embedding = nn.Embedding(
                schema["horizon_count"], latent
            )
            self.control_projection = nn.Linear(
                schema["control_count"], latent, bias=False
            )
            self.predictor = nn.Sequential(
                nn.Linear(latent, latent * 2),
                nn.GELU(),
                nn.Linear(latent * 2, latent),
            )
            self.current_predictor = nn.Sequential(
                nn.Linear(latent, latent * 2),
                nn.GELU(),
                nn.Linear(latent * 2, latent),
            )
            self.reconstruction = nn.Linear(
                latent,
                schema["target_block_size"]
                * schema["feature_count"],
            )
            self.context_reconstruction = nn.Linear(
                latent,
                schema["lookback"] * schema["feature_count"],
            )

        def trainable_parameters(self) -> Any:
            return (
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        def _represent_context(
            self, batch: Mapping[str, Any], visible: Any
        ) -> Tuple[Any, Any, Any]:
            masked = batch["fine_context"] * visible[..., None]
            fine = self.online(
                masked,
                batch["entity_ids"],
                batch["kind_ids"],
                batch["adjacency"],
            )
            coarse_visible = torch.stack(
                [
                    visible[
                        :,
                        start : start + schema["coarse_factor"],
                    ].all(dim=1)
                    for start in range(
                        0,
                        schema["lookback"],
                        schema["coarse_factor"],
                    )
                ],
                dim=1,
            )
            coarse = self.online(
                batch["coarse_context"] * coarse_visible[..., None],
                batch["entity_ids"],
                batch["kind_ids"],
                batch["adjacency"],
            )
            base = 0.5 * (fine[:, -1] + coarse[:, -1])
            return base, base.flatten(start_dim=1), fine

        def _future_target(self, batch: Mapping[str, Any]) -> Any:
            fine_values = batch["fine_targets"].mean(dim=2)
            coarse_values = batch["coarse_targets"].mean(dim=2)
            fine = self.target(
                fine_values,
                batch["entity_ids"],
                batch["kind_ids"],
                batch["adjacency"],
            )
            coarse = self.target(
                coarse_values,
                batch["entity_ids"],
                batch["kind_ids"],
                batch["adjacency"],
            )
            return 0.5 * (fine + coarse)

        def _current_target(self, batch: Mapping[str, Any]) -> Any:
            return self.target(
                batch["fine_context"],
                batch["entity_ids"],
                batch["kind_ids"],
                batch["adjacency"],
            )

        def _predict_from_base(
            self, base: Any, target_controls: Any
        ) -> Any:
            horizon_ids = torch.arange(
                schema["horizon_count"],
                dtype=torch.long,
                device=base.device,
            )
            queries = (
                base[:, None]
                + self.horizon_embedding(horizon_ids)[None, :, None]
                + self.control_projection(
                    target_controls.mean(dim=2)
                )[:, :, None, :]
            )
            return self.predictor(queries)

        def _decode(self, predicted: Any) -> Any:
            batch = predicted.shape[0]
            decoded = self.reconstruction(predicted)
            return decoded.reshape(
                batch,
                schema["horizon_count"],
                schema["entity_count"],
                schema["target_block_size"],
                schema["feature_count"],
            ).permute(0, 1, 3, 2, 4)

        def decode_context_base(self, base: Any) -> Any:
            batch = base.shape[0]
            decoded = self.context_reconstruction(base)
            return decoded.reshape(
                batch,
                schema["entity_count"],
                schema["lookback"],
                schema["feature_count"],
            ).permute(0, 2, 1, 3)

        def forward_training(
            self,
            batch: Mapping[str, Any],
            visible: Any,
            target_mask: Any,
        ) -> Mapping[str, Any]:
            base, embedding, fine = self._represent_context(
                batch, visible
            )
            predicted = self._predict_from_base(
                base, batch["target_controls"]
            )
            current_predicted = self.current_predictor(fine)
            with torch.no_grad():
                target = self._future_target(batch)
                current_target = self._current_target(batch)
            return {
                "predicted": predicted,
                "target": target,
                "current_predicted": current_predicted,
                "current_target": current_target,
                "target_mask": target_mask,
                "reconstructed": self._decode(predicted),
                "reconstructed_context": self.decode_context_base(base),
                "base": base,
                "embedding": embedding,
            }

        def forward_prediction(
            self, batch: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            visible = torch.ones(
                batch["fine_context"].shape[:-1],
                dtype=torch.bool,
                device=batch["fine_context"].device,
            )
            target_mask = torch.zeros_like(visible)
            return self.forward_training(
                batch, visible, target_mask
            )

        def update_target(self, decay: float) -> None:
            with torch.no_grad():
                for target, online in zip(
                    self.target.parameters(),
                    self.online.parameters(),
                ):
                    target.mul_(decay).add_(
                        online, alpha=1.0 - decay
                    )

    return Network()


def _batch_to_torch(
    torch: Any,
    tokens: HybridGraphTokens,
    selection: NDArray[np.int64],
    device: str,
) -> Mapping[str, Any]:
    def values(array: NDArray[Any]) -> Any:
        return torch.as_tensor(
            array[selection],
            dtype=torch.float32,
            device=device,
        )

    return {
        "fine_context": values(tokens.fine_context),
        "fine_targets": values(tokens.fine_targets),
        "coarse_context": values(tokens.coarse_context),
        "coarse_targets": values(tokens.coarse_targets),
        "target_controls": values(tokens.target_controls),
        "feature_mask": torch.as_tensor(
            tokens.feature_mask,
            dtype=torch.bool,
            device=device,
        ),
        "entity_ids": torch.as_tensor(
            tokens.entity_ids,
            dtype=torch.long,
            device=device,
        ),
        "kind_ids": torch.as_tensor(
            tokens.kind_ids,
            dtype=torch.long,
            device=device,
        ),
        "adjacency": torch.as_tensor(
            tokens.typed_adjacency,
            dtype=torch.bool,
            device=device,
        ),
    }


def _loss_components(
    torch: Any,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: HybridJepaConfig,
) -> Mapping[str, Any]:
    predicted = output["predicted"]
    future_latent = torch.nn.functional.l1_loss(
        predicted, output["target"]
    )
    target_mask = output["target_mask"][..., None].to(
        dtype=predicted.dtype
    )
    masked_latent_error = (
        output["current_predicted"] - output["current_target"]
    ).abs() * target_mask
    masked_latent = masked_latent_error.sum() / (
        target_mask.sum() * float(predicted.shape[-1])
    ).clamp(min=1.0)
    latent = future_latent + masked_latent
    feature_mask = batch["feature_mask"][
        None, None, None, :, :
    ].to(dtype=predicted.dtype)
    squared = (
        output["reconstructed"] - batch["fine_targets"]
    ).square() * feature_mask
    denominator = feature_mask.sum() * float(
        predicted.shape[0]
        * predicted.shape[1]
        * batch["fine_targets"].shape[2]
    )
    reconstruction = squared.sum() / denominator.clamp(min=1.0)
    context_feature_mask = batch["feature_mask"][
        None, None, :, :
    ].to(dtype=predicted.dtype)
    context_selection = target_mask * context_feature_mask
    context_squared = (
        output["reconstructed_context"] - batch["fine_context"]
    ).square() * context_selection
    context_reconstruction = context_squared.sum() / (
        context_selection.sum().clamp(min=1.0)
    )
    base = output["base"]
    observed_entities = batch["feature_mask"].any(dim=1)
    if len(base) < 2 or not bool(observed_entities.any()):
        variance = torch.ones((), device=predicted.device)
        covariance = torch.zeros((), device=predicted.device)
    else:
        centered = base - base.mean(dim=0, keepdim=True)
        standard_deviation = torch.sqrt(
            centered.square().mean(dim=0) + 1e-4
        )
        per_entity_variance = torch.relu(
            1.0 - standard_deviation
        ).mean(dim=1)
        variance = per_entity_variance[observed_entities].mean()
        covariance_matrix = torch.einsum(
            "bed,bef->edf", centered, centered
        ) / float(len(base) - 1)
        diagonal = torch.diagonal(
            covariance_matrix, dim1=1, dim2=2
        )
        off_diagonal = covariance_matrix - torch.diag_embed(diagonal)
        per_entity_covariance = off_diagonal.square().sum(
            dim=(1, 2)
        ) / float(
            max(base.shape[-1], 1)
        )
        covariance = per_entity_covariance[observed_entities].mean()
    total = (
        latent
        + config.reconstruction_weight * reconstruction
        + config.context_reconstruction_weight
        * context_reconstruction
        + config.variance_weight * variance
        + config.covariance_weight * covariance
    )
    return {
        "total": total,
        "latent": latent,
        "reconstruction": reconstruction,
        "context_reconstruction": context_reconstruction,
        "variance": variance,
        "covariance": covariance,
    }
