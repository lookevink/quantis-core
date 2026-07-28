"""Inspectable linear joint-embedding prediction over a telemetry graph."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphStateWindows,
)


CONTEXT_SCOPES = (
    "entity_local",
    "one_hop",
    "all_entities",
)
REQUIRED_MODEL_SCOPES = frozenset(CONTEXT_SCOPES)


@dataclass(frozen=True)
class GraphJepaTrainingConfig:
    """Small-data configuration for the linear graph-JEPA tracer."""

    latent_dimension: int = 2
    ridge: float = 1e-3
    context_scope: str = "one_hop"
    entity_latent_dimensions: Optional[Mapping[str, int]] = None

    def __post_init__(self) -> None:
        if self.latent_dimension < 1:
            raise ValueError(
                "graph JEPA latent dimension must be positive"
            )
        if self.ridge <= 0.0:
            raise ValueError("graph JEPA ridge must be positive")
        if self.context_scope not in CONTEXT_SCOPES:
            raise ValueError("unsupported graph JEPA context scope")
        if self.entity_latent_dimensions is not None:
            for entity_id, width in (
                self.entity_latent_dimensions.items()
            ):
                if (
                    not entity_id
                    or isinstance(width, bool)
                    or width < 1
                    or width > self.latent_dimension
                ):
                    raise ValueError(
                        "graph JEPA entity widths must be positive "
                        "and no larger than latent_dimension"
                    )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "latent_dimension": self.latent_dimension,
            "ridge": self.ridge,
            "context_scope": self.context_scope,
        }
        if self.entity_latent_dimensions is not None:
            payload["entity_latent_dimensions"] = {
                entity_id: int(width)
                for entity_id, width in sorted(
                    self.entity_latent_dimensions.items()
                )
            }
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "GraphJepaTrainingConfig":
        return cls(
            latent_dimension=int(payload["latent_dimension"]),
            ridge=float(payload["ridge"]),
            context_scope=str(payload["context_scope"]),
            entity_latent_dimensions=(
                {
                    str(entity_id): int(width)
                    for entity_id, width in dict(
                        payload["entity_latent_dimensions"]
                    ).items()
                }
                if payload.get("entity_latent_dimensions")
                is not None
                else None
            ),
        )

    def latent_dimension_for(self, entity_id: str) -> int:
        if self.entity_latent_dimensions is None:
            return self.latent_dimension
        return int(
            self.entity_latent_dimensions.get(
                entity_id, self.latent_dimension
            )
        )


@dataclass(frozen=True)
class GraphJepaPrediction:
    """Predicted and target entity tokens plus decoded raw blocks."""

    predicted_tokens: NDArray[np.float64]
    target_tokens: NDArray[np.float64]
    decoded_target_blocks: NDArray[np.float64]
    reconstructed_target_blocks: NDArray[np.float64]


@dataclass(frozen=True)
class _EntityEncoder:
    location: NDArray[np.float64]
    components: NDArray[np.float64]


@dataclass(frozen=True)
class _EntityPredictor:
    context_entity_ids: Tuple[str, ...]
    location: NDArray[np.float64]
    scale: NDArray[np.float64]
    weights: NDArray[np.float64]


class LinearGraphJepaWorldModel:
    """Frozen PCA entity encoders and graph-conditioned latent predictors."""

    def __init__(self, config: GraphJepaTrainingConfig) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._entity_ids: Tuple[str, ...] = ()
        self._entity_kinds: Tuple[str, ...] = ()
        self._local_feature_keys: Tuple[Tuple[str, ...], ...] = ()
        self._control_feature_names: Tuple[str, ...] = ()
        self._horizons: Tuple[int, ...] = ()
        self._target_block_size = 0
        self._lookback = 0
        self._observation_mask = np.zeros(
            (0, 0), dtype=np.bool_
        )
        self._encoders: Dict[str, _EntityEncoder] = {}
        self._predictors: Dict[str, _EntityPredictor] = {}

    def fit(
        self, windows: GraphStateWindows
    ) -> "LinearGraphJepaWorldModel":
        """Fit training-only entity PCA encoders and latent predictors."""

        lookback = windows.contexts.shape[1]
        block_size = windows.target_block_size
        if lookback % block_size != 0:
            raise ValueError(
                "graph JEPA lookback must divide into target-sized patches"
            )
        self._graph = windows.graph
        self._entity_ids = windows.entity_ids
        self._entity_kinds = windows.entity_kinds
        self._local_feature_keys = windows.local_feature_keys
        self._control_feature_names = windows.control_feature_names
        self._horizons = windows.horizons
        self._target_block_size = block_size
        self._lookback = lookback
        self._observation_mask = windows.observation_mask.copy()
        if self.config.entity_latent_dimensions is not None:
            unknown_widths = (
                set(self.config.entity_latent_dimensions)
                - set(windows.entity_ids)
            )
            if unknown_widths:
                raise ValueError(
                    "graph JEPA entity widths reference unknown "
                    f"entities: {sorted(unknown_widths)}"
                )
        self._encoders = {}
        self._predictors = {}

        for entity_position, entity_id in enumerate(
            windows.entity_ids
        ):
            mask = windows.observation_mask[entity_position]
            if not np.any(mask):
                continue
            context_blocks = _context_blocks(
                windows, entity_position
            )
            target_blocks = _target_blocks(
                windows, entity_position
            )
            encoder_rows = np.concatenate(
                (
                    context_blocks.reshape(
                        -1, context_blocks.shape[-1]
                    ),
                    target_blocks.reshape(
                        -1, target_blocks.shape[-1]
                    ),
                ),
                axis=0,
            )
            location = np.mean(encoder_rows, axis=0)
            centered = encoder_rows - location
            _, _, right = np.linalg.svd(
                centered, full_matrices=False
            )
            components = np.zeros(
                (
                    encoder_rows.shape[1],
                    self.config.latent_dimension,
                ),
                dtype=np.float64,
            )
            retained = min(
                self.config.latent_dimension_for(entity_id),
                right.shape[0],
                right.shape[1],
            )
            components[:, :retained] = right[:retained].T
            self._encoders[entity_id] = _EntityEncoder(
                location=np.asarray(location, dtype=np.float64),
                components=components,
            )

        context_tokens = self._encode_context(windows)
        for entity_position, entity_id in enumerate(
            windows.entity_ids
        ):
            encoder = self._encoders.get(entity_id)
            if encoder is None:
                continue
            context_entity_ids = self._context_entity_ids(entity_id)
            design = _prediction_design(
                context_tokens,
                windows,
                context_entity_ids,
            )
            target = _encode(
                _target_blocks(
                    windows, entity_position
                ).reshape(
                    len(windows.contexts) * len(windows.horizons),
                    -1,
                ),
                encoder,
            )
            location, scale, weights = _fit_ridge(
                design,
                target,
                self.config.ridge,
            )
            self._predictors[entity_id] = _EntityPredictor(
                context_entity_ids=context_entity_ids,
                location=location,
                scale=scale,
                weights=weights,
            )
        return self

    def predict(
        self, windows: GraphStateWindows
    ) -> GraphJepaPrediction:
        """Predict future tokens without fitting on validation targets."""

        self._validate_schema(windows)
        sample_count = len(windows.contexts)
        horizon_count = len(windows.horizons)
        entity_count = len(windows.entity_ids)
        slot_count = windows.contexts.shape[3]
        latent_dimension = self.config.latent_dimension
        context_tokens = self._encode_context(windows)
        predicted = np.zeros(
            (
                sample_count,
                horizon_count,
                entity_count,
                latent_dimension,
            ),
            dtype=np.float64,
        )
        targets = np.zeros_like(predicted)
        decoded = np.zeros_like(windows.target_blocks)
        reconstructed = np.zeros_like(windows.target_blocks)
        for entity_position, entity_id in enumerate(
            windows.entity_ids
        ):
            encoder = self._encoders.get(entity_id)
            predictor = self._predictors.get(entity_id)
            if encoder is None or predictor is None:
                continue
            raw_targets = _target_blocks(
                windows, entity_position
            )
            target_tokens = _encode(
                raw_targets.reshape(
                    sample_count * horizon_count, -1
                ),
                encoder,
            ).reshape(
                sample_count,
                horizon_count,
                latent_dimension,
            )
            design = _prediction_design(
                context_tokens,
                windows,
                predictor.context_entity_ids,
            )
            prediction = _apply_ridge(
                design,
                predictor.location,
                predictor.scale,
                predictor.weights,
            ).reshape(
                sample_count,
                horizon_count,
                latent_dimension,
            )
            predicted[:, :, entity_position, :] = prediction
            targets[:, :, entity_position, :] = target_tokens
            mask = windows.observation_mask[entity_position]
            feature_count = int(np.count_nonzero(mask))
            decoded_values = _decode(
                prediction.reshape(-1, latent_dimension),
                encoder,
            ).reshape(
                sample_count,
                horizon_count,
                windows.target_block_size,
                feature_count,
            )
            reconstructed_values = _decode(
                target_tokens.reshape(-1, latent_dimension),
                encoder,
            ).reshape(
                sample_count,
                horizon_count,
                windows.target_block_size,
                feature_count,
            )
            decoded[
                :, :, :, entity_position, mask
            ] = decoded_values
            reconstructed[
                :, :, :, entity_position, mask
            ] = reconstructed_values
        if decoded.shape[-1] != slot_count:
            raise AssertionError("decoded graph slots changed")
        return GraphJepaPrediction(
            predicted_tokens=predicted,
            target_tokens=targets,
            decoded_target_blocks=decoded,
            reconstructed_target_blocks=reconstructed,
        )

    def to_dict(self) -> Dict[str, Any]:
        self._require_fitted()
        assert self._graph is not None
        return {
            "schema_version": 1,
            "kind": "linear_graph_jepa_world_model",
            "config": self.config.to_dict(),
            "graph": self._graph.to_dict(),
            "schema": {
                "entity_ids": list(self._entity_ids),
                "entity_kinds": list(self._entity_kinds),
                "local_feature_keys": [
                    list(values)
                    for values in self._local_feature_keys
                ],
                "control_feature_names": list(
                    self._control_feature_names
                ),
                "horizons": list(self._horizons),
                "target_block_size": self._target_block_size,
                "lookback": self._lookback,
                "observation_mask": (
                    self._observation_mask.astype(int).tolist()
                ),
            },
            "encoders": {
                entity_id: {
                    "location": encoder.location.tolist(),
                    "components": encoder.components.tolist(),
                }
                for entity_id, encoder in self._encoders.items()
            },
            "predictors": {
                entity_id: {
                    "context_entity_ids": list(
                        predictor.context_entity_ids
                    ),
                    "location": predictor.location.tolist(),
                    "scale": predictor.scale.tolist(),
                    "weights": predictor.weights.tolist(),
                }
                for entity_id, predictor in self._predictors.items()
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LinearGraphJepaWorldModel":
        if (
            payload.get("schema_version") != 1
            or payload.get("kind")
            != "linear_graph_jepa_world_model"
        ):
            raise ValueError("unsupported linear graph JEPA model")
        model = cls(
            GraphJepaTrainingConfig.from_dict(
                dict(payload["config"])
            )
        )
        schema = dict(payload["schema"])
        model._graph = DeclaredTelemetryGraph.from_dict(
            dict(payload["graph"])
        )
        model._entity_ids = tuple(
            str(value) for value in schema["entity_ids"]
        )
        model._entity_kinds = tuple(
            str(value) for value in schema["entity_kinds"]
        )
        model._local_feature_keys = tuple(
            tuple(str(value) for value in values)
            for values in schema["local_feature_keys"]
        )
        model._control_feature_names = tuple(
            str(value)
            for value in schema["control_feature_names"]
        )
        model._horizons = tuple(
            int(value) for value in schema["horizons"]
        )
        model._target_block_size = int(
            schema["target_block_size"]
        )
        model._lookback = int(schema["lookback"])
        model._observation_mask = np.asarray(
            schema["observation_mask"], dtype=np.bool_
        )
        model._encoders = {
            str(entity_id): _EntityEncoder(
                location=np.asarray(
                    dict(raw)["location"], dtype=np.float64
                ),
                components=np.asarray(
                    dict(raw)["components"], dtype=np.float64
                ),
            )
            for entity_id, raw in dict(payload["encoders"]).items()
        }
        model._predictors = {
            str(entity_id): _EntityPredictor(
                context_entity_ids=tuple(
                    str(value)
                    for value in dict(raw)["context_entity_ids"]
                ),
                location=np.asarray(
                    dict(raw)["location"], dtype=np.float64
                ),
                scale=np.asarray(
                    dict(raw)["scale"], dtype=np.float64
                ),
                weights=np.asarray(
                    dict(raw)["weights"], dtype=np.float64
                ),
            )
            for entity_id, raw in dict(payload["predictors"]).items()
        }
        model._require_fitted()
        return model

    def _encode_context(
        self, windows: GraphStateWindows
    ) -> NDArray[np.float64]:
        patch_count = (
            windows.contexts.shape[1]
            // windows.target_block_size
        )
        tokens = np.zeros(
            (
                len(windows.contexts),
                patch_count,
                len(windows.entity_ids),
                self.config.latent_dimension,
            ),
            dtype=np.float64,
        )
        for entity_position, entity_id in enumerate(
            windows.entity_ids
        ):
            encoder = self._encoders.get(entity_id)
            if encoder is None:
                continue
            blocks = _context_blocks(windows, entity_position)
            tokens[:, :, entity_position, :] = _encode(
                blocks.reshape(-1, blocks.shape[-1]),
                encoder,
            ).reshape(
                len(windows.contexts),
                patch_count,
                self.config.latent_dimension,
            )
        return tokens

    def _context_entity_ids(
        self, entity_id: str
    ) -> Tuple[str, ...]:
        assert self._graph is not None
        candidates: Tuple[str, ...]
        if self.config.context_scope == "entity_local":
            candidates = (entity_id,)
        elif self.config.context_scope == "one_hop":
            candidates = self._graph.neighboring_entity_ids(
                entity_id
            )
        else:
            candidates = self._entity_ids
        return tuple(
            candidate
            for candidate in candidates
            if candidate in self._encoders
        )

    def _validate_schema(
        self, windows: GraphStateWindows
    ) -> None:
        self._require_fitted()
        if (
            windows.entity_ids != self._entity_ids
            or windows.entity_kinds != self._entity_kinds
            or windows.local_feature_keys
            != self._local_feature_keys
            or windows.control_feature_names
            != self._control_feature_names
            or windows.horizons != self._horizons
            or windows.target_block_size
            != self._target_block_size
            or windows.contexts.shape[1] != self._lookback
            or not np.array_equal(
                windows.observation_mask,
                self._observation_mask,
            )
            or windows.graph != self._graph
        ):
            raise ValueError(
                "graph JEPA input schema differs from fitted model"
            )

    def _require_fitted(self) -> None:
        if (
            self._graph is None
            or not self._entity_ids
            or not self._encoders
            or set(self._encoders) != set(self._predictors)
        ):
            raise RuntimeError("graph JEPA model is not fitted")


def evaluate_linear_graph_jepa(
    models: Mapping[str, LinearGraphJepaWorldModel],
    training: GraphStateWindows,
    validation: GraphStateWindows,
    *,
    validation_window_case_ids: Sequence[str],
    maximum_graph_to_all_error_ratio: float = 1.05,
    maximum_reconstruction_normalized_mse: float = 0.1,
) -> Mapping[str, Any]:
    """Evaluate localized latent prediction and equal-encoder controls."""

    if set(models) != REQUIRED_MODEL_SCOPES:
        raise ValueError(
            "graph JEPA evaluation requires all context scopes"
        )
    if len(validation_window_case_ids) != len(
        validation.contexts
    ):
        raise ValueError(
            "graph JEPA validation case identities must align"
        )
    if maximum_graph_to_all_error_ratio < 1.0:
        raise ValueError(
            "graph-to-all error ratio must be at least one"
        )
    if not (
        0.0 < maximum_reconstruction_normalized_mse < 1.0
    ):
        raise ValueError(
            "graph JEPA reconstruction gate must be between zero and one"
        )
    predictions = {
        scope: model.predict(validation)
        for scope, model in models.items()
    }
    training_prediction = models["one_hop"].predict(training)
    families = tuple(
        _family_id(case_id)
        for case_id in validation_window_case_ids
    )
    repeated_families = np.repeat(
        np.asarray(families, dtype=object),
        len(validation.horizons)
        * validation.target_block_size,
    )
    representation_errors: Dict[str, list[float]] = {
        "training_mean": [],
        "pca_target_reconstruction": [],
        **{scope: [] for scope in CONTEXT_SCOPES},
    }
    family_errors: Dict[str, Dict[str, list[float]]] = {
        name: {} for name in representation_errors
    }
    target_results: Dict[str, Dict[str, Any]] = {
        name: {} for name in representation_errors
    }
    for entity_position, feature_keys in enumerate(
        training.local_feature_keys
    ):
        for slot_position, feature_key in enumerate(feature_keys):
            training_target = training.target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            validation_target = validation.target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            variance = float(np.var(training_target))
            if variance <= 1e-12:
                for results in target_results.values():
                    results[feature_key] = {
                        "status": "insufficient_training_variation"
                    }
                continue
            candidate_values = {
                "training_mean": np.full(
                    len(validation_target),
                    float(np.mean(training_target)),
                    dtype=np.float64,
                ),
                "pca_target_reconstruction": (
                    predictions["one_hop"]
                    .reconstructed_target_blocks[
                        :, :, :, entity_position, slot_position
                    ]
                    .reshape(-1)
                ),
                **{
                    scope: prediction.decoded_target_blocks[
                        :, :, :, entity_position, slot_position
                    ].reshape(-1)
                    for scope, prediction in predictions.items()
                },
            }
            for name, candidate in candidate_values.items():
                squared_error = np.square(
                    candidate - validation_target
                )
                normalized_mse = float(
                    np.mean(squared_error) / variance
                )
                per_family = {
                    family: float(
                        np.mean(
                            squared_error[
                                repeated_families == family
                            ]
                        )
                        / variance
                    )
                    for family in sorted(set(repeated_families))
                }
                target_results[name][feature_key] = {
                    "status": "completed",
                    "entity_id": training.entity_ids[
                        entity_position
                    ],
                    "training_variance": variance,
                    "validation_normalized_mse": normalized_mse,
                    "family_normalized_mse": per_family,
                }
                representation_errors[name].append(normalized_mse)
                for family, error in per_family.items():
                    family_errors[name].setdefault(
                        family, []
                    ).append(error)

    representations: Dict[str, Dict[str, Any]] = {
        name: {
            "mean_validation_normalized_mse": (
                float(np.mean(errors)) if errors else None
            ),
            "completed_target_count": len(errors),
            "family_normalized_mse": {
                family: float(np.mean(values))
                for family, values in sorted(
                    family_errors[name].items()
                )
            },
            "targets": target_results[name],
        }
        for name, errors in representation_errors.items()
    }
    latent_errors = _latent_prediction_errors(
        models,
        training,
        predictions,
        training_prediction,
    )
    for scope, latent_error in latent_errors.items():
        representations[scope][
            "mean_validation_latent_normalized_mse"
        ] = latent_error

    mean_error = _representation_error(
        representations, "training_mean"
    )
    reconstruction_error = _representation_error(
        representations, "pca_target_reconstruction"
    )
    local_error = _representation_error(
        representations, "entity_local"
    )
    graph_error = _representation_error(
        representations, "one_hop"
    )
    all_error = _representation_error(
        representations, "all_entities"
    )
    gates = {
        "target_pca_retains_raw_state": {
            "observed": reconstruction_error,
            "maximum": maximum_reconstruction_normalized_mse,
            "passed": (
                reconstruction_error
                <= maximum_reconstruction_normalized_mse
            ),
        },
        "one_hop_prediction_beats_training_mean": {
            "observed": graph_error,
            "reference": mean_error,
            "passed": graph_error < mean_error,
        },
        "one_hop_prediction_retains_all_entity_performance": {
            "observed_error_ratio": (
                graph_error / all_error
                if all_error > 0.0
                else (1.0 if graph_error == 0.0 else float("inf"))
            ),
            "maximum": maximum_graph_to_all_error_ratio,
            "passed": (
                graph_error
                <= maximum_graph_to_all_error_ratio * all_error
                + 1e-12
            ),
        },
        "one_hop_prediction_beats_entity_local": {
            "observed": graph_error,
            "reference": local_error,
            "passed": graph_error < local_error,
        },
    }
    supported = all(
        bool(gate["passed"]) for gate in gates.values()
    )
    raw_context_dimension = int(
        training.contexts.shape[1]
        * np.count_nonzero(training.observation_mask)
    )
    observed_entity_count = int(
        np.count_nonzero(
            np.any(training.observation_mask, axis=1)
        )
    )
    active_latent_width = sum(
        models["one_hop"].config.latent_dimension_for(entity_id)
        for entity_position, entity_id in enumerate(
            training.entity_ids
        )
        if np.any(training.observation_mask[entity_position])
    )
    latent_context_dimension = int(
        (
            training.contexts.shape[1]
            // training.target_block_size
        )
        * active_latent_width
    )
    return {
        "schema_version": 1,
        "kind": "linear_graph_jepa_development_assessment",
        "status": "supported" if supported else "not_supported",
        "decision": (
            "collect_observability_rich_corpus"
            if supported
            else "improve_graph_representation_before_collection"
        ),
        "evidence_boundary": (
            "linear frozen-PCA graph-JEPA development on inspected "
            "fault-free schedule families; no intervention, causal, "
            "or world-model claim"
        ),
        "validation_families": sorted(set(families)),
        "compression": {
            "raw_context_dimension": raw_context_dimension,
            "latent_context_dimension": latent_context_dimension,
            "context_ratio": (
                raw_context_dimension / latent_context_dimension
            ),
            "latent_dimension_per_observed_entity": (
                models["one_hop"].config.latent_dimension
            ),
            "active_latent_width_per_patch": active_latent_width,
            "entity_latent_dimensions": {
                entity_id: (
                    models["one_hop"].config.latent_dimension_for(
                        entity_id
                    )
                )
                for entity_position, entity_id in enumerate(
                    training.entity_ids
                )
                if np.any(
                    training.observation_mask[entity_position]
                )
            },
            "observed_entity_count": observed_entity_count,
        },
        "representations": representations,
        "gates": gates,
    }


def write_linear_graph_jepa_artifacts(
    models: Mapping[str, LinearGraphJepaWorldModel],
    assessment: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, Path]:
    """Serialize all controlled models and the development assessment."""

    if set(models) != REQUIRED_MODEL_SCOPES:
        raise ValueError(
            "graph JEPA artifacts require all context scopes"
        )
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite graph JEPA artifacts: {output}"
        )
    output.mkdir(parents=True)
    paths: Dict[str, Path] = {}
    for scope, model in models.items():
        path = output / f"{scope}-model.json"
        path.write_text(
            json.dumps(
                model.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        paths[f"{scope}_model"] = path
    assessment_path = output / "assessment.json"
    assessment_path.write_text(
        json.dumps(
            assessment,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    paths["assessment"] = assessment_path
    report_path = output / "report.md"
    report_path.write_text(_assessment_report(assessment))
    paths["report"] = report_path
    return paths


def _context_blocks(
    windows: GraphStateWindows,
    entity_position: int,
) -> NDArray[np.float64]:
    mask = windows.observation_mask[entity_position]
    feature_count = int(np.count_nonzero(mask))
    patch_count = (
        windows.contexts.shape[1] // windows.target_block_size
    )
    return np.asarray(
        windows.contexts[:, :, entity_position, mask].reshape(
            len(windows.contexts),
            patch_count,
            windows.target_block_size * feature_count,
        ),
        dtype=np.float64,
    )


def _target_blocks(
    windows: GraphStateWindows,
    entity_position: int,
) -> NDArray[np.float64]:
    mask = windows.observation_mask[entity_position]
    feature_count = int(np.count_nonzero(mask))
    return np.asarray(
        windows.target_blocks[
            :, :, :, entity_position, mask
        ].reshape(
            len(windows.contexts),
            len(windows.horizons),
            windows.target_block_size * feature_count,
        ),
        dtype=np.float64,
    )


def _encode(
    values: NDArray[np.float64],
    encoder: _EntityEncoder,
) -> NDArray[np.float64]:
    return np.asarray(
        (values - encoder.location) @ encoder.components,
        dtype=np.float64,
    )


def _decode(
    tokens: NDArray[np.float64],
    encoder: _EntityEncoder,
) -> NDArray[np.float64]:
    return np.asarray(
        tokens @ encoder.components.T + encoder.location,
        dtype=np.float64,
    )


def _prediction_design(
    context_tokens: NDArray[np.float64],
    windows: GraphStateWindows,
    context_entity_ids: Tuple[str, ...],
) -> NDArray[np.float64]:
    positions = tuple(
        windows.entity_ids.index(entity_id)
        for entity_id in context_entity_ids
    )
    context = context_tokens[:, :, positions, :].reshape(
        len(windows.contexts), -1
    )
    horizon_count = len(windows.horizons)
    repeated_context = np.broadcast_to(
        context[:, None, :],
        (len(context), horizon_count, context.shape[1]),
    )
    controls = np.mean(windows.target_controls, axis=2)
    horizon_one_hot = np.broadcast_to(
        np.eye(horizon_count, dtype=np.float64)[None, :, :],
        (len(context), horizon_count, horizon_count),
    )
    return np.asarray(
        np.concatenate(
            (repeated_context, controls, horizon_one_hot),
            axis=2,
        ).reshape(len(context) * horizon_count, -1),
        dtype=np.float64,
    )


def _fit_ridge(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    ridge: float,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    location = np.mean(design, axis=0)
    scale = np.std(design, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (design - location) / scale
    with_intercept = np.column_stack(
        (normalized, np.ones(len(normalized)))
    )
    penalty = (
        np.eye(with_intercept.shape[1], dtype=np.float64) * ridge
    )
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(
        with_intercept.T @ with_intercept + penalty,
        with_intercept.T @ target,
    )
    return (
        np.asarray(location, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _apply_ridge(
    design: NDArray[np.float64],
    location: NDArray[np.float64],
    scale: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    normalized = (design - location) / scale
    with_intercept = np.column_stack(
        (normalized, np.ones(len(normalized)))
    )
    return np.asarray(
        with_intercept @ weights,
        dtype=np.float64,
    )


def _latent_prediction_errors(
    models: Mapping[str, LinearGraphJepaWorldModel],
    training: GraphStateWindows,
    predictions: Mapping[str, GraphJepaPrediction],
    training_prediction: GraphJepaPrediction,
) -> Mapping[str, Optional[float]]:
    errors: Dict[str, list[float]] = {
        scope: [] for scope in CONTEXT_SCOPES
    }
    for entity_position in range(len(training.entity_ids)):
        if not np.any(training.observation_mask[entity_position]):
            continue
        for latent_position in range(
            models["one_hop"].config.latent_dimension
        ):
            training_target = training_prediction.target_tokens[
                :, :, entity_position, latent_position
            ].reshape(-1)
            variance = float(np.var(training_target))
            if variance <= 1e-12:
                continue
            for scope in CONTEXT_SCOPES:
                squared_error = np.square(
                    predictions[scope].predicted_tokens[
                        :, :, entity_position, latent_position
                    ].reshape(-1)
                    - predictions[scope].target_tokens[
                        :, :, entity_position, latent_position
                    ].reshape(-1)
                )
                errors[scope].append(
                    float(np.mean(squared_error) / variance)
                )
    return {
        scope: float(np.mean(values)) if values else None
        for scope, values in errors.items()
    }


def _representation_error(
    representations: Mapping[str, Mapping[str, Any]],
    name: str,
) -> float:
    value = representations[name][
        "mean_validation_normalized_mse"
    ]
    if value is None:
        raise ValueError(
            "graph JEPA evaluation has no variable raw targets"
        )
    return float(value)


def _family_id(case_id: str) -> str:
    match = re.search(r"-f([0-9]+)-", case_id)
    if match is None:
        raise ValueError(
            f"cannot derive graph JEPA family: {case_id}"
        )
    return f"f{int(match.group(1)):02d}"


def _assessment_report(assessment: Mapping[str, Any]) -> str:
    representations = dict(assessment["representations"])
    lines = [
        "# Linear graph-JEPA development",
        "",
        f"Result: **{str(assessment['status']).upper()}**",
        "",
        f"Next decision: `{assessment['decision']}`",
        "",
        "## Decoded held-out normalized MSE",
        "",
    ]
    for name in (
        "training_mean",
        "pca_target_reconstruction",
        "entity_local",
        "one_hop",
        "all_entities",
    ):
        value = float(
            dict(representations[name])[
                "mean_validation_normalized_mse"
            ]
        )
        lines.append(f"- `{name}`: `{value:.6f}`")
    compression = dict(assessment["compression"])
    lines.extend(
        (
            "",
            "## Compression",
            "",
            f"- Raw context: `{compression['raw_context_dimension']}`",
            (
                "- Graph latent context: "
                f"`{compression['latent_context_dimension']}`"
            ),
            f"- Ratio: `{float(compression['context_ratio']):.3f}:1`",
            "",
            "## Development gates",
            "",
        )
    )
    for name, raw_gate in dict(assessment["gates"]).items():
        gate = dict(raw_gate)
        lines.append(
            f"- {'PASS' if gate['passed'] else 'FAIL'} — `{name}`"
        )
    lines.extend(
        (
            "",
            "## Evidence boundary",
            "",
            str(assessment["evidence_boundary"]),
            "",
        )
    )
    return "\n".join(lines)
