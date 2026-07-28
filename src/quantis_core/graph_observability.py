"""Held-out raw-state gates for graph-structured telemetry."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .graph_telemetry import GraphStateWindows


REPRESENTATION_NAMES = (
    "training_mean",
    "persistence",
    "entity_local_ridge",
    "one_hop_graph_ridge",
    "flat_raw_context_ridge",
)


def evaluate_graph_observability(
    training: GraphStateWindows,
    validation: GraphStateWindows,
    *,
    training_window_case_ids: Sequence[str],
    validation_window_case_ids: Sequence[str],
    ridge: float,
    maximum_graph_to_flat_error_ratio: float = 1.05,
) -> Mapping[str, Any]:
    """Assess raw future-state observability on held-out families."""

    _validate_inputs(
        training,
        validation,
        training_window_case_ids,
        validation_window_case_ids,
        ridge,
        maximum_graph_to_flat_error_ratio,
    )
    validation_families = tuple(
        _family_id(case_id) for case_id in validation_window_case_ids
    )
    representations: Dict[str, Dict[str, Any]] = {
        name: {
            "targets": {},
            "completed_target_count": 0,
            "mean_validation_normalized_mse": None,
            "family_normalized_mse": {},
        }
        for name in REPRESENTATION_NAMES
    }
    flat_training = _context_matrix(
        training, tuple(range(len(training.entity_ids)))
    )
    flat_validation = _context_matrix(
        validation, tuple(range(len(validation.entity_ids)))
    )
    flat_training_design, flat_validation_design = _conditioned_designs(
        flat_training,
        flat_validation,
        training,
        validation,
    )
    target_errors: Dict[str, list[float]] = {
        name: [] for name in REPRESENTATION_NAMES
    }
    family_errors: Dict[str, Dict[str, list[float]]] = {
        name: {} for name in REPRESENTATION_NAMES
    }
    horizon_count = len(training.horizons)
    repeated_validation_families = np.repeat(
        np.asarray(validation_families, dtype=object),
        horizon_count,
    )

    for entity_position, feature_keys in enumerate(
        training.local_feature_keys
    ):
        local_training = _context_matrix(
            training, (entity_position,)
        )
        local_validation = _context_matrix(
            validation, (entity_position,)
        )
        local_designs = _conditioned_designs(
            local_training,
            local_validation,
            training,
            validation,
        )
        neighbor_ids = training.graph.neighboring_entity_ids(
            training.entity_ids[entity_position]
        )
        neighbor_positions = tuple(
            training.entity_ids.index(entity_id)
            for entity_id in neighbor_ids
        )
        graph_training = _context_matrix(
            training, neighbor_positions
        )
        graph_validation = _context_matrix(
            validation, neighbor_positions
        )
        graph_designs = _conditioned_designs(
            graph_training,
            graph_validation,
            training,
            validation,
        )
        for slot_position, feature_key in enumerate(feature_keys):
            training_target = np.mean(
                training.target_blocks[
                    :, :, :, entity_position, slot_position
                ],
                axis=2,
            ).reshape(-1)
            validation_target = np.mean(
                validation.target_blocks[
                    :, :, :, entity_position, slot_position
                ],
                axis=2,
            ).reshape(-1)
            training_variance = float(np.var(training_target))
            if training_variance <= 1e-12:
                for representation in representations.values():
                    representation["targets"][feature_key] = {
                        "status": "insufficient_training_variation"
                    }
                continue
            persistence = np.repeat(
                validation.contexts[
                    :, -1, entity_position, slot_position
                ][:, None],
                horizon_count,
                axis=1,
            ).reshape(-1)
            predictions = {
                "training_mean": np.full(
                    len(validation_target),
                    float(np.mean(training_target)),
                    dtype=np.float64,
                ),
                "persistence": persistence,
                "entity_local_ridge": _ridge_predict(
                    local_designs[0],
                    training_target,
                    local_designs[1],
                    ridge,
                ),
                "one_hop_graph_ridge": _ridge_predict(
                    graph_designs[0],
                    training_target,
                    graph_designs[1],
                    ridge,
                ),
                "flat_raw_context_ridge": _ridge_predict(
                    flat_training_design,
                    training_target,
                    flat_validation_design,
                    ridge,
                ),
            }
            for name, prediction in predictions.items():
                squared_error = np.square(
                    prediction - validation_target
                )
                normalized_mse = float(
                    np.mean(squared_error) / training_variance
                )
                per_family = {
                    family: float(
                        np.mean(
                            squared_error[
                                repeated_validation_families == family
                            ]
                        )
                        / training_variance
                    )
                    for family in sorted(
                        set(repeated_validation_families)
                    )
                }
                representations[name]["targets"][feature_key] = {
                    "status": "completed",
                    "entity_id": training.entity_ids[
                        entity_position
                    ],
                    "training_variance": training_variance,
                    "validation_normalized_mse": normalized_mse,
                    "family_normalized_mse": per_family,
                }
                target_errors[name].append(normalized_mse)
                for family, error in per_family.items():
                    family_errors[name].setdefault(
                        family, []
                    ).append(error)

    for name in REPRESENTATION_NAMES:
        errors = target_errors[name]
        representations[name]["completed_target_count"] = len(errors)
        representations[name][
            "mean_validation_normalized_mse"
        ] = float(np.mean(errors)) if errors else None
        representations[name]["family_normalized_mse"] = {
            family: float(np.mean(errors_by_target))
            for family, errors_by_target in sorted(
                family_errors[name].items()
            )
        }

    mean_error = _aggregate_error(
        representations, "training_mean"
    )
    persistence_error = _aggregate_error(
        representations, "persistence"
    )
    graph_error = _aggregate_error(
        representations, "one_hop_graph_ridge"
    )
    flat_error = _aggregate_error(
        representations, "flat_raw_context_ridge"
    )
    gates = {
        "flat_raw_beats_training_mean": {
            "observed": flat_error,
            "reference": mean_error,
            "passed": flat_error < mean_error,
        },
        "one_hop_graph_beats_persistence": {
            "observed": graph_error,
            "reference": persistence_error,
            "passed": graph_error < persistence_error,
        },
        "one_hop_graph_retains_flat_raw_performance": {
            "observed_error_ratio": (
                graph_error / flat_error
                if flat_error > 0.0
                else (1.0 if graph_error == 0.0 else float("inf"))
            ),
            "maximum": maximum_graph_to_flat_error_ratio,
            "passed": (
                graph_error
                <= maximum_graph_to_flat_error_ratio * flat_error
                + 1e-12
            ),
        },
    }
    supported = all(
        bool(gate["passed"]) for gate in gates.values()
    )
    return {
        "schema_version": 1,
        "kind": "graph_jepa_observability_pilot",
        "status": "supported" if supported else "not_supported",
        "decision": (
            "train_graph_jepa"
            if supported
            else "add_explicit_operational_state"
        ),
        "evidence_boundary": (
            "development reuse of inspected fault-free schedule-family "
            "transfer; not publication-confirmation or causal evidence"
        ),
        "training_window_count": len(training.contexts),
        "validation_window_count": len(validation.contexts),
        "entity_ids": list(training.entity_ids),
        "validation_families": sorted(set(validation_families)),
        "representations": representations,
        "gates": gates,
    }


def write_graph_observability_assessment(
    assessment: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, Path]:
    """Write the machine-readable assessment and concise report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    assessment_path = output / "assessment.json"
    report_path = output / "report.md"
    assessment_path.write_text(
        json.dumps(
            assessment,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    representations = assessment["representations"]
    gates = assessment["gates"]
    lines = [
        "# Graph JEPA observability pilot",
        "",
        f"Result: **{str(assessment['status']).upper()}**",
        "",
        f"Next decision: `{assessment['decision']}`",
        "",
        "## Held-out normalized MSE",
        "",
    ]
    for name in REPRESENTATION_NAMES:
        value = representations[name][
            "mean_validation_normalized_mse"
        ]
        lines.append(f"- `{name}`: `{float(value):.6f}`")
    lines.extend(("", "## Development gates", ""))
    for name, gate in gates.items():
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
    report_path.write_text("\n".join(lines))
    return {"assessment": assessment_path, "report": report_path}


def _validate_inputs(
    training: GraphStateWindows,
    validation: GraphStateWindows,
    training_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
    ridge: float,
    maximum_ratio: float,
) -> None:
    if ridge <= 0.0:
        raise ValueError("graph observability ridge must be positive")
    if maximum_ratio < 1.0:
        raise ValueError(
            "graph-to-flat error ratio must be at least one"
        )
    if (
        len(training_case_ids) != len(training.contexts)
        or len(validation_case_ids) != len(validation.contexts)
    ):
        raise ValueError(
            "graph observability case identities must align with windows"
        )
    if (
        training.entity_ids != validation.entity_ids
        or training.entity_kinds != validation.entity_kinds
        or training.local_feature_keys
        != validation.local_feature_keys
        or not np.array_equal(
            training.observation_mask,
            validation.observation_mask,
        )
        or training.control_feature_names
        != validation.control_feature_names
        or training.horizons != validation.horizons
        or training.target_block_size != validation.target_block_size
        or training.graph != validation.graph
    ):
        raise ValueError(
            "training and validation graph schemas must match"
        )


def _context_matrix(
    windows: GraphStateWindows,
    entity_positions: Tuple[int, ...],
) -> NDArray[np.float64]:
    columns = []
    for entity_position in entity_positions:
        mask = windows.observation_mask[entity_position]
        if np.any(mask):
            columns.append(
                windows.contexts[:, :, entity_position, mask].reshape(
                    len(windows.contexts), -1
                )
            )
    if not columns:
        return np.zeros((len(windows.contexts), 0), dtype=np.float64)
    return np.concatenate(columns, axis=1)


def _conditioned_designs(
    training_context: NDArray[np.float64],
    validation_context: NDArray[np.float64],
    training: GraphStateWindows,
    validation: GraphStateWindows,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        _conditioned_design(training_context, training),
        _conditioned_design(validation_context, validation),
    )


def _conditioned_design(
    context: NDArray[np.float64],
    windows: GraphStateWindows,
) -> NDArray[np.float64]:
    sample_count = len(context)
    horizon_count = len(windows.horizons)
    controls = np.mean(windows.target_controls, axis=2)
    repeated_context = np.broadcast_to(
        context[:, None, :],
        (sample_count, horizon_count, context.shape[1]),
    )
    horizon_one_hot = np.broadcast_to(
        np.eye(horizon_count, dtype=np.float64)[None, :, :],
        (sample_count, horizon_count, horizon_count),
    )
    return np.asarray(
        np.concatenate(
            (repeated_context, controls, horizon_one_hot),
            axis=2,
        ).reshape(sample_count * horizon_count, -1),
        dtype=np.float64,
    )


def _ridge_predict(
    training: NDArray[np.float64],
    target: NDArray[np.float64],
    validation: NDArray[np.float64],
    ridge: float,
) -> NDArray[np.float64]:
    location = np.mean(training, axis=0)
    scale = np.std(training, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized_training = (training - location) / scale
    normalized_validation = (validation - location) / scale
    design = np.column_stack(
        (normalized_training, np.ones(len(training)))
    )
    validation_design = np.column_stack(
        (normalized_validation, np.ones(len(validation)))
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return np.asarray(
        validation_design @ weights,
        dtype=np.float64,
    )


def _aggregate_error(
    representations: Mapping[str, Mapping[str, Any]],
    name: str,
) -> float:
    value = representations[name][
        "mean_validation_normalized_mse"
    ]
    if value is None:
        raise ValueError("graph observability has no variable targets")
    return float(value)


def _family_id(case_id: str) -> str:
    match = re.search(r"-f([0-9]+)-", case_id)
    if match is None:
        raise ValueError(
            f"cannot derive graph observability family: {case_id}"
        )
    return f"f{int(match.group(1)):02d}"
