"""Deterministic matched synthetic trajectories for the Phase-0 tracer."""

from typing import Tuple

import numpy as np

from .action_conditioned_dynamics import (
    ActionConditionedCaseManifest,
    ActionConditionedRun,
    InterventionAction,
)
from .graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
)

SYNTHETIC_ACTION_LIBRARY = (
    ("api_rejection", "source"),
    ("redis_enqueue_delay", "source_to_middle"),
    ("worker_pause", "middle"),
    ("redis_dequeue_delay", "middle_to_sink"),
    ("postgres_lock", "sink"),
)


def causal_chain_graph() -> DeclaredTelemetryGraph:
    """Return the fixed source-to-middle-to-sink tracer graph."""

    return DeclaredTelemetryGraph(
        entities=(
            GraphEntity("source", "node", "service"),
            GraphEntity(
                "source_to_middle",
                "edge",
                "dependency",
                "source",
                "middle",
            ),
            GraphEntity("middle", "node", "service"),
            GraphEntity(
                "middle_to_sink",
                "edge",
                "dependency",
                "middle",
                "sink",
            ),
            GraphEntity("sink", "node", "service"),
        ),
        bindings=(),
    )


def synthetic_action_runs(
    count: int,
    *,
    split: str,
    seed: int,
) -> Tuple[ActionConditionedRun, ...]:
    """Return complete treatment/control pairs with shared exogenous noise."""

    if count < 2 or count % 2:
        raise ValueError(
            "synthetic runs require complete treatment/control pairs"
        )
    return tuple(
        _synthetic_run(
            pair_index=index // 2,
            split=split,
            seed=seed,
            with_action=index % 2 == 0,
        )
        for index in range(count)
    )


def _synthetic_run(
    *,
    pair_index: int,
    split: str,
    seed: int,
    with_action: bool,
) -> ActionConditionedRun:
    rng = np.random.default_rng(seed + pair_index)
    point_count = 36
    action_start = 10 + pair_index % 4
    action_stop = action_start + 8
    magnitude = 0.5 + 0.25 * (pair_index % 3)
    action_kind, target_entity = SYNTHETIC_ACTION_LIBRARY[
        pair_index % len(SYNTHETIC_ACTION_LIBRARY)
    ]
    target_position = causal_chain_graph().entity_ids.index(
        target_entity
    )
    pair_id = f"{split}-pair-{pair_index:03d}"
    case_id = (
        f"{pair_id}-action"
        if with_action
        else f"{pair_id}-control"
    )
    actions = (
        (
            InterventionAction(
                action_id=f"{case_id}-reject",
                action_kind=action_kind,
                target_entity=target_entity,
                start_index=action_start,
                stop_index=action_stop,
                magnitude=magnitude,
            ),
        )
        if with_action
        else ()
    )
    manifest = ActionConditionedCaseManifest(
        case_id=case_id,
        matched_pair_id=pair_id,
        split=split,
        point_count=point_count,
        logical_window_period_nano=1_000_000_000,
        topology_id="causal-chain",
        worker_replicas=1,
        workload_seed=seed + pair_index,
        intervention_seed=seed * 10 + pair_index,
        actions=actions,
    )
    controls = (
        1.0
        + 0.2
        * np.sin(
            np.arange(point_count, dtype=np.float64) * 0.4
            + pair_index
        )
    ).reshape(point_count, 1)
    states = np.zeros((point_count, 5, 1), dtype=np.float64)
    states[0, :, 0] = rng.normal(0.0, 0.03, size=5)
    for point_index in range(point_count - 1):
        action_value = (
            magnitude
            if (
                with_action
                and action_start
                <= point_index
                < action_stop
            )
            else 0.0
        )
        current = states[point_index, :, 0]
        next_state = states[point_index + 1, :, 0]
        next_state[0] = (
            0.55 * current[0]
            + 0.12 * controls[point_index, 0]
        )
        next_state[1] = 0.50 * current[1] + 0.45 * current[0]
        next_state[2] = 0.55 * current[2] + 0.40 * current[1]
        next_state[3] = 0.50 * current[3] + 0.45 * current[2]
        next_state[4] = 0.55 * current[4] + 0.40 * current[3]
        next_state[target_position] += 1.4 * action_value
        next_state += rng.normal(0.0, 0.005, size=5)
    return ActionConditionedRun(
        manifest=manifest,
        graph=causal_chain_graph(),
        observations=states,
        controls=controls,
        state_feature_names=("pressure",),
        control_feature_names=("request_demand",),
    )
