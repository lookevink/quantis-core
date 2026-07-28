"""Run the deterministic Phase-0 action-conditioned tracer bullet."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
    ActionTrajectoryCompiler,
    GraphVarxConfig,
    GraphVarxDynamics,
    InterventionAction,
    RolloutCandidate,
    persistence_rollout,
    rank_action_candidates,
    validate_matched_action_pairs,
)
from quantis_core.action_dynamics_development import (
    assess_action_dynamics_evidence,
)
from quantis_core.action_dynamics_synthetic import (
    SYNTHETIC_ACTION_LIBRARY,
    synthetic_action_runs,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


CANDIDATE_ONSETS = (10, 11, 12, 13)
CANDIDATE_MAGNITUDES = (0.5, 0.75, 1.0)
CANDIDATE_DURATION = 8


def run_synthetic_tracer(
    output_directory: Path,
) -> Dict[str, Any]:
    """Fit, attribute, assess, and write the Phase-0 synthetic run."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"synthetic tracer output is not empty: {output}"
        )
    training_runs = synthetic_action_runs(
        30, split="training", seed=1_000
    )
    validation_runs = synthetic_action_runs(
        10, split="validation", seed=9_000
    )
    training_pair_validation = validate_matched_action_pairs(
        training_runs
    )
    validation_pair_validation = validate_matched_action_pairs(
        validation_runs
    )
    fitted_compiler = ActionTrajectoryCompiler(
        context_length=4,
        rollout_horizon=8,
    ).fit(training_runs)
    training = fitted_compiler.transform(training_runs)
    fitted_action_model = GraphVarxDynamics(
        GraphVarxConfig(ridge=1e-3, include_actions=True)
    ).fit(training)
    fitted_action_agnostic_model = GraphVarxDynamics(
        GraphVarxConfig(ridge=1e-3, include_actions=False)
    ).fit(training)
    compiler_artifact = fitted_compiler.to_dict()
    action_model_artifact = fitted_action_model.to_dict()
    action_agnostic_model_artifact = (
        fitted_action_agnostic_model.to_dict()
    )
    compiler = ActionTrajectoryCompiler.from_dict(
        compiler_artifact
    )
    action_model = GraphVarxDynamics.from_dict(
        action_model_artifact
    )
    action_agnostic_model = GraphVarxDynamics.from_dict(
        action_agnostic_model_artifact
    )
    validation = compiler.transform(validation_runs)
    action_model.ensure_compatible(validation)
    action_agnostic_model.ensure_compatible(validation)
    action_prediction = action_model.rollout(
        validation.histories,
        validation.future_controls,
        validation.future_actions,
        validation.graph,
    )
    action_agnostic_prediction = action_agnostic_model.rollout(
        validation.histories,
        validation.future_controls,
        validation.future_actions,
        validation.graph,
    )
    persistence_prediction = persistence_rollout(
        validation.histories, compiler.rollout_horizon
    )
    attribution_rows: List[Dict[str, Any]] = []
    no_action_candidate_id = "none"
    propagation_case_id = "validation-pair-000-action"
    for pair_index in range(len(validation_runs) // 2):
        action_case_id = (
            f"validation-pair-{pair_index:03d}-action"
        )
        control_case_id = (
            f"validation-pair-{pair_index:03d}-control"
        )
        transition_index = 9
        action_position = _window_position(
            validation, action_case_id, transition_index
        )
        control_position = _window_position(
            validation, control_case_id, transition_index
        )
        action_truth = validation_runs[
            pair_index * 2
        ].manifest.actions[0]
        true_id = _candidate_id(
            action_truth.action_kind,
            action_truth.target_entity,
            action_truth.start_index,
            action_truth.magnitude,
        )
        action_candidates = _candidate_grid(
            compiler=compiler,
            point_count=validation_runs[
                pair_index * 2
            ].manifest.point_count,
            transition_index=transition_index,
            rollout_horizon=compiler.rollout_horizon,
            graph=validation.graph,
        )
        action_result = rank_action_candidates(
            model=action_model,
            history=validation.histories[action_position],
            future_controls=validation.future_controls[
                action_position
            ],
            observed_future=validation.future_states[
                action_position
            ],
            candidates=action_candidates,
            graph=validation.graph,
            no_action_candidate_id=no_action_candidate_id,
        )
        control_result = rank_action_candidates(
            model=action_model,
            history=validation.histories[control_position],
            future_controls=validation.future_controls[
                control_position
            ],
            observed_future=validation.future_states[
                control_position
            ],
            candidates=action_candidates,
            graph=validation.graph,
            no_action_candidate_id=no_action_candidate_id,
        )
        attribution_rows.extend(
            (
                {
                    "case_id": action_case_id,
                    "true_candidate_id": true_id,
                    "observed_future": validation.future_states[
                        action_position
                    ].tolist(),
                    "candidate_ids": list(
                        action_result.candidate_ids
                    ),
                    "candidate_distribution": {
                        "mean": (
                            action_result.candidate_distribution.mean.tolist()
                        ),
                        "variance": (
                            action_result.candidate_distribution.variance.tolist()
                        ),
                    },
                },
                {
                    "case_id": control_case_id,
                    "true_candidate_id": no_action_candidate_id,
                    "observed_future": validation.future_states[
                        control_position
                    ].tolist(),
                    "candidate_ids": list(
                        control_result.candidate_ids
                    ),
                    "candidate_distribution": {
                        "mean": (
                            control_result.candidate_distribution.mean.tolist()
                        ),
                        "variance": (
                            control_result.candidate_distribution.variance.tolist()
                        ),
                    },
                },
            )
        )

    evidence = {
        "schema_version": 2,
        "kind": "action_dynamics_phase_zero_evidence",
        "observed_future": validation.future_states.tolist(),
        "predictions": {
            "action_conditioned": {
                "mean": action_prediction.mean.tolist(),
                "variance": action_prediction.variance.tolist(),
            },
            "action_agnostic": {
                "mean": action_agnostic_prediction.mean.tolist(),
                "variance": (
                    action_agnostic_prediction.variance.tolist()
                ),
            },
            "persistence": {
                "mean": persistence_prediction.mean.tolist(),
                "variance": (
                    persistence_prediction.variance.tolist()
                ),
            },
        },
        "propagation_case_id": propagation_case_id,
        "attribution_rows": attribution_rows,
    }
    assessment = assess_action_dynamics_evidence(evidence)
    protocol = {
        "schema_version": 1,
        "kind": "action_dynamics_phase_zero_protocol",
        "training_run_count": len(training_runs),
        "training_pair_count": len(training_runs) // 2,
        "validation_run_count": len(validation_runs),
        "validation_pair_count": len(validation_runs) // 2,
        "context_length": compiler.context_length,
        "rollout_horizon": compiler.rollout_horizon,
        "action_at_t_predicts_state": "t+1",
        "matched_twins_share_exogenous_noise": True,
        "candidate_library": [
            {
                "action_kind": action_kind,
                "target_entity": target_entity,
            }
            for action_kind, target_entity
            in SYNTHETIC_ACTION_LIBRARY
        ],
        "candidate_onsets": list(CANDIDATE_ONSETS),
        "candidate_magnitudes": list(CANDIDATE_MAGNITUDES),
        "candidate_duration": CANDIDATE_DURATION,
        "training_pair_validation": training_pair_validation,
        "validation_pair_validation": validation_pair_validation,
        "evidence_boundary": (
            "synthetic tracer bullet; not lab or world-model evidence"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "protocol.json": protocol,
        "compiler.json": compiler_artifact,
        "action-model.json": action_model_artifact,
        "action-agnostic-model.json": (
            action_agnostic_model_artifact
        ),
        "evidence.json": evidence,
        "assessment.json": assessment,
    }
    for filename, payload in artifacts.items():
        _write_json(output / filename, payload)
    artifact_manifest = {
        "schema_version": 1,
        "kind": "action_dynamics_phase_zero_artifact_manifest",
        "sha256": {
            filename: _sha256(output / filename)
            for filename in sorted(artifacts)
        },
    }
    _write_json(
        output / "artifact-manifest.json", artifact_manifest
    )
    return {
        "protocol": protocol,
        "assessment": assessment,
        "artifacts": sorted(
            (*artifacts, "artifact-manifest.json")
        ),
    }


def _window_position(
    windows: ActionConditionedWindows,
    trajectory_id: str,
    transition_index: int,
) -> int:
    for position, (candidate_id, candidate_transition) in enumerate(
        zip(
            windows.trajectory_ids,
            windows.transition_indices,
        )
    ):
        if (
            candidate_id == trajectory_id
            and candidate_transition == transition_index
        ):
            return position
    raise ValueError("synthetic evaluation window is missing")


def _candidate_grid(
    *,
    compiler: ActionTrajectoryCompiler,
    point_count: int,
    transition_index: int,
    rollout_horizon: int,
    graph: DeclaredTelemetryGraph,
) -> Sequence[RolloutCandidate]:
    candidates = [
        RolloutCandidate(
            "none",
            compiler.compile_action_trajectory(
                point_count=point_count,
                actions=(),
                graph=graph,
            )[
                transition_index : transition_index
                + rollout_horizon
            ],
        )
    ]
    for action_kind, target_entity in SYNTHETIC_ACTION_LIBRARY:
        for onset in CANDIDATE_ONSETS:
            for magnitude in CANDIDATE_MAGNITUDES:
                candidate_id = _candidate_id(
                    action_kind,
                    target_entity,
                    onset,
                    magnitude,
                )
                action = InterventionAction(
                    action_id=candidate_id,
                    action_kind=action_kind,
                    target_entity=target_entity,
                    start_index=onset,
                    stop_index=onset + CANDIDATE_DURATION,
                    magnitude=magnitude,
                )
                full_trajectory = (
                    compiler.compile_action_trajectory(
                        point_count=point_count,
                        actions=(action,),
                        graph=graph,
                    )
                )
                candidates.append(
                    RolloutCandidate(
                        candidate_id,
                        full_trajectory[
                            transition_index : transition_index
                            + rollout_horizon
                        ],
                    )
                )
    return tuple(candidates)


def _candidate_id(
    action_kind: str,
    target_entity: str,
    onset: int,
    magnitude: float,
) -> str:
    return (
        f"{action_kind}@{target_entity}:"
        f"onset={onset}:magnitude={magnitude:.2f}"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "synthetic-phase-zero-v1"
        ),
    )
    arguments = parser.parse_args(argv)
    result = run_synthetic_tracer(arguments.output)
    print(json.dumps(result["assessment"], indent=2))
    return (
        0
        if result["assessment"]["status"] == "supported"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
