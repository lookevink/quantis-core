"""Deterministic preprocessing for edge-dynamics development experiments."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .action_conditioned_dynamics import (
    ActionConditionedRun,
    ActionConditionedWindows,
    ActionTrajectoryCompiler,
)
from .action_dynamics_corpus import LoadedActionDynamicsCorpus
from .action_dynamics_real_corpus import (
    AttributionQuery,
    build_development_validation_queries,
)
from .graph_telemetry import DeclaredTelemetryGraph


EDGE_ROLE_NAMES = ("fit", "selection", "calibration", "evaluation")
_CONTEXT_LENGTH = 20
_ROLLOUT_HORIZON = 10


@dataclass(frozen=True)
class EdgePairRoles:
    """Whole-pair roles for the open development tournament."""

    fit_pair_ids: Tuple[str, ...]
    selection_pair_ids: Tuple[str, ...]
    calibration_pair_ids: Tuple[str, ...]
    evaluation_pair_ids: Tuple[str, ...]

    def pair_ids(self, role: str) -> Tuple[str, ...]:
        """Return the pair IDs assigned to one role."""

        values = {
            "fit": self.fit_pair_ids,
            "selection": self.selection_pair_ids,
            "calibration": self.calibration_pair_ids,
            "evaluation": self.evaluation_pair_ids,
        }
        try:
            return values[role]
        except KeyError as error:
            raise ValueError(f"unknown edge role: {role}") from error

    def to_dict(self) -> Dict[str, Any]:
        """Return an auditable JSON representation."""

        return {
            "schema_version": 1,
            "kind": "edge_dynamics_pair_roles",
            "roles": {
                role: list(self.pair_ids(role))
                for role in EDGE_ROLE_NAMES
            },
            "pair_counts": {
                role: len(self.pair_ids(role))
                for role in EDGE_ROLE_NAMES
            },
        }


@dataclass(frozen=True)
class PreparedAttributionQueries:
    """Evaluation queries materialized at the common rollout seam."""

    query_ids: Tuple[str, ...]
    histories: NDArray[np.float32]
    future_controls: NDArray[np.float32]
    observed_future: NDArray[np.float32]
    candidate_actions: NDArray[np.float32]
    candidate_ids: Tuple[str, ...]
    candidate_action_kinds: Tuple[str, ...]
    candidate_target_entities: Tuple[str, ...]
    expected_action_kinds: Tuple[str, ...]
    expected_target_entities: Tuple[str, ...]
    expected_variant_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        query_count = len(self.query_ids)
        candidate_count = len(self.candidate_ids)
        if (
            query_count == 0
            or candidate_count == 0
            or self.histories.shape[0] != query_count
            or self.future_controls.shape[0] != query_count
            or self.observed_future.shape[0] != query_count
            or self.candidate_actions.shape[:2]
            != (query_count, candidate_count)
            or len(self.candidate_action_kinds) != candidate_count
            or len(self.candidate_target_entities) != candidate_count
            or len(self.expected_action_kinds) != query_count
            or len(self.expected_target_entities) != query_count
            or len(self.expected_variant_ids) != query_count
        ):
            raise ValueError("prepared attribution queries do not align")


@dataclass(frozen=True)
class PreparedEdgeDynamicsData:
    """One reusable normalized cache for all candidate experiments."""

    source_corpus_sha256: str
    roles: EdgePairRoles
    compiler_artifact: Mapping[str, Any]
    windows: Mapping[str, ActionConditionedWindows]
    attribution_queries: PreparedAttributionQueries

    @property
    def graph(self) -> DeclaredTelemetryGraph:
        """Return the shared declared graph."""

        return self.windows["fit"].graph


def assign_edge_pair_roles(
    corpus: LoadedActionDynamicsCorpus,
) -> EdgePairRoles:
    """Assign 4/1/1 training pairs per action-topology cell."""

    pair_cell = _pair_cells(corpus.training_runs)
    cells: Dict[Tuple[str, int], list[str]] = {}
    for pair_id, cell in pair_cell.items():
        cells.setdefault(cell, []).append(pair_id)
    if len(cells) != 15 or any(len(pair_ids) != 6 for pair_ids in cells.values()):
        raise ValueError(
            "edge role assignment requires six training pairs in 15 cells"
        )
    assignments: Dict[str, list[str]] = {
        "fit": [],
        "selection": [],
        "calibration": [],
    }
    for cell in sorted(cells):
        ordered = sorted(
            cells[cell],
            key=lambda pair_id: (
                _role_digest(corpus.identity.corpus_sha256, pair_id),
                pair_id,
            ),
        )
        assignments["fit"].extend(ordered[:4])
        assignments["selection"].append(ordered[4])
        assignments["calibration"].append(ordered[5])
    roles = EdgePairRoles(
        fit_pair_ids=tuple(sorted(assignments["fit"])),
        selection_pair_ids=tuple(sorted(assignments["selection"])),
        calibration_pair_ids=tuple(sorted(assignments["calibration"])),
        evaluation_pair_ids=tuple(sorted(corpus.validation_pair_ids)),
    )
    _validate_roles(roles)
    return roles


def prepare_edge_dynamics_data(
    corpus: LoadedActionDynamicsCorpus,
) -> PreparedEdgeDynamicsData:
    """Normalize once and materialize the common experiment inputs."""

    roles = assign_edge_pair_roles(corpus)
    runs_by_role = {
        role: tuple(
            run
            for run in corpus.runs
            if run.manifest.matched_pair_id in set(roles.pair_ids(role))
        )
        for role in EDGE_ROLE_NAMES
    }
    compiler = ActionTrajectoryCompiler(
        context_length=_CONTEXT_LENGTH,
        rollout_horizon=_ROLLOUT_HORIZON,
    ).fit(runs_by_role["fit"])
    windows = {
        role: _as_float32(compiler.transform(runs_by_role[role]))
        for role in EDGE_ROLE_NAMES
    }
    queries = build_development_validation_queries(
        runs_by_role["evaluation"], corpus.runs[0].graph
    )
    prepared_queries = _prepare_attribution_queries(
        queries=queries,
        validation_runs=runs_by_role["evaluation"],
        validation=windows["evaluation"],
        compiler=compiler,
    )
    return PreparedEdgeDynamicsData(
        source_corpus_sha256=corpus.identity.corpus_sha256,
        roles=roles,
        compiler_artifact=compiler.to_dict(),
        windows=windows,
        attribution_queries=prepared_queries,
    )


def write_edge_dynamics_cache(
    data: PreparedEdgeDynamicsData,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Write one non-overwriting preprocessing cache and manifest."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite edge cache: {output}")
    output.mkdir(parents=True)
    for role in EDGE_ROLE_NAMES:
        window = data.windows[role]
        np.savez(
            output / f"{role}.npz",
            histories=window.histories,
            future_states=window.future_states,
            future_controls=window.future_controls,
            future_actions=window.future_actions,
            trajectory_ids=np.asarray(window.trajectory_ids),
            matched_pair_ids=np.asarray(window.matched_pair_ids),
            transition_indices=window.transition_indices,
        )
    query = data.attribution_queries
    np.savez(
        output / "attribution-queries.npz",
        histories=query.histories,
        future_controls=query.future_controls,
        observed_future=query.observed_future,
        candidate_actions=query.candidate_actions,
    )
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "edge_dynamics_preprocessing_cache",
        "source_corpus_sha256": data.source_corpus_sha256,
        "roles": data.roles.to_dict(),
        "compiler": dict(data.compiler_artifact),
        "window_counts": {
            role: len(data.windows[role].histories)
            for role in EDGE_ROLE_NAMES
        },
        "attribution": {
            "query_ids": list(query.query_ids),
            "candidate_ids": list(query.candidate_ids),
            "candidate_action_kinds": list(
                query.candidate_action_kinds
            ),
            "candidate_target_entities": list(
                query.candidate_target_entities
            ),
            "expected_action_kinds": list(
                query.expected_action_kinds
            ),
            "expected_target_entities": list(
                query.expected_target_entities
            ),
            "expected_variant_ids": list(query.expected_variant_ids),
        },
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(_pretty_json(metadata))
    artifact_hashes = {
        path.name: _file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "edge_dynamics_preprocessing_manifest",
        "source_corpus_sha256": data.source_corpus_sha256,
        "sha256": artifact_hashes,
    }
    (output / "artifact-manifest.json").write_text(_pretty_json(manifest))
    return manifest


def load_edge_dynamics_cache(
    directory: Path,
) -> PreparedEdgeDynamicsData:
    """Restore a strict preprocessing cache without raw-corpus parsing."""

    root = Path(directory)
    metadata = _read_object(root / "metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "edge_dynamics_preprocessing_cache"
    ):
        raise ValueError("unsupported edge preprocessing cache")
    manifest = _read_object(root / "artifact-manifest.json")
    recorded_hashes = manifest.get("sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("edge cache manifest is invalid")
    for filename, expected in recorded_hashes.items():
        if (
            not isinstance(filename, str)
            or not isinstance(expected, str)
            or _file_sha256(root / filename) != expected
        ):
            raise ValueError("edge cache content identity mismatch")
    compiler = dict(metadata["compiler"])
    schema = dict(compiler["semantic_schema"])
    graph = DeclaredTelemetryGraph.from_dict(dict(schema["graph"]))
    state_names = tuple(str(value) for value in schema["state_feature_names"])
    control_names = tuple(
        str(value) for value in schema["control_feature_names"]
    )
    action_names = tuple(
        str(value) for value in schema["action_feature_names"]
    )
    windows: Dict[str, ActionConditionedWindows] = {}
    for role in EDGE_ROLE_NAMES:
        with np.load(root / f"{role}.npz", allow_pickle=False) as arrays:
            windows[role] = ActionConditionedWindows(
                histories=arrays["histories"],
                future_states=arrays["future_states"],
                future_controls=arrays["future_controls"],
                future_actions=arrays["future_actions"],
                trajectory_ids=tuple(
                    str(value) for value in arrays["trajectory_ids"]
                ),
                matched_pair_ids=tuple(
                    str(value) for value in arrays["matched_pair_ids"]
                ),
                transition_indices=arrays["transition_indices"],
                entity_names=graph.entity_ids,
                state_feature_names=state_names,
                control_feature_names=control_names,
                action_feature_names=action_names,
                graph=graph,
            )
    raw_roles = dict(dict(metadata["roles"])["roles"])
    roles = EdgePairRoles(
        fit_pair_ids=tuple(str(value) for value in raw_roles["fit"]),
        selection_pair_ids=tuple(
            str(value) for value in raw_roles["selection"]
        ),
        calibration_pair_ids=tuple(
            str(value) for value in raw_roles["calibration"]
        ),
        evaluation_pair_ids=tuple(
            str(value) for value in raw_roles["evaluation"]
        ),
    )
    query_metadata = dict(metadata["attribution"])
    with np.load(
        root / "attribution-queries.npz", allow_pickle=False
    ) as arrays:
        prepared_queries = PreparedAttributionQueries(
            query_ids=tuple(
                str(value) for value in query_metadata["query_ids"]
            ),
            histories=arrays["histories"],
            future_controls=arrays["future_controls"],
            observed_future=arrays["observed_future"],
            candidate_actions=arrays["candidate_actions"],
            candidate_ids=tuple(
                str(value) for value in query_metadata["candidate_ids"]
            ),
            candidate_action_kinds=tuple(
                str(value)
                for value in query_metadata["candidate_action_kinds"]
            ),
            candidate_target_entities=tuple(
                str(value)
                for value in query_metadata[
                    "candidate_target_entities"
                ]
            ),
            expected_action_kinds=tuple(
                str(value)
                for value in query_metadata["expected_action_kinds"]
            ),
            expected_target_entities=tuple(
                str(value)
                for value in query_metadata[
                    "expected_target_entities"
                ]
            ),
            expected_variant_ids=tuple(
                str(value)
                for value in query_metadata["expected_variant_ids"]
            ),
        )
    return PreparedEdgeDynamicsData(
        source_corpus_sha256=str(metadata["source_corpus_sha256"]),
        roles=roles,
        compiler_artifact=compiler,
        windows=windows,
        attribution_queries=prepared_queries,
    )


def _pair_cells(
    runs: Sequence[ActionConditionedRun],
) -> Mapping[str, Tuple[str, int]]:
    grouped: Dict[str, list[ActionConditionedRun]] = {}
    for run in runs:
        grouped.setdefault(run.manifest.matched_pair_id, []).append(run)
    cells: Dict[str, Tuple[str, int]] = {}
    for pair_id, pair in grouped.items():
        treatment = [run for run in pair if len(run.manifest.actions) == 1]
        if len(pair) != 2 or len(treatment) != 1:
            raise ValueError("edge role pairs require treatment and control")
        action = treatment[0].manifest.actions[0]
        cells[pair_id] = (
            action.action_kind,
            treatment[0].manifest.worker_replicas,
        )
    return cells


def _role_digest(corpus_sha256: str, pair_id: str) -> str:
    return hashlib.sha256(
        f"{corpus_sha256}:{pair_id}".encode("utf-8")
    ).hexdigest()


def _validate_roles(roles: EdgePairRoles) -> None:
    expected_counts = {
        "fit": 60,
        "selection": 15,
        "calibration": 15,
        "evaluation": 30,
    }
    all_ids: list[str] = []
    for role, expected in expected_counts.items():
        pair_ids = roles.pair_ids(role)
        if len(pair_ids) != expected or len(set(pair_ids)) != expected:
            raise ValueError("edge role pair counts are invalid")
        all_ids.extend(pair_ids)
    if len(set(all_ids)) != 120:
        raise ValueError("edge roles must be pair-disjoint")


def _as_float32(
    windows: ActionConditionedWindows,
) -> ActionConditionedWindows:
    return ActionConditionedWindows(
        histories=np.asarray(windows.histories, dtype=np.float32),
        future_states=np.asarray(
            windows.future_states, dtype=np.float32
        ),
        future_controls=np.asarray(
            windows.future_controls, dtype=np.float32
        ),
        future_actions=np.asarray(
            windows.future_actions, dtype=np.float32
        ),
        trajectory_ids=windows.trajectory_ids,
        matched_pair_ids=windows.matched_pair_ids,
        transition_indices=windows.transition_indices,
        entity_names=windows.entity_names,
        state_feature_names=windows.state_feature_names,
        control_feature_names=windows.control_feature_names,
        action_feature_names=windows.action_feature_names,
        graph=windows.graph,
    )


def _prepare_attribution_queries(
    *,
    queries: Sequence[AttributionQuery],
    validation_runs: Sequence[ActionConditionedRun],
    validation: ActionConditionedWindows,
    compiler: ActionTrajectoryCompiler,
) -> PreparedAttributionQueries:
    run_by_id = {
        run.manifest.case_id: run for run in validation_runs
    }
    candidate_ids = tuple(
        candidate.candidate_id for candidate in queries[0].candidates
    )
    candidate_kinds = tuple(
        (
            candidate.actions[0].action_kind
            if candidate.actions
            else ""
        )
        for candidate in queries[0].candidates
    )
    candidate_targets = tuple(
        (
            candidate.actions[0].target_entity
            if candidate.actions
            else ""
        )
        for candidate in queries[0].candidates
    )
    histories = []
    controls = []
    observed = []
    action_batches = []
    for query in queries:
        if (
            tuple(
                candidate.candidate_id
                for candidate in query.candidates
            )
            != candidate_ids
        ):
            raise ValueError("attribution candidate order drifted")
        matching = np.flatnonzero(
            np.asarray(validation.trajectory_ids)
            == query.validation_case_id
        )
        matching = matching[
            validation.transition_indices[matching]
            == query.transition_index
        ]
        if len(matching) != 1:
            raise ValueError(
                "attribution query is not a unique evaluation window"
            )
        index = int(matching[0])
        source_run = run_by_id[query.validation_case_id]
        histories.append(validation.histories[index])
        controls.append(validation.future_controls[index])
        observed.append(validation.future_states[index])
        action_batches.append(
            np.stack(
                [
                    compiler.compile_action_trajectory(
                        point_count=source_run.manifest.point_count,
                        actions=candidate.actions,
                        graph=validation.graph,
                    )[
                        query.transition_index : query.transition_index
                        + validation.future_states.shape[1]
                    ]
                    for candidate in query.candidates
                ],
                axis=0,
            )
        )
    return PreparedAttributionQueries(
        query_ids=tuple(query.query_id for query in queries),
        histories=np.asarray(histories, dtype=np.float32),
        future_controls=np.asarray(controls, dtype=np.float32),
        observed_future=np.asarray(observed, dtype=np.float32),
        candidate_actions=np.asarray(action_batches, dtype=np.float32),
        candidate_ids=candidate_ids,
        candidate_action_kinds=candidate_kinds,
        candidate_target_entities=candidate_targets,
        expected_action_kinds=tuple(
            query.expected_action_kind or "" for query in queries
        ),
        expected_target_entities=tuple(
            query.expected_target_entity or "" for query in queries
        ),
        expected_variant_ids=tuple(
            query.expected_variant_candidate_id or ""
            for query in queries
        ),
    )


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
