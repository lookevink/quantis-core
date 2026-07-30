"""Load role-isolated richer-regime shards into action windows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .action_conditioned_dynamics import (
    ActionConditionedRun,
    ActionConditionedWindows,
    ActionTrajectoryCompiler,
)
from .action_dynamics_corpus import (
    CONTROL_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    _development_graph,
    _observations,
)
from .action_dynamics_lab import LabActionCaptureManifest
from .richer_regime_retry import WORKLOAD_FAMILIES


_LAB_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "action_case",
    "sample_period_seconds",
    "request_schedule",
    "api_request_queue_size",
    "image_digests",
    "observation_schema_sha256",
    "protocol_sha256",
    "prepared_plan_sha256",
    "graph_observation_schema_sha256",
    "corpus_role",
}


@dataclass(frozen=True)
class LoadedRicherRegimeWindows:
    """Normalized windows and exact shard identities by evidence role."""

    windows: Mapping[str, ActionConditionedWindows]
    runs: Mapping[str, Tuple[ActionConditionedRun, ...]]
    source_assessment_sha256s: Mapping[str, Mapping[str, str]]
    compiler_artifact: Mapping[str, Any]


def load_richer_regime_windows(
    role_directories: Mapping[str, Path],
) -> LoadedRicherRegimeWindows:
    """Load qualified role roots without allowing cross-role fallback."""

    if (
        set(role_directories) not in ({"fit"}, {"fit", "selection"})
        or any(not Path(path).is_dir() for path in role_directories.values())
    ):
        raise ValueError("richer-regime role directories are invalid")
    runs_by_role: Dict[str, Tuple[ActionConditionedRun, ...]] = {}
    hashes_by_role: Dict[str, Mapping[str, str]] = {}
    for role in role_directories:
        root = Path(role_directories[role])
        role_runs = []
        family_hashes = {}
        for family in WORKLOAD_FAMILIES:
            shard = root / role / family
            assessment_path = shard / "shard-assessment.json"
            assessment = _read_object(assessment_path)
            if (
                assessment.get("status") != "qualified"
                or assessment.get("corpus_role") != role
                or assessment.get("workload_family") != family
            ):
                raise ValueError(
                    f"richer-regime shard is not qualified: "
                    f"{role}/{family}"
                )
            family_hashes[family] = _file_sha256(assessment_path)
            manifests = shard / "inputs" / "manifests"
            captures = shard / "cases"
            prepared_ids = {
                path.stem for path in manifests.glob("*.json")
            }
            captured_ids = {
                path.name
                for path in captures.iterdir()
                if path.is_dir()
            }
            if prepared_ids != captured_ids or not prepared_ids:
                raise ValueError(
                    "richer-regime prepared and captured cases differ"
                )
            for case_id in sorted(prepared_ids):
                raw = _read_object(manifests / f"{case_id}.json")
                captured = _read_object(
                    captures / case_id / "capture-manifest.json"
                )
                if raw != captured:
                    raise ValueError(
                        "richer-regime captured manifest drifted"
                    )
                if (
                    raw.get("retry_corpus_role") != role
                    or raw.get("workload_family") != family
                ):
                    raise ValueError(
                        "richer-regime manifest role drifted"
                    )
                manifest = LabActionCaptureManifest.from_dict(
                    {key: raw[key] for key in _LAB_MANIFEST_KEYS}
                )
                observations = _observations(
                    captures / case_id,
                    manifest,
                    _development_graph(),
                )
                controls = np.column_stack(
                    (
                        np.asarray(
                            manifest.request_schedule,
                            dtype=np.float64,
                        ),
                        np.full(
                            manifest.action_case.point_count,
                            float(
                                manifest.action_case.worker_replicas
                            ),
                            dtype=np.float64,
                        ),
                    )
                )
                role_runs.append(
                    ActionConditionedRun(
                        manifest=manifest.action_case,
                        graph=_development_graph(),
                        observations=observations,
                        controls=controls,
                        state_feature_names=STATE_FEATURE_NAMES,
                        control_feature_names=CONTROL_FEATURE_NAMES,
                    )
                )
        expected_pairs = 90 if role == "fit" else 45
        pair_ids = {
            run.manifest.matched_pair_id for run in role_runs
        }
        if (
            len(pair_ids) != expected_pairs
            or len(role_runs) != expected_pairs * 2
        ):
            raise ValueError(
                f"richer-regime {role} balance is incomplete"
            )
        runs_by_role[role] = tuple(role_runs)
        hashes_by_role[role] = family_hashes
    fit_pairs = {
        run.manifest.matched_pair_id for run in runs_by_role["fit"]
    }
    if "selection" in runs_by_role:
        selection_pairs = {
            run.manifest.matched_pair_id
            for run in runs_by_role["selection"]
        }
        if fit_pairs & selection_pairs:
            raise ValueError("richer-regime roles overlap by matched pair")
    compiler = ActionTrajectoryCompiler(
        context_length=20, rollout_horizon=10
    ).fit(runs_by_role["fit"])
    windows = {
        role: compiler.transform(runs)
        for role, runs in runs_by_role.items()
    }
    return LoadedRicherRegimeWindows(
        windows=windows,
        runs=runs_by_role,
        source_assessment_sha256s=hashes_by_role,
        compiler_artifact=compiler.to_dict(),
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
