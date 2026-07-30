"""Prepare, collect, qualify, and score the frozen MPRM selection campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedRun,
    ActionTrajectoryCompiler,
    MixtureTrajectoryDistribution,
)
from quantis_core.action_dynamics_corpus import (
    CONTROL_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    _development_graph,
    _observations,
)
from quantis_core.action_dynamics_lab import (
    ActionCollectionProtocol,
    LabActionCaptureManifest,
    assess_action_pair_metric_series,
    load_action_case_metric_series,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
)
from quantis_core.mprm_jepa import (
    MprmJepaProtocol,
    build_mprm_selection_plan,
    prepare_mprm_selection_campaign,
    qualify_mprm_selection_campaign,
)

from assess_mprm_jepa import assess_stored_mprm_selection
from collect_pilot import collect_action_cases
from mprm_jepa_model import MeanPreservingResidualJepa
from prototype_multi_hypothesis_jepa import (
    MultiHypothesisJepaPrototype,
)
from run_lab_pilot import (
    APPLICATION_IMAGE,
    RUNTIME_IMAGES,
    _canonical_sha256,
    _observation_schema,
)
from run_mprm_jepa import MODEL_NAMES, preflight_mprm_jepa
from run_richer_regime_retry import _stack_identity


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


def prepare_selection(
    *,
    repository: Path,
    model_freeze_directory: Path,
    output: Path,
    reuse_built_image: bool = False,
) -> Mapping[str, Any]:
    """Prepare all 180 manifests without collecting any case."""

    root = Path(repository).resolve()
    if preflight_mprm_jepa(root)["status"] != "go":
        raise ValueError("MPRM-JEPA repository preflight failed")
    if output.exists():
        raise FileExistsError("MPRM-JEPA selection output exists")
    model_freeze = model_freeze_directory / "model-freeze-manifest.json"
    if not model_freeze.is_file():
        raise ValueError("MPRM-JEPA model freeze is absent")
    lab = root / "lab" / "action_dynamics"
    protocol_path = lab / "mprm-jepa-protocol-v1.json"
    protocol = MprmJepaProtocol.from_dict(_read_object(protocol_path))
    plan = build_mprm_selection_plan(protocol)
    application_image, build_context = _stack_identity(
        lab,
        prepare_only=True,
        reuse_built_image=reuse_built_image,
    )
    schema = _observation_schema(lab)
    action_protocol_path = lab / "development-protocol-v1.json"
    collector_digest = (
        "sha256:" + RUNTIME_IMAGES["collector"].rsplit("sha256:", 1)[1]
    )
    bindings = {
        "candidate_protocol_sha256": _file_sha256(protocol_path),
        "model_freeze_manifest_sha256": _file_sha256(model_freeze),
        "action_protocol_sha256": _file_sha256(action_protocol_path),
        "observation_schema_sha256": _canonical_sha256(schema),
        "application_build_context_sha256": build_context,
        "application_image_digest": application_image,
        "collector_image_digest": collector_digest,
        "attempt_id": "mprm-jepa-selection-v1-attempt-001",
    }
    prepared = prepare_mprm_selection_campaign(
        protocol,
        plan,
        action_library=_read_object(action_protocol_path)[
            "action_library"
        ],
        bindings=bindings,
    )
    inputs = output / "inputs"
    manifests_directory = inputs / "manifests"
    manifests_directory.mkdir(parents=True)
    _write_json(inputs / "campaign-protocol.json", protocol.to_dict())
    _write_json(inputs / "campaign-plan.json", plan)
    _write_json(
        inputs / "protocol.json", prepared["executor_protocol"]
    )
    _write_json(inputs / "plan.json", prepared["executor_plan"])
    _write_json(output / "observation-schema.json", schema)
    for case_id, manifest in prepared["manifests"].items():
        _write_json(manifests_directory / f"{case_id}.json", manifest)
    result = {
        "status": "prepared",
        "decision": "collect_complete_selection_campaign",
        "pair_count": 90,
        "capture_count": 180,
        "campaign_bindings": bindings,
    }
    _write_json(output / "prepared-result.json", result)
    return result


def collect_selection(
    *, repository: Path, output: Path
) -> Mapping[str, Any]:
    """Collect the prepared campaign once with no automatic retry."""

    root = Path(repository).resolve()
    if (output / "captures").exists():
        raise FileExistsError("MPRM-JEPA captures already exist")
    inputs = output / "inputs"
    protocol = _read_object(inputs / "protocol.json")
    binding = protocol["campaign_bindings"]
    result = collect_action_cases(
        protocol_path=inputs / "protocol.json",
        plan_path=inputs / "plan.json",
        manifests_directory=inputs / "manifests",
        captures_directory=output / "captures",
        compose_file=root / "lab" / "action_dynamics" / "compose.yaml",
        project_prefix="quantis-mprm-selection-v1",
        application_image_id=binding["application_image_digest"],
        application_build_context_sha256=binding[
            "application_build_context_sha256"
        ],
        parallel_jobs=6,
        attestation_path=output / "collection-attestation.json",
    )
    return cast(Mapping[str, Any], result)


def qualify_selection(
    *, repository: Path, output: Path
) -> Mapping[str, Any]:
    """Recompute raw action gates and qualify the complete campaign."""

    root = Path(repository).resolve()
    protocol = MprmJepaProtocol.from_dict(
        _read_object(
            root
            / "lab"
            / "action_dynamics"
            / "mprm-jepa-protocol-v1.json"
        )
    )
    plan = _read_object(output / "inputs" / "campaign-plan.json")
    manifests = {
        path.stem: _read_object(path)
        for path in sorted(
            (output / "inputs" / "manifests").glob("*.json")
        )
    }
    action_protocol_path = (
        root
        / "lab"
        / "action_dynamics"
        / "development-protocol-v1.json"
    )
    action_protocol = ActionCollectionProtocol.from_dict(
        _read_object(action_protocol_path)
    )
    bound_action_hash = next(iter(manifests.values()))[
        "campaign_bindings"
    ]["action_protocol_sha256"]
    if _file_sha256(action_protocol_path) != bound_action_hash:
        raise ValueError("MPRM-JEPA bound action protocol drifted")
    typed_manifests = tuple(
        LabActionCaptureManifest.from_dict(
            {key: manifest[key] for key in _LAB_MANIFEST_KEYS}
        )
        for manifest in manifests.values()
    )
    metric_series = {
        manifest.action_case.case_id: load_action_case_metric_series(
            output / "captures" / manifest.action_case.case_id,
            manifest,
        )
        for manifest in typed_manifests
    }
    pair_rows = assess_action_pair_metric_series(
        action_protocol, typed_manifests, metric_series
    )
    pair_assessments = {
        str(row["pair_id"]): dict(row) for row in pair_rows
    }
    captures = {
        case_id: {
            "capture_manifest_sha256": _canonical_sha256(
                _read_object(
                    output
                    / "captures"
                    / case_id
                    / "capture-manifest.json"
                )
            ),
            "runner_log_sha256": _file_sha256(
                output / "captures" / case_id / "runner.log"
            ),
            "metrics_sha256": _file_sha256(
                output
                / "captures"
                / case_id
                / "collector-metrics.jsonl"
            ),
            "logs_sha256": _file_sha256(
                output
                / "captures"
                / case_id
                / "collector-logs.jsonl"
            ),
            "traces_sha256": _file_sha256(
                output
                / "captures"
                / case_id
                / "collector-traces.jsonl"
            ),
            "actions_sha256": _file_sha256(
                output
                / "captures"
                / case_id
                / "collector-actions.jsonl"
            ),
        }
        for case_id in manifests
    }
    qualified = qualify_mprm_selection_campaign(
        protocol,
        plan,
        manifests,
        captures,
        _read_object(output / "collection-attestation.json"),
        pair_assessments,
        _read_object(action_protocol_path)["action_library"],
    )
    _write_json(output / "qualified-corpus.json", qualified)
    return cast(Mapping[str, Any], qualified)


def score_selection(
    *,
    repository: Path,
    model_freeze_directory: Path,
    output: Path,
) -> Mapping[str, Any]:
    """Restore frozen models only after qualification and score once."""

    root = Path(repository).resolve()
    qualified_path = output / "qualified-corpus.json"
    qualified = _read_object(qualified_path)
    if qualified.get("status") != "qualified":
        raise ValueError("MPRM-JEPA selection corpus is not qualified")
    _verify_qualified_sources(output, qualified)
    bound = qualified.get("campaign_bindings")
    model_freeze_path = (
        model_freeze_directory / "model-freeze-manifest.json"
    )
    protocol_path = (
        root / "lab" / "action_dynamics" / "mprm-jepa-protocol-v1.json"
    )
    if (
        not isinstance(bound, dict)
        or bound.get("model_freeze_manifest_sha256")
        != _file_sha256(model_freeze_path)
        or bound.get("candidate_protocol_sha256")
        != _file_sha256(protocol_path)
    ):
        raise ValueError("MPRM-JEPA scoring identity differs from collection")
    manifests = {
        path.stem: _read_object(path)
        for path in sorted(
            (output / "inputs" / "manifests").glob("*.json")
        )
    }
    runs = []
    family_by_pair: Dict[str, str] = {}
    for case_id, raw_manifest in manifests.items():
        manifest = LabActionCaptureManifest.from_dict(
            {
                key: raw_manifest[key]
                for key in _LAB_MANIFEST_KEYS
            }
        )
        family_by_pair[
            manifest.action_case.matched_pair_id
        ] = str(raw_manifest["workload_family"])
        observations = _observations(
            output / "captures" / case_id,
            manifest,
            _development_graph(),
        )
        controls = np.column_stack(
            (
                np.asarray(manifest.request_schedule, dtype=np.float64),
                np.full(
                    manifest.action_case.point_count,
                    float(manifest.action_case.worker_replicas),
                ),
            )
        )
        runs.append(
            ActionConditionedRun(
                manifest=manifest.action_case,
                graph=_development_graph(),
                observations=observations,
                controls=controls,
                state_feature_names=STATE_FEATURE_NAMES,
                control_feature_names=CONTROL_FEATURE_NAMES,
            )
        )
    compiler = ActionTrajectoryCompiler.from_dict(
        _read_object(model_freeze_directory / "compiler-artifact.json")
    )
    windows = compiler.transform(tuple(runs))
    models = model_freeze_directory / "models"
    anchor = ContractiveLowRankDynamics.from_dict(
        _read_object(models / "raw_rank_32_predictive_core.json")
    )
    raw_distribution = anchor.rollout(
        windows.histories,
        windows.future_controls,
        windows.future_actions,
        windows.graph,
    )
    distributions: Dict[str, MixtureTrajectoryDistribution] = {
        "raw_rank_32_predictive_core": MixtureTrajectoryDistribution(
            component_mean=raw_distribution.mean[:, None],
            component_variance=raw_distribution.variance[:, None],
            weight=np.ones(
                (len(raw_distribution.mean), 1), dtype=np.float64
            ),
        )
    }
    for name in (
        "one_component_anchored_jepa_residual",
        "supervised_four_component_mean_preserving_residual_mixture",
        "capacity_matched_anchored_single_gaussian",
        "mprm_jepa_candidate",
    ):
        model = cast(
            MeanPreservingResidualJepa,
            MeanPreservingResidualJepa.load(models, name),
        )
        distributions[name] = model.rollout_anchored(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            raw_distribution.mean,
        )
    unanchored = MultiHypothesisJepaPrototype.load(
        models, "unanchored_four_component_jepa_diagnostic"
    )
    distributions["unanchored_four_component_jepa_diagnostic"] = (
        unanchored.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
        )
    )
    predictions = output / "predictions"
    if predictions.exists():
        raise FileExistsError("MPRM-JEPA predictions already exist")
    predictions.mkdir()
    active = np.any(windows.future_actions[..., 1] > 0.5, axis=2)
    sample_ids = tuple(
        f"{trajectory}:{int(index)}"
        for trajectory, index in zip(
            windows.trajectory_ids, windows.transition_indices
        )
    )
    np.savez_compressed(
        predictions / "selection-inputs.npz",
        observed=windows.future_states,
        action_active=np.broadcast_to(
            active[..., None, None], windows.future_states.shape
        ),
        trajectory_ids=np.asarray(windows.trajectory_ids),
        matched_pair_ids=np.asarray(windows.matched_pair_ids),
        sample_ids=np.asarray(sample_ids),
        pair_workload_families_json=np.asarray(
            json.dumps(family_by_pair, sort_keys=True)
        ),
    )
    for name in MODEL_NAMES:
        distribution = distributions[name]
        np.savez_compressed(
            predictions / f"selection-{name}.npz",
            component_mean=distribution.component_mean,
            component_variance=distribution.component_variance,
            weight=distribution.weight,
        )
    prediction_manifest = {
        "schema_version": 1,
        "kind": "mprm_jepa_prediction_manifest",
        "model_freeze_manifest_sha256": _file_sha256(
            model_freeze_path
        ),
        "qualified_corpus_sha256": qualified[
            "qualified_corpus_sha256"
        ],
        "sha256": {
            path.name: _file_sha256(path)
            for path in sorted(predictions.glob("*.npz"))
        },
    }
    _write_json(
        predictions / "prediction-manifest.json",
        prediction_manifest,
    )
    model_freeze = model_freeze_path
    result = assess_stored_mprm_selection(
        protocol_path=(
            protocol_path
        ),
        model_freeze_manifest=model_freeze,
        qualified_corpus=qualified_path,
        predictions_directory=predictions,
        expected_model_freeze_sha256=_file_sha256(model_freeze),
        expected_qualified_corpus_sha256=str(
            qualified["qualified_corpus_sha256"]
        ),
        expected_prediction_manifest_sha256=_file_sha256(
            predictions / "prediction-manifest.json"
        ),
    )
    _write_json(output / "selection-assessment.json", result)
    return result


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_qualified_sources(
    output: Path, qualified: Mapping[str, Any]
) -> None:
    source = qualified.get("source_content_manifest")
    captures = source.get("capture_sha256") if isinstance(source, dict) else None
    manifests = (
        source.get("manifest_sha256") if isinstance(source, dict) else None
    )
    names = {
        "capture_manifest_sha256": "capture-manifest.json",
        "runner_log_sha256": "runner.log",
        "metrics_sha256": "collector-metrics.jsonl",
        "logs_sha256": "collector-logs.jsonl",
        "traces_sha256": "collector-traces.jsonl",
        "actions_sha256": "collector-actions.jsonl",
    }
    if not isinstance(captures, dict):
        raise ValueError("MPRM-JEPA qualified sources are absent")
    manifest_directory = output / "inputs" / "manifests"
    manifest_paths = {
        path.stem: path
        for path in manifest_directory.glob("*.json")
    }
    if (
        not isinstance(manifests, dict)
        or set(manifest_paths) != set(manifests)
        or any(
            _canonical_sha256(_read_object(manifest_paths[case_id]))
            != expected
            for case_id, expected in manifests.items()
        )
    ):
        raise ValueError("MPRM-JEPA prepared manifest drifted")
    for case_id, evidence in captures.items():
        if not isinstance(evidence, dict):
            raise ValueError("MPRM-JEPA qualified source entry is invalid")
        for key, filename in names.items():
            path = output / "captures" / case_id / filename
            actual = (
                _canonical_sha256(_read_object(path))
                if key == "capture_manifest_sha256"
                else _file_sha256(path)
            )
            if actual != evidence.get(key):
                raise ValueError("MPRM-JEPA qualified source drifted")


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "collect", "qualify", "score")
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--model-freeze-directory",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/mprm-jepa-model-freeze-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/mprm-jepa-selection-v1-attempt-001"
        ),
    )
    parser.add_argument("--reuse-built-image", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.command == "prepare":
        result = prepare_selection(
            repository=parsed.repository,
            model_freeze_directory=parsed.model_freeze_directory,
            output=parsed.output,
            reuse_built_image=parsed.reuse_built_image,
        )
    elif parsed.command == "collect":
        result = collect_selection(
            repository=parsed.repository, output=parsed.output
        )
    elif parsed.command == "qualify":
        result = qualify_selection(
            repository=parsed.repository, output=parsed.output
        )
    else:
        result = score_selection(
            repository=parsed.repository,
            model_freeze_directory=parsed.model_freeze_directory,
            output=parsed.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
