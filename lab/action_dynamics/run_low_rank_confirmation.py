"""Prepare, collect, qualify, and score sealed low-rank confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from quantis_core.action_conditioned_dynamics import (
    ActionTrajectoryCompiler,
    persistence_rollout,
)
from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)
from quantis_core.action_dynamics_lab import (
    LabActionCaptureManifest,
    write_action_collection_assessment,
    write_prepared_action_collection,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
)
from quantis_core.low_rank_confirmation import (
    LowRankConfirmationContract,
)

from assess_low_rank_confirmation import (
    assess_stored_low_rank_confirmation,
)
from collect_pilot import collect_action_cases
from run_development import _require_same_build
from run_lab_pilot import (
    RUNTIME_IMAGES,
    _canonical_sha256,
    _observation_schema,
)
from run_richer_regime_retry import _stack_identity


def preflight_confirmation(repository: Path) -> Mapping[str, Any]:
    """Verify every frozen source, model, and preprocessing identity."""

    root = Path(repository).resolve()
    contract_path = (
        root
        / "lab"
        / "action_dynamics"
        / "low-rank-confirmation-contract-v3.json"
    )
    contract = LowRankConfirmationContract.from_dict(
        _read_object(contract_path)
    )
    source_commit = _git_source_commit(root)
    source_clean = _git_source_is_clean(root)
    base = _read_object(
        root
        / str(
            contract.payload["base_collection_protocol"]["path"]
        )
    )
    protocol = contract.materialize_collection_protocol(
        base, execution_source_commit=source_commit
    )
    candidate = cast(
        Mapping[str, Any], contract.payload["candidate"]
    )
    checks: Dict[str, bool] = {
        "base_protocol": (
            _canonical_sha256(base) == contract.base_protocol_sha256
        ),
        "execution_source_clean": source_clean,
    }
    for name, path_key, hash_key in (
        (
            "candidate_model",
            "model_path",
            "model_sha256",
        ),
        (
            "development_artifact_manifest",
            "development_artifact_manifest_path",
            "development_artifact_manifest_sha256",
        ),
        (
            "compiler_metadata",
            "compiler_metadata_path",
            "compiler_metadata_sha256",
        ),
        (
            "compiler_artifact_manifest",
            "compiler_artifact_manifest_path",
            "compiler_artifact_manifest_sha256",
        ),
    ):
        path = root / str(candidate[path_key])
        checks[name] = (
            path.is_file()
            and _file_sha256(path) == candidate[hash_key]
        )
    execution = contract.payload.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("confirmation execution identities are absent")
    for name, value in execution.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, Mapping)
        ):
            raise ValueError("confirmation execution identity is invalid")
        path = root / str(value.get("path"))
        checks[f"execution:{name}"] = (
            path.is_file()
            and _file_sha256(path) == value.get("sha256")
        )
    model = ContractiveLowRankDynamics.from_dict(
        _read_object(root / str(candidate["model_path"]))
    )
    checks["model_kind"] = model.kind == candidate["kind"]
    checks["model_rank"] = model.config.rank == candidate["rank"]
    checks["model_radius"] = (
        model.spectral_radius
        <= float(candidate["maximum_spectral_radius"])
    )
    checks["fresh_protocol"] = (
        protocol.generator_seed == contract.payload["generator_seed"]
    )
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "decision": (
            "prepare_sealed_confirmation"
            if all(checks.values())
            else "repair_frozen_confirmation_contract"
        ),
        "checks": checks,
        "contract_sha256": _file_sha256(contract_path),
        "model_sha256": candidate["model_sha256"],
        "materialized_collection_protocol_sha256": (
            protocol.canonical_sha256()
        ),
        "execution_source_commit": source_commit,
    }


def prepare_confirmation(
    *,
    repository: Path,
    smoke_evidence: Path,
    pilot_evidence: Path,
    output: Path,
) -> Mapping[str, Any]:
    """Materialize the fresh campaign without collecting a case."""

    root = Path(repository).resolve()
    preflight = preflight_confirmation(root)
    if preflight["status"] != "go":
        raise ValueError("low-rank confirmation preflight failed")
    if output.exists():
        raise FileExistsError("low-rank confirmation output exists")
    lab = root / "lab" / "action_dynamics"
    contract = LowRankConfirmationContract.from_dict(
        _read_object(
            lab / "low-rank-confirmation-contract-v3.json"
        )
    )
    source_commit = _git_source_commit(root)
    if not _git_source_is_clean(root):
        raise ValueError("confirmation preparation requires clean source")
    protocol = contract.materialize_collection_protocol(
        _read_object(lab / "development-protocol-v1.json"),
        execution_source_commit=source_commit,
    )
    application_image, build_context = _stack_identity(
        lab, prepare_only=True, reuse_built_image=True
    )
    _require_same_build(build_context, smoke_evidence, pilot_evidence)
    schema = _observation_schema(lab)
    write_prepared_action_collection(
        protocol,
        output / "inputs",
        image_digests={
            **RUNTIME_IMAGES,
            "application": application_image,
        },
        observation_schema_sha256=_canonical_sha256(schema),
        application_build_context_sha256=build_context,
        qualifying_smoke_directory=smoke_evidence,
        qualifying_pilot_directory=pilot_evidence,
    )
    _write_json(output / "observation-schema.json", schema)
    result = {
        **dict(preflight),
        "status": "prepared",
        "decision": "collect_fresh_confirmation_once",
        "pair_count": 120,
        "capture_count": 240,
        "application_image_digest": application_image,
        "application_build_context_sha256": build_context,
        "execution_source_commit": source_commit,
    }
    _write_json(output / "prepared-result.json", result)
    return result


def collect_confirmation(
    *, repository: Path, output: Path
) -> Mapping[str, Any]:
    """Collect all prepared pairs once with no automatic retry."""

    root = Path(repository).resolve()
    _require_prepared_source(root, output)
    captures = output / "cases"
    if captures.exists():
        raise FileExistsError("low-rank confirmation captures exist")
    manifest_paths = sorted(
        (output / "inputs" / "manifests").glob("*.json")
    )
    if len(manifest_paths) != 240:
        raise ValueError("prepared confirmation coverage is incomplete")
    manifest = LabActionCaptureManifest.from_dict(
        _read_object(manifest_paths[0])
    )
    return collect_action_cases(
        protocol_path=output / "inputs" / "protocol.json",
        plan_path=output / "inputs" / "plan.json",
        manifests_directory=output / "inputs" / "manifests",
        captures_directory=captures,
        compose_file=(
            root / "lab" / "action_dynamics" / "compose.yaml"
        ),
        project_prefix="quantis-low-rank-confirmation-v3",
        application_image_id=manifest.image_digests[
            "application"
        ],
        application_build_context_sha256=str(
            _read_object(output / "inputs" / "plan.json")[
                "application_build_context_sha256"
            ]
        ),
        parallel_jobs=6,
        attestation_path=output / "collection-attestation.json",
    )


def qualify_confirmation(
    *, repository: Path, output: Path
) -> Mapping[str, Any]:
    """Recompute and retain every raw collection qualification gate."""

    _require_prepared_source(Path(repository).resolve(), output)
    result = write_action_collection_assessment(
        output / "inputs",
        output / "cases",
        output / "collection-attestation.json",
        output,
    )
    if result.get("status") != "qualified":
        raise ValueError("fresh confirmation corpus did not qualify")
    return result


def score_confirmation(
    *, repository: Path, output: Path
) -> Mapping[str, Any]:
    """Restore the frozen model and score the qualified campaign once."""

    root = Path(repository).resolve()
    if preflight_confirmation(root)["status"] != "go":
        raise ValueError("low-rank confirmation scoring preflight failed")
    contract_path = (
        root
        / "lab"
        / "action_dynamics"
        / "low-rank-confirmation-contract-v3.json"
    )
    contract = LowRankConfirmationContract.from_dict(
        _read_object(contract_path)
    )
    stored_protocol = _read_object(
        output / "inputs" / "protocol.json"
    )
    raw_claim = stored_protocol.get("claim")
    source_commit = (
        raw_claim.get("execution_source_commit")
        if isinstance(raw_claim, Mapping)
        else None
    )
    if (
        not isinstance(source_commit, str)
        or source_commit != _git_source_commit(root)
        or not _git_source_is_clean(root)
    ):
        raise ValueError("scoring source differs from collected source")
    expected_protocol = contract.materialize_collection_protocol(
        _read_object(
            root
            / str(
                contract.payload["base_collection_protocol"]["path"]
            )
        ),
        execution_source_commit=source_commit,
    )
    if stored_protocol != expected_protocol.to_dict():
        raise ValueError("collected confirmation protocol drifted")
    corpus = load_action_dynamics_development_corpus(output)
    candidate = cast(
        Mapping[str, Any], contract.payload["candidate"]
    )
    compiler_metadata = _read_object(
        root / str(candidate["compiler_metadata_path"])
    )
    compiler_payload = compiler_metadata.get("compiler")
    if not isinstance(compiler_payload, Mapping):
        raise ValueError("frozen compiler artifact is absent")
    compiler = ActionTrajectoryCompiler.from_dict(compiler_payload)
    windows = compiler.transform(corpus.runs)
    model_path = root / str(candidate["model_path"])
    model = ContractiveLowRankDynamics.from_dict(
        _read_object(model_path)
    )
    neutral_actions = np.zeros_like(windows.future_actions)
    neutral_actions[..., 0] = 1.0
    predictions: Dict[str, NDArray[np.float64]] = {
        "candidate": model.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean,
        "action_masked": model.rollout(
            windows.histories,
            windows.future_controls,
            neutral_actions,
            windows.graph,
        ).mean,
        "persistence": persistence_rollout(
            windows.histories, windows.future_states.shape[1]
        ).mean,
    }
    action_kind_by_pair = {
        run.manifest.matched_pair_id: run.manifest.actions[0].action_kind
        for run in corpus.runs
        if run.manifest.actions
    }
    if len(action_kind_by_pair) != 120:
        raise ValueError("confirmation action-family mapping is incomplete")
    predictions_directory = output / "predictions"
    if predictions_directory.exists():
        raise FileExistsError("confirmation predictions already exist")
    predictions_directory.mkdir()
    np.savez_compressed(
        predictions_directory / "confirmation-inputs.npz",
        observed=windows.future_states,
        future_actions=windows.future_actions,
        trajectory_ids=np.asarray(windows.trajectory_ids),
        matched_pair_ids=np.asarray(windows.matched_pair_ids),
        transition_indices=windows.transition_indices,
        action_kind_by_pair_json=np.asarray(
            json.dumps(action_kind_by_pair, sort_keys=True)
        ),
    )
    for name, prediction in predictions.items():
        np.savez_compressed(
            predictions_directory / f"{name}.npz",
            prediction=prediction,
        )
    source_manifest = output / "artifact-manifest.json"
    prediction_manifest = {
        "schema_version": 1,
        "kind": "low_rank_confirmation_prediction_manifest",
        "contract_sha256": _file_sha256(contract_path),
        "model_sha256": _file_sha256(model_path),
        "source_artifact_manifest_sha256": _file_sha256(
            source_manifest
        ),
        "sha256": {
            path.name: _file_sha256(path)
            for path in sorted(predictions_directory.glob("*.npz"))
        },
    }
    _write_json(
        predictions_directory / "prediction-manifest.json",
        prediction_manifest,
    )
    result = assess_stored_low_rank_confirmation(
        contract_path=contract_path,
        model_path=model_path,
        source_artifact_manifest=source_manifest,
        predictions_directory=predictions_directory,
        expected_contract_sha256=_file_sha256(contract_path),
        expected_model_sha256=_file_sha256(model_path),
        expected_source_artifact_manifest_sha256=_file_sha256(
            source_manifest
        ),
        expected_prediction_manifest_sha256=_file_sha256(
            predictions_directory / "prediction-manifest.json"
        ),
    )
    _write_json(output / "confirmation-assessment.json", result)
    retained = {
        path.relative_to(output).as_posix(): _file_sha256(path)
        for path in sorted(
            [
                output / "artifact-manifest.json",
                output / "data-quality.json",
                output / "confirmation-assessment.json",
                *predictions_directory.glob("*"),
            ]
        )
    }
    _write_json(
        output / "confirmation-artifact-manifest.json",
        {
            "schema_version": 1,
            "kind": "low_rank_confirmation_artifact_manifest",
            "sha256": retained,
        },
    )
    return result


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_source_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_source_is_clean(repository: Path) -> bool:
    return (
        subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def _require_prepared_source(repository: Path, output: Path) -> str:
    protocol = _read_object(output / "inputs" / "protocol.json")
    claim = protocol.get("claim")
    source_commit = (
        claim.get("execution_source_commit")
        if isinstance(claim, Mapping)
        else None
    )
    if (
        not isinstance(source_commit, str)
        or source_commit != _git_source_commit(repository)
        or not _git_source_is_clean(repository)
    ):
        raise ValueError(
            "execution source differs from prepared confirmation"
        )
    return source_commit


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preflight", "prepare", "collect", "qualify", "score"),
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--smoke-evidence",
        type=Path,
        default=Path("artifacts/action-dynamics/lab-smoke-v4"),
    )
    parser.add_argument(
        "--pilot-evidence",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/instrumentation-pilot-v4"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "low-rank-confirmation-v3-attempt-001"
        ),
    )
    parsed = parser.parse_args(arguments)
    if parsed.command == "preflight":
        result = preflight_confirmation(parsed.repository)
    elif parsed.command == "prepare":
        result = prepare_confirmation(
            repository=parsed.repository,
            smoke_evidence=parsed.smoke_evidence,
            pilot_evidence=parsed.pilot_evidence,
            output=parsed.output,
        )
    elif parsed.command == "collect":
        result = collect_confirmation(
            repository=parsed.repository, output=parsed.output
        )
    elif parsed.command == "qualify":
        result = qualify_confirmation(
            repository=parsed.repository, output=parsed.output
        )
    else:
        result = score_confirmation(
            repository=parsed.repository, output=parsed.output
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if parsed.command == "score":
        return 0 if result.get("status") == "confirmed" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
