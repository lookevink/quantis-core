import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from quantis_core.action_dynamics_lab import prepare_action_collection
from quantis_core.low_rank_confirmation import (
    LowRankConfirmationContract,
    assess_low_rank_confirmation_arrays,
    paired_sign_flip_p_value,
)


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _contract() -> LowRankConfirmationContract:
    path = (
        _repository()
        / "lab"
        / "action_dynamics"
        / "low-rank-confirmation-contract-v2.json"
    )
    return LowRankConfirmationContract.from_dict(
        json.loads(path.read_text())
    )


def test_confirmation_contract_materializes_fresh_complete_campaign() -> None:
    repository = _repository()
    base_path = (
        repository
        / "lab"
        / "action_dynamics"
        / "development-protocol-v1.json"
    )
    contract = _contract()
    protocol = contract.materialize_collection_protocol(
        json.loads(base_path.read_text()),
        execution_source_commit="a" * 40,
    )
    manifests, _ = prepare_action_collection(
        protocol,
        image_digests={
            "application": "sha256:" + "1" * 64,
            "collector": "sha256:" + "2" * 64,
        },
        observation_schema_sha256="3" * 64,
    )
    base_protocol = json.loads(base_path.read_text())
    base_manifests, _ = prepare_action_collection(
        contract.base_protocol(base_protocol),
        image_digests={
            "application": "sha256:" + "1" * 64,
            "collector": "sha256:" + "2" * 64,
        },
        observation_schema_sha256="3" * 64,
    )

    assert len(manifests) == 240
    assert len({item.action_case.matched_pair_id for item in manifests}) == 120
    assert not {
        item.action_case.matched_pair_id for item in manifests
    } & {
        item.action_case.matched_pair_id for item in base_manifests
    }
    assert protocol.claim["confirmation_contract"] == contract.to_dict()
    assert protocol.claim["execution_source_commit"] == "a" * 40


def test_confirmation_assessment_requires_action_value_and_pair_significance() -> None:
    pair_ids = tuple(f"pair-{index}" for index in range(120))
    action_families = (
        "worker_pause",
        "postgres_lock",
        "redis_enqueue_delay",
        "redis_dequeue_delay",
        "api_rejection",
    )
    action_kinds = {
        pair_id: action_families[index // 24]
        for index, pair_id in enumerate(pair_ids)
    }
    candidate = np.full(120, 0.4, dtype=np.float64)
    action_masked = np.full(120, 1.0, dtype=np.float64)
    persistence = np.full(120, 0.8, dtype=np.float64)

    result = assess_low_rank_confirmation_arrays(
        pair_ids=pair_ids,
        action_kind_by_pair=action_kinds,
        candidate_action_mse=candidate,
        action_masked_action_mse=action_masked,
        persistence_action_mse=persistence,
        candidate_downstream_effect_mse=0.2,
        action_masked_downstream_effect_mse=0.5,
        persistence_downstream_effect_mse=0.4,
        spectral_radius=0.87,
        parameter_count=34_503,
        serialized_size_bytes=860_000,
        rollout_finite=True,
        seed=26073042,
        draws=99_999,
    )

    assert result["status"] == "confirmed"
    assert result["decision"] == "confirm_learnable_action_dynamics"
    assert all(result["gates"].values())

    failed = assess_low_rank_confirmation_arrays(
        pair_ids=pair_ids,
        action_kind_by_pair=action_kinds,
        candidate_action_mse=np.full(120, 0.95),
        action_masked_action_mse=action_masked,
        persistence_action_mse=persistence,
        candidate_downstream_effect_mse=0.39,
        action_masked_downstream_effect_mse=0.5,
        persistence_downstream_effect_mse=0.4,
        spectral_radius=0.87,
        parameter_count=34_503,
        serialized_size_bytes=860_000,
        rollout_finite=True,
        seed=26073042,
        draws=99_999,
    )
    assert failed["status"] == "not_confirmed"
    assert not all(failed["gates"].values())


def test_sign_flip_test_is_deterministic_and_rejects_wrong_contract() -> None:
    differences = np.full(20, -0.5, dtype=np.float64)

    first = paired_sign_flip_p_value(
        differences, seed=26073042, draws=99_999
    )
    second = paired_sign_flip_p_value(
        differences, seed=26073042, draws=99_999
    )

    assert first == second
    assert first <= 0.05
    with pytest.raises(ValueError, match="contract"):
        paired_sign_flip_p_value(
            differences, seed=26073042, draws=999
        )


def test_confirmation_assessment_rejects_incomplete_campaign() -> None:
    pair_ids = tuple(f"pair-{index}" for index in range(20))
    with pytest.raises(ValueError, match="pair identities"):
        assess_low_rank_confirmation_arrays(
            pair_ids=pair_ids,
            action_kind_by_pair={
                pair_id: "worker_pause" for pair_id in pair_ids
            },
            candidate_action_mse=np.full(20, 0.1),
            action_masked_action_mse=np.full(20, 1.0),
            persistence_action_mse=np.full(20, 1.0),
            candidate_downstream_effect_mse=0.1,
            action_masked_downstream_effect_mse=1.0,
            persistence_downstream_effect_mse=1.0,
            spectral_radius=0.87,
            parameter_count=34_503,
            serialized_size_bytes=860_000,
            rollout_finite=True,
            seed=26073042,
            draws=99_999,
        )


def test_contract_rejects_base_protocol_drift() -> None:
    contract = _contract()
    base_path = (
        _repository()
        / "lab"
        / "action_dynamics"
        / "development-protocol-v1.json"
    )
    base = json.loads(base_path.read_text())
    base["generator_seed"] += 1
    assert hashlib.sha256(
        json.dumps(
            base, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest() != contract.base_protocol_sha256

    with pytest.raises(ValueError, match="base protocol"):
        contract.materialize_collection_protocol(
            base, execution_source_commit="a" * 40
        )


def test_independent_assessor_rejects_model_not_bound_to_contract(
    tmp_path: Path,
) -> None:
    repository = _repository()
    contract_path = (
        repository
        / "lab"
        / "action_dynamics"
        / "low-rank-confirmation-contract-v2.json"
    )
    model = tmp_path / "different-model.json"
    source = tmp_path / "artifact-manifest.json"
    predictions = tmp_path / "predictions"
    prediction_manifest = predictions / "prediction-manifest.json"
    model.write_text("{}\n")
    source.write_text("{}\n")
    predictions.mkdir()
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    prediction_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "low_rank_confirmation_prediction_manifest",
                "contract_sha256": hashlib.sha256(
                    contract_path.read_bytes()
                ).hexdigest(),
                "model_sha256": model_sha,
                "source_artifact_manifest_sha256": source_sha,
                "sha256": {},
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(
                repository
                / "lab"
                / "action_dynamics"
                / "assess_low_rank_confirmation.py"
            ),
            "--contract",
            str(contract_path),
            "--model",
            str(model),
            "--source-artifact-manifest",
            str(source),
            "--predictions",
            str(predictions),
            "--expected-contract-sha256",
            hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "--expected-model-sha256",
            model_sha,
            "--expected-source-artifact-manifest-sha256",
            source_sha,
            "--expected-prediction-manifest-sha256",
            hashlib.sha256(prediction_manifest.read_bytes()).hexdigest(),
        ],
        cwd=repository,
        env={"PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "assessed model differs from contract" in result.stderr
