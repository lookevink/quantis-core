import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import cast

import numpy as np
import pytest

from quantis_core.mprm_jepa import (
    MprmJepaProtocol,
    build_mprm_selection_plan,
    canonicalize_mixture_weights,
    mean_preserving_component_means,
    paired_randomization_p_value,
    prepare_mprm_selection_campaign,
    qualify_mprm_selection_campaign,
)
from quantis_core.richer_regime_retry import (
    build_richer_regime_plan,
    RicherRegimeRetryProtocol,
)


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _protocol() -> MprmJepaProtocol:
    payload = json.loads(
        (
            _repository()
            / "lab"
            / "action_dynamics"
            / "mprm-jepa-protocol-v1.json"
        ).read_text()
    )
    return MprmJepaProtocol.from_dict(payload)


def _action_library() -> dict[str, object]:
    restored = json.loads(
        (
            _repository()
            / "lab"
            / "action_dynamics"
            / "development-protocol-v1.json"
        ).read_text()
    )["action_library"]
    return cast(dict[str, object], restored)


def test_frozen_selection_plan_is_complete_and_fresh() -> None:
    protocol = _protocol()

    plan = build_mprm_selection_plan(protocol)

    old_protocol = RicherRegimeRetryProtocol.from_dict(
        json.loads(
            (
                _repository()
                / "lab"
                / "action_dynamics"
                / "richer-regime-retry-protocol-v1.json"
            ).read_text()
        )
    )
    old_ids = {
        pair["pair_id"] for pair in build_richer_regime_plan(old_protocol)["pairs"]
    }
    assert plan == build_mprm_selection_plan(protocol)
    assert len(plan["pairs"]) == 90
    assert len({pair["pair_id"] for pair in plan["pairs"]}) == 90
    assert not old_ids & {pair["pair_id"] for pair in plan["pairs"]}
    assert {
        (
            pair["action_kind"],
            pair["worker_replicas"],
            pair["workload_family"],
        )
        for pair in plan["pairs"]
    } == {
        (action, workers, family)
        for action in protocol.action_kinds
        for workers in protocol.worker_replica_values
        for family in protocol.workload_families
    }


def test_weight_canonicalization_and_mean_preservation_are_transport_safe() -> None:
    raw_weights = np.asarray(
        [[1e-12, 0.2, 0.3, 0.5]], dtype=np.float32
    )
    weights = canonicalize_mixture_weights(raw_weights, floor=1e-9)
    anchor = np.arange(12, dtype=np.float64).reshape(1, 2, 2, 3)
    residual = np.asarray(
        np.random.default_rng(4).normal(size=(1, 4, 2, 2, 3)),
        dtype=np.float64,
    )

    component_mean = mean_preserving_component_means(
        anchor, residual, weights
    )
    recovered = np.sum(
        weights[:, :, None, None, None] * component_mean, axis=1
    )

    assert np.min(weights) >= 1e-9
    assert np.array_equal(np.sum(weights, axis=1), np.ones(1))
    assert np.max(np.abs(recovered - anchor)) <= 1e-10


def test_campaign_qualification_rejects_drift_and_accepts_complete_evidence() -> None:
    protocol = _protocol()
    plan = build_mprm_selection_plan(protocol)
    bindings = {
        "candidate_protocol_sha256": "a" * 64,
        "model_freeze_manifest_sha256": "b" * 64,
        "action_protocol_sha256": "c" * 64,
        "observation_schema_sha256": "d" * 64,
        "application_build_context_sha256": "e" * 64,
        "application_image_digest": "sha256:" + "f" * 64,
        "collector_image_digest": "sha256:" + "1" * 64,
        "attempt_id": "mprm-jepa-selection-v1-attempt-001",
    }
    prepared = prepare_mprm_selection_campaign(
        protocol,
        plan,
        action_library=_action_library(),
        bindings=bindings,
    )
    captures = {
        case_id: {
            "capture_manifest_sha256": prepared["manifest_sha256"][
                case_id
            ],
            "runner_log_sha256": "3" * 64,
            "metrics_sha256": "4" * 64,
            "logs_sha256": "5" * 64,
            "traces_sha256": "6" * 64,
            "actions_sha256": "7" * 64,
        }
        for case_id in prepared["manifests"]
    }
    pair_assessments = {
        pair["pair_id"]: {
            "schedule_alignment": True,
            "raw_effect_passed": True,
            "recovery_passed": True,
            "count_resolution_passed": True,
            "drain_eligible": True,
            "restart_probe_live": True,
            "mechanistic_recovery_passed": True,
        }
        for pair in plan["pairs"]
    }
    attestation = {
        "schema_version": 1,
        "kind": "mprm_jepa_collection_attestation_v1",
        "campaign_bindings": prepared["campaign_bindings"],
        "protocol_sha256": prepared["executor_protocol_sha256"],
        "plan_sha256": prepared["executor_plan_sha256"],
        "case_count": 180,
        "pair_count": 90,
        "manifest_sha256": prepared["manifest_sha256"],
    }

    qualified = qualify_mprm_selection_campaign(
        protocol,
        plan,
        prepared["manifests"],
        captures,
        attestation,
        pair_assessments,
        _action_library(),
    )
    assert qualified["status"] == "qualified"
    assert qualified["pair_count"] == 90
    assert len(qualified["qualified_corpus_sha256"]) == 64

    drifted = copy.deepcopy(attestation)
    del drifted["campaign_bindings"]["model_freeze_manifest_sha256"]
    with pytest.raises(ValueError, match="attestation"):
        qualify_mprm_selection_campaign(
            protocol,
            plan,
            prepared["manifests"],
            captures,
            drifted,
            pair_assessments,
            _action_library(),
        )

    mutated_manifests = copy.deepcopy(prepared["manifests"])
    first_case = next(iter(mutated_manifests))
    mutated_manifests[first_case]["request_schedule"][0] += 1
    mutated_captures = copy.deepcopy(captures)
    mutated_captures[first_case]["capture_manifest_sha256"] = (
        hashlib.sha256(
            json.dumps(
                mutated_manifests[first_case],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    mutated_attestation = copy.deepcopy(attestation)
    mutated_attestation["manifest_sha256"][first_case] = (
        mutated_captures[first_case]["capture_manifest_sha256"]
    )
    with pytest.raises(ValueError, match="manifest content"):
        qualify_mprm_selection_campaign(
            protocol,
            plan,
            mutated_manifests,
            mutated_captures,
            mutated_attestation,
            pair_assessments,
            _action_library(),
        )


def test_paired_randomization_is_deterministic_and_one_sided() -> None:
    candidate_minus_raw = np.full(90, -0.02, dtype=np.float64)

    first = paired_randomization_p_value(
        candidate_minus_raw, seed=26072932, draws=99999
    )
    second = paired_randomization_p_value(
        candidate_minus_raw, seed=26072932, draws=99999
    )

    assert first == second
    assert 0.0 < first <= 0.05


def test_independent_assessor_rejects_prediction_tampering(
    tmp_path: Path,
) -> None:
    repository = _repository()
    protocol_path = (
        repository
        / "lab"
        / "action_dynamics"
        / "mprm-jepa-protocol-v1.json"
    )
    model_freeze = tmp_path / "model-freeze-manifest.json"
    model_freeze.write_text("{}\n")
    model_hash = hashlib.sha256(model_freeze.read_bytes()).hexdigest()
    source: dict[str, object] = {}
    source_hash = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    qualified = tmp_path / "qualified-corpus.json"
    qualified.write_text(
        json.dumps(
            {
                "status": "qualified",
                "qualified_corpus_sha256": source_hash,
                "campaign_bindings": {
                    "model_freeze_manifest_sha256": model_hash,
                    "candidate_protocol_sha256": hashlib.sha256(
                        protocol_path.read_bytes()
                    ).hexdigest(),
                },
                "source_content_manifest": source,
            }
        )
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    array_path = predictions / "selection-inputs.npz"
    array_path.write_bytes(b"original")
    prediction_manifest = predictions / "prediction-manifest.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "sha256": {
                    array_path.name: hashlib.sha256(
                        array_path.read_bytes()
                    ).hexdigest()
                }
            }
        )
    )
    prediction_hash = hashlib.sha256(
        prediction_manifest.read_bytes()
    ).hexdigest()
    array_path.write_bytes(b"tampered")

    result = subprocess.run(
        [
            sys.executable,
            str(
                repository
                / "lab"
                / "action_dynamics"
                / "assess_mprm_jepa.py"
            ),
            "--protocol",
            str(protocol_path),
            "--model-freeze-manifest",
            str(model_freeze),
            "--qualified-corpus",
            str(qualified),
            "--predictions",
            str(predictions),
            "--expected-model-freeze-sha256",
            model_hash,
            "--expected-qualified-corpus-sha256",
            source_hash,
            "--expected-prediction-manifest-sha256",
            prediction_hash,
        ],
        cwd=repository,
        env={"PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "frozen input hash differs" in result.stderr
