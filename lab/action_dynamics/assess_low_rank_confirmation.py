"""Independently assess stored low-rank confirmation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
)
from quantis_core.low_rank_confirmation import (
    LowRankConfirmationContract,
    assess_low_rank_confirmation_arrays,
    downstream_effect_mse,
    pair_balanced_action_mse,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


def assess_stored_low_rank_confirmation(
    *,
    contract_path: Path,
    model_path: Path,
    source_artifact_manifest: Path,
    predictions_directory: Path,
    expected_contract_sha256: str,
    expected_model_sha256: str,
    expected_source_artifact_manifest_sha256: str,
    expected_prediction_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Verify every identity and recompute the frozen decision."""

    expected = {
        contract_path: expected_contract_sha256,
        model_path: expected_model_sha256,
        source_artifact_manifest: (
            expected_source_artifact_manifest_sha256
        ),
        predictions_directory / "prediction-manifest.json": (
            expected_prediction_manifest_sha256
        ),
    }
    if any(_file_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("frozen input hash differs")
    contract = LowRankConfirmationContract.from_dict(
        _read_object(contract_path)
    )
    candidate = contract.payload["candidate"]
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("model_sha256") != expected_model_sha256
    ):
        raise ValueError("assessed model differs from contract")
    prediction_manifest = _read_object(
        predictions_directory / "prediction-manifest.json"
    )
    recorded = prediction_manifest.get("sha256")
    if (
        prediction_manifest.get("schema_version") != 1
        or prediction_manifest.get("kind")
        != "low_rank_confirmation_prediction_manifest"
        or prediction_manifest.get("contract_sha256")
        != expected_contract_sha256
        or prediction_manifest.get("model_sha256")
        != expected_model_sha256
        or prediction_manifest.get("source_artifact_manifest_sha256")
        != expected_source_artifact_manifest_sha256
        or not isinstance(recorded, dict)
    ):
        raise ValueError("prediction manifest identity differs")
    for filename, digest in recorded.items():
        path = predictions_directory / str(filename)
        if (
            not isinstance(digest, str)
            or not path.is_file()
            or _file_sha256(path) != digest
        ):
            raise ValueError("stored confirmation prediction drifted")

    with np.load(
        predictions_directory / "confirmation-inputs.npz",
        allow_pickle=False,
    ) as arrays:
        observed = arrays["observed"]
        future_actions = arrays["future_actions"]
        trajectory_ids = tuple(
            str(value) for value in arrays["trajectory_ids"]
        )
        matched_pair_ids = tuple(
            str(value) for value in arrays["matched_pair_ids"]
        )
        transition_indices = arrays["transition_indices"]
        action_kind_by_pair = json.loads(
            str(arrays["action_kind_by_pair_json"])
        )
    predictions = {}
    for name in ("candidate", "action_masked", "persistence"):
        with np.load(
            predictions_directory / f"{name}.npz",
            allow_pickle=False,
        ) as arrays:
            predictions[name] = arrays["prediction"]
    model = ContractiveLowRankDynamics.from_dict(
        _read_object(model_path)
    )
    pair_order: Optional[tuple[str, ...]] = None
    pair_losses: Dict[str, NDArray[np.float64]] = {}
    downstream: Dict[str, float] = {}
    graph = DeclaredTelemetryGraph.from_dict(
        dict(model.to_dict()["graph"])
    )
    for name, prediction in predictions.items():
        current_order, losses = pair_balanced_action_mse(
            prediction=prediction,
            observed=observed,
            future_actions=future_actions,
            matched_pair_ids=matched_pair_ids,
        )
        if pair_order is None:
            pair_order = current_order
        elif current_order != pair_order:
            raise ValueError("prediction pair order differs")
        pair_losses[name] = losses
        downstream[name] = downstream_effect_mse(
            prediction=prediction,
            observed=observed,
            future_actions=future_actions,
            trajectory_ids=trajectory_ids,
            matched_pair_ids=matched_pair_ids,
            transition_indices=transition_indices,
            graph=graph,
        )
    if pair_order is None:
        raise ValueError("stored confirmation has no pairs")
    gates = dict(contract.payload["decision_gates"])
    assessment = dict(
        assess_low_rank_confirmation_arrays(
            pair_ids=pair_order,
            action_kind_by_pair={
                str(key): str(value)
                for key, value in action_kind_by_pair.items()
            },
            candidate_action_mse=pair_losses["candidate"],
            action_masked_action_mse=pair_losses["action_masked"],
            persistence_action_mse=pair_losses["persistence"],
            candidate_downstream_effect_mse=downstream["candidate"],
            action_masked_downstream_effect_mse=downstream[
                "action_masked"
            ],
            persistence_downstream_effect_mse=downstream[
                "persistence"
            ],
            spectral_radius=model.spectral_radius,
            parameter_count=model.parameter_count,
            serialized_size_bytes=len(
                json.dumps(
                    model.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ),
            rollout_finite=all(
                np.all(np.isfinite(value))
                for value in predictions.values()
            ),
            seed=int(gates["paired_sign_flip_seed"]),
            draws=int(gates["paired_sign_flip_draws"]),
        )
    )
    assessment["identities"] = {
        "contract_sha256": expected_contract_sha256,
        "model_sha256": expected_model_sha256,
        "source_artifact_manifest_sha256": (
            expected_source_artifact_manifest_sha256
        ),
        "prediction_manifest_sha256": (
            expected_prediction_manifest_sha256
        ),
    }
    assessment["normalized_mse_overall"] = {
        name: float(np.mean(np.square(prediction - observed)))
        for name, prediction in predictions.items()
    }
    return assessment


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source-artifact-manifest", type=Path, required=True
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--expected-contract-sha256", required=True
    )
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument(
        "--expected-source-artifact-manifest-sha256", required=True
    )
    parser.add_argument(
        "--expected-prediction-manifest-sha256", required=True
    )
    parsed = parser.parse_args(arguments)
    result = assess_stored_low_rank_confirmation(
        contract_path=parsed.contract,
        model_path=parsed.model,
        source_artifact_manifest=parsed.source_artifact_manifest,
        predictions_directory=parsed.predictions,
        expected_contract_sha256=parsed.expected_contract_sha256,
        expected_model_sha256=parsed.expected_model_sha256,
        expected_source_artifact_manifest_sha256=(
            parsed.expected_source_artifact_manifest_sha256
        ),
        expected_prediction_manifest_sha256=(
            parsed.expected_prediction_manifest_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
