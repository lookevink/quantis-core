"""Retained pure-assessment entry point for complete LeJEPA result bundles."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    COMPLETE_LEJEPA_REPRESENTATION_NAMES,
    assess_complete_lejepa_gates,
)
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.graph_telemetry import DeclaredTelemetryGraph

from prototype_complete_lejepa import (
    _action_sanity_from_predictions,
    _attribution_scores_from_predictions,
    _downstream_pair_errors,
    _forecast_scores,
    _protocol_checks,
    _state_probe,
)


def reassess_complete_lejepa_evidence(directory: Path) -> Dict[str, Any]:
    """Recompute every gate from stored arrays without invoking a model."""

    root = Path(directory)
    metadata = json.loads((root / "evidence-metadata.json").read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind")
        != "complete_lejepa_assessment_evidence"
    ):
        raise ValueError("unsupported complete LeJEPA assessment evidence")
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    entity_names = tuple(str(value) for value in metadata["entity_names"])
    state_names = tuple(
        str(value) for value in metadata["state_feature_names"]
    )
    control_names = tuple(
        str(value) for value in metadata["control_feature_names"]
    )
    action_names = tuple(
        str(value) for value in metadata["action_feature_names"]
    )
    ownership = np.asarray(metadata["ownership_mask"], dtype=np.bool_)
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        if any(
            not np.all(np.isfinite(stored[name]))
            for name in stored.files
        ):
            raise ValueError("complete LeJEPA evidence contains non-finite arrays")
        windows: Dict[str, ActionConditionedWindows] = {}
        for role, raw_identity in metadata["roles"].items():
            identity = dict(raw_identity)
            windows[role] = ActionConditionedWindows(
                histories=stored[f"histories__{role}"],
                future_states=stored[f"target__{role}"],
                future_controls=stored[f"controls__{role}"],
                future_actions=stored[f"actions__{role}"],
                trajectory_ids=tuple(
                    str(value) for value in identity["trajectory_ids"]
                ),
                matched_pair_ids=tuple(
                    str(value) for value in identity["matched_pair_ids"]
                ),
                transition_indices=np.asarray(
                    identity["transition_indices"], dtype=np.int64
                ),
                entity_names=entity_names,
                state_feature_names=state_names,
                control_feature_names=control_names,
                action_feature_names=action_names,
                graph=graph,
            )
        forecast_scores = {
            name: {
                role: _forecast_scores(
                    stored[f"prediction__{name}__{role}"],
                    windows[role],
                )
                for role in (
                    "selection",
                    "iid_evaluation",
                    "transfer_evaluation",
                )
            }
            for name in COMPLETE_LEJEPA_REPRESENTATION_NAMES
        }
        raw_scores = {
            role: _forecast_scores(
                stored[f"raw_prediction__{role}"], windows[role]
            )
            for role in (
                "selection",
                "iid_evaluation",
                "transfer_evaluation",
            )
        }
        ridge_curves = {}
        for name in COMPLETE_LEJEPA_REPRESENTATION_NAMES:
            rows = []
            for ridge in metadata["ridge_values"]:
                prediction = np.load(
                    root
                    / "ridge-selection-evidence"
                    / f"{name}__ridge_{float(ridge):.4g}.npy",
                    allow_pickle=False,
                )
                if not np.all(np.isfinite(prediction)):
                    raise ValueError(
                        "complete LeJEPA ridge evidence is non-finite"
                    )
                scores = _forecast_scores(
                    prediction, windows["selection"]
                )
                rows.append(
                    {
                        "ridge": float(ridge),
                        "raw_safe": all(
                            scores[key]
                            <= 1.05 * raw_scores["selection"][key]
                            for key in (
                                "overall_mse",
                                "action_overlap_mse",
                                "downstream_effect_mse",
                            )
                        ),
                        **scores,
                    }
                )
            ridge_curves[name] = rows
        state_probes = {
            name: _state_probe(
                stored[f"representation__{name}__fit"],
                windows["fit"],
                stored[
                    f"representation__{name}__transfer_evaluation"
                ],
                windows["transfer_evaluation"],
                ownership,
            )
            for name in COMPLETE_LEJEPA_REPRESENTATION_NAMES
        }
        query_metadata = dict(metadata["queries"])
        queries = PreparedAttributionQueries(
            query_ids=tuple(
                str(value) for value in query_metadata["query_ids"]
            ),
            histories=stored["query_histories"],
            future_controls=stored["query_future_controls"],
            observed_future=stored["query_observed_future"],
            candidate_actions=stored["query_candidate_actions"],
            candidate_ids=tuple(
                str(value) for value in query_metadata["candidate_ids"]
            ),
            candidate_action_kinds=tuple(
                str(value)
                for value in query_metadata["candidate_action_kinds"]
            ),
            candidate_target_entities=tuple(
                str(value)
                for value in query_metadata["candidate_target_entities"]
            ),
            expected_action_kinds=tuple(
                str(value)
                for value in query_metadata["expected_action_kinds"]
            ),
            expected_target_entities=tuple(
                str(value)
                for value in query_metadata["expected_target_entities"]
            ),
            expected_variant_ids=tuple(
                str(value)
                for value in query_metadata["expected_variant_ids"]
            ),
        )
        attribution = {
            name: _attribution_scores_from_predictions(
                stored[f"attribution_prediction__{name}"],
                queries,
                ownership,
            )
            for name in COMPLETE_LEJEPA_REPRESENTATION_NAMES
        }
        action_sanity = {
            name: _action_sanity_from_predictions(
                {
                    variant: stored[
                        f"action_sanity__{name}__{variant}"
                    ]
                    for variant in ("correct", "no_action", "shuffled")
                },
                windows["transfer_evaluation"],
                ownership,
            )
            for name in COMPLETE_LEJEPA_REPRESENTATION_NAMES
        }
        restoration = {
            name: bool(
                np.allclose(
                    stored[f"restored__{name}"],
                    stored[
                        f"representation__{name}__transfer_evaluation"
                    ][:8],
                    atol=1e-7,
                )
            )
            for name in COMPLETE_LEJEPA_REPRESENTATION_NAMES
        }
        pair_errors = {
            name: _downstream_pair_errors(
                stored[
                    f"prediction__{name}__transfer_evaluation"
                ],
                windows["transfer_evaluation"],
            )
            for name in COMPLETE_LEJEPA_REPRESENTATION_NAMES
        }
    return dict(
        assess_complete_lejepa_gates(
            forecast_scores=forecast_scores,
            raw_scores=raw_scores,
            state_probes=state_probes,
            attribution=attribution,
            action_sanity=action_sanity,
            restoration_parity=restoration,
            ridge_curves=ridge_curves,
            selected_ridges={
                str(key): float(value)
                for key, value in metadata["selected_ridges"].items()
            },
            transfer_pair_errors=pair_errors,
            protocol_checks=_protocol_checks(root),
        )
    )


def verify_complete_lejepa_bundle(directory: Path) -> Dict[str, Any]:
    """Verify identities and independently recompute the stored assessment."""

    root = Path(directory)
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "complete_lejepa_artifact_manifest"
    ):
        raise ValueError("unsupported complete LeJEPA artifact manifest")
    for filename, expected in manifest["sha256"].items():
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"complete LeJEPA artifact identity mismatch: {filename}"
            )
    report = json.loads((root / "result.json").read_text())
    recorded = report.get("assessment")
    if not isinstance(recorded, dict) or "decision" not in recorded:
        raise ValueError("complete LeJEPA assessment is incomplete")
    reassessed = reassess_complete_lejepa_evidence(root)
    if report.get("frozen_contract_run") is False:
        reassessed = {
            **reassessed,
            "interpretable": False,
            "provisional_decision": reassessed["decision"],
            "decision": "non_interpretable_smoke",
        }
    if reassessed != recorded:
        raise ValueError(
            "stored complete LeJEPA assessment differs from recomputation"
        )
    return reassessed


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parsed = parser.parse_args(arguments)
    print(
        json.dumps(
            verify_complete_lejepa_bundle(parsed.directory),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
