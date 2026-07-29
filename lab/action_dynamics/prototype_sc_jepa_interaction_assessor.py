#!/usr/bin/env python3
"""Recompute ticket 013 from stored arrays without fitted model execution."""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.edge_dynamics.sc_jepa import (
    SC_JEPA_ASSESSMENT_MODEL_NAMES,
    SC_JEPA_ASSESSMENT_ROLE_NAMES,
    SC_JEPA_CELL_NAMES,
    ScJepaConfig,
    ScJepaModel,
    assess_sc_jepa_interaction,
)


def assess_stored_bundle(
    directory: Path, *, verify_manifest: bool = True
) -> Mapping[str, Any]:
    """Verify a completed bundle and recompute its complete assessment."""

    root = Path(directory)
    if verify_manifest:
        verify_artifact_manifest(root)
    metadata = _read_object(root / "assessment-metadata.json")
    model_bundle = _read_object(root / "models.json")
    event_definition = _read_object(root / "event-definition.json")
    protocol = _read_object(root / "protocol.json")
    data_identity = _read_object(root / "data-identity.json")
    _verify_source_identity(root, data_identity)
    if (
        tuple(metadata["model_names"])
        != SC_JEPA_ASSESSMENT_MODEL_NAMES
        or tuple(metadata["role_names"])
        != SC_JEPA_ASSESSMENT_ROLE_NAMES
    ):
        raise ValueError("SC-JEPA assessment names differ")
    with np.load(
        root / "sc-jepa-evidence.npz", allow_pickle=False
    ) as arrays:
        risks = _read_model_arrays(arrays, "risk")
        restored_risks = _read_model_arrays(
            arrays, "restored_risk"
        )
        calibrated = _read_model_arrays(arrays, "calibrated")
        restored_calibrated = _read_model_arrays(
            arrays, "restored_calibrated"
        )
        decisions = _read_decisions(arrays, "alert_decision")
        restored_decisions = _read_decisions(
            arrays, "restored_alert_decision"
        )
        labels = {
            role: arrays[f"labels__{role}"].astype(np.bool_)
            for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
        }
        transition_indices = {
            role: arrays[
                f"transition_indices__{role}"
            ].astype(np.int64)
            for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
        }
        representation_tokens = {
            name: arrays[
                f"representation_tokens__{name}"
            ].copy()
            for name in SC_JEPA_CELL_NAMES
        }
        restored_representation_tokens = {
            name: arrays[
                f"restored_representation_tokens__{name}"
            ].copy()
            for name in SC_JEPA_CELL_NAMES
        }
        representation_patch_values = {
            name: arrays[
                f"representation_patch_values__{name}"
            ].copy()
            for name in SC_JEPA_CELL_NAMES
        }
        restored_representation_patch_values = {
            name: arrays[
                f"restored_representation_patch_values__{name}"
            ].copy()
            for name in SC_JEPA_CELL_NAMES
        }
        representation_code_probabilities = {
            name: _read_optional_representation(
                arrays, "representation_code_probabilities", name
            )
            for name in SC_JEPA_CELL_NAMES
        }
        restored_representation_code_probabilities = {
            name: _read_optional_representation(
                arrays,
                "restored_representation_code_probabilities",
                name,
            )
            for name in SC_JEPA_CELL_NAMES
        }
        state_truth = arrays["state_truth"].astype(np.float64)
        state_fit_truth = arrays["state_fit_truth"].astype(np.float64)
        state_ownership = arrays["state_ownership_mask"].astype(
            np.bool_
        )
        stored_state_scale = arrays["state_scale"].copy()
        stored_state_varying = arrays["state_varying_mask"].astype(
            np.bool_
        )
        raw_state_scale = np.std(state_fit_truth, axis=0)
        state_varying = state_ownership & (raw_state_scale > 1e-8)
        state_scale = np.where(state_varying, raw_state_scale, 1.0)
        if (
            not np.allclose(
                state_scale, stored_state_scale, atol=0.0, rtol=0.0
            )
            or not np.array_equal(
                state_varying, stored_state_varying
            )
        ):
            raise ValueError(
                "SC-JEPA stored state normalization does not derive"
            )
        for probe_name in ("codebook_multi", "matched_pca"):
            probe = dict(
                model_bundle["state_probes"][probe_name]
            )
            if (
                not np.allclose(
                    np.asarray(
                        probe["target_scale"], dtype=np.float64
                    ),
                    state_scale,
                    atol=0.0,
                    rtol=0.0,
                )
                or not np.array_equal(
                    np.asarray(
                        probe["target_varying_mask"], dtype=np.bool_
                    ),
                    state_varying,
                )
                or not np.array_equal(
                    np.asarray(
                        probe["ownership_mask"], dtype=np.bool_
                    ),
                    state_ownership,
                )
            ):
                raise ValueError(
                    "SC-JEPA state probe normalization differs"
                )
        state_predictions = {
            name: arrays[f"state_prediction__{name}"].copy()
            for name in ("codebook_multi", "matched_pca")
        }
        latency_samples = {
            name: arrays[f"latency_samples__{name}"].copy()
            for name in SC_JEPA_ASSESSMENT_MODEL_NAMES
        }
        event_fit_deltas = arrays["event_fit_deltas"].copy()
        event_fit_offsets = arrays["event_fit_offsets"].astype(
            np.int64
        )
        event_fit_ownership = arrays["event_fit_ownership"].astype(
            np.bool_
        )
        event_fit_future_actions = arrays[
            "event_fit_future_actions"
        ].copy()
        fit_identity = dict(
            metadata["role_input_identities"]["fit"]
        )
        if (
            _array_identity(event_fit_future_actions)
            != dict(fit_identity["arrays"]["future_actions"])
        ):
            raise ValueError(
                "SC-JEPA event action evidence differs from fit identity"
            )
        fit_trajectory_ids = tuple(
            str(value) for value in fit_identity["trajectory_ids"]
        )
        action_names = tuple(
            str(value)
            for value in fit_identity["action_feature_names"]
        )
        applicable = action_names.index("applicable")
        treatments = {
            fit_trajectory_ids[index]
            for index in range(len(fit_trajectory_ids))
            if np.any(
                event_fit_future_actions[
                    index, ..., applicable
                ]
                > 0.5
            )
        }
        derived_control_ids = tuple(
            sorted(set(fit_trajectory_ids) - treatments)
        )
        if derived_control_ids != tuple(
            str(value)
            for value in metadata[
                "event_fit_control_trajectory_ids"
            ]
        ):
            raise ValueError(
                "SC-JEPA event control membership does not derive"
            )
        derived_event_definition = _derive_event_definition(
            deltas=event_fit_deltas,
            offsets=event_fit_offsets,
            ownership=event_fit_ownership,
            entity_ids=tuple(
                str(value) for value in event_definition["entity_ids"]
            ),
            feature_names=tuple(
                str(value)
                for value in event_definition["feature_names"]
            ),
            control_trajectory_ids=derived_control_ids,
        )
        causality = {
            name: arrays[name].copy()
            for name in (
                "audit_histories",
                "audit_counterfactual_histories",
                "audit_forbidden",
                "audit_counterfactual_forbidden",
                "audit_original_outputs",
                "audit_counterfactual_outputs",
                "audit_forbidden_keyword_rejections",
            )
        }
    trajectory_ids = {
        role: tuple(
            str(value)
            for value in metadata["trajectory_ids"][role]
        )
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }
    trajectory_onsets = {
        role: {
            str(key): (
                None if value is None else int(value)
            )
            for key, value in dict(
                metadata["trajectory_onsets"][role]
            ).items()
        }
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }
    for role in SC_JEPA_ASSESSMENT_ROLE_NAMES:
        identity = dict(metadata["role_input_identities"][role])
        if (
            tuple(str(value) for value in identity["trajectory_ids"])
            != trajectory_ids[role]
            or not np.array_equal(
                np.asarray(
                    identity["transition_indices"], dtype=np.int64
                ),
                transition_indices[role],
            )
        ):
            raise ValueError(
                "SC-JEPA assessed rows differ from role input identity"
            )
    models = {
        str(name): dict(payload)
        for name, payload in dict(model_bundle["models"]).items()
    }
    restored_models = {
        str(name): dict(payload)
        for name, payload in dict(
            model_bundle["restored_models"]
        ).items()
    }
    raw = dict(model_bundle["raw_low_rank"])
    restored_raw = dict(model_bundle["restored_raw_low_rank"])
    training_bindings = {
        str(name): dict(value)
        for name, value in dict(
            model_bundle["training_bindings"]
        ).items()
    }
    event_binding_valid = dict(model_bundle["event_fit_binding"]) == {
        "fit_input_sha256": fit_identity["identity_sha256"],
        "event_definition_sha256": hashlib.sha256(
            _canonical_json_bytes(event_definition)
        ).hexdigest(),
        "deltas": _array_identity(event_fit_deltas),
        "offsets": _array_identity(event_fit_offsets),
        "ownership": _array_identity(event_fit_ownership),
        "future_actions": _array_identity(event_fit_future_actions),
    }
    entity_pca = dict(model_bundle["entity_pca"])
    state_probes = {
        str(name): dict(payload)
        for name, payload in dict(
            model_bundle["state_probes"]
        ).items()
    }
    reference_bindings = {
        str(name): dict(payload)
        for name, payload in dict(
            model_bundle["reference_bindings"]
        ).items()
    }
    reference_payloads = {
        "entity_pca": entity_pca,
        "codebook_multi_probe": state_probes["codebook_multi"],
        "matched_pca_probe": state_probes["matched_pca"],
    }
    reference_bindings_valid = (
        set(reference_bindings) == set(reference_payloads)
        and all(
            reference_bindings[name]
            == {
                "fit_input_sha256": fit_identity[
                    "identity_sha256"
                ],
                "payload_sha256": hashlib.sha256(
                    _canonical_json_bytes(payload)
                ).hexdigest(),
            }
            for name, payload in reference_payloads.items()
        )
    )
    raw_config_valid = dict(raw["config"]) == {
        "width": 32,
        "hidden_width": 64,
        "alert_steps": 200,
        "checkpoint_interval": 25,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "seed": 13013,
    }
    reference_contracts_valid = (
        raw_config_valid
        and int(entity_pca["width"]) == 32
        and float(state_probes["codebook_multi"]["ridge"]) == 1e-3
        and float(state_probes["matched_pca"]["ridge"]) == 1e-3
        and reference_bindings_valid
    )
    training_counts = {
        name: _parameter_count(
            models[name],
            prefixes=(
                "online_encoder.",
                "online_bottleneck.",
                "fine_predictor.",
                "global_predictor.",
                "decoder.",
                "risk_head.",
            ),
        )
        for name in SC_JEPA_CELL_NAMES
    }
    inference_counts = {
        name: _parameter_count(
            models[name],
            prefixes=(
                "online_encoder.",
                "online_bottleneck.",
                "risk_head.",
            ),
        )
        for name in SC_JEPA_CELL_NAMES
    }
    calibrations = {
        **{
            name: _read_calibration(models[name])
            for name in SC_JEPA_CELL_NAMES
        },
        "raw_low_rank": _read_calibration(raw),
    }
    restored_calibrations = {
        **{
            name: _read_calibration(restored_models[name])
            for name in SC_JEPA_CELL_NAMES
        },
        "raw_low_rank": _read_calibration(restored_raw),
    }
    protocol_checks = _derive_protocol_checks(
        metadata=metadata,
        protocol=protocol,
        data_identity=data_identity,
        event_definition=event_definition,
        derived_event_definition=derived_event_definition,
        event_binding_valid=event_binding_valid,
        models=models,
        restored_models=restored_models,
        raw=raw,
        restored_raw=restored_raw,
        training_bindings=training_bindings,
        training_counts=training_counts,
        inference_counts=inference_counts,
        causality=causality,
        reference_contracts_valid=reference_contracts_valid,
    )
    edge_metrics = _derive_edge_metrics(
        metadata=metadata,
        model_bundle=model_bundle,
        event_definition=event_definition,
        inference_counts=inference_counts,
        latency_samples=latency_samples,
    )
    return assess_sc_jepa_interaction(
        risks=risks,
        restored_risks=restored_risks,
        stored_calibrated_risks=calibrated,
        restored_calibrated_risks=restored_calibrated,
        stored_alert_decisions=decisions,
        restored_alert_decisions=restored_decisions,
        stored_calibrations=calibrations,
        restored_calibrations=restored_calibrations,
        labels=labels,
        trajectory_ids=trajectory_ids,
        transition_indices=transition_indices,
        trajectory_onsets=trajectory_onsets,
        representation_tokens=representation_tokens,
        restored_representation_tokens=(
            restored_representation_tokens
        ),
        representation_patch_values=representation_patch_values,
        restored_representation_patch_values=(
            restored_representation_patch_values
        ),
        representation_code_probabilities=(
            representation_code_probabilities
        ),
        restored_representation_code_probabilities=(
            restored_representation_code_probabilities
        ),
        state_truth=state_truth,
        state_scale=state_scale,
        state_varying_mask=state_varying,
        state_predictions=state_predictions,
        training_parameter_counts=training_counts,
        inference_parameter_counts=inference_counts,
        protocol_checks=protocol_checks,
        edge_metrics=edge_metrics,
    )


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Require canonical stored and recomputed assessments to match."""

    root = Path(directory)
    recomputed = assess_stored_bundle(root)
    if (root / "assessment.json").read_text() != _pretty_json(
        recomputed
    ):
        raise ValueError("stored SC-JEPA assessment does not recompute")
    return recomputed


def _read_model_arrays(
    arrays: Any, prefix: str
) -> Mapping[str, Mapping[str, NDArray[np.float64]]]:
    return {
        role: {
            model: arrays[f"{prefix}__{role}__{model}"].copy()
            for model in SC_JEPA_ASSESSMENT_MODEL_NAMES
        }
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }


def _read_decisions(
    arrays: Any, prefix: str
) -> Mapping[str, Mapping[str, NDArray[np.bool_]]]:
    return {
        role: {
            model: arrays[
                f"{prefix}__{role}__{model}"
            ].astype(np.bool_)
            for model in SC_JEPA_ASSESSMENT_MODEL_NAMES
        }
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }


def _read_optional_representation(
    arrays: Any, prefix: str, model: str
) -> Any:
    marker_prefix = prefix.replace("_code_probabilities", "_has_codes")
    has_codes = bool(arrays[f"{marker_prefix}__{model}"].item())
    key = f"{prefix}__{model}"
    if not has_codes:
        if key in arrays.files:
            raise ValueError(
                "continuous SC-JEPA representation unexpectedly has codes"
            )
        return None
    if key not in arrays.files:
        raise ValueError("codebook SC-JEPA representation is missing codes")
    return arrays[key].copy()


def _read_calibration(
    payload: Mapping[str, Any]
) -> Mapping[str, float]:
    raw = payload.get("calibration")
    expected = {
        "slope",
        "intercept",
        "calibration_brier",
        "alert_threshold",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("SC-JEPA calibration payload is invalid")
    result = {str(key): float(value) for key, value in raw.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("SC-JEPA calibration payload is non-finite")
    return result


def _derive_event_definition(
    *,
    deltas: NDArray[np.float64],
    offsets: NDArray[np.int64],
    ownership: NDArray[np.bool_],
    entity_ids: Tuple[str, ...],
    feature_names: Tuple[str, ...],
    control_trajectory_ids: Tuple[str, ...],
) -> Mapping[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    boundaries = np.asarray(offsets, dtype=np.int64)
    shape = (len(entity_ids), len(feature_names))
    if (
        values.ndim != 3
        or values.shape[1:] != shape
        or ownership.shape != shape
        or not np.any(ownership)
        or not np.all(np.isfinite(values))
        or boundaries.shape != (len(control_trajectory_ids) + 1,)
        or boundaries[0] != 0
        or boundaries[-1] != len(values)
        or np.any(np.diff(boundaries) < 1)
        or len(control_trajectory_ids) < 2
    ):
        raise ValueError("SC-JEPA event fitting evidence is invalid")
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(
        np.abs(values - center[None]), axis=0
    )
    standard_deviation = np.std(values, axis=0)
    scale = np.where(
        mad > 1e-8,
        mad,
        np.where(standard_deviation > 1e-8, standard_deviation, 1.0),
    )
    center = np.where(ownership, center, 0.0)
    scale = np.where(ownership, scale, 1.0)
    maxima = []
    for position in range(len(control_trajectory_ids)):
        local = values[
            boundaries[position] : boundaries[position + 1]
        ]
        standardized = (local - center[None]) / scale[None]
        scores = np.sqrt(
            np.mean(np.square(standardized[:, ownership]), axis=1)
        )
        maxima.append(float(np.max(scores)))
    return {
        "schema_version": 1,
        "kind": "hepa_normalized_effect_event_definition",
        "entity_ids": list(entity_ids),
        "feature_names": list(feature_names),
        "ownership_mask": ownership.astype(int).tolist(),
        "delta_center": center.tolist(),
        "delta_scale": scale.tolist(),
        "threshold": float(
            np.quantile(
                np.asarray(maxima, dtype=np.float64),
                0.95,
                method="higher",
            )
        ),
        "control_trajectory_count": len(control_trajectory_ids),
        "quantile": 0.95,
    }


def _parameter_count(
    payload: Mapping[str, Any], *, prefixes: Tuple[str, ...]
) -> int:
    count = 0
    for name, raw in dict(payload["state_dict"]).items():
        value = dict(raw)
        shape = tuple(int(item) for item in value["shape"])
        array = np.asarray(value["values"])
        if array.shape != shape:
            raise ValueError("SC-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("SC-JEPA state tensor is non-finite")
        key = str(name)
        if key.endswith("ownership"):
            continue
        if any(key.startswith(prefix) for prefix in prefixes):
            count += int(np.prod(shape, dtype=np.int64))
    return count


def _derive_protocol_checks(
    *,
    metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    event_definition: Mapping[str, Any],
    derived_event_definition: Mapping[str, Any],
    event_binding_valid: bool,
    models: Mapping[str, Mapping[str, Any]],
    restored_models: Mapping[str, Mapping[str, Any]],
    raw: Mapping[str, Any],
    restored_raw: Mapping[str, Any],
    training_bindings: Mapping[str, Mapping[str, Any]],
    training_counts: Mapping[str, int],
    inference_counts: Mapping[str, int],
    causality: Mapping[str, NDArray[Any]],
    reference_contracts_valid: bool,
) -> Mapping[str, bool]:
    source = {
        str(role): set(str(value) for value in values)
        for role, values in dict(
            metadata["source_role_pair_ids"]
        ).items()
    }
    role_identities = {
        str(role): dict(value)
        for role, value in dict(
            metadata["role_input_identities"]
        ).items()
    }
    identities_valid = set(role_identities) == {
        "fit",
        "selection",
        "calibration",
        "evaluation_iid",
        "evaluation_transfer",
    }
    for role, identity in role_identities.items():
        stated_digest = str(identity.pop("identity_sha256"))
        identities_valid = (
            identities_valid
            and stated_digest
            == hashlib.sha256(
                _canonical_json_bytes(identity)
            ).hexdigest()
            and int(identity["row_count"])
            == len(identity["matched_pair_ids"])
            == len(identity["trajectory_ids"])
            == len(identity["transition_indices"])
        )
        identity["identity_sha256"] = stated_digest
    used = {
        role: set(str(value) for value in identity["matched_pair_ids"])
        for role, identity in role_identities.items()
    }
    roles = ("fit", "selection", "calibration", "evaluation")
    names_valid = set(source) == set(roles) and set(used) == {
        "fit",
        "selection",
        "calibration",
        "evaluation_iid",
        "evaluation_transfer",
    }
    disjoint = names_valid and all(
        not (source[left] & source[right])
        for position, left in enumerate(roles)
        for right in roles[position + 1 :]
    )
    used_valid = (
        identities_valid
        and names_valid
        and used["fit"] <= source["fit"]
        and used["selection"] <= source["selection"]
        and used["calibration"] <= source["calibration"]
        and used["evaluation_iid"] <= source["evaluation"]
        and used["evaluation_transfer"] <= source["evaluation"]
        and not (
            used["evaluation_iid"] & used["evaluation_transfer"]
        )
    )
    phases = {
        str(key): float(value)
        for key, value in dict(metadata["phases"]).items()
    }
    phase_names = (
        "pretraining_started_unix_seconds",
        "calibration_completed_unix_seconds",
        "evaluation_started_unix_seconds",
        "evaluation_completed_unix_seconds",
    )
    phases_valid = (
        set(phases) == set(phase_names)
        and all(np.isfinite(phases[name]) for name in phase_names)
        and all(
            phases[left] <= phases[right]
            for left, right in zip(phase_names, phase_names[1:])
        )
    )
    configs = {
        name: dict(models[name]["config"])
        for name in SC_JEPA_CELL_NAMES
    }
    expected_flags = {
        "continuous_single": (False, False),
        "continuous_multi": (False, True),
        "codebook_single": (True, False),
        "codebook_multi": (True, True),
    }
    exact_configs = []
    flags_valid = True
    for name in SC_JEPA_CELL_NAMES:
        config = configs[name]
        flags = (
            bool(config["use_codebook"]),
            bool(config["multi_resolution"]),
        )
        flags_valid = flags_valid and flags == expected_flags[name]
        exact_configs.append(
            config
            == vars(
                ScJepaConfig(
                    use_codebook=flags[0],
                    multi_resolution=flags[1],
                )
            )
        )
    configs_match = flags_valid and all(exact_configs)
    sources = {
        str(name): dict(value)
        for name, value in dict(
            data_identity["implementation_sources"]
        ).items()
    }
    source_identity_valid = (
        len(str(data_identity["implementation_commit"])) == 40
        and bool(sources)
        and all(
            bool(identity["matches_head"])
            and identity["head_sha256"] == identity["sha256"]
            for identity in sources.values()
        )
    )
    frozen_protocol = (
        protocol.get("schema_version") == 1
        and protocol.get("kind") == "sc_jepa_interaction_protocol"
        and protocol.get("contract") == "sc-jepa-interaction-v1"
        and protocol.get("interpretable") is True
        and protocol.get("smoke_only") is False
        and int(protocol.get("pretrain_steps", -1)) == 300
        and int(protocol.get("alert_steps", -1)) == 200
        and int(protocol.get("frozen_pretrain_steps", -1)) == 300
        and int(protocol.get("frozen_alert_steps", -1)) == 200
        and int(protocol.get("seed", -1)) == 13013
        and int(protocol.get("expected_pair_count", -1)) == 40
    )
    signature_evidence = dict(metadata["inference_signatures"])
    signatures_valid = (
        tuple(inspect.signature(ScJepaModel.encode).parameters)
        == ("self", "histories", "graph")
        and tuple(
            inspect.signature(ScJepaModel.predict_risk).parameters
        )
        == ("self", "histories", "graph")
        and signature_evidence
        == {
            "sc_jepa_encode": ["self", "histories", "graph"],
            "sc_jepa_predict_risk": [
                "self",
                "histories",
                "graph",
            ],
            "raw_low_rank_predict_risk": [
                "self",
                "histories",
                "graph",
            ],
        }
    )
    causal_counterfactual_valid = (
        np.array_equal(
            causality["audit_histories"],
            causality["audit_counterfactual_histories"],
        )
        and not np.array_equal(
            causality["audit_forbidden"],
            causality["audit_counterfactual_forbidden"],
        )
        and np.allclose(
            causality["audit_original_outputs"],
            causality["audit_counterfactual_outputs"],
            atol=0.0,
            rtol=0.0,
        )
        and np.all(
            causality["audit_forbidden_keyword_rejections"].astype(
                np.bool_
            )
        )
    )
    expected_binding_roles = {
        role: str(identity["identity_sha256"])
        for role, identity in role_identities.items()
    }
    bound_payloads = {**models, "raw_low_rank": raw}
    bindings_valid = set(training_bindings) == set(
        SC_JEPA_ASSESSMENT_MODEL_NAMES
    ) and all(
        training_bindings[name]
        == {
            "model_payload_sha256": hashlib.sha256(
                _canonical_json_bytes(bound_payloads[name])
            ).hexdigest(),
            "fit_input_sha256": expected_binding_roles["fit"],
            "selection_input_sha256": expected_binding_roles[
                "selection"
            ],
            "calibration_input_sha256": expected_binding_roles[
                "calibration"
            ],
        }
        for name in SC_JEPA_ASSESSMENT_MODEL_NAMES
    )
    return {
        "frozen_interpretable_contract": (
            frozen_protocol
            and configs_match
            and source_identity_valid
            and reference_contracts_valid
            and event_binding_valid
        ),
        "implementation_sources_match_commit": source_identity_valid,
        "role_pairs_are_disjoint": disjoint,
        "fit_uses_40_in_distribution_pairs": (
            used_valid and len(used["fit"]) == 40
        ),
        "selection_uses_10_in_distribution_pairs": (
            used_valid and len(used["selection"]) == 10
        ),
        "calibration_uses_10_in_distribution_pairs": (
            used_valid and len(used["calibration"]) == 10
        ),
        "evaluation_uses_20_iid_and_10_transfer_pairs": (
            used_valid
            and len(used["evaluation_iid"]) == 20
            and len(used["evaluation_transfer"]) == 10
        ),
        "event_definition_fit_on_40_controls": (
            event_definition == derived_event_definition
            and event_binding_valid
            and int(
                derived_event_definition["control_trajectory_count"]
            )
            == 40
        ),
        "selection_and_calibration_precede_evaluation": phases_valid,
        "evaluation_not_used_for_fitting": (
            used_valid
            and bindings_valid
            and not (
                used["fit"]
                & (
                    used["evaluation_iid"]
                    | used["evaluation_transfer"]
                )
            )
        ),
        "model_accepts_only_histories_and_graph": (
            signatures_valid and causal_counterfactual_valid
        ),
        "factorial_changes_only_frozen_flags": configs_match,
        "fitted_models_are_bound_to_role_inputs": bindings_valid,
        "raw_and_pca_comparators_match_contract": (
            reference_contracts_valid
        ),
        "factorial_capacity_matches": (
            len(set(training_counts.values())) == 1
            and len(set(inference_counts.values())) == 1
        ),
        "restored_model_payloads_match": (
            all(
                models[name] == restored_models[name]
                for name in SC_JEPA_CELL_NAMES
            )
            and raw == restored_raw
        ),
        "serialized_fitted_state_is_finite": _all_numeric_finite(
            (models, restored_models, raw, restored_raw)
        ),
    }


def _derive_edge_metrics(
    *,
    metadata: Mapping[str, Any],
    model_bundle: Mapping[str, Any],
    event_definition: Mapping[str, Any],
    inference_counts: Mapping[str, int],
    latency_samples: Mapping[str, NDArray[np.float64]],
) -> Mapping[str, Mapping[str, float]]:
    peak_rss = float(metadata["peak_rss_bytes"])
    models = dict(model_bundle["models"])
    probes = dict(model_bundle["state_probes"])
    result = {}
    for name in SC_JEPA_ASSESSMENT_MODEL_NAMES:
        samples = np.asarray(
            latency_samples[name], dtype=np.float64
        )
        if samples.ndim != 1 or len(samples) < 1:
            raise ValueError("SC-JEPA latency evidence is invalid")
        if name == "codebook_multi":
            serialized = {
                "model": models[name],
                "event_definition": event_definition,
                "state_probe": probes["codebook_multi"],
            }
        elif name == "raw_low_rank":
            serialized = {
                "model": model_bundle["raw_low_rank"]
            }
        else:
            serialized = {"model": models[name]}
        result[name] = {
            "inference_parameter_count": float(
                inference_counts.get(name, 0)
            ),
            "serialized_candidate_sidecars_bytes": float(
                len(_canonical_json_bytes(serialized))
            ),
            "batch_one_cpu_latency_ms": float(np.mean(samples)),
            "batch_one_cpu_p95_latency_ms": float(
                np.quantile(samples, 0.95)
            ),
            "peak_rss_bytes": peak_rss,
            "latency_repetitions": float(len(samples)),
        }
    return result


def verify_artifact_manifest(root: Path) -> None:
    """Verify every retained file against the artifact manifest."""

    manifest = _read_object(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "sc_jepa_artifact_manifest"
    ):
        raise ValueError("unsupported SC-JEPA artifact manifest")
    recorded = dict(manifest["files"])
    expected_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "artifact-manifest.json"
    }
    if set(recorded) != expected_files:
        raise ValueError("SC-JEPA artifact manifest file set mismatch")
    for relative, identity in recorded.items():
        path = root / relative
        if (
            int(identity["bytes"]) != path.stat().st_size
            or str(identity["sha256"]) != _file_sha256(path)
        ):
            raise ValueError(
                "SC-JEPA artifact content identity mismatch"
            )


def _verify_source_identity(
    root: Path, data_identity: Mapping[str, Any]
) -> None:
    sources = dict(data_identity["implementation_sources"])
    if not sources:
        raise ValueError("SC-JEPA implementation source identity is empty")
    for relative, raw_identity in sources.items():
        identity = dict(raw_identity)
        source_path = str(identity["path"])
        retained = root / "reproduction" / Path(source_path).name
        retained_bytes = retained.read_bytes() if retained.is_file() else b""
        identity_bytes = (
            retained_bytes
            if identity["scope"] == "file"
            else _source_symbol_bytes(
                retained_bytes, str(identity["symbol"])
            )
        )
        if (
            not retained.is_file()
            or int(identity["bytes"]) != len(identity_bytes)
            or str(identity["sha256"])
            != hashlib.sha256(identity_bytes).hexdigest()
        ):
            raise ValueError(
                "SC-JEPA retained source identity does not match"
            )


def _source_symbol_bytes(source: bytes, symbol: str) -> bytes:
    text = source.decode("utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == symbol:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            return segment.encode("utf-8")
    raise ValueError(f"SC-JEPA dependency symbol is missing: {symbol}")


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_identity(values: NDArray[Any]) -> Mapping[str, Any]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    assessment = verify_stored_assessment(arguments.directory)
    print(_pretty_json(assessment), end="")


if __name__ == "__main__":
    main()
