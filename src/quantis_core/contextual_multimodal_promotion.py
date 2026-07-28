"""Frozen confirmation gates for the contextual metrics + logs JEPA."""

import hashlib
import json
import math
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .demand_conditioning import canonical_request_schedule
from .contextual_multimodal_world_model import (
    ContextualMultimodalJepaWorldModelDetector,
)


def assess_contextual_multimodal_promotion(
    training_result_path: Path,
    promotion_protocol_path: Path,
    *,
    repeat_training_result_path: Path,
    training_attestation_path: Path,
    repeat_training_attestation_path: Path,
    repository: Path,
    preregistered_git_commit: str,
) -> Mapping[str, Any]:
    """Verify provenance and assess two distinct training artifacts."""

    training_result_path = training_result_path.resolve()
    repeat_training_result_path = (
        repeat_training_result_path.resolve()
    )
    training_attestation_path = training_attestation_path.resolve()
    repeat_training_attestation_path = (
        repeat_training_attestation_path.resolve()
    )
    if (
        training_result_path == repeat_training_result_path
        or os.path.samefile(
            training_result_path,
            repeat_training_result_path,
        )
    ):
        raise ValueError(
            "promotion requires two distinct training result files"
        )
    if (
        training_attestation_path
        == repeat_training_attestation_path
        or os.path.samefile(
            training_attestation_path,
            repeat_training_attestation_path,
        )
    ):
        raise ValueError(
            "promotion requires two distinct execution attestations"
        )
    protocol = verify_contextual_multimodal_preregistration(
        repository,
        promotion_protocol_path,
        preregistered_git_commit,
    )
    result_bytes = training_result_path.read_bytes()
    repeat_result_bytes = repeat_training_result_path.read_bytes()
    first_execution = _validate_execution_attestation(
        training_attestation_path,
        training_result_path,
        result_bytes,
        protocol,
    )
    repeat_execution = _validate_execution_attestation(
        repeat_training_attestation_path,
        repeat_training_result_path,
        repeat_result_bytes,
        protocol,
    )
    if first_execution["execution_id"] == (
        repeat_execution["execution_id"]
    ):
        raise ValueError(
            "promotion execution attestations reuse one execution ID"
        )
    if int(first_execution["completed_unix_nano"]) > int(
        repeat_execution["started_unix_nano"]
    ):
        raise ValueError(
            "promotion execution attestations are not sequential"
        )
    return _assess_contextual_multimodal_promotion_bytes(
        result_bytes,
        protocol,
        repeat_result_bytes=repeat_result_bytes,
        preregistered_git_commit=preregistered_git_commit,
        execution_ids=(
            str(first_execution["execution_id"]),
            str(repeat_execution["execution_id"]),
        ),
        attestation_sha256s=(
            hashlib.sha256(
                training_attestation_path.read_bytes()
            ).hexdigest(),
            hashlib.sha256(
                repeat_training_attestation_path.read_bytes()
            ).hexdigest(),
        ),
    )


def _assess_contextual_multimodal_promotion_bytes(
    result_bytes: bytes,
    protocol: Mapping[str, Any],
    *,
    repeat_result_bytes: bytes,
    preregistered_git_commit: str,
    execution_ids: Tuple[str, str],
    attestation_sha256s: Tuple[str, str],
) -> Mapping[str, Any]:
    """Apply gates after the public entry point verifies provenance."""

    result = _parse_json_mapping(result_bytes)
    repeat_result = _parse_json_mapping(repeat_result_bytes)
    _validate_inputs(result, protocol)
    _validate_inputs(repeat_result, protocol)
    if not preregistered_git_commit:
        raise ValueError("promotion requires a preregistered commit")
    determinism_verified = result_bytes == repeat_result_bytes
    metrics = dict(result["metrics"])
    contextual = _alert_rate(metrics, "contextual_multimodal")
    metrics_only = _alert_rate(metrics, "metrics_only")
    capacity = _alert_rate(
        metrics,
        "capacity_matched_metrics_only",
    )
    shuffled = _alert_rate(metrics, "shuffled_logs")
    thresholds = dict(protocol["gates"])
    families = list(
        dict(result["schedule_transfer"])["validation_families"]
    )
    contextual_family_rates = [
        float(dict(family["contextual_multimodal"])["alert_rate"])
        for family in families
    ]
    metrics_family_rates = [
        float(dict(family["metrics_only"])["alert_rate"])
        for family in families
    ]
    no_worse_fraction = sum(
        candidate <= comparator
        for candidate, comparator in zip(
            contextual_family_rates,
            metrics_family_rates,
        )
    ) / len(families)
    diagnostics = dict(dict(result["model"])["diagnostics"])
    gates: Dict[str, Dict[str, Any]] = {
        "protocol_preregistered": {
            "observed": preregistered_git_commit,
            "required": "verified_git_commit",
            "passed": True,
        },
        "byte_identical_repeat": {
            "observed": determinism_verified,
            "required": True,
            "passed": determinism_verified,
        },
        "distinct_sequential_training_executions": {
            "observed": list(execution_ids),
            "required": "two_distinct_non_overlapping_executions",
            "passed": True,
        },
        "validation_alert_rate_at_most_maximum": {
            "observed": contextual,
            "maximum": float(
                thresholds["maximum_validation_alert_rate"]
            ),
            "passed": contextual
            <= float(thresholds["maximum_validation_alert_rate"]),
        },
        "maximum_schedule_family_alert_rate": {
            "observed": max(contextual_family_rates),
            "maximum": float(
                thresholds[
                    "maximum_schedule_family_alert_rate"
                ]
            ),
            "passed": max(contextual_family_rates)
            <= float(
                thresholds[
                    "maximum_schedule_family_alert_rate"
                ]
            ),
        },
        "no_worse_than_metrics_only": {
            "observed": contextual,
            "comparator": metrics_only,
            "passed": contextual <= metrics_only,
        },
        "no_worse_than_capacity_matched_metrics_only": {
            "observed": contextual,
            "comparator": capacity,
            "passed": contextual <= capacity,
        },
        "strictly_better_than_shuffled_logs": {
            "observed": contextual,
            "comparator": shuffled,
            "passed": contextual < shuffled,
        },
        "no_worse_schedule_family_fraction": {
            "observed": no_worse_fraction,
            "minimum": float(
                thresholds[
                    "minimum_no_worse_schedule_family_fraction"
                ]
            ),
            "passed": no_worse_fraction
            >= float(
                thresholds[
                    "minimum_no_worse_schedule_family_fraction"
                ]
            ),
        },
        "metric_active_latent_rank": {
            "observed": float(
                diagnostics["metric_effective_rank"]
            ),
            "minimum": float(
                thresholds["minimum_metric_effective_rank"]
            ),
            "passed": float(
                diagnostics["metric_effective_rank"]
            )
            >= float(
                thresholds["minimum_metric_effective_rank"]
            ),
        },
        "log_active_latent_rank": {
            "observed": float(diagnostics["log_effective_rank"]),
            "minimum": float(
                thresholds["minimum_log_effective_rank"]
            ),
            "passed": float(diagnostics["log_effective_rank"])
            >= float(thresholds["minimum_log_effective_rank"]),
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "schema_version": 1,
        "kind": "contextual_multimodal_jepa_promotion_assessment",
        "status": "passed" if passed else "failed",
        "publication_eligible": passed,
        "gates": gates,
        "training_result_sha256": hashlib.sha256(
            result_bytes
        ).hexdigest(),
        "repeat_training_result_sha256": hashlib.sha256(
            repeat_result_bytes
        ).hexdigest(),
        "training_execution_attestation_sha256s": list(
            attestation_sha256s
        ),
        "eligible_model_artifact_sha256": dict(
            result["protocol"]
        )["model_artifact_sha256"],
        "promotion_protocol_sha256": _canonical_sha256(protocol),
        "evidence_boundary": (
            "eligibility applies only to the separately serialized "
            "contextual metrics-plus-logs model"
        ),
        "preregistered_git_commit": preregistered_git_commit,
    }


def validate_contextual_multimodal_promotion_corpus(
    corpus: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Bind a promotion label to the frozen inner corpus provenance."""

    _validate_protocol(protocol)
    if (
        corpus.get("schema_version") != 2
        or corpus.get("kind")
        != "contextual_multimodal_telemetry_corpus"
    ):
        raise ValueError("unsupported contextual promotion corpus")
    expected_training = list(protocol["training_case_ids"])
    expected_validation = list(protocol["validation_case_ids"])
    contextual_protocol = dict(corpus["protocol"])
    if contextual_protocol.get("evidence_assignment") != (
        "deferred_to_training_protocol"
    ):
        raise ValueError(
            "contextual corpus has an invalid evidence assignment"
        )
    if contextual_protocol.get("training_case_ids") != expected_training:
        raise ValueError(
            "inner contextual training cases differ from protocol"
        )
    if contextual_protocol.get("validation_case_ids") != (
        expected_validation
    ):
        raise ValueError(
            "inner contextual validation cases differ from protocol"
        )
    design = dict(protocol["corpus"])
    if (
        contextual_protocol.get("target_horizons")
        != design["target_horizons"]
        or contextual_protocol.get("target_block_size")
        != design["target_block_size"]
    ):
        raise ValueError(
            "contextual target design differs from protocol"
        )

    base = dict(corpus["base_corpus"])
    if base.get("kind") != "multimodal_telemetry_corpus":
        raise ValueError("promotion corpus base is not multimodal")
    _validate_split_spec(
        dict(dict(base["protocol"])["split_spec"]),
        expected_training,
        expected_validation,
        design,
    )
    metric_corpus = dict(base["metric_corpus"])
    metric_protocol = dict(metric_corpus["protocol"])
    _validate_split_spec(
        dict(metric_protocol["split_spec"]),
        expected_training,
        expected_validation,
        design,
    )
    metric_vocabulary = dict(protocol["metric_vocabulary"])
    if (
        metric_protocol.get("feature_schema_id")
        != metric_vocabulary["feature_schema_id"]
        or metric_protocol.get("feature_spec_sha256")
        != metric_vocabulary["feature_spec_sha256"]
    ):
        raise ValueError(
            "metric feature schema differs from protocol"
        )
    vocabulary = dict(protocol["log_vocabulary"])
    raw_log_spec = dict(base["log_feature_spec"])
    if _canonical_sha256(raw_log_spec) != vocabulary[
        "feature_schema_id"
    ]:
        raise ValueError(
            "raw log feature schema differs from protocol"
        )
    raw_feature_names = [
        str(dict(feature)["name"])
        for feature in raw_log_spec["features"]
    ]
    if raw_feature_names != list(vocabulary["raw_feature_names"]):
        raise ValueError(
            "raw log feature names differ from protocol"
        )
    preprocessing = dict(corpus["preprocessing"])
    transformer = dict(
        dict(preprocessing["logs"])["transformer"]
    )
    if transformer.get("features") != list(
        vocabulary["semantic_feature_names"]
    ):
        raise ValueError(
            "semantic log features differ from protocol"
        )
    if dict(base["protocol"]).get("log_window_assignment") != (
        "event_time_metric_boundaries"
    ):
        raise ValueError(
            "promotion logs are not assigned by metric event time"
        )

    expected_case_ids = expected_training + expected_validation
    metric_runs = dict(metric_protocol["runs"])
    log_runs = dict(dict(base["protocol"])["runs"])
    if list(metric_runs) != expected_case_ids or list(log_runs) != (
        expected_case_ids
    ):
        raise ValueError(
            "inner corpus run order or membership differs from protocol"
        )
    schedule_by_case_id = _expected_schedule_by_case_id(protocol)
    for case_id in expected_case_ids:
        run = dict(metric_runs[case_id])
        worker_replicas = _case_worker_replicas(case_id)
        if (
            run.get("case_id") != case_id
            or run.get("canonical_request_schedule")
            != list(schedule_by_case_id[case_id])
            or run.get("application_build_context_sha256")
            != design["application_build_context_sha256"]
            or run.get("application_api_request_queue_size")
            != design[
                "expected_application_api_request_queue_size"
            ]
            or run.get("normal_interval")
            != [0, design["point_count"]]
            or run.get("normal_point_count")
            != design["point_count"]
            or run.get("topology_id")
            != f"workers-{worker_replicas}"
            or run.get("worker_replicas") != worker_replicas
        ):
            raise ValueError(
                f"inner metric provenance differs for {case_id}"
            )
        if dict(log_runs[case_id]).get("case_id") != case_id:
            raise ValueError(
                f"inner log provenance differs for {case_id}"
            )


def verify_contextual_multimodal_preregistration(
    repository: Path,
    protocol_path: Path,
    commit: str,
) -> Mapping[str, Any]:
    """Verify protocol and frozen bytes at one committed Git state."""

    repository = repository.resolve()
    resolved_commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != resolved_commit:
        raise ValueError(
            "preregistered commit must be a full immutable Git ID"
        )
    protocol_path = protocol_path.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol: Mapping[str, Any] = dict(json.loads(protocol_bytes))
    _validate_protocol(protocol)
    protocol_relative = str(protocol_path.relative_to(repository))
    if _git_bytes(
        repository,
        commit,
        protocol_relative,
    ) != protocol_bytes:
        raise ValueError(
            "working protocol differs from preregistration commit"
        )
    frozen_files = protocol.get("frozen_files")
    if not isinstance(frozen_files, Mapping) or not frozen_files:
        raise ValueError("promotion protocol has no frozen files")
    for relative_path, expected_sha256 in frozen_files.items():
        if not isinstance(relative_path, str) or not isinstance(
            expected_sha256,
            str,
        ):
            raise ValueError("invalid frozen file entry")
        working = (repository / relative_path).read_bytes()
        if hashlib.sha256(working).hexdigest() != expected_sha256:
            raise ValueError(
                f"working frozen file hash mismatch: {relative_path}"
            )
        if _git_bytes(
            repository,
            commit,
            relative_path,
        ) != working:
            raise ValueError(
                f"frozen file differs from commit: {relative_path}"
            )
    return protocol


def write_contextual_multimodal_promotion_assessment(
    assessment: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, Path]:
    """Write machine-readable and human-readable promotion evidence."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "promotion.json"
    report_path = output / "report.md"
    json_path.write_text(
        json.dumps(
            assessment,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    lines = [
        "# Contextual metrics + logs JEPA promotion assessment",
        "",
        f"Status: **{str(assessment['status']).upper()}**",
        "",
        "This decision applies only to the separately serialized "
        "contextual metrics-plus-logs model.",
        "",
        "## Preregistered gates",
        "",
    ]
    for name, raw_gate in dict(assessment["gates"]).items():
        gate = dict(raw_gate)
        lines.append(
            f"- {'PASS' if gate['passed'] else 'FAIL'} — {name}"
        )
    report_path.write_text("\n".join(lines) + "\n")
    return {"promotion": json_path, "report": report_path}


def _validate_execution_attestation(
    attestation_path: Path,
    training_result_path: Path,
    result_bytes: bytes,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    if attestation_path.parent != training_result_path.parent:
        raise ValueError(
            "execution attestation is not beside its training result"
        )
    attestation = _parse_json_mapping(
        attestation_path.read_bytes()
    )
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind")
        != (
            "contextual_multimodal_jepa_"
            "training_execution_attestation"
        )
    ):
        raise ValueError("unsupported training execution attestation")
    execution_id = attestation.get("execution_id")
    try:
        parsed_execution_id = uuid.UUID(str(execution_id))
    except ValueError as error:
        raise ValueError(
            "training execution ID is invalid"
        ) from error
    if (
        parsed_execution_id.version != 4
        or str(parsed_execution_id) != execution_id
    ):
        raise ValueError(
            "training execution ID must be canonical UUIDv4"
        )
    started = attestation.get("started_unix_nano")
    completed = attestation.get("completed_unix_nano")
    process_id = attestation.get("process_id")
    if (
        not isinstance(started, int)
        or isinstance(started, bool)
        or started <= 0
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or completed <= started
        or not isinstance(process_id, int)
        or isinstance(process_id, bool)
        or process_id <= 0
    ):
        raise ValueError(
            "training execution timing or process ID is invalid"
        )
    result = _parse_json_mapping(result_bytes)
    result_protocol = dict(result["protocol"])
    expected = {
        "output_directory": str(
            training_result_path.parent.resolve()
        ),
        "training_result_sha256": hashlib.sha256(
            result_bytes
        ).hexdigest(),
        "corpus_metadata_sha256": result_protocol[
            "corpus_metadata_sha256"
        ],
        "model_artifact_sha256": result_protocol[
            "model_artifact_sha256"
        ],
        "promotion_protocol_sha256": _canonical_sha256(protocol),
    }
    if any(
        attestation.get(name) != value
        for name, value in expected.items()
    ):
        raise ValueError(
            "training execution attestation is not bound to result"
        )
    return attestation


def _validate_inputs(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    _validate_protocol(protocol)
    if (
        result.get("schema_version") != 1
        or result.get("kind")
        != (
            "contextual_multimodal_jepa_world_model_"
            "promotion_confirmation"
        )
    ):
        raise ValueError("unsupported contextual training result")
    if result.get("evidence_mode") != "promotion_confirmation":
        raise ValueError(
            "result evidence mode is not promotion confirmation"
        )
    if dict(result["config"]) != dict(protocol["training_config"]):
        raise ValueError(
            "training configuration differs from promotion protocol"
        )
    result_protocol = dict(result["protocol"])
    if result_protocol.get("model_selection_status") != (
        "fixed_promotion_confirmation"
    ):
        raise ValueError("result is not promotion confirmation evidence")
    if result_protocol.get("validation_use") != (
        "fixed_confirmation_no_adaptive_reuse"
    ):
        raise ValueError("validation use is not confirmation-only")
    if result_protocol.get("promotion_protocol_sha256") != (
        _canonical_sha256(protocol)
    ):
        raise ValueError(
            "training result is not bound to promotion protocol"
        )
    if result_protocol.get("training_uses_validation_windows") is not False:
        raise ValueError("promotion training used validation windows")
    if dict(result_protocol["cross_validation"]).get("status") != (
        "disabled"
    ):
        raise ValueError(
            "promotion confirmation cannot perform model selection"
        )
    if dict(result["cross_validation"]).get("status") != "disabled":
        raise ValueError(
            "top-level cross-validation was not disabled"
        )
    selection = dict(result["selection"])
    if (
        selection.get("status") != "not_assessed"
        or selection.get("publication_eligible") is not False
    ):
        raise ValueError(
            "promotion training contains adaptive selection evidence"
        )
    for split in ("training", "validation"):
        observed = list(result_protocol[f"{split}_case_ids"])
        expected = list(protocol[f"{split}_case_ids"])
        if observed != expected:
            raise ValueError(
                f"{split} cases differ from promotion protocol"
            )
    if list(result_protocol.get("controls", ())) != list(
        protocol["controls"]
    ):
        raise ValueError(
            "serialized controls differ from promotion protocol"
        )
    if dict(result_protocol.get("training_runtime", {})) != dict(
        protocol["training_runtime"]
    ):
        raise ValueError(
            "training runtime differs from promotion protocol"
        )
    corpus_artifact = dict(result["corpus"])
    if result_protocol.get("corpus_metadata_sha256") != (
        _canonical_sha256(corpus_artifact)
    ):
        raise ValueError("corpus metadata hash does not match artifact")
    validate_contextual_multimodal_promotion_corpus(
        corpus_artifact, protocol
    )
    model_artifacts = {
        "model": dict(result["model"]),
        "metrics_only_model": dict(result["metrics_only_model"]),
        "capacity_matched_metrics_only_model": dict(
            result["capacity_matched_metrics_only_model"]
        ),
        "shuffled_log_model": dict(
            result["shuffled_log_model"]
        ),
        "log_only_model": dict(result["log_only_model"]),
    }
    if result_protocol.get("model_artifact_sha256") != (
        _canonical_sha256(model_artifacts["model"])
    ):
        raise ValueError("model artifact hash does not match artifact")
    expected_control_hashes = {
        name: _canonical_sha256(model_artifacts[name])
        for name in (
            "metrics_only_model",
            "capacity_matched_metrics_only_model",
            "shuffled_log_model",
            "log_only_model",
        )
    }
    if dict(
        result_protocol.get("control_artifact_sha256s", {})
    ) != expected_control_hashes:
        raise ValueError(
            "control artifact hashes do not match artifacts"
        )
    shuffled_artifact = model_artifacts["shuffled_log_model"]
    if dict(shuffled_artifact["control_protocol"]) != dict(
        protocol["shuffled_log_control"]
    ):
        raise ValueError(
            "shuffled-log control differs from promotion protocol"
        )
    metric_feature_names = list(
        dict(protocol["metric_vocabulary"])[
            "semantic_feature_names"
        ]
    )
    semantic_feature_names = list(
        dict(protocol["log_vocabulary"])[
            "semantic_feature_names"
        ]
    )
    training_config = dict(protocol["training_config"])
    expected_dimensions = {
        "model": (
            training_config["metric_latent_dimension"],
            training_config["log_latent_dimension"],
        ),
        "metrics_only_model": (
            training_config["metric_latent_dimension"],
            0,
        ),
        "capacity_matched_metrics_only_model": (
            training_config["metric_latent_dimension"]
            + training_config["log_latent_dimension"],
            0,
        ),
        "shuffled_log_model": (
            training_config["metric_latent_dimension"],
            training_config["log_latent_dimension"],
        ),
        "log_only_model": (
            0,
            training_config["log_latent_dimension"],
        ),
    }
    preprocessing = dict(corpus_artifact["preprocessing"])
    for artifact_name, artifact in model_artifacts.items():
        metric_dimension, log_dimension = expected_dimensions[
            artifact_name
        ]
        _validate_model_artifact(
            artifact,
            training_config,
            metric_latent_dimension=int(metric_dimension),
            log_latent_dimension=int(log_dimension),
            metric_feature_names=metric_feature_names,
            log_feature_names=semantic_feature_names,
            preprocessing=preprocessing,
        )
    metrics = dict(result["metrics"])
    _validate_metrics(metrics)
    families = list(
        dict(result["schedule_transfer"])["validation_families"]
    )
    if not families:
        raise ValueError(
            "promotion result has no validation schedule families"
        )
    corpus = dict(protocol.get("corpus", {}))
    expected_family_count = int(
        corpus.get("validation_family_count", len(families))
    )
    if len(families) != expected_family_count:
        raise ValueError(
            "promotion validation family count differs from protocol"
        )
    _validate_schedule_transfer_families(families, protocol)
    for family in families:
        for model_name in (
            "contextual_multimodal",
            "metrics_only",
            "capacity_matched_metrics_only",
            "shuffled_logs",
        ):
            _validate_score_metrics(
                dict(family[model_name]),
                f"schedule family {model_name}",
            )
    _validate_family_aggregates(families, metrics)
    diagnostics = dict(model_artifacts["model"]["diagnostics"])
    _validate_effective_rank(
        diagnostics.get("metric_effective_rank"),
        int(training_config["metric_latent_dimension"]),
        "metric",
    )
    _validate_effective_rank(
        diagnostics.get("log_effective_rank"),
        int(training_config["log_latent_dimension"]),
        "log",
    )


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("kind")
        != "contextual_multimodal_jepa_promotion_v1"
    ):
        raise ValueError("unsupported contextual promotion protocol")


def _validate_model_artifact(
    artifact: Mapping[str, Any],
    training_config: Mapping[str, Any],
    *,
    metric_latent_dimension: int,
    log_latent_dimension: int,
    metric_feature_names: Sequence[str],
    log_feature_names: Sequence[str],
    preprocessing: Mapping[str, Any],
) -> None:
    try:
        ContextualMultimodalJepaWorldModelDetector.from_dict(
            artifact
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "promotion model artifact is incomplete or inconsistent"
        ) from error
    if (
        artifact.get("schema_version") != 1
        or artifact.get("kind")
        != "contextual_multimodal_jepa_world_model_v1"
    ):
        raise ValueError("unsupported promotion model artifact")
    expected_config = {
        name: training_config[name]
        for name in (
            "pretraining_epochs",
            "predictor_refinement_epochs",
            "learning_rate",
            "ema_decay",
            "weight_decay",
            "loss",
            "huber_delta",
            "auxiliary_loss_weight",
            "rollout_loss_weight",
            "calibration_quantile",
            "seed",
        )
    }
    observed_config = {
        name: artifact.get(name) for name in expected_config
    }
    v2_recipe_defaults = {
        "modality_mask_probability": 0.0,
        "log_self_loss_multiplier": 1.0,
        "cross_modal_loss_multiplier": 1.0,
    }
    observed_v2_recipe = {
        name: artifact.get(name, default)
        for name, default in v2_recipe_defaults.items()
    }
    if (
        observed_config != expected_config
        or observed_v2_recipe != v2_recipe_defaults
        or artifact.get("metric_latent_dimension")
        != metric_latent_dimension
        or artifact.get("log_latent_dimension")
        != log_latent_dimension
    ):
        raise ValueError(
            "serialized model configuration differs from protocol"
        )
    if (
        list(artifact.get("metric_feature_names", ()))
        != list(metric_feature_names)
        or list(artifact.get("log_feature_names", ()))
        != list(log_feature_names)
        or list(artifact.get("control_feature_names", ()))
        != ["request_demand", "worker_replicas"]
    ):
        raise ValueError(
            "serialized model features differ from protocol"
        )
    if dict(artifact.get("preprocessing", {})) != dict(
        preprocessing
    ):
        raise ValueError(
            "serialized model preprocessing differs from corpus"
        )


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    expected_models = {
        "contextual_multimodal",
        "metrics_only",
        "capacity_matched_metrics_only",
        "shuffled_logs",
        "log_only",
        "modality_dropout",
    }
    if set(metrics) != expected_models:
        raise ValueError(
            "promotion metrics omit required controls or diagnostics"
        )
    for model_name in expected_models - {"modality_dropout"}:
        splits = dict(metrics[model_name])
        if set(splits) != {"training", "validation"}:
            raise ValueError(
                f"{model_name} metrics have invalid split coverage"
            )
        for split_name, raw_metrics in splits.items():
            _validate_score_metrics(
                dict(raw_metrics),
                f"{model_name} {split_name}",
            )
    dropout = dict(metrics["modality_dropout"])
    if set(dropout) != {
        "metric_context_only",
        "log_context_only",
    }:
        raise ValueError(
            "promotion modality-dropout diagnostics are incomplete"
        )
    for diagnostic_name, raw_metrics in dropout.items():
        _validate_score_metrics(
            dict(raw_metrics),
            f"modality dropout {diagnostic_name}",
        )


def _validate_family_aggregates(
    families: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> None:
    for model_name in (
        "contextual_multimodal",
        "metrics_only",
        "capacity_matched_metrics_only",
        "shuffled_logs",
    ):
        family_metrics = [
            dict(family[model_name]) for family in families
        ]
        aggregate = dict(
            dict(metrics[model_name])["validation"]
        )
        if (
            sum(
                int(item["window_count"])
                for item in family_metrics
            )
            != aggregate["window_count"]
            or sum(int(item["alerts"]) for item in family_metrics)
            != aggregate["alerts"]
        ):
            raise ValueError(
                f"{model_name} family metrics do not match "
                "validation aggregate"
            )


def _validate_score_metrics(
    metrics: Mapping[str, Any],
    label: str,
) -> None:
    window_count = metrics.get("window_count")
    alerts = metrics.get("alerts")
    alert_rate = metrics.get("alert_rate")
    if (
        not isinstance(window_count, int)
        or isinstance(window_count, bool)
        or window_count <= 0
        or not isinstance(alerts, int)
        or isinstance(alerts, bool)
        or alerts < 0
        or alerts > window_count
        or not _is_finite_number(alert_rate)
    ):
        raise ValueError(f"{label} contains invalid alert metrics")
    assert isinstance(window_count, int)
    assert isinstance(alerts, int)
    assert isinstance(alert_rate, (int, float))
    rate = float(alert_rate)
    if (
        rate < 0.0
        or rate > 1.0
        or not math.isclose(
            rate,
            alerts / window_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{label} contains inconsistent alert rate")


def _validate_effective_rank(
    value: object,
    maximum: int,
    label: str,
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"{label} effective rank is invalid"
        )
    rank = float(value)
    if (
        not math.isfinite(rank)
        or rank < 0.0
        or rank > maximum
    ):
        raise ValueError(
            f"{label} effective rank is invalid"
        )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_split_spec(
    split: Mapping[str, Any],
    expected_training: Sequence[str],
    expected_validation: Sequence[str],
    design: Mapping[str, Any],
) -> None:
    if (
        split.get("training_case_ids") != list(expected_training)
        or split.get("validation_case_ids")
        != list(expected_validation)
        or split.get("lookback") != design["lookback"]
        or split.get("expected_application_api_request_queue_size")
        != design["expected_application_api_request_queue_size"]
    ):
        raise ValueError(
            "inner corpus split differs from promotion protocol"
        )


def _expected_schedule_by_case_id(
    protocol: Mapping[str, Any],
) -> Mapping[str, Tuple[int, ...]]:
    design = dict(protocol["corpus"])
    schedule_families = list(design["schedule_families"])
    expected: Dict[str, Tuple[int, ...]] = {}
    for case_id in (
        list(protocol["training_case_ids"])
        + list(protocol["validation_case_ids"])
    ):
        family_index = _case_family_index(case_id)
        family = dict(schedule_families[family_index - 1])
        expected[str(case_id)] = canonical_request_schedule(
            int(family["requests_per_window"]),
            tuple(
                int(value)
                for value in family["load_pattern_offsets"]
            ),
        )
    return expected


def _case_family_index(case_id: str) -> int:
    marker = "-f"
    try:
        return int(case_id.split(marker, 1)[1][:2])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"promotion case has invalid family identity: {case_id}"
        ) from error


def _case_worker_replicas(case_id: str) -> int:
    try:
        return int(case_id.rsplit("-w", 1)[1].split("-", 1)[0])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"promotion case has invalid topology identity: {case_id}"
        ) from error


def _schedule_sha256(schedule: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(schedule),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_schedule_transfer_families(
    families: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    design = dict(protocol["corpus"])
    training_family_count = int(design["training_family_count"])
    schedules = _expected_schedule_by_case_id(protocol)
    expected: Dict[str, set[str]] = {}
    for case_id in protocol["validation_case_ids"]:
        schedule_hash = _schedule_sha256(schedules[str(case_id)])
        expected.setdefault(schedule_hash, set()).add(str(case_id))
        if _case_family_index(str(case_id)) <= training_family_count:
            raise ValueError(
                "validation case belongs to a training family"
            )
    observed: Dict[str, set[str]] = {}
    all_case_ids = []
    for raw_family in families:
        family = dict(raw_family)
        schedule_hash = str(family["schedule_sha256"])
        case_ids = [str(value) for value in family["case_ids"]]
        if schedule_hash in observed:
            raise ValueError(
                "promotion schedule-transfer families are duplicated"
            )
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "promotion family contains duplicate cases"
            )
        observed[schedule_hash] = set(case_ids)
        all_case_ids.extend(case_ids)
    if (
        observed != expected
        or len(all_case_ids) != len(set(all_case_ids))
    ):
        raise ValueError(
            "promotion schedule-transfer case coverage differs "
            "from protocol"
        )


def _alert_rate(
    metrics: Mapping[str, Any],
    model_name: str,
) -> float:
    return float(
        dict(dict(metrics[model_name])["validation"])[
            "alert_rate"
        ]
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_json_mapping(content: bytes) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    def reject_duplicate_keys(
        pairs: Sequence[Tuple[str, Any]],
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate JSON key: {key}")
            parsed[key] = value
        return parsed

    parsed = json.loads(
        content,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(parsed, Mapping):
        raise ValueError("promotion result must be a JSON object")
    return dict(parsed)


def _git_bytes(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
