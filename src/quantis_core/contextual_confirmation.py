"""Publication-oriented confirmation for contextual metrics + logs JEPA."""

import hashlib
import itertools
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


CONFIRMATION_PROTOCOL_KIND = (
    "contextual_multimodal_jepa_confirmation_v2"
)


@dataclass(frozen=True)
class ConfirmationCollectionCase:
    """One preregistered case assigned to a balanced collection lane."""

    case_id: str
    family: int
    worker_replicas: int
    batch: int
    lane: int
    split: str


def confirmation_case_ids(
    protocol: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Derive ordered case identities from the self-contained protocol."""

    _validate_confirmation_protocol_shape(protocol)
    corpus = dict(protocol["corpus"])
    prefix = str(corpus["case_prefix"])
    seed_label = int(corpus["seed_label"])
    workers = tuple(int(value) for value in corpus["worker_replicas"])
    training_family_count = int(corpus["training_family_count"])
    validation_family_count = int(corpus["validation_family_count"])

    def cases(start: int, stop: int) -> Tuple[str, ...]:
        return tuple(
            f"{prefix}-f{family:02d}-w{worker}-{seed_label}"
            for family in range(start, stop)
            for worker in workers
        )

    boundary = training_family_count + 1
    return (
        cases(1, boundary),
        cases(
            boundary,
            boundary + validation_family_count,
        ),
    )


def plan_parallel_confirmation_collection(
    protocol: Mapping[str, Any],
) -> Tuple[ConfirmationCollectionCase, ...]:
    """Assign all cases to balanced, full-width collection batches."""

    training, validation = confirmation_case_ids(protocol)
    corpus = dict(protocol["corpus"])
    workers = tuple(int(value) for value in corpus["worker_replicas"])
    jobs = int(dict(protocol["collection"])["parallel_jobs"])
    if jobs != len(workers):
        raise ValueError(
            "confirmation collection requires one lane per topology"
        )
    if tuple(sorted(workers)) != tuple(range(1, jobs + 1)):
        raise ValueError(
            "confirmation worker replicas must be contiguous lanes"
        )
    training_set = set(training)
    family_count = int(corpus["training_family_count"]) + int(
        corpus["validation_family_count"]
    )
    family_order = tuple(
        int(value)
        for value in dict(protocol["collection"])["family_order"]
    )
    if (
        len(family_order) != family_count
        or set(family_order) != set(range(1, family_count + 1))
    ):
        raise ValueError(
            "confirmation collection family order is not a permutation"
        )
    plans = []
    for batch, family in enumerate(family_order, start=1):
        for worker_replicas in workers:
            lane = (
                (worker_replicas - 1 + family - 1) % jobs
            ) + 1
            case_id = (
                f"{corpus['case_prefix']}-f{family:02d}"
                f"-w{worker_replicas}-{corpus['seed_label']}"
            )
            plans.append(
                ConfirmationCollectionCase(
                    case_id=case_id,
                    family=family,
                    worker_replicas=worker_replicas,
                    batch=batch,
                    lane=lane,
                    split=(
                        "training"
                        if case_id in training_set
                        else "validation"
                    ),
                )
            )
    return tuple(plans)


def assess_confirmation_results(
    results: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    determinism_verified: bool,
) -> Mapping[str, Any]:
    """Aggregate fixed seeds at the preregistered schedule-family unit."""

    _validate_confirmation_protocol_shape(protocol)
    expected_seeds = tuple(
        int(value) for value in protocol["training_seeds"]
    )
    observed_seeds = tuple(
        int(dict(result["config"])["seed"])
        for result in results
    )
    if (
        len(observed_seeds) != len(set(observed_seeds))
        or set(observed_seeds) != set(expected_seeds)
    ):
        raise ValueError(
            "confirmation training seed coverage differs from protocol"
        )
    result_by_seed = {
        int(dict(result["config"])["seed"]): result
        for result in results
    }
    ordered_results = tuple(
        result_by_seed[seed] for seed in expected_seeds
    )
    thresholds = dict(protocol["gates"])
    family_rates = _family_alert_rates(ordered_results)
    contextual_family = _mean_by_family(
        family_rates["contextual_multimodal"]
    )
    metric_family = _mean_by_family(
        family_rates["metrics_only"]
    )
    capacity_family = _mean_by_family(
        family_rates["capacity_matched_metrics_only"]
    )
    shuffled_family = _mean_by_family(
        family_rates["shuffled_logs"]
    )
    capacity_alert_test = _paired_randomization(
        tuple(
            capacity_family[family] - contextual_family[family]
            for family in sorted(contextual_family)
        )
    )
    shuffled_alert_test = _paired_randomization(
        tuple(
            shuffled_family[family] - contextual_family[family]
            for family in sorted(contextual_family)
        )
    )
    maximum_p_value = float(
        thresholds["maximum_paired_randomization_p_value"]
    )
    no_worse_fraction = sum(
        contextual_family[family] <= metric_family[family]
        for family in contextual_family
    ) / len(contextual_family)
    seed_wins = [
        _aggregate_alert_rate(result, "contextual_multimodal")
        < _aggregate_alert_rate(
            result,
            "capacity_matched_metrics_only",
        )
        and _aggregate_alert_rate(result, "contextual_multimodal")
        < _aggregate_alert_rate(result, "shuffled_logs")
        for result in ordered_results
    ]
    seed_win_fraction = sum(seed_wins) / len(seed_wins)

    transfer = _transfer_summary(ordered_results, protocol)
    contextual_transfer = transfer["contextual_multimodal"]
    raw_transfer = transfer["raw_context_ridge"]
    pca_dimension = int(
        dict(protocol["representation_transfer"])[
            "pca_context_dimension"
        ]
    )
    pca_name = (
        f"pca_{pca_dimension}_context_ridge"
    )
    contextual_to_raw = (
        contextual_transfer["mean_error"]
        / max(raw_transfer["mean_error"], 1e-12)
    )
    capacity_probe_test = _paired_randomization(
        tuple(
            transfer["capacity_matched_metrics_only"]["families"][
                family
            ]
            - contextual_transfer["families"][family]
            for family in sorted(contextual_transfer["families"])
        )
    )
    shuffled_probe_test = _paired_randomization(
        tuple(
            transfer["shuffled_logs"]["families"][family]
            - contextual_transfer["families"][family]
            for family in sorted(contextual_transfer["families"])
        )
    )
    pca_probe_test = _paired_randomization(
        tuple(
            transfer[pca_name]["families"][family]
            - contextual_transfer["families"][family]
            for family in sorted(contextual_transfer["families"])
        )
    )
    metric_rank = min(
        float(dict(dict(result["model"])["diagnostics"])[
            "metric_effective_rank"
        ])
        for result in ordered_results
    )
    log_rank = min(
        float(dict(dict(result["model"])["diagnostics"])[
            "log_effective_rank"
        ])
        for result in ordered_results
    )
    contextual_alert_rate = sum(
        _aggregate_alert_rate(result, "contextual_multimodal")
        for result in ordered_results
    ) / len(ordered_results)
    maximum_family_alert_rate = max(
        contextual_family.values()
    )
    gates: Dict[str, Dict[str, Any]] = {
        "deterministic_primary_repeat": {
            "observed": determinism_verified,
            "required": True,
            "passed": determinism_verified,
        },
        "validation_alert_rate_at_most_maximum": {
            "observed": contextual_alert_rate,
            "maximum": float(
                thresholds["maximum_validation_alert_rate"]
            ),
            "passed": contextual_alert_rate
            <= float(thresholds["maximum_validation_alert_rate"]),
        },
        "every_schedule_family_at_most_maximum": {
            "observed": maximum_family_alert_rate,
            "maximum": float(
                thresholds["maximum_schedule_family_alert_rate"]
            ),
            "passed": maximum_family_alert_rate
            <= float(
                thresholds["maximum_schedule_family_alert_rate"]
            ),
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
        "alert_better_than_capacity_matched": {
            **capacity_alert_test,
            "maximum_p_value": maximum_p_value,
            "passed": capacity_alert_test["improvement"] > 0.0
            and capacity_alert_test["p_value"] <= maximum_p_value,
        },
        "alert_better_than_shuffled_logs": {
            **shuffled_alert_test,
            "maximum_p_value": maximum_p_value,
            "passed": shuffled_alert_test["improvement"] > 0.0
            and shuffled_alert_test["p_value"] <= maximum_p_value,
        },
        "seed_stability": {
            "observed": seed_win_fraction,
            "minimum": float(
                thresholds["minimum_seed_win_fraction"]
            ),
            "passed": seed_win_fraction
            >= float(thresholds["minimum_seed_win_fraction"]),
        },
        "metric_active_latent_rank": {
            "observed": metric_rank,
            "minimum": float(
                thresholds["minimum_metric_effective_rank"]
            ),
            "passed": metric_rank
            >= float(thresholds["minimum_metric_effective_rank"]),
        },
        "log_active_latent_rank": {
            "observed": log_rank,
            "minimum": float(
                thresholds["minimum_log_effective_rank"]
            ),
            "passed": log_rank
            >= float(thresholds["minimum_log_effective_rank"]),
        },
        "completed_frozen_probe_targets": {
            "observed": contextual_transfer[
                "minimum_completed_targets"
            ],
            "minimum": int(
                thresholds["minimum_completed_probe_targets"]
            ),
            "passed": contextual_transfer[
                "minimum_completed_targets"
            ]
            >= int(thresholds["minimum_completed_probe_targets"]),
        },
        "compressed_state_retains_raw_probe_performance": {
            "observed_error_ratio": contextual_to_raw,
            "maximum": float(
                thresholds[
                    "maximum_contextual_to_raw_probe_error_ratio"
                ]
            ),
            "passed": contextual_to_raw
            <= float(
                thresholds[
                    "maximum_contextual_to_raw_probe_error_ratio"
                ]
            ),
        },
        "compressed_state_predicts_better_than_mean": {
            "observed": contextual_transfer["mean_error"],
            "maximum": float(
                thresholds[
                    "maximum_contextual_probe_normalized_mse"
                ]
            ),
            "passed": contextual_transfer["mean_error"]
            <= float(
                thresholds[
                    "maximum_contextual_probe_normalized_mse"
                ]
            ),
        },
        "frozen_state_better_than_capacity_matched": {
            **capacity_probe_test,
            "maximum_p_value": maximum_p_value,
            "passed": capacity_probe_test["improvement"] > 0.0
            and capacity_probe_test["p_value"] <= maximum_p_value,
        },
        "frozen_state_better_than_shuffled_logs": {
            **shuffled_probe_test,
            "maximum_p_value": maximum_p_value,
            "passed": shuffled_probe_test["improvement"] > 0.0
            and shuffled_probe_test["p_value"] <= maximum_p_value,
        },
        "frozen_state_better_than_pca_context": {
            **pca_probe_test,
            "maximum_p_value": maximum_p_value,
            "passed": pca_probe_test["improvement"] > 0.0
            and pca_probe_test["p_value"] <= maximum_p_value,
        },
    }
    claim_supported = all(
        bool(gate["passed"]) for gate in gates.values()
    )
    publication_ready = determinism_verified
    compression_supported = bool(
        gates[
            "compressed_state_retains_raw_probe_performance"
        ]["passed"]
        and gates[
            "compressed_state_predicts_better_than_mean"
        ]["passed"]
        and gates["completed_frozen_probe_targets"]["passed"]
        and gates["frozen_state_better_than_pca_context"][
            "passed"
        ]
    )
    next_step_policy = dict(protocol["next_step_policy"])
    if claim_supported:
        decision = str(next_step_policy["supported_claim"])
        reason = (
            "aligned logs and compact frozen state passed all "
            "confirmation gates"
        )
    elif compression_supported:
        decision = str(next_step_policy["compression_only"])
        reason = (
            "compact state transferred, but aligned logs did not "
            "clear every paired control"
        )
    else:
        decision = str(next_step_policy["unsupported_claim"])
        reason = (
            "the compact representation did not retain sufficient "
            "held-out operational state"
        )
    return {
        "schema_version": 2,
        "kind": "contextual_multimodal_jepa_confirmation_assessment",
        "status": (
            "supported" if claim_supported else "not_supported"
        ),
        "claim_supported": claim_supported,
        "publication_ready": publication_ready,
        "training_seeds": list(expected_seeds),
        "gates": gates,
        "statistics": {
            "unit": "schedule_family",
            "family_count": len(contextual_family),
            "seed_aggregation": (
                "mean_within_family_before_test"
            ),
        },
        "next_step": {
            "decision": decision,
            "reason": reason,
        },
    }


def assess_contextual_confirmation(
    training_result_paths: Sequence[Path],
    training_attestation_paths: Sequence[Path],
    *,
    collection_attestation_path: Path,
    repeat_training_result_path: Path,
    repeat_training_attestation_path: Path,
    confirmation_protocol_path: Path,
    repository: Path,
    preregistered_git_commit: str,
) -> Mapping[str, Any]:
    """Verify provenance and assess the full multi-seed confirmation."""

    from .contextual_multimodal_promotion import (
        _parse_json_mapping,
        _validate_execution_attestation,
        _validate_inputs,
        verify_contextual_multimodal_preregistration,
    )

    protocol = verify_contextual_multimodal_preregistration(
        repository,
        confirmation_protocol_path,
        preregistered_git_commit,
    )
    _validate_confirmation_protocol_shape(protocol)
    collection_attestation_path = Path(
        collection_attestation_path
    ).resolve()
    collection_attestation = _parse_json_mapping(
        collection_attestation_path.read_bytes()
    )
    validate_confirmation_collection_attestation(
        collection_attestation,
        protocol,
    )
    expected_seeds = tuple(
        int(seed) for seed in protocol["training_seeds"]
    )
    if (
        len(training_result_paths) != len(expected_seeds)
        or len(training_attestation_paths) != len(expected_seeds)
    ):
        raise ValueError(
            "confirmation requires one result and attestation per seed"
        )

    resolved_results = tuple(
        Path(path).resolve() for path in training_result_paths
    )
    resolved_attestations = tuple(
        Path(path).resolve() for path in training_attestation_paths
    )
    repeat_result_path = Path(
        repeat_training_result_path
    ).resolve()
    repeat_attestation_path = Path(
        repeat_training_attestation_path
    ).resolve()
    if (
        len(set(resolved_results + (repeat_result_path,)))
        != len(resolved_results) + 1
        or len(
            set(resolved_attestations + (repeat_attestation_path,))
        )
        != len(resolved_attestations) + 1
    ):
        raise ValueError(
            "confirmation evidence paths must all be distinct"
        )
    for first, second in itertools.combinations(
        resolved_results + (repeat_result_path,),
        2,
    ):
        if os.path.samefile(first, second):
            raise ValueError(
                "confirmation results must be distinct files"
            )
    for first, second in itertools.combinations(
        resolved_attestations + (repeat_attestation_path,),
        2,
    ):
        if os.path.samefile(first, second):
            raise ValueError(
                "confirmation attestations must be distinct files"
            )

    results_by_seed: Dict[int, Mapping[str, Any]] = {}
    bytes_by_seed: Dict[int, bytes] = {}
    executions = []
    result_sha256s: Dict[str, str] = {}
    attestation_sha256s: Dict[str, str] = {}
    for result_path, attestation_path in zip(
        resolved_results,
        resolved_attestations,
    ):
        result_bytes = result_path.read_bytes()
        result = _parse_json_mapping(result_bytes)
        _validate_inputs(result, protocol)
        seed = int(dict(result["config"])["seed"])
        if seed in results_by_seed:
            raise ValueError(
                f"confirmation repeats training seed {seed}"
            )
        execution = _validate_execution_attestation(
            attestation_path,
            result_path,
            result_bytes,
            protocol,
        )
        results_by_seed[seed] = result
        bytes_by_seed[seed] = result_bytes
        executions.append(execution)
        result_sha256s[str(seed)] = hashlib.sha256(
            result_bytes
        ).hexdigest()
        attestation_sha256s[str(seed)] = hashlib.sha256(
            attestation_path.read_bytes()
        ).hexdigest()
    if set(results_by_seed) != set(expected_seeds):
        raise ValueError(
            "confirmation training seed coverage differs from protocol"
        )

    repeat_bytes = repeat_result_path.read_bytes()
    repeat_result = _parse_json_mapping(repeat_bytes)
    _validate_inputs(repeat_result, protocol)
    primary_seed = int(protocol["determinism_repeat_seed"])
    if int(dict(repeat_result["config"])["seed"]) != primary_seed:
        raise ValueError(
            "confirmation repeat does not use the frozen primary seed"
        )
    repeat_execution = _validate_execution_attestation(
        repeat_attestation_path,
        repeat_result_path,
        repeat_bytes,
        protocol,
    )
    executions.append(repeat_execution)
    execution_ids = [
        str(execution["execution_id"]) for execution in executions
    ]
    if len(execution_ids) != len(set(execution_ids)):
        raise ValueError(
            "confirmation execution attestations reuse an execution ID"
        )
    ordered_executions = sorted(
        executions,
        key=lambda execution: int(execution["started_unix_nano"]),
    )
    for previous, following in zip(
        ordered_executions,
        ordered_executions[1:],
    ):
        if int(previous["completed_unix_nano"]) > int(
            following["started_unix_nano"]
        ):
            raise ValueError(
                "confirmation training executions overlap"
            )

    determinism_verified = (
        bytes_by_seed[primary_seed] == repeat_bytes
    )
    assessment = dict(
        assess_confirmation_results(
            [
                results_by_seed[seed]
                for seed in expected_seeds
            ],
            protocol,
            determinism_verified=determinism_verified,
        )
    )
    assessment.update(
        {
            "claim": protocol["claim"],
            "evidence_boundary": protocol["evidence_boundary"],
            "preregistered_git_commit": preregistered_git_commit,
            "collection_attestation_sha256": hashlib.sha256(
                collection_attestation_path.read_bytes()
            ).hexdigest(),
            "collection_execution_id": str(
                collection_attestation["execution_id"]
            ),
            "confirmation_protocol_sha256": (
                _canonical_sha256(protocol)
            ),
            "training_result_sha256s": result_sha256s,
            "repeat_training_result_sha256": hashlib.sha256(
                repeat_bytes
            ).hexdigest(),
            "training_execution_ids": execution_ids,
            "training_execution_attestation_sha256s": (
                attestation_sha256s
            ),
            "repeat_training_execution_attestation_sha256": (
                hashlib.sha256(
                    repeat_attestation_path.read_bytes()
                ).hexdigest()
            ),
        }
    )
    return assessment


def write_contextual_confirmation_assessment(
    assessment: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, Path]:
    """Write the publication-oriented confirmation decision package."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "confirmation.json"
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
        "# Contextual metrics + logs JEPA confirmation",
        "",
        f"Result: **{str(assessment['status']).upper()}**",
        "",
        f"Publication ready: **{bool(assessment['publication_ready'])}**",
        "",
        "## Narrow claim",
        "",
        str(assessment["claim"]),
        "",
        "## Evidence boundary",
        "",
        str(assessment["evidence_boundary"]),
        "",
        "## Preregistered gates",
        "",
    ]
    for name, raw_gate in dict(assessment["gates"]).items():
        gate = dict(raw_gate)
        lines.append(
            f"- {'PASS' if gate['passed'] else 'FAIL'} — {name}"
        )
    next_step = dict(assessment["next_step"])
    lines.extend(
        [
            "",
            "## Subsequent world-model step",
            "",
            f"Decision: `{next_step['decision']}`",
            "",
            str(next_step["reason"]),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    return {"confirmation": json_path, "report": report_path}


def _family_alert_rates(
    results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Tuple[float, ...]]]:
    model_names = (
        "contextual_multimodal",
        "metrics_only",
        "capacity_matched_metrics_only",
        "shuffled_logs",
    )
    by_model: Dict[str, Dict[str, list[float]]] = {
        name: {} for name in model_names
    }
    expected_families = None
    for result in results:
        observed = {}
        for family in dict(result["schedule_transfer"])[
            "validation_families"
        ]:
            raw_family = dict(family)
            family_id = _case_family(
                str(list(raw_family["case_ids"])[0])
            )
            observed[family_id] = raw_family
        if expected_families is None:
            expected_families = set(observed)
        elif set(observed) != expected_families:
            raise ValueError(
                "confirmation schedule family coverage differs by seed"
            )
        for family_id, raw_family in observed.items():
            for model_name in model_names:
                by_model[model_name].setdefault(
                    family_id, []
                ).append(
                    float(
                        dict(raw_family[model_name])["alert_rate"]
                    )
                )
    return {
        model_name: {
            family: tuple(values)
            for family, values in families.items()
        }
        for model_name, families in by_model.items()
    }


def _mean_by_family(
    values: Mapping[str, Sequence[float]],
) -> Mapping[str, float]:
    return {
        family: sum(observed) / len(observed)
        for family, observed in values.items()
    }


def _aggregate_alert_rate(
    result: Mapping[str, Any],
    model_name: str,
) -> float:
    return float(
        dict(
            dict(dict(result["metrics"])[model_name])[
                "validation"
            ]
        )["alert_rate"]
    )


def _transfer_summary(
    results: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    pca_dimension = int(
        dict(protocol["representation_transfer"])[
            "pca_context_dimension"
        ]
    )
    pca_name = (
        f"pca_{pca_dimension}_context_ridge"
    )
    required = (
        "contextual_multimodal",
        "capacity_matched_metrics_only",
        "shuffled_logs",
        "raw_context_ridge",
        pca_name,
    )
    accumulated: Dict[str, Dict[str, Any]] = {}
    for name in required:
        seed_errors = []
        completed = []
        family_values: Dict[str, list[float]] = {}
        for result in results:
            representations = dict(
                dict(result["representation_transfer"])[
                    "representations"
                ]
            )
            representation = dict(representations[name])
            seed_errors.append(
                float(
                    representation[
                        "mean_validation_normalized_mse"
                    ]
                )
            )
            completed.append(
                int(representation["completed_target_count"])
            )
            per_family: Dict[str, list[float]] = {}
            for target in dict(representation["targets"]).values():
                raw_target = dict(target)
                if raw_target.get("status") != "completed":
                    continue
                for family, value in dict(
                    raw_target["family_normalized_mse"]
                ).items():
                    per_family.setdefault(str(family), []).append(
                        float(value)
                    )
            for family, values in per_family.items():
                family_values.setdefault(family, []).append(
                    sum(values) / len(values)
                )
        accumulated[name] = {
            "mean_error": sum(seed_errors) / len(seed_errors),
            "minimum_completed_targets": min(completed),
            "families": {
                family: sum(values) / len(values)
                for family, values in family_values.items()
            },
        }
    family_sets = {
        tuple(sorted(summary["families"]))
        for summary in accumulated.values()
    }
    if len(family_sets) != 1 or not next(iter(family_sets), ()):
        raise ValueError(
            "confirmation frozen probes have inconsistent families"
        )
    return accumulated


def _paired_randomization(
    differences: Sequence[float],
) -> Mapping[str, float]:
    if not differences or any(
        not math.isfinite(value) for value in differences
    ):
        raise ValueError(
            "paired confirmation differences must be finite"
        )
    observed = sum(differences) / len(differences)
    exceedances = 0
    assignment_count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = sum(
            sign * value
            for sign, value in zip(signs, differences)
        ) / len(differences)
        exceedances += permuted >= observed - 1e-15
        assignment_count += 1
    return {
        "improvement": observed,
        "p_value": exceedances / assignment_count,
    }


def _case_family(case_id: str) -> str:
    marker = "-f"
    try:
        return f"f{int(case_id.split(marker, 1)[1][:2]):02d}"
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"confirmation case has invalid family: {case_id}"
        ) from error


def _validate_confirmation_protocol_shape(
    protocol: Mapping[str, Any],
) -> None:
    if (
        protocol.get("schema_version") != 2
        or protocol.get("kind") != CONFIRMATION_PROTOCOL_KIND
    ):
        raise ValueError(
            "unsupported contextual confirmation protocol"
        )
    corpus = dict(protocol.get("corpus", {}))
    family_count = int(corpus.get("training_family_count", 0)) + int(
        corpus.get("validation_family_count", 0)
    )
    families = list(corpus.get("schedule_families", ()))
    workers = tuple(
        int(value) for value in corpus.get("worker_replicas", ())
    )
    if (
        family_count < 2
        or len(families) != family_count
        or not workers
        or len(workers) != len(set(workers))
        or any(worker < 1 for worker in workers)
    ):
        raise ValueError(
            "contextual confirmation corpus design is incomplete"
        )


def validate_confirmation_collection_attestation(
    attestation: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Validate one balanced collection execution against its protocol."""

    plans = plan_parallel_confirmation_collection(protocol)
    cases = list(attestation.get("cases", ()))
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind")
        != (
            "contextual_multimodal_jepa_confirmation_"
            "collection_attestation"
        )
        or attestation.get("parallel_jobs")
        != protocol["collection"]["parallel_jobs"]
        or attestation.get("batch_count")
        != len({plan.batch for plan in plans})
        or attestation.get("case_count") != len(plans)
        or attestation.get("application_build_context_sha256")
        != protocol["corpus"]["application_build_context_sha256"]
        or not isinstance(attestation.get("application_image_id"), str)
        or not attestation.get("application_image_id")
        or attestation.get("protocol_sha256")
        != _canonical_sha256(protocol)
        or len(cases) != len(plans)
    ):
        raise ValueError(
            "confirmation collection attestation differs from protocol"
        )
    try:
        execution_id = uuid.UUID(str(attestation["execution_id"]))
    except (KeyError, ValueError) as error:
        raise ValueError(
            "confirmation collection execution ID is invalid"
        ) from error
    if (
        execution_id.version != 4
        or str(execution_id) != attestation["execution_id"]
    ):
        raise ValueError(
            "confirmation collection execution ID must be UUIDv4"
        )
    overall_started = attestation.get("started_unix_nano")
    overall_completed = attestation.get("completed_unix_nano")
    if (
        not isinstance(overall_started, int)
        or isinstance(overall_started, bool)
        or not isinstance(overall_completed, int)
        or isinstance(overall_completed, bool)
        or overall_started <= 0
        or overall_completed <= overall_started
    ):
        raise ValueError(
            "confirmation collection execution timing is invalid"
        )
    observed = {
        str(case["case_id"]): dict(case) for case in cases
    }
    if len(observed) != len(plans):
        raise ValueError(
            "confirmation collection case IDs are not unique"
        )
    for plan in plans:
        case = observed.get(plan.case_id)
        if case is None or any(
            case.get(name) != expected
            for name, expected in {
                "family": plan.family,
                "worker_replicas": plan.worker_replicas,
                "split": plan.split,
                "batch": plan.batch,
                "lane": plan.lane,
            }.items()
        ):
            raise ValueError(
                "confirmation collection lane plan differs from protocol"
            )
        started = case.get("started_unix_nano")
        completed = case.get("completed_unix_nano")
        if (
            not isinstance(started, int)
            or isinstance(started, bool)
            or not isinstance(completed, int)
            or isinstance(completed, bool)
            or started <= 0
            or completed <= started
        ):
            raise ValueError(
                "confirmation collection case timing is invalid"
            )
        if started < overall_started or completed > overall_completed:
            raise ValueError(
                "confirmation case timing is outside collection execution"
            )
    for batch in range(1, int(attestation["batch_count"])):
        prior_completed = max(
            int(case["completed_unix_nano"])
            for case in cases
            if int(case["batch"]) == batch
        )
        next_started = min(
            int(case["started_unix_nano"])
            for case in cases
            if int(case["batch"]) == batch + 1
        )
        if prior_completed > next_started:
            raise ValueError(
                "confirmation collection batches overlap"
            )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
