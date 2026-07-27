import copy
import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quantis_core.contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)
from quantis_core.contextual_multimodal_promotion import (
    assess_contextual_multimodal_promotion,
)
from quantis_core.contextual_multimodal_world_model import (
    ContextualMultimodalJepaWorldModelDetector,
)
from quantis_core.demand_conditioning import (
    canonical_request_schedule,
)


def _protocol():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "lab"
            / "fault_matrix"
            / "contextual-jepa-promotion-v1.json"
        ).read_text()
    )


def _result(protocol=None):
    protocol = _protocol() if protocol is None else protocol
    corpus = _corpus(protocol)
    semantic_names = protocol["log_vocabulary"][
        "semantic_feature_names"
    ]
    preprocessing = corpus["preprocessing"]
    config = protocol["training_config"]
    model = _model_artifact(
        protocol,
        config["metric_latent_dimension"],
        config["log_latent_dimension"],
        preprocessing,
    )
    metrics_only = _model_artifact(
        protocol,
        config["metric_latent_dimension"],
        0,
        preprocessing,
    )
    capacity_matched = _model_artifact(
        protocol,
        config["metric_latent_dimension"]
        + config["log_latent_dimension"],
        0,
        preprocessing,
    )
    shuffled = _model_artifact(
        protocol,
        config["metric_latent_dimension"],
        config["log_latent_dimension"],
        preprocessing,
    )
    shuffled["control_protocol"] = protocol[
        "shuffled_log_control"
    ]
    log_only = _model_artifact(
        protocol,
        0,
        config["log_latent_dimension"],
        preprocessing,
    )
    result = {
        "schema_version": 1,
        "kind": (
            "contextual_multimodal_jepa_world_model_"
            "promotion_confirmation"
        ),
        "evidence_mode": "promotion_confirmation",
        "config": config,
        "corpus": corpus,
        "model": model,
        "metrics_only_model": metrics_only,
        "capacity_matched_metrics_only_model": (
            capacity_matched
        ),
        "shuffled_log_model": shuffled,
        "log_only_model": log_only,
        "metrics": {
            "contextual_multimodal": {
                "training": _score(0.01),
                "validation": _score(0.02, 200),
            },
            "metrics_only": {
                "training": _score(0.02),
                "validation": _score(0.035, 200),
            },
            "capacity_matched_metrics_only": {
                "training": _score(0.02),
                "validation": _score(0.04, 200),
            },
            "shuffled_logs": {
                "training": _score(0.03),
                "validation": _score(0.05, 200),
            },
            "log_only": {
                "training": _score(0.04),
                "validation": _score(0.04, 200),
            },
            "modality_dropout": {
                "metric_context_only": _score(0.03, 200),
                "log_context_only": _score(0.04, 200),
            },
        },
        "schedule_transfer": {
            "validation_families": _validation_families(protocol)
        },
        "cross_validation": {"status": "disabled"},
        "selection": {
            "status": "not_assessed",
            "publication_eligible": False,
        },
        "protocol": {
            "model_selection_status": "fixed_promotion_confirmation",
            "validation_use": (
                "fixed_confirmation_no_adaptive_reuse"
            ),
            "promotion_protocol_sha256": _canonical_sha256(
                protocol
            ),
            "training_case_ids": protocol["training_case_ids"],
            "validation_case_ids": protocol[
                "validation_case_ids"
            ],
            "training_uses_validation_windows": False,
            "cross_validation": {"status": "disabled"},
            "controls": protocol["controls"],
            "training_runtime": protocol["training_runtime"],
            "corpus_metadata_sha256": _canonical_sha256(
                corpus
            ),
            "model_artifact_sha256": _canonical_sha256(model),
            "control_artifact_sha256s": {
                "metrics_only_model": _canonical_sha256(
                    metrics_only
                ),
                "capacity_matched_metrics_only_model": (
                    _canonical_sha256(capacity_matched)
                ),
                "shuffled_log_model": _canonical_sha256(
                    shuffled
                ),
                "log_only_model": _canonical_sha256(log_only),
            },
        },
    }
    assert model["log_feature_names"] == semantic_names
    return result


def _model_artifact(
    protocol,
    metric_latent_dimension,
    log_latent_dimension,
    preprocessing,
):
    artifact = copy.deepcopy(
        _fitted_model_artifact(
            metric_latent_dimension,
            log_latent_dimension,
        )
    )
    artifact["preprocessing"] = preprocessing
    return artifact


@functools.lru_cache(maxsize=None)
def _fitted_model_artifact(
    metric_latent_dimension,
    log_latent_dimension,
):
    protocol = _protocol()
    config = protocol["training_config"]
    detector = ContextualMultimodalJepaWorldModelDetector(
        metric_latent_dimension=metric_latent_dimension,
        log_latent_dimension=log_latent_dimension,
        pretraining_epochs=config["pretraining_epochs"],
        predictor_refinement_epochs=(
            config["predictor_refinement_epochs"]
        ),
        learning_rate=config["learning_rate"],
        ema_decay=config["ema_decay"],
        weight_decay=config["weight_decay"],
        loss=config["loss"],
        huber_delta=config["huber_delta"],
        auxiliary_loss_weight=config["auxiliary_loss_weight"],
        rollout_loss_weight=config["rollout_loss_weight"],
        calibration_quantile=config["calibration_quantile"],
        seed=config["seed"],
    ).fit(_model_windows(protocol))
    return detector.to_dict()


def _model_windows(protocol):
    sample_count = 24
    lookback = protocol["corpus"]["lookback"]
    horizons = tuple(protocol["corpus"]["target_horizons"])
    block_size = protocol["corpus"]["target_block_size"]
    metric_names = tuple(
        protocol["metric_vocabulary"]["semantic_feature_names"]
    )
    log_names = tuple(
        protocol["log_vocabulary"]["semantic_feature_names"]
    )
    control_names = ("request_demand", "worker_replicas")
    generator = np.random.default_rng(991)
    return ContextualMultimodalModelWindows(
        metric_contexts=generator.normal(
            size=(sample_count, lookback, len(metric_names))
        ),
        log_contexts=generator.normal(
            size=(sample_count, lookback, len(log_names))
        ),
        metric_target_blocks=generator.normal(
            size=(
                sample_count,
                len(horizons),
                block_size,
                len(metric_names),
            )
        ),
        log_target_blocks=generator.normal(
            size=(
                sample_count,
                len(horizons),
                block_size,
                len(log_names),
            )
        ),
        target_controls=generator.normal(
            size=(
                sample_count,
                len(horizons),
                block_size,
                len(control_names),
            )
        ),
        point_indices=np.arange(sample_count, dtype=np.int64),
        metric_feature_names=metric_names,
        log_feature_names=log_names,
        control_feature_names=control_names,
        horizons=horizons,
        target_block_size=block_size,
    )


def _score(alert_rate, window_count=1000):
    return {
        "window_count": window_count,
        "alerts": round(alert_rate * window_count),
        "alert_rate": alert_rate,
    }


def _corpus(protocol):
    design = protocol["corpus"]
    case_ids = (
        protocol["training_case_ids"]
        + protocol["validation_case_ids"]
    )
    split = {
        "training_case_ids": protocol["training_case_ids"],
        "validation_case_ids": protocol["validation_case_ids"],
        "reserved_case_ids": [],
        "lookback": design["lookback"],
        "expected_application_api_request_queue_size": design[
            "expected_application_api_request_queue_size"
        ],
    }
    metric_runs = {}
    log_runs = {}
    for case_id in case_ids:
        family_index = int(case_id.split("-f", 1)[1][:2])
        worker_replicas = int(
            case_id.rsplit("-w", 1)[1].split("-", 1)[0]
        )
        family = design["schedule_families"][family_index - 1]
        schedule = canonical_request_schedule(
            family["requests_per_window"],
            family["load_pattern_offsets"],
        )
        metric_runs[case_id] = {
            "case_id": case_id,
            "canonical_request_schedule": list(schedule),
            "application_build_context_sha256": design[
                "application_build_context_sha256"
            ],
            "application_api_request_queue_size": design[
                "expected_application_api_request_queue_size"
            ],
            "normal_interval": [0, design["point_count"]],
            "normal_point_count": design["point_count"],
            "topology_id": f"workers-{worker_replicas}",
            "worker_replicas": worker_replicas,
        }
        log_runs[case_id] = {"case_id": case_id}
    raw_spec = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "lab"
            / "fault_matrix"
            / "contextual-promotion-log-feature-spec.json"
        ).read_text()
    )
    return {
        "schema_version": 2,
        "kind": "contextual_multimodal_telemetry_corpus",
        "protocol": {
            "evidence_assignment": "deferred_to_training_protocol",
            "training_case_ids": protocol["training_case_ids"],
            "validation_case_ids": protocol[
                "validation_case_ids"
            ],
            "target_horizons": design["target_horizons"],
            "target_block_size": design["target_block_size"],
        },
        "preprocessing": {
            "logs": {
                "transformer": {
                    "features": protocol["log_vocabulary"][
                        "semantic_feature_names"
                    ]
                }
            }
        },
        "base_corpus": {
            "kind": "multimodal_telemetry_corpus",
            "log_feature_spec": raw_spec,
            "protocol": {
                "split_spec": split,
                "log_window_assignment": (
                    "event_time_metric_boundaries"
                ),
                "runs": log_runs,
            },
            "metric_corpus": {
                "protocol": {
                    "split_spec": split,
                    "runs": metric_runs,
                    "feature_schema_id": protocol[
                        "metric_vocabulary"
                    ]["feature_schema_id"],
                    "feature_spec_sha256": protocol[
                        "metric_vocabulary"
                    ]["feature_spec_sha256"],
                }
            },
        },
    }


def _validation_families(protocol):
    design = protocol["corpus"]
    families = []
    for family_index in range(
        design["training_family_count"] + 1,
        design["training_family_count"]
        + design["validation_family_count"]
        + 1,
    ):
        family = design["schedule_families"][family_index - 1]
        schedule = canonical_request_schedule(
            family["requests_per_window"],
            family["load_pattern_offsets"],
        )
        cases = [
            case_id
            for case_id in protocol["validation_case_ids"]
            if f"-f{family_index:02d}-" in case_id
        ]
        families.append(
            {
                "schedule_sha256": hashlib.sha256(
                    json.dumps(
                        list(schedule),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "case_ids": cases,
                "contextual_multimodal": {
                    **_score(
                        0.01 if len(families) == 0 else 0.03,
                        100,
                    ),
                },
                "metrics_only": {
                    **_score(
                        0.03 if len(families) == 0 else 0.04,
                        100,
                    ),
                },
                "capacity_matched_metrics_only": _score(
                    0.04, 100
                ),
                "shuffled_logs": _score(0.05, 100),
            }
        )
    return families


def _canonical_sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _encoded(payload):
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode()


@pytest.fixture
def promotion_case(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    frozen_path = repository / "frozen.txt"
    frozen_path.write_text("frozen promotion dependency\n")
    protocol = _protocol()
    protocol["frozen_files"] = {
        "frozen.txt": hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest()
    }
    protocol_path = repository / "protocol.json"
    protocol_path.write_bytes(_encoded(protocol))
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@quantis.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Quantis Tests"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "add", "protocol.json", "frozen.txt"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "Freeze promotion"],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    training_directory = repository / "training-a"
    repeat_directory = repository / "training-b"
    training_directory.mkdir()
    repeat_directory.mkdir()
    return {
        "repository": repository,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "commit": commit,
        "training_path": (
            training_directory / "promotion-training.json"
        ),
        "repeat_path": (
            repeat_directory / "promotion-training.json"
        ),
        "training_attestation_path": (
            training_directory / "execution-attestation.json"
        ),
        "repeat_attestation_path": (
            repeat_directory / "execution-attestation.json"
        ),
    }


def _assess(promotion_case, result, repeat=None):
    repeat_result = result if repeat is None else repeat
    promotion_case["training_path"].write_bytes(_encoded(result))
    promotion_case["repeat_path"].write_bytes(
        _encoded(repeat_result)
    )
    _write_execution_attestations(
        promotion_case,
        result,
        repeat_result,
    )
    return assess_contextual_multimodal_promotion(
        promotion_case["training_path"],
        promotion_case["protocol_path"],
        repeat_training_result_path=promotion_case["repeat_path"],
        training_attestation_path=promotion_case[
            "training_attestation_path"
        ],
        repeat_training_attestation_path=promotion_case[
            "repeat_attestation_path"
        ],
        repository=promotion_case["repository"],
        preregistered_git_commit=promotion_case["commit"],
    )


def _write_execution_attestations(
    promotion_case,
    result,
    repeat,
):
    first_start = 1_000_000_000
    first_completed = first_start + 100
    second_start = first_completed + 100
    second_completed = second_start + 100
    for (
        attestation_path,
        result_path,
        payload,
        execution_id,
        started,
        completed,
    ) in (
        (
            promotion_case["training_attestation_path"],
            promotion_case["training_path"],
            result,
            "00000000-0000-4000-8000-000000000001",
            first_start,
            first_completed,
        ),
        (
            promotion_case["repeat_attestation_path"],
            promotion_case["repeat_path"],
            repeat,
            "00000000-0000-4000-8000-000000000002",
            second_start,
            second_completed,
        ),
    ):
        attestation_path.write_bytes(
            _encoded(
                {
                    "schema_version": 1,
                    "kind": (
                        "contextual_multimodal_jepa_"
                        "training_execution_attestation"
                    ),
                    "execution_id": execution_id,
                    "process_id": os.getpid(),
                    "started_unix_nano": started,
                    "completed_unix_nano": completed,
                    "output_directory": str(
                        result_path.parent.resolve()
                    ),
                    "training_result_sha256": hashlib.sha256(
                        result_path.read_bytes()
                    ).hexdigest(),
                    "corpus_metadata_sha256": payload["protocol"][
                        "corpus_metadata_sha256"
                    ],
                    "model_artifact_sha256": payload["protocol"][
                        "model_artifact_sha256"
                    ],
                    "promotion_protocol_sha256": (
                        _canonical_sha256(
                            promotion_case["protocol"]
                        )
                    ),
                }
            )
        )


def test_promotion_requires_fresh_fixed_candidate_and_all_controls(
    promotion_case,
) -> None:
    assessment = _assess(
        promotion_case,
        _result(promotion_case["protocol"]),
    )

    assert assessment["status"] == "passed"
    assert assessment["publication_eligible"] is True
    assert all(
        gate["passed"] for gate in assessment["gates"].values()
    )


def test_promotion_fails_when_aligned_logs_do_not_beat_shuffle(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    result["metrics"]["shuffled_logs"]["validation"] = _score(
        0.02, 200
    )
    for family in result["schedule_transfer"][
        "validation_families"
    ]:
        family["shuffled_logs"] = _score(0.02, 100)

    assessment = _assess(promotion_case, result)

    assert assessment["status"] == "failed"
    assert assessment["publication_eligible"] is False
    assert assessment["gates"][
        "strictly_better_than_shuffled_logs"
    ]["passed"] is False


def test_promotion_rejects_incomplete_validation_families(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    result["schedule_transfer"]["validation_families"].pop()

    with pytest.raises(ValueError, match="family count"):
        _assess(promotion_case, result)


def test_promotion_rejects_duplicate_family_case_coverage(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    first = result["schedule_transfer"]["validation_families"][0]
    second = result["schedule_transfer"]["validation_families"][1]
    second["case_ids"] = first["case_ids"]

    with pytest.raises(ValueError, match="case coverage"):
        _assess(promotion_case, result)


def test_promotion_binds_repeat_evidence_to_assessed_result(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    repeat = _result(promotion_case["protocol"])
    repeat["repeat_execution_marker"] = True

    assessment = _assess(promotion_case, result, repeat)

    assert assessment["status"] == "failed"
    assert assessment["gates"]["byte_identical_repeat"][
        "passed"
    ] is False


def test_public_assessor_rejects_same_evidence_file(
    promotion_case,
) -> None:
    result_path = promotion_case["training_path"]
    result_path.write_bytes(
        _encoded(_result(promotion_case["protocol"]))
    )

    with pytest.raises(ValueError, match="distinct training"):
        assess_contextual_multimodal_promotion(
            result_path,
            promotion_case["protocol_path"],
            repeat_training_result_path=result_path,
            training_attestation_path=promotion_case[
                "training_attestation_path"
            ],
            repeat_training_attestation_path=promotion_case[
                "repeat_attestation_path"
            ],
            repository=promotion_case["repository"],
            preregistered_git_commit=promotion_case["commit"],
        )


def test_public_assessor_rejects_reused_execution_id(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    promotion_case["training_path"].write_bytes(_encoded(result))
    promotion_case["repeat_path"].write_bytes(_encoded(result))
    _write_execution_attestations(
        promotion_case,
        result,
        result,
    )
    first = json.loads(
        promotion_case["training_attestation_path"].read_text()
    )
    repeated = json.loads(
        promotion_case["repeat_attestation_path"].read_text()
    )
    repeated["execution_id"] = first["execution_id"]
    promotion_case["repeat_attestation_path"].write_bytes(
        _encoded(repeated)
    )

    with pytest.raises(ValueError, match="reuse one execution ID"):
        assess_contextual_multimodal_promotion(
            promotion_case["training_path"],
            promotion_case["protocol_path"],
            repeat_training_result_path=promotion_case[
                "repeat_path"
            ],
            training_attestation_path=promotion_case[
                "training_attestation_path"
            ],
            repeat_training_attestation_path=promotion_case[
                "repeat_attestation_path"
            ],
            repository=promotion_case["repository"],
            preregistered_git_commit=promotion_case["commit"],
        )


def test_public_assessor_rejects_symbolic_commit(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    promotion_case["training_path"].write_bytes(_encoded(result))
    promotion_case["repeat_path"].write_bytes(_encoded(result))
    _write_execution_attestations(
        promotion_case,
        result,
        result,
    )

    with pytest.raises(ValueError, match="full immutable Git ID"):
        assess_contextual_multimodal_promotion(
            promotion_case["training_path"],
            promotion_case["protocol_path"],
            repeat_training_result_path=promotion_case[
                "repeat_path"
            ],
            training_attestation_path=promotion_case[
                "training_attestation_path"
            ],
            repeat_training_attestation_path=promotion_case[
                "repeat_attestation_path"
            ],
            repository=promotion_case["repository"],
            preregistered_git_commit="HEAD",
        )


def test_promotion_rejects_missing_control_artifact(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    del result["metrics_only_model"]

    with pytest.raises(KeyError, match="metrics_only_model"):
        _assess(promotion_case, result)


def test_promotion_rejects_truncated_model_artifact(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    result["model"]["metric_encoder_weights"].pop()
    result["protocol"]["model_artifact_sha256"] = (
        _canonical_sha256(result["model"])
    )

    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        _assess(promotion_case, result)


def test_promotion_rejects_non_finite_or_inconsistent_metrics(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    result["metrics"]["contextual_multimodal"]["validation"][
        "alert_rate"
    ] = -0.1

    with pytest.raises(ValueError, match="alert rate"):
        _assess(promotion_case, result)


def test_failed_promotion_cli_returns_nonzero(
    promotion_case,
) -> None:
    result = _result(promotion_case["protocol"])
    result["metrics"]["shuffled_logs"]["validation"] = _score(
        0.02, 200
    )
    for family in result["schedule_transfer"][
        "validation_families"
    ]:
        family["shuffled_logs"] = _score(0.02, 100)
    promotion_case["training_path"].write_bytes(_encoded(result))
    promotion_case["repeat_path"].write_bytes(_encoded(result))
    _write_execution_attestations(
        promotion_case,
        result,
        result,
    )
    output = promotion_case["repository"] / "assessment"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantis_core",
            "assess-contextual-multimodal-jepa-promotion",
            "--training-result",
            str(promotion_case["training_path"]),
            "--repeat-training-result",
            str(promotion_case["repeat_path"]),
            "--training-attestation",
            str(promotion_case["training_attestation_path"]),
            "--repeat-training-attestation",
            str(promotion_case["repeat_attestation_path"]),
            "--promotion-protocol",
            str(promotion_case["protocol_path"]),
            "--repository",
            str(promotion_case["repository"]),
            "--preregistered-git-commit",
            promotion_case["commit"],
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "promotion: FAILED" in completed.stdout
    assert json.loads((output / "promotion.json").read_text())[
        "publication_eligible"
    ] is False
