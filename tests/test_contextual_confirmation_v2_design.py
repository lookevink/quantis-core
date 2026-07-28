import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from quantis_core.contextual_confirmation import (
    confirmation_case_ids,
    plan_parallel_confirmation_collection,
)
from quantis_core.demand_conditioning import canonical_request_schedule


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (
            _repository()
            / "lab"
            / "fault_matrix"
            / "contextual-jepa-confirmation-v2.json"
        ).read_text()
    )


def test_confirmation_v2_freezes_publishable_narrow_claim():
    protocol = _protocol()

    assert protocol["schema_version"] == 2
    assert protocol["kind"] == (
        "contextual_multimodal_jepa_confirmation_v2"
    )
    assert protocol["corpus"]["training_family_count"] == 12
    assert protocol["corpus"]["validation_family_count"] == 12
    assert len(protocol["corpus"]["schedule_families"]) == 24
    assert protocol["collection"]["parallel_jobs"] == 3
    assert protocol["collection"]["family_order"] == [
        1,
        13,
        2,
        14,
        3,
        15,
        4,
        16,
        5,
        17,
        6,
        18,
        7,
        19,
        8,
        20,
        9,
        21,
        10,
        22,
        11,
        23,
        12,
        24,
    ]
    assert protocol["training_seeds"] == [89, 97, 101, 103, 107]
    assert protocol["determinism_repeat_seed"] == 89
    assert protocol["statistics"] == {
        "unit": "schedule_family",
        "test": "exact_one_sided_paired_sign_randomization",
        "maximum_p_value": 0.05,
        "seed_aggregation": "mean_within_family_before_test",
    }
    assert protocol["representation_transfer"][
        "context_latent_dimension"
    ] == 12
    assert protocol["representation_transfer"][
        "raw_context_dimension"
    ] == 108
    assert protocol["representation_transfer"][
        "target_block_reduction"
    ] == "mean"
    assert protocol["next_step_policy"]["supported_claim"] == (
        "action_conditioned_intervention_corpus"
    )


def test_confirmation_schedules_are_unique_and_unexposed():
    protocol = _protocol()
    schedules = {
        canonical_request_schedule(
            family["requests_per_window"],
            family["load_pattern_offsets"],
        )
        for family in protocol["corpus"]["schedule_families"]
    }
    development = json.loads(
        (
            _repository()
            / "artifacts"
            / "jepa-world-model-v2"
            / "contextual-development-v2"
            / "training"
            / "candidates"
            / "v2_log_latent_1"
            / "corpus.json"
        ).read_text()
    )
    exposed = {
        tuple(run["canonical_request_schedule"])
        for run in development["base_corpus"]["metric_corpus"][
            "protocol"
        ]["runs"].values()
    }
    families = protocol["corpus"]["schedule_families"]
    marginal = lambda family: (
        family["requests_per_window"],
        len(family["load_pattern_offsets"]),
    )

    assert len(schedules) == 24
    assert schedules.isdisjoint(exposed)
    assert Counter(map(marginal, families[:12])) == Counter(
        map(marginal, families[12:])
    )


def test_confirmation_case_identity_and_lane_plan_are_balanced():
    protocol = _protocol()
    training, validation = confirmation_case_ids(protocol)
    plans = plan_parallel_confirmation_collection(protocol)

    assert len(training) == 36
    assert len(validation) == 36
    assert set(training).isdisjoint(validation)
    assert len(plans) == 72
    assert {plan.batch for plan in plans} == set(range(1, 25))
    for family in range(1, 25):
        family_plans = [
            plan for plan in plans if plan.family == family
        ]
        assert {plan.worker_replicas for plan in family_plans} == {
            1,
            2,
            3,
        }
        assert {plan.lane for plan in family_plans} == {1, 2, 3}
    for worker_replicas in (1, 2, 3):
        lane_counts = {
            lane: sum(
                plan.worker_replicas == worker_replicas
                and plan.lane == lane
                for plan in plans
            )
            for lane in (1, 2, 3)
        }
        assert lane_counts == {1: 8, 2: 8, 3: 8}


def test_confirmation_preparer_matches_protocol(tmp_path):
    repository = _repository()
    protocol_path = (
        repository
        / "lab"
        / "fault_matrix"
        / "contextual-jepa-confirmation-v2.json"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "lab/fault_matrix/prepare_contextual_confirmation_corpus.py",
            "--protocol",
            str(protocol_path),
            "--output",
            str(tmp_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    split = json.loads((tmp_path / "split.json").read_text())
    protocol = _protocol()
    training, validation = confirmation_case_ids(protocol)
    manifests = sorted((tmp_path / "manifests").glob("*.json"))
    assert split["training_case_ids"] == list(training)
    assert split["validation_case_ids"] == list(validation)
    assert len(manifests) == 72


def test_confirmation_runner_does_not_consume_corpus_during_design():
    repository = _repository()
    runner = (
        repository
        / "lab"
        / "fault_matrix"
        / "run-contextual-multimodal-jepa-confirmation-v2.sh"
    ).read_text()

    assert "collect-contextual-confirmation" in runner
    assert "--parallel-jobs 3" in runner
    assert "train-contextual-confirmation-v2" in runner
    assert "assess-contextual-confirmation-v2" in runner
    assert not (
        repository
        / "artifacts"
        / "jepa-world-model-v2"
        / "contextual-confirmation-v2"
    ).exists()
