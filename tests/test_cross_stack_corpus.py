import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from quantis_core.cross_stack_corpus import (
    CorpusRecord,
    MinimumDiversityContract,
    assess_corpus_diversity,
    assess_serialized_inventory,
    discover_corpus_inventory,
)


INTERVENTIONS = (
    "service_pause",
    "persistence_contention",
    "message_production_delay",
    "message_consumption_delay",
    "request_rejection",
)
TOPOLOGIES = ("small", "medium", "large")
WORKLOADS = ("steady", "ramp_or_burst", "periodic_or_multiphase")


def test_discovery_extracts_identity_and_preserves_source_equivalence(
    tmp_path: Path,
) -> None:
    normal = {
        "metric_corpus": {
            "protocol": {
                "application_build_context_sha256": "a" * 64,
                "application_image_id": "sha256:" + "b" * 64,
                "feature_schema_id": "c" * 64,
                "runs": {
                    "normal-f01-w1": {"capture_sha256": "d" * 64},
                    "normal-f01-w2": {"capture_sha256": "e" * 64},
                    "normal-f01-w3": {"capture_sha256": "f" * 64},
                },
            }
        }
    }
    contextual = {"base_corpus": normal, "preprocessing": {"kind": "contextual"}}
    action_protocol = {
        "workload": {"schedule_kind": "seeded_explicit_uniform_integer"},
        "action_library": {
            "worker_pause": {},
            "postgres_lock": {},
            "redis_enqueue_delay": {},
            "redis_dequeue_delay": {},
            "api_rejection": {},
        },
        "design": {
            "worker_replica_values": [1, 2, 3],
            "replicates_per_cell": 8,
        },
    }
    action_plan = {
        "application_build_context_sha256": "1" * 64,
        "assignments": [{}] * 240,
    }
    action_quality = {
        "counts": {"pair_count": 120, "case_count": 240},
        "case_file_sha256s": {
            "capture-0": {
                "capture-manifest.json": "3" * 64,
                "collector-metrics.jsonl": "4" * 64,
            }
        },
        "pair_counts_by_action": {
            name: 24
            for name in (
                "worker_pause",
                "postgres_lock",
                "redis_enqueue_delay",
                "redis_dequeue_delay",
                "api_rejection",
            )
        },
        "pair_counts_by_topology": {
            "workers-1": 40,
            "workers-2": 40,
            "workers-3": 40,
        },
    }
    action_attestation = {
        "application_image_id": "sha256:" + "2" * 64,
        "application_build_context_sha256": "1" * 64,
        "pair_count": 120,
        "case_count": 240,
        "cases": [
            {"case_id": "capture-0", "manifest_sha256": "5" * 64}
        ],
    }
    _write_json(tmp_path / "normal.json", normal)
    _write_json(tmp_path / "contextual.json", contextual)
    _write_json(tmp_path / "action-protocol.json", action_protocol)
    _write_json(tmp_path / "action-plan.json", action_plan)
    _write_json(tmp_path / "action-quality.json", action_quality)
    _write_json(tmp_path / "action-attestation.json", action_attestation)
    catalog = {
        "schema_version": 1,
        "corpora": [
            {
                "corpus_id": "normal",
                "extractor": "multimodal",
                "paths": {"corpus": "normal.json"},
                "campaign_id": "normal-campaign",
                "stack_id": "checkout",
                "status": "open_development",
                "assigned_role": "fit",
            },
            {
                "corpus_id": "contextual",
                "extractor": "multimodal",
                "paths": {"corpus": "contextual.json"},
                "campaign_id": "contextual-cache",
                "source_campaign_id": "normal-campaign",
                "stack_id": "checkout",
                "status": "derived",
            },
            {
                "corpus_id": "action",
                "extractor": "action_dynamics",
                "paths": {
                    "protocol": "action-protocol.json",
                    "plan": "action-plan.json",
                    "data_quality": "action-quality.json",
                    "attestation": "action-attestation.json",
                },
                "campaign_id": "action-campaign",
                "stack_id": "checkout",
                "status": "open_development",
                "assigned_role": "fit",
            },
        ],
    }
    _write_json(tmp_path / "catalog.json", catalog)

    records, identities = discover_corpus_inventory(
        tmp_path, tmp_path / "catalog.json"
    )

    by_id = {record.corpus_id: record for record in records}
    assert by_id["normal"].run_count == 3
    assert by_id["normal"].topology_levels == TOPOLOGIES
    assert by_id["contextual"].source_campaign_id == "normal-campaign"
    assert (
        by_id["contextual"].raw_capture_identity
        == by_id["normal"].raw_capture_identity
    )
    assert by_id["action"].matched_pair_count == 120
    assert by_id["action"].intervention_targets == INTERVENTIONS
    assert by_id["action"].topology_levels == TOPOLOGIES
    assert by_id["action"].workload_families == ("steady",)
    assert by_id["action"].minimum_pairs_per_observed_cell == 8
    assert set(identities) == {
        "normal.json",
        "contextual.json",
        "action-protocol.json",
        "action-plan.json",
        "action-quality.json",
        "action-attestation.json",
        "catalog.json",
    }
    assert all(len(digest) == 64 for digest in identities.values())

    first_action_identity = by_id["action"].raw_capture_identity
    action_quality["case_file_sha256s"]["capture-0"][
        "collector-metrics.jsonl"
    ] = "6" * 64
    _write_json(tmp_path / "action-quality.json", action_quality)
    changed, _ = discover_corpus_inventory(
        tmp_path, tmp_path / "catalog.json"
    )
    changed_by_id = {record.corpus_id: record for record in changed}
    assert changed_by_id["action"].raw_capture_identity != first_action_identity


def test_assessment_does_not_count_runs_topologies_or_derived_caches_as_stacks() -> None:
    records = (
        _record(
            "normal",
            campaign="normal",
            stack="checkout",
            run_count=30,
            interventions=(),
            workloads=("steady",),
            minimum_pairs=0,
            complete=False,
        ),
        _record(
            "contextual",
            campaign="contextual",
            source_campaign="normal",
            stack="checkout",
            status="derived",
            run_count=30,
            interventions=(),
            workloads=("steady",),
            minimum_pairs=0,
            complete=False,
        ),
        _record(
            "action",
            campaign="action",
            stack="checkout",
            run_count=240,
            pair_count=120,
            workloads=("steady",),
            minimum_pairs=8,
            complete=False,
        ),
        _record(
            "edge-cache",
            campaign="edge-cache",
            source_campaign="action",
            stack="checkout",
            status="derived",
            run_count=240,
            pair_count=120,
            workloads=("steady",),
            minimum_pairs=8,
            complete=False,
        ),
        _record(
            "confirmation",
            campaign="confirmation",
            stack="checkout",
            status="result_bearing_confirmation",
            role=None,
            run_count=72,
            interventions=(),
            workloads=("steady",),
            minimum_pairs=0,
            complete=False,
        ),
    )

    result = assess_corpus_diversity(records, MinimumDiversityContract())

    assert result["decision"] == "collect_cross_stack_corpus_before_jepa"
    assert result["inventory"]["primary_open_campaign_count"] == 2
    assert result["inventory"]["distinct_existing_stack_count"] == 1
    assert result["inventory"]["qualifying_complete_stack_count"] == 0
    assert result["inventory"]["eligible_stack_ids"] == ["checkout"]
    assert result["source_campaign_equivalence_classes"] == {
        "action": ["action", "edge-cache"],
        "confirmation": ["confirmation"],
        "normal": ["contextual", "normal"],
    }
    assert result["exclusions"]["derived"] == ["contextual", "edge-cache"]
    assert result["exclusions"]["result_bearing_confirmation"] == [
        "confirmation"
    ]
    assert result["gaps"]["additional_distinct_stacks"] == 5
    assert result["gaps"]["existing_stack_completion_pairs"] == 90
    assert result["gaps"]["new_stack_pairs"] == 675
    assert result["gaps"]["minimum_additional_pairs"] == 765
    assert result["gaps"]["minimum_additional_trajectories"] == 1530
    assert result["gaps"]["factor_coverage"]["checkout"] == {
        "best_corpus_id": "action",
        "intervention_targets": {"observed": 5, "required": 5},
        "topology_levels": {"observed": 3, "required": 3},
        "workload_families": {"observed": 1, "required": 3},
        "minimum_pairs_per_observed_cell": {"observed": 8, "required": 3},
        "missing_factorial_cells": 30,
        "completion_pairs": 90,
    }


def test_strict_six_stack_role_split_is_ready_only_with_complete_factorials() -> None:
    records = tuple(
        _record(
            f"stack-{index}",
            campaign=f"campaign-{index}",
            stack=f"stack-{index}",
            role=role,
            pair_count=135,
            run_count=270,
            workloads=WORKLOADS,
            minimum_pairs=3,
            complete=True,
        )
        for index, role in enumerate(
            ("fit", "fit", "fit", "selection", "calibration", "evaluation")
        )
    )

    ready = assess_corpus_diversity(records, MinimumDiversityContract())

    assert ready["decision"] == "cross_stack_tracer_corpus_ready"
    assert ready["ready"] is True
    assert ready["role_counts"] == {
        "fit": 3,
        "selection": 1,
        "calibration": 1,
        "evaluation": 1,
    }
    assert ready["gaps"]["minimum_additional_pairs"] == 0

    leaked = records + (
        _record(
            "leaked-derived-copy",
            campaign="leaked-derived-copy",
            source_campaign="campaign-5",
            stack="other",
            status="derived",
            role="fit",
            pair_count=135,
            run_count=270,
            workloads=WORKLOADS,
            minimum_pairs=3,
            complete=True,
        ),
    )
    rejected = assess_corpus_diversity(
        leaked, MinimumDiversityContract()
    )
    assert rejected["ready"] is False
    assert "derived_record_assigned_role" in rejected["invalidators"]


def test_ready_rejects_raw_overlap_and_noncanonical_factor_families() -> None:
    roles = ("fit", "fit", "fit", "selection", "calibration", "evaluation")
    records = [
        _record(
            f"stack-{index}",
            campaign=f"campaign-{index}",
            stack=f"stack-{index}",
            role=role,
            pair_count=135,
            run_count=270,
            workloads=WORKLOADS,
            minimum_pairs=3,
            complete=True,
        )
        for index, role in enumerate(roles)
    ]
    records[0] = CorpusRecord.from_dict(
        {
            **records[0].to_dict(),
            "raw_capture_identity": "a" * 64,
            "raw_capture_fingerprints": ["capture-a", "capture-shared"],
        }
    )
    records[5] = CorpusRecord.from_dict(
        {
            **records[5].to_dict(),
            "raw_capture_identity": "b" * 64,
            "raw_capture_fingerprints": ["capture-shared", "capture-c"],
        }
    )

    overlapped = assess_corpus_diversity(
        tuple(records), MinimumDiversityContract()
    )

    assert overlapped["ready"] is False
    assert "raw_capture_crosses_roles" in overlapped["invalidators"]

    records[5] = _record(
        "stack-5",
        campaign="campaign-5",
        stack="stack-5",
        role="evaluation",
        pair_count=135,
        run_count=270,
        workloads=WORKLOADS,
        minimum_pairs=3,
        complete=True,
    )
    records.append(
        _record(
            "supplement-with-extra-family",
            campaign="supplement-with-extra-family",
            stack="stack-0",
            role="fit",
            pair_count=3,
            run_count=6,
            workloads=("stack_specific",),
            minimum_pairs=3,
            complete=False,
        )
    )
    mismatched = assess_corpus_diversity(
        tuple(records), MinimumDiversityContract()
    )
    assert mismatched["ready"] is False
    assert "factor_family_mismatch" in mismatched["invalidators"]


def test_stored_inventory_reassessment_is_deterministic_and_strict() -> None:
    records = [
        _record(
            "action",
            campaign="action",
            stack="checkout",
            pair_count=120,
            run_count=240,
            workloads=("steady",),
            minimum_pairs=8,
            complete=False,
        ).to_dict()
    ]
    contract = MinimumDiversityContract()
    first = assess_serialized_inventory(
        {"schema_version": 1, "corpora": records},
        contract.to_dict(),
    )
    second = assess_serialized_inventory(
        json.loads(json.dumps({"schema_version": 1, "corpora": records})),
        json.loads(json.dumps(contract.to_dict())),
    )

    assert first == second
    assert first["contract_sha256"] == second["contract_sha256"]
    with pytest.raises(ValueError, match="inventory schema"):
        assess_serialized_inventory(
            {"schema_version": 2, "corpora": records},
            contract.to_dict(),
        )


def _record(
    corpus_id: str,
    *,
    campaign: str,
    stack: str,
    source_campaign: str = "",
    status: str = "open_development",
    role: Optional[str] = "fit",
    run_count: int = 0,
    pair_count: int = 0,
    interventions: Tuple[str, ...] = INTERVENTIONS,
    workloads: Tuple[str, ...] = WORKLOADS,
    minimum_pairs: int = 3,
    complete: bool = True,
) -> CorpusRecord:
    return CorpusRecord(
        corpus_id=corpus_id,
        campaign_id=campaign,
        source_campaign_id=source_campaign,
        stack_id=stack,
        status=status,
        assigned_role=role,
        source_paths=(f"{corpus_id}.json",),
        raw_capture_identity=corpus_id.rjust(64, "0")[-64:],
        raw_capture_fingerprints=(
            corpus_id.rjust(64, "0")[-64:],
        ),
        application_build_context_sha256="a" * 64,
        application_image_id="sha256:" + "b" * 64,
        semantic_schema_id="c" * 64,
        run_count=run_count,
        matched_pair_count=pair_count,
        intervention_targets=interventions,
        topology_levels=TOPOLOGIES,
        workload_families=workloads,
        minimum_pairs_per_observed_cell=minimum_pairs,
        complete_factorial=complete,
        notes=(),
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
