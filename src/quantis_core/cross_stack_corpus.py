"""Read-only corpus identity and diversity assessment for cross-stack JEPA."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


PORTABLE_INTERVENTION_TARGETS = (
    "service_pause",
    "persistence_contention",
    "message_production_delay",
    "message_consumption_delay",
    "request_rejection",
)
CANONICAL_TOPOLOGY_LEVELS = ("small", "medium", "large")
CANONICAL_WORKLOAD_FAMILIES = (
    "steady",
    "ramp_or_burst",
    "periodic_or_multiphase",
)
EVIDENCE_ROLES = ("fit", "selection", "calibration", "evaluation")
EVIDENCE_STATUSES = (
    "open_development",
    "derived",
    "derived_confirmation",
    "result_bearing_confirmation",
    "qualification",
    "synthetic",
)

_ACTION_TO_PORTABLE_TARGET = {
    "worker_pause": "service_pause",
    "postgres_lock": "persistence_contention",
    "redis_enqueue_delay": "message_production_delay",
    "redis_dequeue_delay": "message_consumption_delay",
    "api_rejection": "request_rejection",
}


@dataclass(frozen=True)
class CorpusRecord:
    """Normalized identity and factorial coverage for one corpus artifact."""

    corpus_id: str
    campaign_id: str
    source_campaign_id: str
    stack_id: str
    status: str
    assigned_role: Optional[str]
    source_paths: Tuple[str, ...]
    raw_capture_identity: str
    raw_capture_fingerprints: Tuple[str, ...]
    application_build_context_sha256: str
    application_image_id: str
    semantic_schema_id: str
    run_count: int
    matched_pair_count: int
    intervention_targets: Tuple[str, ...]
    topology_levels: Tuple[str, ...]
    workload_families: Tuple[str, ...]
    minimum_pairs_per_observed_cell: int
    complete_factorial: bool
    notes: Tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("corpus_id", self.corpus_id),
            ("campaign_id", self.campaign_id),
            ("stack_id", self.stack_id),
        ):
            if not value:
                raise ValueError(f"{name} cannot be empty")
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError("unsupported corpus evidence status")
        if self.assigned_role is not None and self.assigned_role not in EVIDENCE_ROLES:
            raise ValueError("unsupported corpus evidence role")
        if self.run_count < 0 or self.matched_pair_count < 0:
            raise ValueError("corpus counts must be nonnegative")
        if self.minimum_pairs_per_observed_cell < 0:
            raise ValueError("minimum cell repetition count must be nonnegative")
        for values, label in (
            (self.source_paths, "source paths"),
            (self.raw_capture_fingerprints, "raw capture fingerprints"),
            (self.intervention_targets, "intervention targets"),
            (self.topology_levels, "topology levels"),
            (self.workload_families, "workload families"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "campaign_id": self.campaign_id,
            "source_campaign_id": self.source_campaign_id,
            "stack_id": self.stack_id,
            "status": self.status,
            "assigned_role": self.assigned_role,
            "source_paths": list(self.source_paths),
            "raw_capture_identity": self.raw_capture_identity,
            "raw_capture_fingerprints": list(
                self.raw_capture_fingerprints
            ),
            "application_build_context_sha256": (
                self.application_build_context_sha256
            ),
            "application_image_id": self.application_image_id,
            "semantic_schema_id": self.semantic_schema_id,
            "run_count": self.run_count,
            "matched_pair_count": self.matched_pair_count,
            "intervention_targets": list(self.intervention_targets),
            "topology_levels": list(self.topology_levels),
            "workload_families": list(self.workload_families),
            "minimum_pairs_per_observed_cell": (
                self.minimum_pairs_per_observed_cell
            ),
            "complete_factorial": self.complete_factorial,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorpusRecord":
        role_value = payload.get("assigned_role")
        return cls(
            corpus_id=str(payload["corpus_id"]),
            campaign_id=str(payload["campaign_id"]),
            source_campaign_id=str(payload.get("source_campaign_id", "")),
            stack_id=str(payload["stack_id"]),
            status=str(payload["status"]),
            assigned_role=None if role_value is None else str(role_value),
            source_paths=tuple(str(value) for value in payload["source_paths"]),
            raw_capture_identity=str(payload["raw_capture_identity"]),
            raw_capture_fingerprints=tuple(
                str(value)
                for value in payload["raw_capture_fingerprints"]
            ),
            application_build_context_sha256=str(
                payload.get("application_build_context_sha256", "")
            ),
            application_image_id=str(payload.get("application_image_id", "")),
            semantic_schema_id=str(payload.get("semantic_schema_id", "")),
            run_count=int(payload["run_count"]),
            matched_pair_count=int(payload["matched_pair_count"]),
            intervention_targets=tuple(
                str(value) for value in payload["intervention_targets"]
            ),
            topology_levels=tuple(
                str(value) for value in payload["topology_levels"]
            ),
            workload_families=tuple(
                str(value) for value in payload["workload_families"]
            ),
            minimum_pairs_per_observed_cell=int(
                payload["minimum_pairs_per_observed_cell"]
            ),
            complete_factorial=bool(payload["complete_factorial"]),
            notes=tuple(str(value) for value in payload.get("notes", ())),
        )


@dataclass(frozen=True)
class MinimumDiversityContract:
    """Strict role-separated exploratory cross-stack tracer floor."""

    fit_stack_count: int = 3
    selection_stack_count: int = 1
    calibration_stack_count: int = 1
    evaluation_stack_count: int = 1
    intervention_targets: Tuple[str, ...] = PORTABLE_INTERVENTION_TARGETS
    topology_levels: Tuple[str, ...] = CANONICAL_TOPOLOGY_LEVELS
    workload_families: Tuple[str, ...] = CANONICAL_WORKLOAD_FAMILIES
    matched_pairs_per_cell: int = 3

    def __post_init__(self) -> None:
        counts = (
            self.fit_stack_count,
            self.selection_stack_count,
            self.calibration_stack_count,
            self.evaluation_stack_count,
            self.matched_pairs_per_cell,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("minimum diversity counts must be positive")

    @property
    def total_stack_count(self) -> int:
        return (
            self.fit_stack_count
            + self.selection_stack_count
            + self.calibration_stack_count
            + self.evaluation_stack_count
        )

    @property
    def pairs_per_stack(self) -> int:
        return (
            len(self.intervention_targets)
            * len(self.topology_levels)
            * len(self.workload_families)
            * self.matched_pairs_per_cell
        )

    def role_requirements(self) -> Dict[str, int]:
        return {
            "fit": self.fit_stack_count,
            "selection": self.selection_stack_count,
            "calibration": self.calibration_stack_count,
            "evaluation": self.evaluation_stack_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "cross_stack_jepa_minimum_diversity_contract",
            "role_stack_counts": self.role_requirements(),
            "intervention_targets": list(self.intervention_targets),
            "topology_levels": list(self.topology_levels),
            "workload_families": list(self.workload_families),
            "matched_pairs_per_cell": self.matched_pairs_per_cell,
            "pairs_per_stack": self.pairs_per_stack,
            "total_stack_count": self.total_stack_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "MinimumDiversityContract":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported diversity contract schema")
        role_counts = _object(payload["role_stack_counts"])
        return cls(
            fit_stack_count=int(role_counts["fit"]),
            selection_stack_count=int(role_counts["selection"]),
            calibration_stack_count=int(role_counts["calibration"]),
            evaluation_stack_count=int(role_counts["evaluation"]),
            intervention_targets=tuple(
                str(value) for value in payload["intervention_targets"]
            ),
            topology_levels=tuple(
                str(value) for value in payload["topology_levels"]
            ),
            workload_families=tuple(
                str(value) for value in payload["workload_families"]
            ),
            matched_pairs_per_cell=int(payload["matched_pairs_per_cell"]),
        )


def assess_corpus_diversity(
    records: Sequence[CorpusRecord],
    contract: MinimumDiversityContract,
) -> Dict[str, Any]:
    """Classify corpus evidence and compute the strict tracer acquisition gap."""

    ordered = tuple(sorted(records, key=lambda item: item.corpus_id))
    if len({item.corpus_id for item in ordered}) != len(ordered):
        raise ValueError("corpus ids must be unique")

    primary_open = tuple(
        item
        for item in ordered
        if item.status == "open_development" and not item.source_campaign_id
    )
    existing_stack_ids = sorted({item.stack_id for item in primary_open})
    qualifying = tuple(
        item for item in primary_open if _qualifies(item, contract)
    )
    qualifying_by_stack = _best_record_by_stack(qualifying, contract)

    exclusions: Dict[str, List[str]] = {}
    for item in ordered:
        if item.status != "open_development":
            exclusions.setdefault(item.status, []).append(item.corpus_id)

    invalidators: List[str] = []
    if any(
        item.assigned_role is not None
        for item in ordered
        if item.status in ("derived", "derived_confirmation")
    ):
        invalidators.append("derived_record_assigned_role")
    if any(
        item.assigned_role is not None
        for item in ordered
        if item.status
        in ("result_bearing_confirmation", "qualification", "synthetic")
    ):
        invalidators.append("ineligible_record_assigned_role")
    if any(
        item.assigned_role is not None
        and (
            not set(item.intervention_targets)
            <= set(contract.intervention_targets)
            or not set(item.topology_levels) <= set(contract.topology_levels)
            or not set(item.workload_families)
            <= set(contract.workload_families)
        )
        for item in primary_open
    ):
        invalidators.append("factor_family_mismatch")

    campaign_roles: Dict[str, Set[str]] = {}
    stack_roles: Dict[str, Set[str]] = {}
    raw_capture_roles: Dict[str, Set[str]] = {}
    for item in primary_open:
        if item.assigned_role is None:
            continue
        for fingerprint in item.raw_capture_fingerprints:
            raw_capture_roles.setdefault(fingerprint, set()).add(
                item.assigned_role
            )
    for item in qualifying:
        if item.assigned_role is None:
            invalidators.append("qualifying_stack_without_role")
            continue
        campaign_roles.setdefault(item.campaign_id, set()).add(
            item.assigned_role
        )
        stack_roles.setdefault(item.stack_id, set()).add(item.assigned_role)
    if any(len(roles) > 1 for roles in campaign_roles.values()):
        invalidators.append("campaign_crosses_roles")
    if any(len(roles) > 1 for roles in stack_roles.values()):
        invalidators.append("stack_crosses_roles")
    if any(len(roles) > 1 for roles in raw_capture_roles.values()):
        invalidators.append("raw_capture_crosses_roles")

    role_stacks = {
        role: {
            item.stack_id
            for item in qualifying
            if item.assigned_role == role
        }
        for role in EVIDENCE_ROLES
    }
    role_counts = {
        role: len(role_stacks[role])
        for role in EVIDENCE_ROLES
    }
    requirements = contract.role_requirements()
    role_gaps = {
        role: max(0, requirements[role] - role_counts[role])
        for role in EVIDENCE_ROLES
    }

    existing_completion_pairs = sum(
        _stack_completion_gap(
            tuple(
                item
                for item in primary_open
                if item.stack_id == stack_id
            ),
            contract,
        )
        for stack_id in existing_stack_ids[: contract.total_stack_count]
    )
    additional_stacks = max(
        0, contract.total_stack_count - len(existing_stack_ids)
    )
    new_stack_pairs = additional_stacks * contract.pairs_per_stack
    minimum_additional_pairs = (
        existing_completion_pairs + new_stack_pairs
    )
    factor_coverage = {
        stack_id: _factor_coverage(
            _best_partial_record(
                tuple(
                    item
                    for item in primary_open
                    if item.stack_id == stack_id
                ),
                contract,
            ),
            contract,
        )
        for stack_id in existing_stack_ids[: contract.total_stack_count]
    }
    equivalence_classes: Dict[str, List[str]] = {}
    for item in ordered:
        root_campaign = item.source_campaign_id or item.campaign_id
        equivalence_classes.setdefault(root_campaign, []).append(
            item.corpus_id
        )
    ready = (
        not invalidators
        and len(qualifying_by_stack) >= contract.total_stack_count
        and all(gap == 0 for gap in role_gaps.values())
    )
    if ready:
        existing_completion_pairs = 0
        new_stack_pairs = 0
        minimum_additional_pairs = 0

    contract_payload = contract.to_dict()
    return {
        "schema_version": 1,
        "kind": "cross_stack_jepa_corpus_diversity_assessment",
        "contract_sha256": _canonical_sha256(contract_payload),
        "decision": (
            "cross_stack_tracer_corpus_ready"
            if ready
            else "collect_cross_stack_corpus_before_jepa"
        ),
        "ready": ready,
        "inventory": {
            "corpus_count": len(ordered),
            "primary_open_campaign_count": len(primary_open),
            "distinct_existing_stack_count": len(existing_stack_ids),
            "qualifying_complete_stack_count": len(qualifying_by_stack),
            "eligible_stack_ids": existing_stack_ids,
            "qualifying_stack_ids": sorted(qualifying_by_stack),
        },
        "role_counts": role_counts,
        "role_requirements": requirements,
        "source_campaign_equivalence_classes": {
            campaign: sorted(corpora)
            for campaign, corpora in sorted(equivalence_classes.items())
        },
        "exclusions": {
            key: sorted(values)
            for key, values in sorted(exclusions.items())
        },
        "invalidators": sorted(set(invalidators)),
        "gaps": {
            "role_stack_gaps": role_gaps,
            "additional_distinct_stacks": additional_stacks,
            "existing_stack_completion_pairs": existing_completion_pairs,
            "new_stack_pairs": new_stack_pairs,
            "minimum_additional_pairs": minimum_additional_pairs,
            "minimum_additional_trajectories": (
                minimum_additional_pairs * 2
            ),
            "factor_coverage": factor_coverage,
        },
        "minimum_design": contract_payload,
        "corpora": [item.to_dict() for item in ordered],
    }


def assess_serialized_inventory(
    inventory: Mapping[str, Any],
    contract_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reassess stored JSON-compatible inputs without opening source corpora."""

    if inventory.get("schema_version") != 1:
        raise ValueError("unsupported inventory schema")
    records = tuple(
        CorpusRecord.from_dict(_object(item))
        for item in _sequence(inventory["corpora"])
    )
    contract = MinimumDiversityContract.from_dict(contract_payload)
    return assess_corpus_diversity(records, contract)


def discover_corpus_inventory(
    workspace_root: Path,
    catalog_path: Path,
) -> Tuple[Tuple[CorpusRecord, ...], Dict[str, str]]:
    """Extract normalized records from a declarative candidate-corpus catalog."""

    root = Path(workspace_root)
    catalog_file = Path(catalog_path)
    catalog = _read_object(catalog_file)
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported corpus catalog schema")
    identities: Dict[str, str] = {
        _relative_or_name(catalog_file, root): _sha256_file(catalog_file)
    }
    records: List[CorpusRecord] = []
    for raw_declaration in _sequence(catalog["corpora"]):
        declaration = _object(raw_declaration)
        paths = {
            str(name): root / str(value)
            for name, value in _object(declaration["paths"]).items()
        }
        for path in paths.values():
            identities[_relative_or_name(path, root)] = _sha256_file(path)
        extractor = str(declaration["extractor"])
        if extractor == "multimodal":
            extracted = _extract_multimodal(paths)
        elif extractor == "action_dynamics":
            extracted = _extract_action_dynamics(paths)
        elif extractor == "event_cache":
            extracted = _extract_event_cache(paths)
        elif extractor == "edge_cache":
            extracted = _extract_edge_cache(paths)
        else:
            raise ValueError(f"unsupported corpus extractor: {extractor}")
        records.append(_record_from_declaration(declaration, paths, extracted))
    return tuple(records), dict(sorted(identities.items()))


def _record_from_declaration(
    declaration: Mapping[str, Any],
    paths: Mapping[str, Path],
    extracted: Mapping[str, Any],
) -> CorpusRecord:
    role_value = declaration.get("assigned_role")
    return CorpusRecord(
        corpus_id=str(declaration["corpus_id"]),
        campaign_id=str(declaration["campaign_id"]),
        source_campaign_id=str(
            declaration.get("source_campaign_id", "")
        ),
        stack_id=str(declaration["stack_id"]),
        status=str(declaration["status"]),
        assigned_role=None if role_value is None else str(role_value),
        source_paths=tuple(
            sorted(
                str(value)
                for value in _object(declaration["paths"]).values()
            )
        ),
        raw_capture_identity=str(extracted["raw_capture_identity"]),
        raw_capture_fingerprints=tuple(
            str(value)
            for value in extracted["raw_capture_fingerprints"]
        ),
        application_build_context_sha256=str(
            extracted.get("application_build_context_sha256", "")
        ),
        application_image_id=str(
            extracted.get("application_image_id", "")
        ),
        semantic_schema_id=str(extracted.get("semantic_schema_id", "")),
        run_count=int(extracted["run_count"]),
        matched_pair_count=int(extracted["matched_pair_count"]),
        intervention_targets=tuple(extracted["intervention_targets"]),
        topology_levels=tuple(extracted["topology_levels"]),
        workload_families=tuple(extracted["workload_families"]),
        minimum_pairs_per_observed_cell=int(
            extracted["minimum_pairs_per_observed_cell"]
        ),
        complete_factorial=bool(extracted["complete_factorial"]),
        notes=tuple(
            str(value) for value in declaration.get("notes", ())
        ),
    )


def _extract_multimodal(paths: Mapping[str, Path]) -> Dict[str, Any]:
    payload = _read_object(paths["corpus"])
    protocol = _find_run_protocol(payload)
    runs = _object(protocol["runs"])
    capture_ids = sorted(
        str(_object(value).get("capture_sha256", key))
        for key, value in runs.items()
    )
    replica_counts = sorted(
        {
            int(match.group(1))
            for case_id in runs
            for match in [re.search(r"-w([0-9]+)(?:-|$)", str(case_id))]
            if match is not None
        }
    )
    return {
        "raw_capture_identity": _canonical_sha256(capture_ids),
        "raw_capture_fingerprints": tuple(capture_ids),
        "application_build_context_sha256": str(
            protocol.get("application_build_context_sha256", "")
        ),
        "application_image_id": str(
            protocol.get("application_image_id", "")
        ),
        "semantic_schema_id": str(
            protocol.get("feature_schema_id", "")
        ),
        "run_count": len(runs),
        "matched_pair_count": 0,
        "intervention_targets": (),
        "topology_levels": _canonical_levels(replica_counts),
        "workload_families": ("steady",),
        "minimum_pairs_per_observed_cell": 0,
        "complete_factorial": False,
    }


def _extract_action_dynamics(paths: Mapping[str, Path]) -> Dict[str, Any]:
    protocol = _read_object(paths["protocol"])
    plan = _read_object(paths["plan"])
    quality = _read_object(paths["data_quality"])
    attestation = _read_object(paths["attestation"])
    action_library = _object(protocol["action_library"])
    portable_targets = tuple(
        target
        for target in PORTABLE_INTERVENTION_TARGETS
        if target
        in {
            _ACTION_TO_PORTABLE_TARGET[name]
            for name in action_library
            if name in _ACTION_TO_PORTABLE_TARGET
        }
    )
    design = _object(protocol["design"])
    replica_values = sorted(
        int(value) for value in _sequence(design["worker_replica_values"])
    )
    assignments = _sequence(plan["assignments"])
    capture_ids = sorted(
        str(_object(value).get("case_id", position))
        for position, value in enumerate(assignments)
    )
    case_file_sha256s = quality.get("case_file_sha256s", {})
    attested_cases = attestation.get("cases", ())
    case_files_by_id = _object(case_file_sha256s)
    attested_by_id = {
        str(_object(value).get("case_id", "")): str(
            _object(value).get("manifest_sha256", "")
        )
        for value in _sequence(attested_cases)
    }
    all_case_ids = sorted(
        set(capture_ids) | set(case_files_by_id) | set(attested_by_id)
    )
    raw_capture_fingerprints = tuple(
        _canonical_sha256(
            {
                "case_id": case_id,
                "case_file_sha256s": case_files_by_id.get(case_id, {}),
                "attested_manifest_sha256": attested_by_id.get(
                    case_id, ""
                ),
            }
        )
        for case_id in all_case_ids
    )
    semantic_schema_id = ""
    case_specs = plan.get("case_specs")
    if isinstance(case_specs, Mapping) and case_specs:
        first_case = _object(next(iter(case_specs.values())))
        semantic_schema_id = str(
            first_case.get(
                "graph_observation_schema_sha256",
                first_case.get("observation_schema_sha256", ""),
            )
        )
    workload = _object(protocol["workload"])
    schedule_kind = str(workload.get("schedule_kind", ""))
    workload_families: Tuple[str, ...] = ()
    if schedule_kind == "seeded_explicit_uniform_integer":
        workload_families = ("steady",)
    counts = _object(quality["counts"])
    return {
        "raw_capture_identity": _canonical_sha256(
            raw_capture_fingerprints
        ),
        "raw_capture_fingerprints": raw_capture_fingerprints,
        "application_build_context_sha256": str(
            plan.get(
                "application_build_context_sha256",
                attestation.get("application_build_context_sha256", ""),
            )
        ),
        "application_image_id": str(
            attestation.get("application_image_id", "")
        ),
        "semantic_schema_id": semantic_schema_id,
        "run_count": int(counts["case_count"]),
        "matched_pair_count": int(counts["pair_count"]),
        "intervention_targets": portable_targets,
        "topology_levels": _canonical_levels(replica_values),
        "workload_families": workload_families,
        "minimum_pairs_per_observed_cell": int(
            design["replicates_per_cell"]
        ),
        "complete_factorial": (
            portable_targets == PORTABLE_INTERVENTION_TARGETS
            and _canonical_levels(replica_values)
            == CANONICAL_TOPOLOGY_LEVELS
            and workload_families == CANONICAL_WORKLOAD_FAMILIES
        ),
    }


def _extract_event_cache(paths: Mapping[str, Path]) -> Dict[str, Any]:
    payload = _read_object(paths["events"])
    captures = _object(payload["capture_sha256"])
    replica_counts = sorted(
        {
            int(match.group(1))
            for case_id in captures
            for match in [re.search(r"-w([0-9]+)(?:-|$)", str(case_id))]
            if match is not None
        }
    )
    return {
        "raw_capture_identity": _canonical_sha256(
            sorted(str(value) for value in captures.values())
        ),
        "raw_capture_fingerprints": tuple(
            sorted(str(value) for value in captures.values())
        ),
        "run_count": len(captures),
        "matched_pair_count": 0,
        "intervention_targets": (),
        "topology_levels": _canonical_levels(replica_counts),
        "workload_families": ("steady",),
        "minimum_pairs_per_observed_cell": 0,
        "complete_factorial": False,
    }


def _extract_edge_cache(paths: Mapping[str, Path]) -> Dict[str, Any]:
    payload = _read_object(paths["metadata"])
    role_counts = _object(_object(payload["roles"])["pair_counts"])
    pair_count = sum(int(value) for value in role_counts.values())
    return {
        "raw_capture_identity": str(payload["source_corpus_sha256"]),
        "raw_capture_fingerprints": (
            str(payload["source_corpus_sha256"]),
        ),
        "semantic_schema_id": str(
            _object(payload["compiler"]).get("semantic_schema_sha256", "")
        ),
        "run_count": pair_count * 2,
        "matched_pair_count": pair_count,
        "intervention_targets": PORTABLE_INTERVENTION_TARGETS,
        "topology_levels": CANONICAL_TOPOLOGY_LEVELS,
        "workload_families": ("steady",),
        "minimum_pairs_per_observed_cell": 8,
        "complete_factorial": False,
    }


def _qualifies(
    record: CorpusRecord, contract: MinimumDiversityContract
) -> bool:
    return (
        record.status == "open_development"
        and not record.source_campaign_id
        and record.assigned_role is not None
        and record.complete_factorial
        and set(record.intervention_targets)
        == set(contract.intervention_targets)
        and set(record.topology_levels) == set(contract.topology_levels)
        and set(record.workload_families) == set(contract.workload_families)
        and record.minimum_pairs_per_observed_cell
        >= contract.matched_pairs_per_cell
        and record.matched_pair_count >= contract.pairs_per_stack
    )


def _best_record_by_stack(
    records: Sequence[CorpusRecord],
    contract: MinimumDiversityContract,
) -> Dict[str, CorpusRecord]:
    best: Dict[str, CorpusRecord] = {}
    for item in records:
        previous = best.get(item.stack_id)
        if previous is None or _coverage_score(item, contract) > _coverage_score(
            previous, contract
        ):
            best[item.stack_id] = item
    return best


def _stack_completion_gap(
    records: Sequence[CorpusRecord],
    contract: MinimumDiversityContract,
) -> int:
    if not records:
        return contract.pairs_per_stack
    return min(_record_completion_gap(item, contract) for item in records)


def _best_partial_record(
    records: Sequence[CorpusRecord],
    contract: MinimumDiversityContract,
) -> CorpusRecord:
    if not records:
        raise ValueError("cannot select from an empty stack record set")
    return max(records, key=lambda item: _coverage_score(item, contract))


def _factor_coverage(
    record: CorpusRecord,
    contract: MinimumDiversityContract,
) -> Dict[str, Any]:
    observed_interventions = len(
        set(record.intervention_targets) & set(contract.intervention_targets)
    )
    observed_topologies = len(
        set(record.topology_levels) & set(contract.topology_levels)
    )
    observed_workloads = len(
        set(record.workload_families) & set(contract.workload_families)
    )
    observed_cells = (
        observed_interventions
        * observed_topologies
        * observed_workloads
    )
    required_cells = (
        len(contract.intervention_targets)
        * len(contract.topology_levels)
        * len(contract.workload_families)
    )
    return {
        "best_corpus_id": record.corpus_id,
        "intervention_targets": {
            "observed": observed_interventions,
            "required": len(contract.intervention_targets),
        },
        "topology_levels": {
            "observed": observed_topologies,
            "required": len(contract.topology_levels),
        },
        "workload_families": {
            "observed": observed_workloads,
            "required": len(contract.workload_families),
        },
        "minimum_pairs_per_observed_cell": {
            "observed": record.minimum_pairs_per_observed_cell,
            "required": contract.matched_pairs_per_cell,
        },
        "missing_factorial_cells": required_cells - observed_cells,
        "completion_pairs": _record_completion_gap(record, contract),
    }


def _record_completion_gap(
    record: CorpusRecord,
    contract: MinimumDiversityContract,
) -> int:
    observed_interventions = len(
        set(record.intervention_targets) & set(contract.intervention_targets)
    )
    observed_topologies = len(
        set(record.topology_levels) & set(contract.topology_levels)
    )
    observed_workloads = len(
        set(record.workload_families) & set(contract.workload_families)
    )
    observed_cells = (
        observed_interventions
        * observed_topologies
        * observed_workloads
    )
    required_cells = (
        len(contract.intervention_targets)
        * len(contract.topology_levels)
        * len(contract.workload_families)
    )
    missing_cells = required_cells - observed_cells
    repetition_gap = max(
        0,
        contract.matched_pairs_per_cell
        - record.minimum_pairs_per_observed_cell,
    )
    return (
        missing_cells * contract.matched_pairs_per_cell
        + observed_cells * repetition_gap
    )


def _coverage_score(
    record: CorpusRecord,
    contract: MinimumDiversityContract,
) -> Tuple[int, int]:
    return (-_record_completion_gap(record, contract), record.matched_pair_count)


def _find_run_protocol(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "runs" in payload and isinstance(payload["runs"], Mapping):
        return payload
    for value in payload.values():
        if isinstance(value, Mapping):
            try:
                return _find_run_protocol(value)
            except ValueError:
                pass
    raise ValueError("multimodal corpus has no run protocol")


def _canonical_levels(values: Sequence[int]) -> Tuple[str, ...]:
    count = min(len(set(values)), len(CANONICAL_TOPOLOGY_LEVELS))
    return CANONICAL_TOPOLOGY_LEVELS[:count]


def _read_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    return dict(_object(payload))


def _object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError("expected JSON array")
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
