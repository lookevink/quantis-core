"""Adversarial validity audit for the retained richer-regime retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from quantis_core.action_dynamics_lab import (
    ActionCollectionProtocol,
    LabActionCaptureManifest,
    assess_action_pair_metric_series,
    load_action_case_metric_series,
)
from quantis_core.richer_regime_retry import WORKLOAD_FAMILIES


_LAB_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "action_case",
    "sample_period_seconds",
    "request_schedule",
    "api_request_queue_size",
    "image_digests",
    "observation_schema_sha256",
    "protocol_sha256",
    "prepared_plan_sha256",
    "graph_observation_schema_sha256",
    "corpus_role",
}
_PAIR_GATES = (
    "schedule_alignment",
    "raw_effect_passed",
    "recovery_passed",
    "count_resolution_passed",
    "drain_eligible",
    "restart_probe_live",
    "mechanistic_recovery_passed",
)


def audit_retained_campaign(
    *,
    fit_campaign: Path,
    selection_campaign: Path,
    amendment_path: Path,
    action_protocol_path: Path,
    output: Path,
) -> Mapping[str, Any]:
    """Audit pair validity and content-address every consumed source file."""

    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite richer-regime validity audit: {output}"
        )
    action_protocol = ActionCollectionProtocol.from_dict(
        _read_object(action_protocol_path)
    )
    role_results = {
        "fit": _audit_role_captures(
            fit_campaign, "fit", action_protocol
        ),
        "selection": _audit_role_captures(
            selection_campaign, "selection", action_protocol
        ),
    }
    amendment = audit_collection_amendment(
        fit_campaign=fit_campaign,
        selection_campaign=selection_campaign,
        amendment_path=amendment_path,
    )
    output.mkdir(parents=True)
    source_manifest = build_consumed_source_manifest(
        fit_campaign=fit_campaign,
        selection_campaign=selection_campaign,
        amendment_path=amendment_path,
        action_protocol_path=action_protocol_path,
    )
    _write_json(output / "source-content-manifest.json", source_manifest)
    all_pairs_valid = all(
        result["failed_pair_count"] == 0
        and result["corpus_complete"] is True
        for result in role_results.values()
    )
    amendment_bound = bool(amendment["selection_manifests_bind_amendment"])
    audit = {
        "schema_version": 1,
        "kind": "richer_regime_retained_validity_audit",
        "scientific_status": (
            "admissible"
            if all_pairs_valid and amendment_bound
            else "inconclusive_methodology_failure"
        ),
        "stored_model_decision_admissible": (
            all_pairs_valid and amendment_bound
        ),
        "roles": role_results,
        "amendment": amendment,
        "action_protocol_sha256": _file_sha256(
            action_protocol_path
        ),
        "source_content_manifest_sha256": _file_sha256(
            output / "source-content-manifest.json"
        ),
        "claim_boundary": (
            "This post-run audit can invalidate retained evidence; it "
            "cannot retroactively repair pair qualification or bind an "
            "unenforced recollection amendment."
        ),
    }
    _write_json(output / "validity-audit.json", audit)
    (output / "report.md").write_text(_report(audit))
    artifact_manifest = {
        "schema_version": 1,
        "kind": "richer_regime_validity_audit_artifact_manifest",
        "sha256": {
            path.relative_to(output).as_posix(): _file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    _write_json(output / "artifact-manifest.json", artifact_manifest)
    return audit


def summarize_action_pair_assessments(
    pair_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Summarize frozen action-specific effect and recovery assessments."""

    if not pair_results:
        raise ValueError(
            "richer-regime validity audit requires pair assessments"
        )
    failures = []
    action_counts: Dict[str, Dict[str, int]] = {}
    reason_counts: Dict[str, int] = {}
    for result in sorted(
        pair_results, key=lambda value: str(value["pair_id"])
    ):
        pair_id = str(result["pair_id"])
        action_kind = str(result["action_kind"])
        counts = action_counts.setdefault(
            action_kind, {"pair_count": 0, "failed_pair_count": 0}
        )
        counts["pair_count"] += 1
        failed_gates = [
            gate for gate in _PAIR_GATES if result.get(gate) is not True
        ]
        if failed_gates:
            counts["failed_pair_count"] += 1
            for gate in failed_gates:
                reason_counts[gate] = reason_counts.get(gate, 0) + 1
            failures.append(
                {
                    "matched_pair_id": pair_id,
                    "action_kind": action_kind,
                    "failed_gates": failed_gates,
                    "active_effect": result["active_effect"],
                    "minimum_effect": result["minimum_effect"],
                    "recovery_ratio": result["recovery_ratio"],
                    "maximum_recovery_ratio": result[
                        "maximum_recovery_ratio"
                    ],
                    "mechanistic_effect": result[
                        "mechanistic_effect"
                    ],
                    "mechanistic_recovery_ratio": result[
                        "mechanistic_recovery_ratio"
                    ],
                }
            )
    pair_count = len(pair_results)
    return {
        "pair_count": pair_count,
        "qualified_pair_count": pair_count - len(failures),
        "failed_pair_count": len(failures),
        "failed_pair_rate": len(failures) / pair_count,
        "failure_counts_by_reason": reason_counts,
        "counts_by_action_kind": action_counts,
        "failures": failures,
    }


def _audit_role_captures(
    campaign: Path,
    role: str,
    action_protocol: ActionCollectionProtocol,
) -> Mapping[str, Any]:
    manifests = []
    metric_series = {}
    pair_ids_by_family: Dict[str, set[str]] = {}
    for family in WORKLOAD_FAMILIES:
        shard = campaign / role / family
        manifest_directory = shard / "inputs" / "manifests"
        family_pair_ids = set()
        for path in sorted(manifest_directory.glob("*.json")):
            raw = _read_object(path)
            manifest = LabActionCaptureManifest.from_dict(
                {key: raw[key] for key in _LAB_MANIFEST_KEYS}
            )
            case_id = manifest.action_case.case_id
            family_pair_ids.add(
                manifest.action_case.matched_pair_id
            )
            manifests.append(manifest)
            metric_series[case_id] = load_action_case_metric_series(
                shard / "cases" / case_id,
                manifest,
            )
        pair_ids_by_family[family] = family_pair_ids
    pair_results = assess_action_pair_metric_series(
        action_protocol,
        tuple(manifests),
        metric_series,
    )
    summary = dict(summarize_action_pair_assessments(pair_results))
    completeness = assess_role_pair_completeness(
        pair_ids_by_family,
        expected_pairs_per_family=30 if role == "fit" else 15,
    )
    summary.update(completeness)
    return summary


def assess_role_pair_completeness(
    pair_ids_by_family: Mapping[str, set[str]],
    *,
    expected_pairs_per_family: int,
) -> Mapping[str, Any]:
    """Require the frozen family balance and cross-family uniqueness."""

    if (
        set(pair_ids_by_family) != set(WORKLOAD_FAMILIES)
        or expected_pairs_per_family < 1
    ):
        raise ValueError("richer-regime family completeness input is invalid")
    counts = {
        family: len(pair_ids_by_family[family])
        for family in WORKLOAD_FAMILIES
    }
    all_pair_ids = [
        pair_id
        for family in WORKLOAD_FAMILIES
        for pair_id in pair_ids_by_family[family]
    ]
    complete = (
        all(count == expected_pairs_per_family for count in counts.values())
        and len(set(all_pair_ids)) == len(all_pair_ids)
    )
    return {
        "corpus_complete": complete,
        "pair_counts_by_workload_family": counts,
        "expected_pairs_per_workload_family": (
            expected_pairs_per_family
        ),
    }


def audit_collection_amendment(
    *,
    fit_campaign: Path,
    selection_campaign: Path,
    amendment_path: Path,
) -> Mapping[str, Any]:
    """Check documentary hashes and manifest-level amendment binding."""

    amendment = _read_object(amendment_path)
    failed_plan = (
        fit_campaign
        / "selection"
        / "steady"
        / "inputs"
        / "plan.json"
    )
    referenced_hashes_valid = bool(
        amendment.get("parent_campaign_protocol_file_sha256")
        == _file_sha256(fit_campaign / "campaign" / "protocol.json")
        and amendment.get("parent_campaign_plan_file_sha256")
        == _file_sha256(fit_campaign / "campaign" / "plan.json")
        and failed_plan.is_file()
        and amendment.get("failed_execution_plan_file_sha256")
        == _file_sha256(failed_plan)
    )
    amendment_sha256 = _file_sha256(amendment_path)
    selection_manifests = [
        path
        for family in WORKLOAD_FAMILIES
        for path in (
            selection_campaign
            / "selection"
            / family
            / "inputs"
            / "manifests"
        ).glob("*.json")
    ]
    embedded = bool(selection_manifests) and all(
        _read_object(path).get("collection_amendment_sha256")
        == amendment_sha256
        for path in selection_manifests
    )
    return {
        "amendment_sha256": amendment_sha256,
        "referenced_hashes_valid": referenced_hashes_valid,
        "selection_manifest_count": len(selection_manifests),
        "selection_manifests_bind_amendment": embedded,
        "operational_conclusion": (
            "document_hashes_match_but_execution_binding_absent"
            if referenced_hashes_valid and not embedded
            else "bound"
            if referenced_hashes_valid and embedded
            else "invalid"
        ),
    }


def build_consumed_source_manifest(
    *,
    fit_campaign: Path,
    selection_campaign: Path,
    amendment_path: Path,
    action_protocol_path: Path,
) -> Mapping[str, Any]:
    """Content-address every file consumed by the retained retry."""

    sources = {
        "amendment": [amendment_path],
        "action_protocol": [action_protocol_path],
        "fit_campaign": [
            fit_campaign / "campaign",
            fit_campaign / "fit",
            fit_campaign / "selection" / "steady",
        ],
        "selection_campaign": [
            selection_campaign / "campaign",
            selection_campaign / "selection",
        ],
    }
    hashes = {}
    for label, paths in sources.items():
        for source in paths:
            candidates = (
                [source]
                if source.is_file()
                else sorted(path for path in source.rglob("*") if path.is_file())
            )
            for path in candidates:
                relative = (
                    path.name
                    if source.is_file()
                    else path.relative_to(source).as_posix()
                )
                hashes[f"{label}/{source.name}/{relative}"] = _file_sha256(
                    path
                )
    return {
        "schema_version": 1,
        "kind": "richer_regime_consumed_source_content_manifest",
        "sha256": dict(sorted(hashes.items())),
    }


def verify_validity_audit(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Verify a pinned validity-audit bundle and return its conclusion."""

    root = Path(directory)
    manifest_path = root / "artifact-manifest.json"
    if _file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("validity-audit artifact identity differs")
    manifest = _read_object(manifest_path)
    raw_hashes = manifest.get("sha256")
    if (
        manifest.get("kind")
        != "richer_regime_validity_audit_artifact_manifest"
        or not isinstance(raw_hashes, dict)
    ):
        raise ValueError("validity-audit manifest is invalid")
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if set(raw_hashes) != expected_paths or any(
        raw_hashes[name] != _file_sha256(root / name)
        for name in expected_paths
    ):
        raise ValueError("validity-audit member hashes differ")
    audit = _read_object(root / "validity-audit.json")
    if (
        audit.get("kind") != "richer_regime_retained_validity_audit"
        or audit.get("scientific_status")
        != "inconclusive_methodology_failure"
        or audit.get("stored_model_decision_admissible") is not False
    ):
        raise ValueError("validity-audit conclusion differs")
    return audit


def _report(audit: Mapping[str, Any]) -> str:
    fit = audit["roles"]["fit"]
    selection = audit["roles"]["selection"]
    amendment = audit["amendment"]
    return (
        "# Richer-regime retained validity audit\n\n"
        f"Scientific status: `{audit['scientific_status']}`\n\n"
        f"- Fit: {fit['failed_pair_count']} / {fit['pair_count']} pairs "
        "failed the frozen action-specific gates.\n"
        f"- Selection: {selection['failed_pair_count']} / "
        f"{selection['pair_count']} pairs failed.\n"
        "- Selection manifests bind the recollection amendment: "
        f"`{amendment['selection_manifests_bind_amendment']}`.\n\n"
        "The stored model metrics remain reproducible diagnostics, but "
        "they are not admissible evidence for the richer-regime model "
        "decision. No retained source artifact was modified.\n"
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-campaign",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/richer-regime-retry-v1"
        ),
    )
    parser.add_argument(
        "--action-protocol",
        type=Path,
        default=Path(
            "lab/action_dynamics/development-protocol-v1.json"
        ),
    )
    parser.add_argument(
        "--selection-campaign",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/richer-regime-retry-v2"
        ),
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "lab/action_dynamics/"
            "richer-regime-collection-amendment-v2.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "richer-regime-retry-v1-validity-audit-v4"
        ),
    )
    parsed = parser.parse_args(arguments)
    audit = audit_retained_campaign(
        fit_campaign=parsed.fit_campaign,
        selection_campaign=parsed.selection_campaign,
        amendment_path=parsed.amendment,
        action_protocol_path=parsed.action_protocol,
        output=parsed.output,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
