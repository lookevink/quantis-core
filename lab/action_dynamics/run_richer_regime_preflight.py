"""Run fit-only mechanism preflights for the richer-regime campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from quantis_core.action_dynamics_corpus import (
    EVENT_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    _development_graph,
    _observations,
)
from quantis_core.action_dynamics_lab import (
    LabActionCaptureManifest,
)
from quantis_core.richer_regime_preflight import (
    RicherRegimeFitEvidence,
    assess_richer_regime_fit_preflight,
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


def run_fit_preflight(
    *, campaign_directory: Path, output: Path
) -> Mapping[str, Any]:
    """Load only fit shards and write the frozen mechanism decision."""

    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite richer-regime preflight: {output}"
        )
    root = Path(campaign_directory)
    campaign_plan = _read_object(root / "campaign" / "plan.json")
    raw_pairs = campaign_plan.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("richer-regime campaign plan is invalid")
    pair_metadata = {
        str(pair["pair_id"]): dict(pair)
        for pair in raw_pairs
        if isinstance(pair, dict)
    }
    graph = _development_graph()
    metric_feature_count = len(STATE_FEATURE_NAMES) - len(
        EVENT_FEATURE_NAMES
    )
    metric_context_rows = []
    event_context_rows = []
    target_rows = []
    demand_rows = []
    topology_rows = []
    family_rows = []
    replicate_rows = []
    schedule_features = []
    schedule_families = []
    schedule_replicates = []
    source_hashes = {}
    for family in WORKLOAD_FAMILIES:
        shard = root / "fit" / family
        assessment_path = shard / "shard-assessment.json"
        assessment = _read_object(assessment_path)
        if (
            assessment.get("status") != "qualified"
            or assessment.get("corpus_role") != "fit"
            or assessment.get("workload_family") != family
        ):
            raise ValueError(
                f"fit shard is not qualified: {family}"
            )
        source_hashes[family] = _file_sha256(assessment_path)
        manifests = shard / "inputs" / "manifests"
        for path in sorted(manifests.glob("*.json")):
            raw_manifest = _read_object(path)
            raw_case = raw_manifest.get("action_case")
            if (
                not isinstance(raw_case, dict)
                or raw_case.get("actions") != []
            ):
                continue
            pair_id = str(raw_case["matched_pair_id"])
            metadata = pair_metadata.get(pair_id)
            if (
                metadata is None
                or metadata.get("corpus_role") != "fit"
                or metadata.get("workload_family") != family
                or metadata.get("replicate") not in {0, 1}
            ):
                raise ValueError(
                    "control manifest differs from campaign ownership"
                )
            lab_manifest = LabActionCaptureManifest.from_dict(
                {
                    key: raw_manifest[key]
                    for key in _LAB_MANIFEST_KEYS
                }
            )
            capture = shard / "cases" / path.stem
            observations = _observations(
                capture, lab_manifest, graph
            ).sum(axis=1)
            metrics = observations[:, :metric_feature_count]
            events = observations[:, metric_feature_count:]
            transition_count = len(metrics) - 1
            metric_context_rows.append(metrics[:-1])
            event_context_rows.append(events[:-1])
            target_rows.append(metrics[1:])
            demand_rows.append(
                np.asarray(
                    lab_manifest.request_schedule[:-1],
                    dtype=np.float64,
                )
            )
            topology_rows.append(
                np.full(
                    transition_count,
                    float(lab_manifest.action_case.worker_replicas),
                    dtype=np.float64,
                )
            )
            family_rows.extend([family] * transition_count)
            replicate_rows.extend(
                [int(metadata["replicate"])] * transition_count
            )
            schedule_features.append(
                _schedule_features(lab_manifest.request_schedule)
            )
            schedule_families.append(family)
            schedule_replicates.append(
                int(metadata["replicate"])
            )
    evidence = RicherRegimeFitEvidence(
        metric_context=np.concatenate(
            metric_context_rows, axis=0
        ),
        event_context=np.concatenate(
            event_context_rows, axis=0
        ),
        targets=np.concatenate(target_rows, axis=0),
        demand=np.concatenate(demand_rows, axis=0),
        topology=np.concatenate(topology_rows, axis=0),
        workload_families=tuple(family_rows),
        replicates=np.asarray(replicate_rows, dtype=np.int64),
        regime_classification_accuracy=_regime_accuracy(
            np.asarray(schedule_features, dtype=np.float64),
            tuple(schedule_families),
            np.asarray(schedule_replicates, dtype=np.int64),
        ),
    )
    assessment = dict(
        assess_richer_regime_fit_preflight(evidence)
    )
    assessment["source_shard_assessment_sha256s"] = source_hashes
    assessment["counts"] = {
        "control_run_count": len(schedule_features),
        "transition_count": len(evidence.metric_context),
        "fit_replicate_count": 2,
        "workload_family_count": 3,
    }
    output.mkdir(parents=True)
    _write_json(output / "fit-preflight-v1.json", assessment)
    (output / "report.md").write_text(_report(assessment))
    return assessment


def _schedule_features(
    schedule: Sequence[int],
) -> NDArray[np.float64]:
    values = np.asarray(schedule[:-1], dtype=np.float64)
    positions = np.arange(len(values), dtype=np.float64)
    slope = float(
        np.polyfit(positions, values, deg=1)[0]
    )
    phase_means = [
        float(np.mean(values[start : start + 12]))
        for start in range(0, 96, 12)
    ]
    return np.asarray(
        (
            np.mean(values),
            np.std(values),
            np.max(values),
            slope,
            np.max(values[36:48]) - np.mean(values[:12]),
            np.mean(phase_means[1::2])
            - np.mean(phase_means[::2]),
        ),
        dtype=np.float64,
    )


def _regime_accuracy(
    features: NDArray[np.float64],
    families: tuple[str, ...],
    replicates: NDArray[np.int64],
) -> float:
    training = replicates == 0
    probe = replicates == 1
    mean = np.mean(features[training], axis=0)
    scale = np.std(features[training], axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (features - mean) / scale
    family_array = np.asarray(families, dtype=object)
    centroids = {
        family: np.mean(
            normalized[training & (family_array == family)], axis=0
        )
        for family in WORKLOAD_FAMILIES
    }
    predicted = [
        min(
            WORKLOAD_FAMILIES,
            key=lambda family: float(
                np.sum(
                    np.square(
                        normalized[index] - centroids[family]
                    )
                )
            ),
        )
        for index in np.flatnonzero(probe)
    ]
    observed = family_array[probe].tolist()
    return float(
        np.mean(
            np.asarray(predicted, dtype=object)
            == np.asarray(observed, dtype=object)
        )
    )


def _report(assessment: Mapping[str, Any]) -> str:
    measurements = assessment["measurements"]
    recommendations = assessment["recommendations"]
    gates = assessment["gates"]
    lines = [
        "# Richer-regime fit-only preflight",
        "",
        f"Status: `{assessment['status']}`",
        "",
        "This report used fit replicas 0 and 1 only. Selection, "
        "calibration, and evaluation evidence remained unopened.",
        "",
        "## Measurements",
        "",
    ]
    lines.extend(
        f"- `{name}`: {float(value):.6f}"
        for name, value in measurements.items()
    )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in gates.items()
    )
    lines.extend(["", "## Retry decisions", ""])
    lines.extend(
        f"- `{name}`: `{decision}`"
        for name, decision in recommendations.items()
    )
    lines.append("")
    return "\n".join(lines)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
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
        "--campaign",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/richer-regime-retry-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "richer-regime-retry-v1/preflight"
        ),
    )
    parsed = parser.parse_args(arguments)
    assessment = run_fit_preflight(
        campaign_directory=parsed.campaign,
        output=parsed.output,
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment.get("status") == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
