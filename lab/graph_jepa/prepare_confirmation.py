"""Prepare the preregistered observability-rich graph corpus inputs."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from quantis_core.contextual_confirmation import (
    ConfirmationCollectionCase,
)
from quantis_core.fault_matrix import FaultMatrixCaseManifest
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec


IMAGES = {
    "python": (
        "python@sha256:"
        "57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
    ),
    "redis": (
        "redis@sha256:"
        "e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2"
    ),
    "postgres": (
        "postgres@sha256:"
        "742f40ea20b9ff2ff31db5458d127452988a2164df9e17441e191f3b72252193"
    ),
    "collector": (
        "ghcr.io/open-telemetry/opentelemetry-collector-releases/"
        "opentelemetry-collector-contrib:0.153.0@sha256:"
        "93aad750175cbf1a973ae1c5886c3371f4d800f61be25cdd26870b8441ffe9fa"
    ),
}
ScheduleFamilies = Tuple[Tuple[int, Tuple[int, ...]], ...]


def generate_schedule_families(
    specification: Mapping[str, Any],
) -> ScheduleFamilies:
    """Expand the frozen integer-only LCG schedule definition."""

    if specification.get("algorithm") != "lcg32_offsets_v1":
        raise ValueError("unsupported graph schedule generator")
    state = int(specification["seed"])
    count = int(specification["family_count"])
    baseline_min = int(specification["baseline_min"])
    baseline_count = int(specification["baseline_count"])
    baseline_stride = int(specification["baseline_stride"])
    period_min = int(specification["period_min"])
    period_count = int(specification["period_count"])
    offset_radius = int(specification["offset_radius"])
    families = []
    for family_index in range(count):
        requests_per_window = baseline_min + (
            family_index * baseline_stride
        ) % baseline_count
        period = period_min + family_index % period_count
        offsets = []
        for _ in range(period):
            state = (
                1_664_525 * state + 1_013_904_223
            ) & 0xFFFF_FFFF
            offsets.append(
                ((state >> 16) % (2 * offset_radius + 1))
                - offset_radius
            )
        if requests_per_window + min(offsets) < 1:
            raise ValueError(
                "generated schedule has nonpositive request demand"
            )
        families.append(
            (requests_per_window, tuple(offsets))
        )
    payload = [
        {
            "requests_per_window": baseline,
            "load_pattern_offsets": list(offsets),
        }
        for baseline, offsets in families
    ]
    if _canonical_sha256(payload) != specification.get(
        "expanded_sha256"
    ):
        raise ValueError("generated graph schedules changed")
    return tuple(families)


def plan_collection(
    protocol: Mapping[str, Any],
) -> Tuple[ConfirmationCollectionCase, ...]:
    """Assign each topology-balanced family to three rotating lanes."""

    corpus = dict(protocol["corpus"])
    collection = dict(protocol["collection"])
    workers = tuple(
        int(value) for value in corpus["worker_replicas"]
    )
    jobs = int(collection["parallel_jobs"])
    if workers != tuple(range(1, jobs + 1)):
        raise ValueError(
            "graph collection requires one lane per topology"
        )
    family_count = int(corpus["training_family_count"]) + int(
        corpus["validation_family_count"]
    )
    family_order = tuple(
        int(value) for value in collection["family_order"]
    )
    if (
        len(family_order) != family_count
        or set(family_order) != set(range(1, family_count + 1))
    ):
        raise ValueError(
            "graph family order must be a complete permutation"
        )
    plans = []
    for batch, family in enumerate(family_order, start=1):
        for worker_replicas in workers:
            plans.append(
                ConfirmationCollectionCase(
                    case_id=(
                        f"{corpus['case_prefix']}-f{family:02d}"
                        f"-w{worker_replicas}-"
                        f"{corpus['seed_label']}"
                    ),
                    family=family,
                    worker_replicas=worker_replicas,
                    batch=batch,
                    lane=(
                        (
                            worker_replicas
                            - 1
                            + family
                            - 1
                        )
                        % jobs
                    )
                    + 1,
                    split=(
                        "training"
                        if family
                        <= int(corpus["training_family_count"])
                        else "validation"
                    ),
                )
            )
    return tuple(plans)


def prepare_from_protocol(
    protocol: Mapping[str, Any],
    output: Path,
) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("kind")
        != "observability_rich_graph_jepa_confirmation_v1"
    ):
        raise ValueError(
            "unsupported observability graph confirmation protocol"
        )
    corpus = dict(protocol["corpus"])
    families = generate_schedule_families(
        dict(corpus["schedule_generator"])
    )
    training_family_count = int(
        corpus["training_family_count"]
    )
    workers = tuple(
        int(value) for value in corpus["worker_replicas"]
    )
    manifests_directory = output / "manifests"
    split_path = output / "split.json"
    if manifests_directory.exists() or split_path.exists():
        raise FileExistsError(
            f"refusing to overwrite graph corpus inputs: {output}"
        )
    manifests_directory.mkdir(parents=True)
    training_case_ids: list[str] = []
    validation_case_ids: list[str] = []
    for family_index, (
        requests_per_window,
        offsets,
    ) in enumerate(families, start=1):
        for worker_replicas in workers:
            case_id = (
                f"{corpus['case_prefix']}-f{family_index:02d}"
                f"-w{worker_replicas}-{corpus['seed_label']}"
            )
            manifest = FaultMatrixCaseManifest(
                case_id=case_id,
                fault_kind="none",
                point_count=int(corpus["point_count"]),
                sample_period_seconds=float(
                    corpus["sample_period_seconds"]
                ),
                logical_window_period_nano=1_000_000_000,
                baseline_interval=(
                    0,
                    int(corpus["point_count"]),
                ),
                routine_noise_interval=(
                    int(corpus["point_count"]),
                    int(corpus["point_count"]),
                ),
                structural_interval=(
                    int(corpus["point_count"]),
                    int(corpus["point_count"]),
                ),
                affected_features=(),
                requests_per_window=requests_per_window,
                routine_noise_delay_ms=0,
                load_pattern_offsets=offsets,
                images=IMAGES,
                schema_version=2,
                topology_id=f"workers-{worker_replicas}",
                worker_replicas=worker_replicas,
            )
            _write_json(
                manifests_directory / f"{case_id}.json",
                manifest.to_dict(),
            )
            destination = (
                training_case_ids
                if family_index <= training_family_count
                else validation_case_ids
            )
            destination.append(case_id)
    split = TelemetryCorpusSplitSpec(
        training_case_ids=tuple(training_case_ids),
        validation_case_ids=tuple(validation_case_ids),
        reserved_case_ids=(),
        lookback=int(corpus["lookback"]),
        expected_application_api_request_queue_size=int(
            corpus[
                "expected_application_api_request_queue_size"
            ]
        ),
    )
    _write_json(split_path, split.to_dict())


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    prepare_from_protocol(
        json.loads(parsed.protocol.read_text()),
        parsed.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
