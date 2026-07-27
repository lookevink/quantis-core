"""Prepare deterministic manifests for the JEPA normal-only corpus."""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from quantis_core.fault_matrix import FaultMatrixCaseManifest
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec


POINT_COUNT = 340
LOOKBACK = 6
TRAINING_FAMILY_COUNT = 8
ScheduleFamilies = Tuple[Tuple[int, Tuple[int, ...]], ...]
SCHEDULE_FAMILIES: Tuple[Tuple[int, Tuple[int, ...]], ...] = (
    (5, (0, 1, -1)),
    (6, (0, 2, -1, 1)),
    (7, (-2, 0, 1, 0, -1)),
    (8, (0, 1, 2, -1)),
    (9, (-3, 0, 2, -1, 1)),
    (10, (0, -2, 1, 3, -1)),
    (6, (1, -1, 2, 0, -2)),
    (8, (-1, 2, 0, -2, 1, 0)),
    (11, (-3, -1, 2, 0, 1)),
    (12, (-4, 0, 3, -2, 1, 0, 2)),
)
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


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="prepare the JEPA normal-only corpus inputs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    prepare_normal_corpus(
        parsed.output,
        case_prefix="jepa-normal",
        seed_label=13,
        schedule_families=SCHEDULE_FAMILIES,
        sample_period_seconds=0.05,
    )
    return 0


def prepare_normal_corpus(
    output: Path,
    *,
    case_prefix: str,
    seed_label: int,
    schedule_families: ScheduleFamilies,
    sample_period_seconds: float,
) -> None:
    """Write a fresh normal-only corpus from explicit schedule families."""

    if len(schedule_families) != 10:
        raise ValueError("normal corpus requires ten schedule families")
    manifests_directory = output / "manifests"
    split_path = output / "split.json"
    if manifests_directory.exists() or split_path.exists():
        raise FileExistsError(
            f"refusing to overwrite prepared corpus inputs: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifests_directory.mkdir()

    training_case_ids: List[str] = []
    validation_case_ids: List[str] = []
    for family_index, (
        requests_per_window,
        load_pattern_offsets,
    ) in enumerate(schedule_families, start=1):
        for worker_replicas in (1, 2, 3):
            case_id = (
                f"{case_prefix}-f{family_index:02d}"
                f"-w{worker_replicas}-{seed_label}"
            )
            manifest = FaultMatrixCaseManifest(
                case_id=case_id,
                fault_kind="none",
                point_count=POINT_COUNT,
                sample_period_seconds=sample_period_seconds,
                logical_window_period_nano=1_000_000_000,
                baseline_interval=(0, POINT_COUNT),
                routine_noise_interval=(POINT_COUNT, POINT_COUNT),
                structural_interval=(POINT_COUNT, POINT_COUNT),
                affected_features=(),
                requests_per_window=requests_per_window,
                routine_noise_delay_ms=0,
                load_pattern_offsets=load_pattern_offsets,
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
                if family_index <= TRAINING_FAMILY_COUNT
                else validation_case_ids
            )
            destination.append(case_id)

    split = TelemetryCorpusSplitSpec(
        training_case_ids=tuple(training_case_ids),
        validation_case_ids=tuple(validation_case_ids),
        reserved_case_ids=(),
        lookback=LOOKBACK,
    )
    _write_json(split_path, split.to_dict())
    print(
        f"Prepared {len(training_case_ids)} training runs and "
        f"{len(validation_case_ids)} validation runs"
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
