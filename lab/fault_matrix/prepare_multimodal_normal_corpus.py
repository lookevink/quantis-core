"""Prepare fresh manifests for the 30-run multimodal JEPA corpus."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from prepare_jepa_normal_corpus import (
    ScheduleFamilies,
    prepare_normal_corpus,
)


SCHEDULE_FAMILIES: ScheduleFamilies = (
    (5, (1, -1, 0, 2, -2)),
    (7, (2, -2, 1, -1, 0, 3)),
    (9, (-4, 1, -1, 3, 0, -2)),
    (11, (2, 0, -3, 1, -1, 3, -2)),
    (6, (-1, 3, 0, -2, 2, 1)),
    (8, (3, -1, -3, 2, 0, -2, 1)),
    (10, (-2, 4, -1, 0, 2, -3)),
    (12, (-5, 1, 4, -2, 0, 2, -1)),
    (9, (1, -3, 3, -1, 2, -2, 0)),
    (13, (-6, 2, -1, 5, -3, 1, 0, 3)),
)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="prepare the multimodal JEPA normal-only corpus inputs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    prepare_normal_corpus(
        parsed.output,
        case_prefix="multimodal-normal-v2",
        seed_label=48,
        schedule_families=SCHEDULE_FAMILIES,
        sample_period_seconds=0.1,
        expected_application_api_request_queue_size=128,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
