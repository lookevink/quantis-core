"""Prepare untouched manifests for contextual JEPA promotion."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from prepare_jepa_normal_corpus import (
    ScheduleFamilies,
    prepare_normal_corpus,
)


SCHEDULE_FAMILIES: ScheduleFamilies = (
    (6, (2, -3, 1, 0, 3, -2, -1)),
    (8, (-3, 2, 1, -2, 4, 0, -1)),
    (10, (3, -5, 2, -1, 0, 4, -2, 1)),
    (12, (-4, 3, -2, 5, 0, -1, 2, -3)),
    (7, (1, 4, -3, 0, -2, 2, -1, 3)),
    (9, (-2, 5, -4, 1, 3, -1, 0, 2)),
    (11, (4, -6, 1, 3, -2, 0, 2, -1)),
    (13, (-5, 4, -1, 2, -3, 6, 0, -2, 1)),
    (10, (5, -4, 0, 2, -5, 3, -1, 4, -2)),
    (14, (-7, 3, 5, -2, 1, -4, 6, 0, -1, 2)),
)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "prepare the untouched contextual JEPA promotion corpus"
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    prepare_normal_corpus(
        parsed.output,
        case_prefix="contextual-promotion-v1",
        seed_label=73,
        schedule_families=SCHEDULE_FAMILIES,
        sample_period_seconds=0.1,
        expected_application_api_request_queue_size=128,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
