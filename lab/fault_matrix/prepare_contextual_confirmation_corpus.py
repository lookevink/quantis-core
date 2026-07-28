"""Prepare untouched manifests for contextual JEPA confirmation v2."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from prepare_jepa_normal_corpus import (
    ScheduleFamilies,
    prepare_normal_corpus,
)


def prepare_from_protocol(
    protocol: Mapping[str, Any],
    output: Path,
) -> None:
    if (
        protocol.get("schema_version") != 2
        or protocol.get("kind")
        != "contextual_multimodal_jepa_confirmation_v2"
    ):
        raise ValueError("unsupported contextual confirmation protocol")
    corpus = dict(protocol["corpus"])
    families: ScheduleFamilies = tuple(
        (
            int(family["requests_per_window"]),
            tuple(
                int(value)
                for value in family["load_pattern_offsets"]
            ),
        )
        for family in corpus["schedule_families"]
    )
    prepare_normal_corpus(
        output,
        case_prefix=str(corpus["case_prefix"]),
        seed_label=int(corpus["seed_label"]),
        schedule_families=families,
        sample_period_seconds=float(
            corpus["sample_period_seconds"]
        ),
        expected_application_api_request_queue_size=int(
            corpus["expected_application_api_request_queue_size"]
        ),
        training_family_count=int(
            corpus["training_family_count"]
        ),
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "prepare the untouched contextual JEPA confirmation corpus"
        )
    )
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
