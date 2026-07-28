"""Compile fresh graph confirmation captures into one verified tensor cache."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Sequence

from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
)
from quantis_core.observability_graph_corpus import (
    compile_observability_graph_corpus,
    write_observability_graph_cache,
)
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.otlp_logs import read_otlp_log_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captures-directory", type=Path, required=True
    )
    parser.add_argument(
        "--manifests-directory", type=Path, required=True
    )
    parser.add_argument(
        "--metric-feature-spec", type=Path, required=True
    )
    parser.add_argument(
        "--log-feature-spec", type=Path, required=True
    )
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    if re.fullmatch(r"[0-9a-f]{40}", parsed.source_git_commit) is None:
        raise ValueError("source git commit must be a full SHA-1")
    if parsed.output.exists():
        raise FileExistsError(
            f"refusing to overwrite graph cache: {parsed.output}"
        )

    manifests = tuple(
        FaultMatrixCaseManifest.from_dict(
            json.loads(path.read_text())
        )
        for path in sorted(
            parsed.manifests_directory.glob("*.json")
        )
    )
    runs = tuple(
        FaultMatrixRun(
            manifest=manifest,
            capture=read_otlp_capture(
                parsed.captures_directory
                / manifest.case_id
                / "collector-output.jsonl"
            ),
        )
        for manifest in manifests
    )
    log_captures = {
        manifest.case_id: read_otlp_log_capture(
            parsed.captures_directory
            / manifest.case_id
            / "collector-logs.jsonl"
        )
        for manifest in manifests
    }
    metric_spec = OtlpFeatureSpec.from_dict(
        json.loads(parsed.metric_feature_spec.read_text())
    )
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(parsed.log_feature_spec.read_text())
    )
    split_spec = TelemetryCorpusSplitSpec.from_dict(
        json.loads(parsed.split_spec.read_text())
    )
    protocol = json.loads(parsed.protocol.read_text())
    corpus = compile_observability_graph_corpus(
        runs,
        log_captures,
        metric_spec,
        log_spec,
        split_spec,
        horizons=tuple(
            int(value)
            for value in protocol["corpus"][
                "target_horizons"
            ]
        ),
        target_block_size=int(
            protocol["corpus"]["target_block_size"]
        ),
        protocol=protocol,
    )
    cache_directory = write_observability_graph_cache(
        corpus, parsed.output
    )
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(cache_directory.iterdir())
        if path.is_file()
    }
    index = {
        "schema_version": 1,
        "kind": "observability_graph_corpus_cache_index",
        "cache_key": cache_directory.name,
        "training_window_count": len(corpus.training.contexts),
        "validation_window_count": len(
            corpus.validation.contexts
        ),
        "entity_count": len(corpus.training.entity_ids),
        "slot_count": corpus.training.contexts.shape[3],
        "lookback": corpus.training.contexts.shape[1],
        "source_git_commit": parsed.source_git_commit,
        "files": files,
    }
    index_path = parsed.output / "cache-index.json"
    index_path.write_text(
        json.dumps(
            index,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"Graph cache: {cache_directory}")
    print(
        "Windows: "
        f"{index['training_window_count']} train / "
        f"{index['validation_window_count']} validation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
