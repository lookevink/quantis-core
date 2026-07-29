"""Fresh-process latency worker for the frozen JEPA-SCORE screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from quantis_core.edge_dynamics.jepa_score import ExactJepaScorer
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


def measure_latency(
    *, bundle_path: Path, inputs_path: Path
) -> Mapping[str, Any]:
    """Load one strict bundle and measure the frozen 20-call rotation."""

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    baseline_rss = _maximum_rss_bytes()
    payload = json.loads(bundle_path.read_text())
    scorer = ExactJepaScorer.from_dict(payload)
    graph = DeclaredTelemetryGraph.from_dict(
        dict(payload["strict_model_payload"]["graph"])
    )
    with np.load(inputs_path, allow_pickle=False) as evidence:
        warmup = np.asarray(
            evidence["warmup_history"], dtype=np.float64
        )
        measurements = np.asarray(
            evidence["measurement_histories"], dtype=np.float64
        )
        trajectory_ids = tuple(
            str(value)
            for value in evidence["measurement_trajectory_ids"]
        )
        transitions = np.asarray(
            evidence["measurement_transitions"], dtype=np.int64
        )
    if (
        warmup.shape != (1, 20, 7, 31)
        or measurements.shape != (20, 20, 7, 31)
        or len(trajectory_ids) != 20
        or transitions.shape != (20,)
        or np.any(transitions != 39)
        or trajectory_ids != tuple(sorted(trajectory_ids))
    ):
        raise ValueError("JEPA-SCORE latency rotation is invalid")
    scorer.score(warmup, graph)
    samples = []
    for history in measurements:
        started = time.perf_counter_ns()
        scorer.score(history[None], graph)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(samples, dtype=np.float64)
    peak_rss = _maximum_rss_bytes()
    return {
        "schema_version": 1,
        "kind": "jepa_score_latency_receipt_v1",
        "samples_ms": values.tolist(),
        "median_ms": float(np.median(values)),
        "p95_ms_higher": float(
            np.quantile(values, 0.95, method="higher")
        ),
        "warmup_count": 1,
        "measurement_count": 20,
        "measurement_trajectory_ids": list(trajectory_ids),
        "measurement_transitions": transitions.tolist(),
        "timer": "time.perf_counter_ns",
        "model_load_excluded": True,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "baseline_rss_bytes": baseline_rss,
        "absolute_peak_rss_bytes": peak_rss,
        "incremental_peak_rss_bytes": max(0, peak_rss - baseline_rss),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cpu_affinity": _cpu_affinity(),
        "power_state": _power_state(),
    }


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _cpu_affinity() -> Optional[list[int]]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return sorted(int(value) for value in getter(0))


def _power_state() -> Optional[str]:
    command = Path("/usr/bin/pmset")
    if not command.exists():
        return None
    result = subprocess.run(
        [str(command), "-g", "batt"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    options = parser.parse_args(arguments)
    result = measure_latency(
        bundle_path=options.bundle, inputs_path=options.inputs
    )
    print(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
