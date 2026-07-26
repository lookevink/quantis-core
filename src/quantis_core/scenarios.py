"""Deterministic infrastructure telemetry scenarios.

The scenario engine is deliberately independent from detector code. Its labels
are an external source of truth used to evaluate both detection and attribution.
"""

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


FEATURE_NAMES: Tuple[str, ...] = (
    "cpu_utilization",
    "memory_utilization",
    "request_latency_ms",
    "active_connections",
    "request_rate",
    "error_rate",
    "queue_depth",
    "cache_hit_ratio",
    "disk_io_ops",
    "network_receive_mbps",
    "gc_pause_ms",
    "thread_count",
)


class Phase(IntEnum):
    """Ground-truth operating phase for a telemetry point."""

    NORMAL = 0
    ROUTINE_NOISE = 1
    STRUCTURAL = 2


@dataclass(frozen=True)
class ScenarioSpec:
    """Serializable controls for one deterministic scenario."""

    seed: int
    length: int = 600
    noise_events: int = 14
    include_structural: bool = True
    structural_start_fraction: float = 0.68
    structural_duration_fraction: float = 0.18

    def __post_init__(self) -> None:
        if self.length < 120:
            raise ValueError("length must be at least 120 points")
        if self.noise_events < 0:
            raise ValueError("noise_events cannot be negative")
        if not 0.4 <= self.structural_start_fraction <= 0.85:
            raise ValueError("structural_start_fraction must be between 0.4 and 0.85")
        if not 0.08 <= self.structural_duration_fraction <= 0.3:
            raise ValueError("structural_duration_fraction must be between 0.08 and 0.3")


@dataclass(frozen=True)
class Scenario:
    """Telemetry plus independently generated evaluation truth."""

    feature_names: Tuple[str, ...]
    telemetry: NDArray[np.float64]
    phases: NDArray[np.int8]
    affected_features: NDArray[np.bool_]
    manifest: Dict[str, Any]


def generate_scenario(spec: ScenarioSpec) -> Scenario:
    """Generate reproducible multivariate telemetry and fault annotations."""

    rng = np.random.default_rng(spec.seed)
    telemetry = _normal_telemetry(spec.length, rng)
    phases = np.full(spec.length, Phase.NORMAL.value, dtype=np.int8)
    affected = np.zeros_like(telemetry, dtype=np.bool_)

    structural_interval = _structural_interval(spec)
    excluded = np.zeros(spec.length, dtype=np.bool_)
    if structural_interval is not None:
        start, stop = structural_interval
        excluded[max(0, start - 8) : min(spec.length, stop + 8)] = True

    noise_points = _sample_noise_points(
        rng=rng,
        length=spec.length,
        count=spec.noise_events,
        excluded=excluded,
    )
    _inject_routine_noise(telemetry, phases, affected, noise_points, rng)

    if structural_interval is not None:
        _inject_structural_drift(
            telemetry,
            phases,
            affected,
            start=structural_interval[0],
            stop=structural_interval[1],
        )

    manifest: Dict[str, Any] = asdict(spec)
    manifest.update(
        {
            "feature_names": list(FEATURE_NAMES),
            "noise_points": [int(point) for point in noise_points],
            "structural_interval": (
                list(structural_interval) if structural_interval is not None else None
            ),
            "generator_version": 2,
        }
    )
    return Scenario(
        feature_names=FEATURE_NAMES,
        telemetry=telemetry,
        phases=phases,
        affected_features=affected,
        manifest=manifest,
    )


def _normal_telemetry(
    length: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    time = np.arange(length, dtype=np.float64)
    load = np.empty(length, dtype=np.float64)
    load[0] = 0.45
    seasonal = 0.11 * np.sin(2.0 * np.pi * time / 75.0)
    demand_shift = 0.06 * np.sin(2.0 * np.pi * time / 190.0 + 0.7)
    innovations = rng.normal(0.0, 0.025, length)
    for index in range(1, length):
        target = 0.47 + seasonal[index] + demand_shift[index]
        load[index] = 0.91 * load[index - 1] + 0.09 * target + innovations[index]
    load = np.clip(load, 0.08, 0.92)

    cpu = 0.12 + 0.72 * load + rng.normal(0.0, 0.012, length)
    memory = 0.31 + 0.32 * load + rng.normal(0.0, 0.008, length)
    latency = 24.0 + 155.0 * np.square(load) + rng.normal(0.0, 2.5, length)
    connections = 45.0 + 330.0 * load + rng.normal(0.0, 4.0, length)
    request_rate = 80.0 + 620.0 * load + rng.normal(0.0, 9.0, length)
    error_rate = np.clip(
        0.002 + 0.018 * np.power(load, 3) + rng.normal(0.0, 0.0008, length),
        0.0,
        None,
    )
    queue_depth = np.clip(
        1.0 + 22.0 * np.square(load) + rng.normal(0.0, 0.6, length),
        0.0,
        None,
    )
    cache_hit_ratio = np.clip(
        0.985 - 0.11 * load + rng.normal(0.0, 0.004, length),
        0.0,
        1.0,
    )
    disk_io_ops = 40.0 + 240.0 * load + rng.normal(0.0, 7.0, length)
    network_receive = 8.0 + 92.0 * load + rng.normal(0.0, 2.5, length)
    gc_pause = np.clip(
        1.0 + 24.0 * np.square(load) + rng.normal(0.0, 0.8, length),
        0.0,
        None,
    )
    thread_count = 8.0 + 46.0 * load + rng.normal(0.0, 1.2, length)
    return np.column_stack(
        (
            cpu,
            memory,
            latency,
            connections,
            request_rate,
            error_rate,
            queue_depth,
            cache_hit_ratio,
            disk_io_ops,
            network_receive,
            gc_pause,
            thread_count,
        )
    ).astype(np.float64)


def _structural_interval(spec: ScenarioSpec) -> Optional[Tuple[int, int]]:
    if not spec.include_structural:
        return None
    start = int(spec.length * spec.structural_start_fraction)
    duration = max(30, int(spec.length * spec.structural_duration_fraction))
    return start, min(spec.length, start + duration)


def _sample_noise_points(
    rng: np.random.Generator,
    length: int,
    count: int,
    excluded: NDArray[np.bool_],
) -> NDArray[np.int64]:
    candidates = np.flatnonzero(~excluded)
    candidates = candidates[(candidates >= 12) & (candidates < length - 2)]
    if count > len(candidates):
        raise ValueError("noise_events exceeds the available non-structural points")
    return np.sort(rng.choice(candidates, size=count, replace=False))


def _inject_routine_noise(
    telemetry: NDArray[np.float64],
    phases: NDArray[np.int8],
    affected: NDArray[np.bool_],
    points: NDArray[np.int64],
    rng: np.random.Generator,
) -> None:
    amplitudes = np.asarray(
        (
            0.18,
            0.16,
            75.0,
            120.0,
            180.0,
            0.025,
            12.0,
            0.08,
            140.0,
            55.0,
            20.0,
            25.0,
        ),
        dtype=np.float64,
    )
    for point in points:
        feature = int(rng.integers(0, telemetry.shape[1]))
        sign = float(rng.choice((-1.0, 1.0)))
        telemetry[point, feature] += sign * amplitudes[feature]
        phases[point] = Phase.ROUTINE_NOISE.value
        affected[point, feature] = True


def _inject_structural_drift(
    telemetry: NDArray[np.float64],
    phases: NDArray[np.int8],
    affected: NDArray[np.bool_],
    start: int,
    stop: int,
) -> None:
    duration = stop - start
    progress = np.linspace(0.03, 1.0, duration, dtype=np.float64)
    smooth_drift = progress * progress * (3.0 - 2.0 * progress)
    deltas = np.zeros((duration, len(FEATURE_NAMES)), dtype=np.float64)
    deltas[:, 1] = 0.34 * smooth_drift
    deltas[:, 2] = 185.0 * smooth_drift
    deltas[:, 3] = -170.0 * smooth_drift
    telemetry[start:stop] += deltas
    phases[start:stop] = Phase.STRUCTURAL.value
    affected[start:stop, 1:4] = True
