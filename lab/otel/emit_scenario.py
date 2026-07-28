"""Emit a deterministic Quantis scenario through an OTLP/HTTP JSON receiver."""

import argparse
import json
import urllib.request
from typing import Any, Dict

from quantis_core.scenarios import ScenarioSpec, generate_scenario


def build_request(seed: int, length: int) -> Dict[str, Any]:
    scenario = generate_scenario(ScenarioSpec(seed=seed, length=length))
    metrics = []
    for feature_index, feature_name in enumerate(scenario.feature_names):
        metrics.append(
            {
                "name": feature_name,
                "unit": "1",
                "gauge": {
                    "dataPoints": [
                        {
                            "timeUnixNano": str((point_index + 1) * 1_000_000_000),
                            "asDouble": float(
                                scenario.telemetry[point_index, feature_index]
                            ),
                        }
                        for point_index in range(length)
                    ]
                },
            }
        )
    metrics.extend(
        [
            {
                "name": "quantis.experiment.request_count",
                "unit": "1",
                "gauge": {
                    "dataPoints": [
                        {
                            "timeUnixNano": "1000000000",
                            "asDouble": 12.0,
                        }
                    ]
                },
            },
            {
                "name": "quantis.experiment.error_count",
                "unit": "1",
                "gauge": {
                    "dataPoints": [
                        {
                            "timeUnixNano": "1000000000",
                            "asDouble": 3.0,
                        }
                    ]
                },
            },
        ]
    )
    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "quantis-lab"},
                        },
                        {
                            "key": "quantis.scenario.seed",
                            "value": {"intValue": str(seed)},
                        },
                    ]
                },
                "scopeMetrics": [
                    {
                        "scope": {
                            "name": "quantis.lab.scenario",
                            "version": "1.0.0",
                        },
                        "metrics": metrics,
                    }
                ],
            }
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default="http://localhost:14318/v1/metrics"
    )
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--length", type=int, default=240)
    arguments = parser.parse_args()

    body = json.dumps(
        build_request(arguments.seed, arguments.length),
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        arguments.endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"collector returned HTTP {response.status}")
    print(
        f"emitted seed={arguments.seed} length={arguments.length} "
        f"bytes={len(body)}"
    )


if __name__ == "__main__":
    main()
