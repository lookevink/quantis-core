"""Summarize the fixed contextual JEPA preflight without exposed-data tuning."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


CANDIDATES = (
    "huber-log1",
    "huber-log2",
    "l1-log1",
    "mse-log1",
)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="summarize contextual JEPA preflight artifacts"
    )
    parser.add_argument("--root", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    summary = summarize(parsed.root)
    (parsed.root / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    (parsed.root / "summary.md").write_text(
        markdown_summary(summary)
    )
    return 0


def summarize(root: Path) -> Mapping[str, Any]:
    candidates: Dict[str, Mapping[str, Any]] = {}
    for name in CANDIDATES:
        development = json.loads(
            (root / name / "development.json").read_text()
        )
        cross_validation = development["cross_validation"][
            "summary"
        ]
        metrics = development["metrics"]
        candidates[name] = {
            "loss": development["config"]["loss"],
            "log_latent_dimension": development["config"][
                "log_latent_dimension"
            ],
            "selection_status": development["selection"]["status"],
            "training_family_alert_rate": cross_validation[
                "contextual_mean_alert_rate"
            ],
            "metrics_only_alert_rate": cross_validation[
                "metrics_only_mean_alert_rate"
            ],
            "no_worse_fold_fraction": cross_validation[
                "no_worse_fold_fraction"
            ],
            "exposed_validation_alert_rate": metrics[
                "contextual_multimodal"
            ]["validation"]["alert_rate"],
            "exposed_shuffled_log_alert_rate": metrics[
                "shuffled_logs"
            ]["validation"]["alert_rate"],
        }
    passing = [
        name
        for name, candidate in candidates.items()
        if candidate["selection_status"] == "passed"
    ]
    selected = (
        min(
            passing,
            key=lambda name: float(
                candidates[name]["training_family_alert_rate"]
            ),
        )
        if passing
        else None
    )
    return {
        "schema_version": 1,
        "kind": "contextual_multimodal_jepa_preflight",
        "selection_uses_exposed_validation": False,
        "selected_candidate": selected,
        "publication_eligible": False,
        "publication_blocker": "new untouched corpus required",
        "candidates": candidates,
    }


def markdown_summary(summary: Mapping[str, Any]) -> str:
    candidates = dict(summary["candidates"])
    lines = [
        "# Contextual multimodal JEPA preflight",
        "",
        "Selection uses training-family folds only. Exposed validation "
        "is diagnostic.",
        "",
        "| Candidate | Fold alert | Metrics only | No-worse folds | "
        "Status |",
        "|---|---:|---:|---:|---|",
    ]
    for name in CANDIDATES:
        candidate = candidates[name]
        lines.append(
            f"| {name} | "
            f"{float(candidate['training_family_alert_rate']):.3%} | "
            f"{float(candidate['metrics_only_alert_rate']):.3%} | "
            f"{float(candidate['no_worse_fold_fraction']):.1%} | "
            f"{candidate['selection_status']} |"
        )
    lines.extend(
        [
            "",
            f"Selected candidate: **{summary['selected_candidate']}**",
            "",
            "Publication eligible: **no**; a new untouched corpus is "
            "required.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
