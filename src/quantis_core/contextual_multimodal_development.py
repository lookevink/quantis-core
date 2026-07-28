"""Fixed v2 candidate sequence and training-only selection policy."""

import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .contextual_multimodal_corpus import (
    ContextualMultimodalTelemetryCorpus,
)
from .contextual_multimodal_training import (
    ContextualMultimodalJepaDevelopmentResult,
    ContextualMultimodalJepaTrainingConfig,
    train_contextual_multimodal_jepa_world_model,
    write_contextual_multimodal_jepa_artifacts,
)

V2_CANDIDATE_RECIPES = (
    ("v2_log_latent_1", 1, 0.0, 1.0, 1.0),
    ("v2_log_latent_2", 2, 0.0, 1.0, 1.0),
    ("v2_log_latent_3", 3, 0.0, 1.0, 1.0),
    (
        "v2_balanced_masked_log_latent_2",
        2,
        0.15,
        0.25,
        1.5,
    ),
    (
        "v2_balanced_masked_log_latent_3",
        3,
        0.15,
        0.25,
        1.5,
    ),
)
_V2_RECIPE_FIELDS = {
    "log_latent_dimension",
    "modality_mask_probability",
    "log_self_loss_multiplier",
    "cross_modal_loss_multiplier",
}


@dataclass(frozen=True)
class ContextualMultimodalJepaV2Candidate:
    """One named recipe fixed before family-held-out scoring."""

    name: str
    config: ContextualMultimodalJepaTrainingConfig


@dataclass(frozen=True)
class ContextualMultimodalJepaV2DevelopmentResult:
    """Fixed candidate results plus training-only selection."""

    candidate_results: Tuple[
        Tuple[str, ContextualMultimodalJepaDevelopmentResult], ...
    ]
    selection: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "contextual_multimodal_jepa_v2_development",
            "candidate_assessments": [
                _assessment_from_result(name, result)
                for name, result in self.candidate_results
            ],
            "selection": dict(self.selection),
            "protocol": {
                "candidate_sequence_fixed_before_scoring": True,
                "selection_uses_exposed_validation": False,
                "exposed_validation_use": "diagnostic_only",
                "publication_eligible": False,
                "next_evidence_step": (
                    "freeze the selected recipe and collect a new "
                    "untouched promotion corpus"
                ),
            },
        }


def default_contextual_multimodal_jepa_v2_candidates(
    base_config: ContextualMultimodalJepaTrainingConfig = (
        ContextualMultimodalJepaTrainingConfig()
    ),
) -> Tuple[ContextualMultimodalJepaV2Candidate, ...]:
    """Return the fixed capacity-then-balance v2 development sequence."""

    return tuple(
        ContextualMultimodalJepaV2Candidate(
            name=name,
            config=replace(
                base_config,
                log_latent_dimension=log_dimension,
                modality_mask_probability=mask_probability,
                log_self_loss_multiplier=log_self_multiplier,
                cross_modal_loss_multiplier=cross_modal_multiplier,
            ),
        )
        for (
            name,
            log_dimension,
            mask_probability,
            log_self_multiplier,
            cross_modal_multiplier,
        ) in V2_CANDIDATE_RECIPES
    )


def select_contextual_multimodal_jepa_v2_candidate(
    assessments: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select only from family-held-out controls and representation gates."""

    if not assessments:
        raise ValueError("v2 candidate assessments cannot be empty")
    leaderboard = [
        _candidate_row(assessment) for assessment in assessments
    ]
    leaderboard.sort(
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["minimum_control_margin"]),
            float(row["contextual_mean_alert_rate"]),
            str(row["candidate"]),
        )
    )
    eligible = [
        row for row in leaderboard if bool(row["eligible"])
    ]
    selected = (
        str(eligible[0]["candidate"]) if eligible else None
    )
    return {
        "schema_version": 1,
        "kind": "contextual_multimodal_jepa_v2_selection",
        "status": "selected" if selected is not None else "failed",
        "selected_candidate": selected,
        "selection_basis": (
            "training_schedule_family_held_out_controls_only"
        ),
        "uses_exposed_validation": False,
        "leaderboard": leaderboard,
        "publication_eligible": False,
        "publication_blocker": (
            "a new untouched promotion corpus is required after "
            "the v2 recipe is frozen"
        ),
    }


def develop_contextual_multimodal_jepa_v2(
    corpus: ContextualMultimodalTelemetryCorpus,
    candidates: Sequence[
        ContextualMultimodalJepaV2Candidate
    ] = (),
) -> ContextualMultimodalJepaV2DevelopmentResult:
    """Train the fixed sequence and select without exposed validation."""

    fixed_candidates = tuple(candidates) or (
        default_contextual_multimodal_jepa_v2_candidates()
    )
    _validate_v2_candidate_sequence(fixed_candidates)
    transformer = dict(
        dict(corpus.preprocessing["logs"])["transformer"]
    )
    if transformer.get("kind") != (
        "dependency_residual_application_logs_v2"
    ):
        raise ValueError(
            "v2 development requires dependency-residual lab logs"
        )
    results = tuple(
        (
            candidate.name,
            train_contextual_multimodal_jepa_world_model(
                corpus,
                candidate.config,
                evidence_mode="development",
            ),
        )
        for candidate in fixed_candidates
    )
    selection = select_contextual_multimodal_jepa_v2_candidate(
        tuple(
            _assessment_from_result(name, result)
            for name, result in results
        )
    )
    return ContextualMultimodalJepaV2DevelopmentResult(
        candidate_results=results,
        selection=selection,
    )


def _validate_v2_candidate_sequence(
    candidates: Sequence[ContextualMultimodalJepaV2Candidate],
) -> None:
    expected_names = tuple(
        recipe[0] for recipe in V2_CANDIDATE_RECIPES
    )
    actual_names = tuple(candidate.name for candidate in candidates)
    if actual_names != expected_names:
        raise ValueError(
            "v2 development requires the canonical fixed candidate "
            "sequence"
        )
    for candidate, recipe in zip(
        candidates,
        V2_CANDIDATE_RECIPES,
    ):
        actual_recipe = (
            candidate.config.log_latent_dimension,
            candidate.config.modality_mask_probability,
            candidate.config.log_self_loss_multiplier,
            candidate.config.cross_modal_loss_multiplier,
        )
        if actual_recipe != recipe[1:]:
            raise ValueError(
                "v2 development requires canonical candidate recipes"
            )
    common_fields = tuple(
        field.name
        for field in fields(
            ContextualMultimodalJepaTrainingConfig
        )
        if field.name not in _V2_RECIPE_FIELDS
    )
    reference = candidates[0].config
    if any(
        getattr(candidate.config, field_name)
        != getattr(reference, field_name)
        for candidate in candidates[1:]
        for field_name in common_fields
    ):
        raise ValueError(
            "v2 canonical candidates must share all base settings"
        )


def write_contextual_multimodal_jepa_v2_artifacts(
    result: ContextualMultimodalJepaV2DevelopmentResult,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Write deterministic candidate evidence and the selected bundle."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite v2 development artifacts: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    candidates_directory = output / "candidates"
    candidates_directory.mkdir(exist_ok=True)
    result_by_name = dict(result.candidate_results)
    for name, candidate_result in result.candidate_results:
        write_contextual_multimodal_jepa_artifacts(
            candidate_result,
            candidates_directory / name,
        )
    paths: Dict[str, Path] = {
        "development": output / "development.json",
        "selection": output / "candidate-selection.json",
        "report": output / "report.md",
    }
    _write_json(paths["development"], result.to_dict())
    _write_json(paths["selection"], dict(result.selection))
    selected_name = result.selection["selected_candidate"]
    if selected_name is not None:
        selected = result_by_name[str(selected_name)]
        paths["selected_model"] = output / "selected-model.json"
        paths["selected_config"] = output / "selected-config.json"
        _write_json(paths["selected_model"], selected.model_artifact)
        _write_json(paths["selected_config"], selected.config.to_dict())
    paths["report"].write_text(_v2_report(result))
    return paths


def _assessment_from_result(
    name: str,
    result: ContextualMultimodalJepaDevelopmentResult,
) -> Dict[str, Any]:
    return {
        "name": name,
        "config": result.config.to_dict(),
        "cross_validation": dict(result.cross_validation),
        "diagnostics": dict(
            dict(result.model_artifact)["diagnostics"]
        ),
    }


def _candidate_row(
    assessment: Mapping[str, Any],
) -> Dict[str, Any]:
    name = str(assessment["name"])
    config = dict(assessment["config"])
    cross_validation = dict(assessment["cross_validation"])
    diagnostics = dict(assessment["diagnostics"])
    if cross_validation.get("uses_exposed_validation") is not False:
        raise ValueError(
            f"{name} cross-validation must exclude exposed validation"
        )
    completed = cross_validation.get("status") == "completed"
    summary = dict(cross_validation.get("summary", {}))
    contextual = float(
        summary.get("contextual_mean_alert_rate", 1.0)
    )
    metrics_only = float(
        summary.get("metrics_only_mean_alert_rate", -1.0)
    )
    capacity = float(
        summary.get(
            "capacity_matched_mean_alert_rate",
            -1.0,
        )
    )
    shuffled = float(
        summary.get(
            "shuffled_logs_mean_alert_rate",
            -1.0,
        )
    )
    no_worse = float(
        summary.get("no_worse_fold_fraction", 0.0)
    )
    metric_dimension = int(config["metric_latent_dimension"])
    log_dimension = int(config["log_latent_dimension"])
    metric_rank = float(
        diagnostics.get("metric_effective_rank", 0.0)
    )
    log_rank = float(
        diagnostics.get("log_effective_rank", 0.0)
    )
    if metric_dimension < 1 or log_dimension < 1:
        raise ValueError(
            f"{name} has invalid active latent dimensions"
        )
    if (
        not math.isfinite(metric_rank)
        or not math.isfinite(log_rank)
        or not 0.0 <= metric_rank <= metric_dimension + 1e-6
        or not 0.0 <= log_rank <= log_dimension + 1e-6
    ):
        raise ValueError(f"{name} has invalid effective rank")
    if completed and (
        any(
            not math.isfinite(rate) or not 0.0 <= rate <= 1.0
            for rate in (
                contextual,
                metrics_only,
                capacity,
                shuffled,
            )
        )
        or not math.isfinite(no_worse)
        or not 0.0 <= no_worse <= 1.0
    ):
        raise ValueError(
            f"{name} has invalid family-held-out metrics"
        )
    gates = {
        "family_held_out_completed": completed,
        "better_than_metrics_only": contextual < metrics_only,
        "no_worse_than_capacity_matched": contextual <= capacity,
        "better_than_shuffled_logs": contextual < shuffled,
        "no_worse_on_at_least_half_of_folds": no_worse >= 0.5,
        "metric_active_latent_rank": (
            metric_rank >= 0.5 * metric_dimension
        ),
        "log_active_latent_rank": (
            log_rank >= 0.5 * log_dimension
        ),
    }
    margin = min(
        metrics_only - contextual,
        capacity - contextual,
        shuffled - contextual,
    )
    return {
        "candidate": name,
        "config": config,
        "contextual_mean_alert_rate": contextual,
        "metrics_only_mean_alert_rate": metrics_only,
        "capacity_matched_mean_alert_rate": capacity,
        "shuffled_logs_mean_alert_rate": shuffled,
        "minimum_control_margin": margin,
        "no_worse_fold_fraction": no_worse,
        "metric_effective_rank": metric_rank,
        "log_effective_rank": log_rank,
        "gates": gates,
        "eligible": all(gates.values()),
    }


def _v2_report(
    result: ContextualMultimodalJepaV2DevelopmentResult,
) -> str:
    selection = result.selection
    lines = [
        "# Contextual metrics + dependency logs JEPA v2",
        "",
        f"Status: **{selection['status']}**",
        "",
        (
            "Candidate selection used only schedule-family-held-out "
            "development folds. Previously exposed validation results "
            "remain diagnostic."
        ),
        "",
        "## Candidate leaderboard",
        "",
    ]
    for row in selection["leaderboard"]:
        lines.append(
            f"- {row['candidate']}: eligible={row['eligible']}, "
            "held-out alert rate="
            f"{float(row['contextual_mean_alert_rate']):.3%}, "
            "minimum control margin="
            f"{float(row['minimum_control_margin']):.3%}, "
            "log effective rank="
            f"{float(row['log_effective_rank']):.3f}"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            (
                "- This is development evidence and cannot support "
                "publication or promotion."
            ),
            (
                "- A selected recipe must be frozen before collecting "
                "an entirely new promotion corpus."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
