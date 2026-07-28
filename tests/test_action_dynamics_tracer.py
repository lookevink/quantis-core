import copy
import json
from pathlib import Path

from lab.action_dynamics.run_synthetic_tracer import (
    run_synthetic_tracer,
)
from quantis_core.action_dynamics_development import (
    assess_action_dynamics_evidence,
)


def test_synthetic_tracer_writes_supported_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    result = run_synthetic_tracer(tmp_path)

    assert result["assessment"]["status"] == "supported"
    assert result["assessment"]["decision"] == (
        "advance_to_instrumented_pilot"
    )
    expected = {
        "protocol.json",
        "compiler.json",
        "action-model.json",
        "action-agnostic-model.json",
        "evidence.json",
        "assessment.json",
        "artifact-manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assessment = json.loads(
        (tmp_path / "assessment.json").read_text()
    )
    compiler = json.loads(
        (tmp_path / "compiler.json").read_text()
    )
    assert assessment == result["assessment"]
    evidence = json.loads(
        (tmp_path / "evidence.json").read_text()
    )
    assert assess_action_dynamics_evidence(evidence) == assessment
    assert compiler["kind"] == "action_trajectory_compiler"
    assert compiler["training_pair_count"] == 15
    artifact_manifest = json.loads(
        (tmp_path / "artifact-manifest.json").read_text()
    )
    assert set(artifact_manifest["sha256"]) == expected - {
        "artifact-manifest.json"
    }

    assert "thresholds" not in evidence
    assert "propagation_delta" not in evidence
    assert "ranked_candidate_ids" not in (
        evidence["attribution_rows"][0]
    )
    assert "negative_log_likelihoods" not in (
        evidence["attribution_rows"][0]
    )

    wrong_propagation = copy.deepcopy(evidence)
    propagation_row = next(
        row
        for row in wrong_propagation["attribution_rows"]
        if row["case_id"]
        == wrong_propagation["propagation_case_id"]
    )
    no_action_position = propagation_row["candidate_ids"].index(
        "none"
    )
    no_action_mean = propagation_row[
        "candidate_distribution"
    ]["mean"][no_action_position]
    propagation_row["candidate_distribution"]["mean"] = [
        copy.deepcopy(no_action_mean)
        for _ in propagation_row["candidate_ids"]
    ]
    reassessed = assess_action_dynamics_evidence(
        wrong_propagation
    )
    assert reassessed["status"] == "not_supported"
    assert not reassessed["gates"][
        "respects_graph_propagation_delay"
    ]["passed"]
