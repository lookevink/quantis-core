import json
from pathlib import Path

import numpy as np
import pytest

from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_loader_rejects_instrumentation_pilot_before_model_use() -> None:
    pilot = (
        REPOSITORY
        / "artifacts"
        / "action-dynamics"
        / "instrumentation-pilot-v4"
    )
    if not pilot.exists():
        pytest.skip("local instrumentation-pilot evidence is absent")

    with pytest.raises(ValueError, match="development"):
        load_action_dynamics_development_corpus(pilot)


def test_loader_rejects_unhashed_files(tmp_path: Path) -> None:
    root = tmp_path / "development"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "protocol.json").write_text(
        json.dumps({"stage": "development"}) + "\n"
    )
    (root / "unexpected.txt").write_text("not attested\n")
    (root / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "action_dynamics_artifact_manifest",
                "sha256": {},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="exact file"):
        load_action_dynamics_development_corpus(root)


def test_loader_rejects_changed_hashed_files(tmp_path: Path) -> None:
    root = tmp_path / "development"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "protocol.json").write_text(
        json.dumps({"stage": "development"}) + "\n"
    )
    (root / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "action_dynamics_artifact_manifest",
                "sha256": {
                    "inputs/protocol.json": "0" * 64,
                },
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="exact file hash"):
        load_action_dynamics_development_corpus(root)


def test_loader_compiles_qualified_120_pair_development_corpus() -> None:
    development = (
        REPOSITORY
        / "artifacts"
        / "action-dynamics"
        / "development-v1"
    )
    if not development.exists():
        pytest.skip("local development-v1 evidence is absent")

    corpus = load_action_dynamics_development_corpus(development)

    assert corpus.summary.run_count == 240
    assert corpus.summary.pair_count == 120
    assert corpus.summary.training_pair_count == 90
    assert corpus.summary.validation_pair_count == 30
    assert len(corpus.training_runs) == 180
    assert len(corpus.validation_runs) == 60
    assert not (
        set(corpus.training_pair_ids)
        & set(corpus.validation_pair_ids)
    )
    first = corpus.runs[0]
    assert first.graph.entity_ids == (
        "api",
        "api_enqueues_queue",
        "checkout_queue",
        "queue_dequeues_to_worker",
        "worker_pool",
        "worker_writes_postgresql",
        "postgresql",
    )
    assert first.observations.shape == (108, 7, 31)
    assert first.controls.shape == (108, 2)
    assert first.control_feature_names == (
        "request_demand",
        "worker_replicas",
    )
    captured_manifest = json.loads(
        (
            development
            / "cases"
            / first.manifest.case_id
            / "capture-manifest.json"
        ).read_text()
    )
    assert np.array_equal(
        first.controls[:, 0],
        np.asarray(
            captured_manifest["request_schedule"],
            dtype=np.float64,
        ),
    )
    assert np.all(first.observations[:, -1, :] == 0.0)
    assert all(
        len(value) == 64
        for value in corpus.identity.to_dict().values()
    )
