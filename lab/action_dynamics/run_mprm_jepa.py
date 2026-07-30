"""Fit and freeze the preregistered MPRM-JEPA candidate and controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy as np

from quantis_core.action_conditioned_dynamics import (
    MixtureTrajectoryDistribution,
)

from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.mprm_jepa import MprmJepaProtocol
from quantis_core.richer_regime_corpus import load_richer_regime_windows

from mprm_jepa_model import MeanPreservingResidualJepa
from prototype_multi_hypothesis_jepa import (
    MultiHypothesisJepaPrototype,
    PrototypeConfig,
    _capacity_matched_width,
)


MODEL_NAMES = (
    "raw_rank_32_predictive_core",
    "one_component_anchored_jepa_residual",
    "supervised_four_component_mean_preserving_residual_mixture",
    "capacity_matched_anchored_single_gaussian",
    "unanchored_four_component_jepa_diagnostic",
    "mprm_jepa_candidate",
)


def preflight_mprm_jepa(
    repository: Path, *, check_runtime: bool = True
) -> Mapping[str, Any]:
    """Verify that fitting may begin without opening selection evidence."""

    root = Path(repository).resolve()
    protocol_path = (
        root / "lab" / "action_dynamics" / "mprm-jepa-protocol-v1.json"
    )
    protocol = MprmJepaProtocol.from_dict(_read_object(protocol_path))
    fit = root / str(protocol.payload["fit_source"]["campaign"])
    audit = root / str(
        protocol.payload["fit_source"]["validity_audit"]
    )
    audit_manifest = audit / "artifact-manifest.json"
    gates = {
        "protocol_frozen": protocol.payload["status"]
        == "frozen_pre_fit_contract",
        "fit_campaign_present": fit.is_dir(),
        "fit_validity_audit_present": audit_manifest.is_file(),
        "fit_validity_audit_identity": (
            audit_manifest.is_file()
            and _file_sha256(audit_manifest)
            == protocol.payload["fit_source"][
                "validity_audit_artifact_manifest_sha256"
            ]
        ),
        "implementation_present": all(
            (root / relative).is_file()
            for relative in (
                "src/quantis_core/mprm_jepa.py",
                "lab/action_dynamics/mprm_jepa_model.py",
                "lab/action_dynamics/run_mprm_jepa.py",
                "lab/action_dynamics/assess_mprm_jepa.py",
                "tests/test_mprm_jepa.py",
            )
        ),
    }
    if audit_manifest.is_file():
        audit_index = _read_object(audit_manifest).get("sha256")
        gates["fit_validity_audit_members"] = (
            isinstance(audit_index, dict)
            and all(
                (audit / relative).is_file()
                and _file_sha256(audit / relative) == expected
                for relative, expected in audit_index.items()
            )
        )
        source_manifest_path = audit / "source-content-manifest.json"
        source_index = (
            _read_object(source_manifest_path).get("sha256")
            if source_manifest_path.is_file()
            else None
        )
        fit_prefix = "fit_campaign/"
        fit_hashes = (
            {
                relative[len(fit_prefix) :]: expected
                for relative, expected in source_index.items()
                if relative.startswith(fit_prefix)
            }
            if isinstance(source_index, dict)
            else {}
        )
        gates["fit_campaign_content_identity"] = bool(fit_hashes) and all(
            (fit / relative).is_file()
            and _file_sha256(fit / relative) == expected
            for relative, expected in fit_hashes.items()
        )
    runtime = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
    }
    if check_runtime:
        import torch

        runtime["torch"] = str(torch.__version__)
        runtime["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        docker = subprocess.run(
            [
                "docker",
                "version",
                "--format",
                "{{.Client.Version}} {{.Server.Version}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        runtime["docker"] = docker
        envelope = protocol.payload["edge_envelope"]
        gates["runtime_identity"] = (
            runtime["python"] == envelope["python"]
            and runtime["architecture"]
            == envelope["runtime_architecture"]
            and runtime["torch"] == envelope["torch"]
            and runtime["cpu"] == envelope["runtime_cpu"]
        )
        gates["docker_available"] = bool(docker)
        source_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        runtime["source_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        gates["source_committed_and_clean"] = not source_status.strip()
    return {
        "schema_version": 1,
        "kind": "mprm_jepa_preflight",
        "status": "go" if all(gates.values()) else "no_go",
        "decision": (
            "fit_and_freeze_models"
            if all(gates.values())
            else "repair_preflight_before_fitting"
        ),
        "protocol_sha256": _file_sha256(protocol_path),
        "gates": gates,
        "runtime": runtime,
    }


def fit_and_freeze_mprm_models(
    *, repository: Path, output: Path
) -> Mapping[str, Any]:
    """Fit only on retained qualified fit evidence and freeze all models."""

    root = Path(repository).resolve()
    preflight = preflight_mprm_jepa(root)
    if preflight["status"] != "go":
        raise ValueError("MPRM-JEPA preflight did not reach go")
    if output.exists() or output.with_name(output.name + ".staging").exists():
        raise FileExistsError("MPRM-JEPA refuses an existing output")
    staging = output.with_name(output.name + ".staging")
    protocol = MprmJepaProtocol.from_dict(
        _read_object(
            root
            / "lab"
            / "action_dynamics"
            / "mprm-jepa-protocol-v1.json"
        )
    )
    fit_campaign = root / str(protocol.payload["fit_source"]["campaign"])
    loaded = load_richer_regime_windows({"fit": fit_campaign})
    fit = loaded.windows["fit"]
    recipe = protocol.payload["recipe"]
    anchor = ContractiveLowRankDynamics(LowRankConfig(rank=32)).fit(fit)
    anchor_distribution = anchor.rollout(
        fit.histories,
        fit.future_controls,
        fit.future_actions,
        fit.graph,
    )
    common = {
        "state_latent_width": recipe["state_latent_width"],
        "context_width": recipe["context_width"],
        "predictor_width": recipe["predictor_width"],
        "epochs": recipe["epochs"],
        "batch_size": recipe["batch_size"],
        "learning_rate": recipe["learning_rate"],
        "weight_decay": recipe["weight_decay"],
        "ema_decay": recipe["ema_decay"],
        "latent_weight": recipe["latent_weight"],
        "target_reconstruction_weight": recipe[
            "target_reconstruction_weight"
        ],
        "context_reconstruction_weight": recipe[
            "context_reconstruction_weight"
        ],
        "variance_floor": recipe["component_variance_floor"],
        "seed": recipe["seed"],
    }
    candidate_config = PrototypeConfig(
        component_count=4, objective="jepa", **common
    )
    probe = MultiHypothesisJepaPrototype(candidate_config)
    import torch
    from prototype_multi_hypothesis_jepa import _build_network

    shape = (
        fit.histories.shape[2],
        fit.histories.shape[3],
        fit.future_states.shape[1],
        fit.future_controls.shape[2],
        fit.future_actions.shape[3],
    )
    probe._shape = shape
    probe._network = _build_network(torch, candidate_config, shape)
    matched_width = _capacity_matched_width(
        torch, fit, probe.parameter_count, int(recipe["epochs"])
    )
    del probe
    anchored_models: Dict[str, MeanPreservingResidualJepa] = {
        "one_component_anchored_jepa_residual": (
            MeanPreservingResidualJepa(
                PrototypeConfig(
                    component_count=1, objective="jepa", **common
                )
            )
        ),
        "supervised_four_component_mean_preserving_residual_mixture": (
            MeanPreservingResidualJepa(
                PrototypeConfig(
                    component_count=4,
                    objective="supervised",
                    **common,
                )
            )
        ),
        "capacity_matched_anchored_single_gaussian": (
            MeanPreservingResidualJepa(
                PrototypeConfig(
                    component_count=1,
                    objective="supervised",
                    **{**common, "predictor_width": matched_width},
                )
            )
        ),
        "mprm_jepa_candidate": MeanPreservingResidualJepa(
            candidate_config
        ),
    }
    unanchored = MultiHypothesisJepaPrototype(candidate_config)
    staging.mkdir(parents=True)
    models_directory = staging / "models"
    models_directory.mkdir()
    _write_json(
        staging / "compiler-artifact.json",
        loaded.compiler_artifact,
    )
    _write_json(
        models_directory / "raw_rank_32_predictive_core.json",
        anchor.to_dict(),
    )
    model_evidence: Dict[str, Mapping[str, Any]] = {
        "raw_rank_32_predictive_core": {
            "parameter_count": anchor.parameter_count,
            "serialized_bytes": (
                models_directory / "raw_rank_32_predictive_core.json"
            ).stat().st_size,
            "config": asdict(LowRankConfig(rank=32)),
        }
    }
    for name, model in anchored_models.items():
        model.fit_anchored(fit, anchor_distribution.mean)
        size = model.save(models_directory, name)
        model_evidence[name] = {
            "parameter_count": model.parameter_count,
            "serialized_bytes": size,
            "config": asdict(model.config),
        }
    unanchored.fit(fit)
    size = unanchored.save(
        models_directory, "unanchored_four_component_jepa_diagnostic"
    )
    model_evidence["unanchored_four_component_jepa_diagnostic"] = {
        "parameter_count": unanchored.parameter_count,
        "serialized_bytes": size,
        "config": asdict(unanchored.config),
    }
    fixture = slice(0, min(8, len(fit.histories)))
    fixture_anchor = anchor_distribution.mean[fixture]
    candidate_model = anchored_models["mprm_jepa_candidate"]
    residual_size = int(
        model_evidence["mprm_jepa_candidate"]["serialized_bytes"]
    )
    anchor_size = int(
        model_evidence["raw_rank_32_predictive_core"][
            "serialized_bytes"
        ]
    )
    compiler_size = (staging / "compiler-artifact.json").stat().st_size
    model_evidence["mprm_jepa_candidate"] = {
        **model_evidence["mprm_jepa_candidate"],
        "residual_serialized_bytes": residual_size,
        "serialized_bytes": residual_size + anchor_size + compiler_size,
        "batch_one_p95_latency_ms": _anchored_p95_latency(
            candidate_model,
            anchor,
            fit,
        ),
    }
    fixture_directory = staging / "restore-fixture"
    fixture_directory.mkdir()
    np.savez_compressed(
        fixture_directory / "inputs.npz",
        histories=fit.histories[fixture],
        controls=fit.future_controls[fixture],
        actions=fit.future_actions[fixture],
        anchor_mean=fixture_anchor,
    )
    expected: Dict[str, MixtureTrajectoryDistribution] = {}
    for name, model in anchored_models.items():
        expected[name] = model.rollout_anchored(
            fit.histories[fixture],
            fit.future_controls[fixture],
            fit.future_actions[fixture],
            fixture_anchor,
        )
    expected["raw_rank_32_predictive_core"] = MixtureTrajectoryDistribution(
        component_mean=fixture_anchor[:, None],
        component_variance=anchor_distribution.variance[fixture, None],
        weight=np.ones((len(fixture_anchor), 1), dtype=np.float64),
    )
    expected["unanchored_four_component_jepa_diagnostic"] = (
        unanchored.rollout(
            fit.histories[fixture],
            fit.future_controls[fixture],
            fit.future_actions[fixture],
        )
    )
    for name, distribution in expected.items():
        np.savez_compressed(
            fixture_directory / f"{name}.npz",
            component_mean=distribution.component_mean,
            component_variance=distribution.component_variance,
            weight=distribution.weight,
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(root / "src")
        + ":"
        + str(root / "lab" / "action_dynamics")
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "verify-restore",
            "--output",
            str(staging),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    parity = json.loads(verified.stdout)
    if not all(parity.values()):
        raise RuntimeError("MPRM-JEPA fresh-process parity failed")
    source_identity = {
        relative: _file_sha256(root / relative)
        for relative in (
            "lab/action_dynamics/mprm-jepa-protocol-v1.json",
            "lab/action_dynamics/mprm_jepa_model.py",
            "lab/action_dynamics/run_mprm_jepa.py",
            "lab/action_dynamics/run_mprm_selection.py",
            "lab/action_dynamics/assess_mprm_jepa.py",
            "lab/action_dynamics/prototype_multi_hypothesis_jepa.py",
            "lab/action_dynamics/collect_pilot.py",
            "lab/action_dynamics/run_lab_pilot.py",
            "lab/action_dynamics/run_richer_regime_retry.py",
            "src/quantis_core/action_conditioned_dynamics.py",
            "src/quantis_core/action_dynamics_corpus.py",
            "src/quantis_core/action_dynamics_lab.py",
            "src/quantis_core/mprm_jepa.py",
            "src/quantis_core/richer_regime_corpus.py",
            "src/quantis_core/edge_dynamics/models.py",
        )
    }
    freeze = {
        "schema_version": 1,
        "kind": "mprm_jepa_model_freeze_manifest",
        "protocol_sha256": preflight["protocol_sha256"],
        "fit_source_assessment_sha256s": (
            loaded.source_assessment_sha256s
        ),
        "compiler_artifact_sha256": _canonical_sha256(
            loaded.compiler_artifact
        ),
        "model_evidence": model_evidence,
        "prediction_parity": parity,
        "fresh_process_prediction_parity": all(parity.values()),
        "runtime_identity": preflight["runtime"],
        "source_sha256": source_identity,
        "artifact_sha256": {
            path.relative_to(staging).as_posix(): _file_sha256(path)
            for path in sorted(
                list(models_directory.iterdir())
                + [staging / "compiler-artifact.json"]
            )
            if path.is_file()
        },
    }
    _write_json(staging / "model-freeze-manifest.json", freeze)
    result = {
        "status": "models_frozen",
        "decision": "freeze_fresh_selection_collection_protocol",
        "model_freeze_manifest_sha256": _file_sha256(
            staging / "model-freeze-manifest.json"
        ),
    }
    _write_json(staging / "result.json", result)
    staging.rename(output)
    return result


def verify_frozen_restore(directory: Path) -> Mapping[str, bool]:
    """Restore every model in this fresh process and compare stored outputs."""

    models = directory / "models"
    fixture = directory / "restore-fixture"
    with np.load(fixture / "inputs.npz", allow_pickle=False) as arrays:
        histories = np.asarray(arrays["histories"])
        controls = np.asarray(arrays["controls"])
        actions = np.asarray(arrays["actions"])
        anchor_mean = np.asarray(arrays["anchor_mean"])
    raw_payload = _read_object(
        models / "raw_rank_32_predictive_core.json"
    )
    raw_model = ContractiveLowRankDynamics.from_dict(raw_payload)
    from quantis_core.graph_telemetry import DeclaredTelemetryGraph

    graph = DeclaredTelemetryGraph.from_dict(raw_payload["graph"])
    actual: Dict[str, MixtureTrajectoryDistribution] = {}
    raw = raw_model.rollout(
        histories, controls, actions, graph
    )
    actual["raw_rank_32_predictive_core"] = (
        MixtureTrajectoryDistribution(
            component_mean=raw.mean[:, None],
            component_variance=raw.variance[:, None],
            weight=np.ones((len(raw.mean), 1), dtype=np.float64),
        )
    )
    for name in (
        "one_component_anchored_jepa_residual",
        "supervised_four_component_mean_preserving_residual_mixture",
        "capacity_matched_anchored_single_gaussian",
        "mprm_jepa_candidate",
    ):
        model = cast(
            MeanPreservingResidualJepa,
            MeanPreservingResidualJepa.load(models, name),
        )
        actual[name] = model.rollout_anchored(
            histories, controls, actions, anchor_mean
        )
    actual["unanchored_four_component_jepa_diagnostic"] = (
        MultiHypothesisJepaPrototype.load(
            models, "unanchored_four_component_jepa_diagnostic"
        ).rollout(histories, controls, actions)
    )
    parity = {}
    for name, distribution in actual.items():
        with np.load(fixture / f"{name}.npz", allow_pickle=False) as arrays:
            parity[name] = bool(
                np.array_equal(
                    distribution.component_mean, arrays["component_mean"]
                )
                and np.array_equal(
                    distribution.component_variance,
                    arrays["component_variance"],
                )
                and np.array_equal(distribution.weight, arrays["weight"])
            )
    return parity


def _anchored_p95_latency(
    model: MeanPreservingResidualJepa,
    anchor: ContractiveLowRankDynamics,
    windows: Any,
) -> float:
    samples = []
    for index in range(101):
        started = time.perf_counter_ns()
        anchor_mean = anchor.rollout(
            windows.histories[:1],
            windows.future_controls[:1],
            windows.future_actions[:1],
            windows.graph,
        ).mean
        model.rollout_anchored(
            windows.histories[:1],
            windows.future_controls[:1],
            windows.future_actions[:1],
            anchor_mean,
        )
        if index:
            samples.append((time.perf_counter_ns() - started) / 1e6)
    return float(np.percentile(np.asarray(samples), 95))


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preflight", "fit-and-freeze", "verify-restore"),
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/mprm-jepa-model-freeze-v1"
        ),
    )
    parsed = parser.parse_args(arguments)
    if parsed.command == "preflight":
        result = preflight_mprm_jepa(parsed.repository)
    elif parsed.command == "verify-restore":
        result = verify_frozen_restore(parsed.output)
    else:
        result = fit_and_freeze_mprm_models(
            repository=parsed.repository, output=parsed.output
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if parsed.command == "verify-restore":
        return 0 if all(result.values()) else 1
    return 0 if result.get("status") in {"go", "models_frozen"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
