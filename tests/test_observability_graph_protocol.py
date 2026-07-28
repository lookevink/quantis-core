import importlib.util
import json
from pathlib import Path
from types import ModuleType


def test_observability_graph_protocol_prepares_fresh_balanced_cases(
    tmp_path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            repository
            / "lab"
            / "graph_jepa"
            / "observability-graph-jepa-confirmation-v1.json"
        ).read_text()
    )
    preparation = _preparation_module(repository)

    preparation.prepare_from_protocol(protocol, tmp_path)
    plans = preparation.plan_collection(protocol)

    manifests = tuple(
        json.loads(path.read_text())
        for path in sorted((tmp_path / "manifests").glob("*.json"))
    )
    split = json.loads((tmp_path / "split.json").read_text())
    assert len(manifests) == 72
    assert len(plans) == 72
    assert {
        len([plan for plan in plans if plan.batch == batch])
        for batch in range(1, 25)
    } == {3}
    assert {
        len([plan for plan in plans if plan.lane == lane])
        for lane in range(1, 4)
    } == {24}
    assert len(split["training_case_ids"]) == 36
    assert len(split["validation_case_ids"]) == 36
    assert split["lookback"] == 20
    assert {
        manifest["worker_replicas"] for manifest in manifests
    } == {1, 2, 3}
    assert {
        len(manifest["load_pattern_offsets"])
        for manifest in manifests
    } == set(range(13, 20))

    prior_schedules = _prior_schedules(repository)
    fresh_schedules = {
        (
            manifest["requests_per_window"],
            tuple(manifest["load_pattern_offsets"]),
        )
        for manifest in manifests
    }
    assert len(fresh_schedules) == 24
    assert not fresh_schedules & prior_schedules


def _preparation_module(repository: Path) -> ModuleType:
    path = (
        repository
        / "lab"
        / "graph_jepa"
        / "prepare_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "quantis_test_graph_preparation", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prior_schedules(repository: Path) -> set[tuple[int, tuple[int, ...]]]:
    schedules: set[tuple[int, tuple[int, ...]]] = set()
    for path in (
        repository / "lab" / "fault_matrix"
    ).glob("*protocol.json"):
        payload = json.loads(path.read_text())
        for family in payload.get("corpus", {}).get(
            "schedule_families", []
        ):
            schedules.add(
                (
                    int(family["requests_per_window"]),
                    tuple(
                        int(value)
                        for value in family[
                            "load_pattern_offsets"
                        ]
                    ),
                )
            )
    return schedules
