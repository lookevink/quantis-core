import json
import subprocess
import sys
from pathlib import Path


def test_contextual_v2_prepares_fresh_development_only_runs(
    tmp_path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "lab/fault_matrix/"
            "prepare_contextual_v2_development_corpus.py",
            "--output",
            str(tmp_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    split = json.loads((tmp_path / "split.json").read_text())
    assert len(split["training_case_ids"]) == 24
    assert len(split["validation_case_ids"]) == 6
    assert split[
        "expected_application_api_request_queue_size"
    ] == 128
    assert all(
        case_id.startswith("contextual-v2-development-")
        for case_id in (
            split["training_case_ids"]
            + split["validation_case_ids"]
        )
    )
    assert not (
        set(split["training_case_ids"])
        & set(split["validation_case_ids"])
    )
    manifests = sorted((tmp_path / "manifests").glob("*.json"))
    assert len(manifests) == 30
    assert {
        json.loads(path.read_text())["point_count"]
        for path in manifests
    } == {340}
