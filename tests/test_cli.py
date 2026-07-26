import subprocess
import sys


def test_evaluate_command_writes_evidence_and_reports_gate_status(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantis_core",
            "evaluate",
            "--quick",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Acceptance: PASS" in completed.stdout
    assert (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "report.md").exists()
