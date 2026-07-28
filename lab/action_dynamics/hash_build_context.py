"""Hash every source copied into the action-dynamics runtime image."""

import hashlib
from pathlib import Path
from typing import Tuple


FILES: Tuple[str, ...] = (
    "action_dynamics/Dockerfile",
    "action_dynamics/application.py",
    "action_dynamics/application_telemetry.py",
    "action_dynamics/interventions.py",
    "action_dynamics/run_capture.py",
    "fault_matrix/requirements.txt",
)


def build_context_bytes() -> bytes:
    """Return an unambiguous byte representation of image inputs."""

    lab = Path(__file__).resolve().parent.parent
    content = bytearray()
    for name in FILES:
        content.extend(name.encode("utf-8"))
        content.extend(b"\0")
        content.extend((lab / name).read_bytes())
        content.extend(b"\0")
    return bytes(content)


def build_context_sha256() -> str:
    """Hash the exact files the Dockerfile copies or executes."""

    return hashlib.sha256(build_context_bytes()).hexdigest()


def main() -> None:
    print(build_context_sha256())


if __name__ == "__main__":
    main()
