"""Hash every source copied into the graph-JEPA application image."""

import hashlib
from pathlib import Path


FILES = (
    "fault_matrix/application_logging.py",
    "fault_matrix/requirements.txt",
    "fault_matrix/run_experiment.py",
    "fault_matrix/service.py",
    "graph_jepa/Dockerfile",
    "graph_jepa/operational_state.py",
    "graph_jepa/run_experiment.py",
    "graph_jepa/service.py",
)


def main() -> None:
    lab = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((lab / name).read_bytes())
        digest.update(b"\0")
    print(digest.hexdigest())


if __name__ == "__main__":
    main()
