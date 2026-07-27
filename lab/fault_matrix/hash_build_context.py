"""Hash every file copied into the fault-matrix application image."""

import hashlib
from pathlib import Path


FILES = (
    "Dockerfile",
    "application_logging.py",
    "requirements.txt",
    "run_experiment.py",
    "service.py",
)


def main() -> None:
    lab = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((lab / name).read_bytes())
        digest.update(b"\0")
    print(digest.hexdigest())


if __name__ == "__main__":
    main()
