#!/usr/bin/env python3
"""Check repository-local Markdown links without requiring network access."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import unquote


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*(?P<target><[^>]+>|[^)\s]+)"
)
SKIPPED_SCHEMES = (
    "http://",
    "https://",
    "mailto:",
    "app://",
)


def markdown_files(root: Path = REPOSITORY_ROOT) -> Iterable[Path]:
    direct = (
        root / "README.md",
        root / "CONTRIBUTING.md",
        root / "artifacts/README.md",
        root / "lab/README.md",
    )
    for path in direct:
        if path.is_file():
            yield path
    for directory in ("docs", "experiments"):
        for path in sorted((root / directory).rglob("*.md")):
            if not path.is_symlink():
                yield path


def check_markdown_links(
    files: Iterable[Path],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> List[str]:
    problems: List[str] = []
    for source in files:
        text = source.read_text()
        for match in MARKDOWN_LINK.finditer(text):
            raw_target = match.group("target").strip("<>")
            if (
                not raw_target
                or raw_target.startswith("#")
                or raw_target.startswith(SKIPPED_SCHEMES)
            ):
                continue
            path_part = unquote(raw_target.split("#", 1)[0])
            if not path_part:
                continue
            target = (
                Path(path_part)
                if Path(path_part).is_absolute()
                else source.parent / path_part
            ).resolve()
            try:
                relative = target.relative_to(repository_root.resolve())
            except ValueError:
                problems.append(
                    f"{source.relative_to(repository_root)}: "
                    f"link escapes repository: {raw_target}"
                )
                continue
            if str(relative).startswith("artifacts/") and not target.exists():
                # Large evidence is intentionally absent from a fresh clone.
                continue
            if not target.exists():
                problems.append(
                    f"{source.relative_to(repository_root)}: "
                    f"missing local link target: {raw_target}"
                )
    return problems


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Markdown files to check; defaults to repository documentation",
    )
    options = parser.parse_args(arguments)
    files = options.paths or list(markdown_files())
    problems = check_markdown_links(files)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
