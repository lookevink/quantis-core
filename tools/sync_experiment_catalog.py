#!/usr/bin/env python3
"""Generate and verify technique-centered experiment capsules.

The catalog is the metadata source of truth. Capsules provide a stable,
human-facing interface without relocating historical source paths that are
bound into immutable experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPOSITORY_ROOT / "experiments/jepa/catalog.json"
CAPSULE_ROOT = CATALOG_PATH.parent
SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
VALID_STATUSES = {"active", "accepted", "rejected", "blocked", "superseded"}
RESERVED_DIRECTORIES = {"__pycache__", "releases"}


class CatalogError(ValueError):
    """Raised when catalog metadata or generated output is invalid."""


def load_catalog(path: Path = CATALOG_PATH) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CatalogError("experiment catalog must be a JSON object")
    return value


def validate_catalog(
    catalog: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> List[Dict[str, Any]]:
    if catalog.get("schema_version") != 1:
        raise CatalogError("unsupported experiment catalog schema")
    if catalog.get("program") != "jepa":
        raise CatalogError("catalog program must be jepa")
    evidence_boundary = catalog.get("evidence_boundary")
    if not isinstance(evidence_boundary, str) or not evidence_boundary.strip():
        raise CatalogError("catalog evidence_boundary must be non-empty")
    raw_experiments = catalog.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise CatalogError("catalog experiments must be a non-empty list")

    experiments: List[Dict[str, Any]] = []
    seen_slugs = set()
    for raw in raw_experiments:
        if not isinstance(raw, dict):
            raise CatalogError("each experiment must be a JSON object")
        experiment = dict(raw)
        slug = experiment.get("slug")
        if not isinstance(slug, str) or SLUG_PATTERN.fullmatch(slug) is None:
            raise CatalogError(f"invalid experiment slug: {slug!r}")
        if slug in seen_slugs:
            raise CatalogError(f"duplicate experiment slug: {slug}")
        seen_slugs.add(slug)

        for field in ("title", "summary", "runner", "spec", "ticket", "artifact"):
            value = experiment.get(field)
            if not isinstance(value, str) or not value.strip():
                raise CatalogError(f"{slug}: {field} must be non-empty")
        findings = experiment.get("findings")
        if not isinstance(findings, list) or not findings:
            raise CatalogError(f"{slug}: findings must be a non-empty list")
        status = experiment.get("status")
        if status not in VALID_STATUSES:
            raise CatalogError(f"{slug}: invalid status {status!r}")
        artifact = str(experiment["artifact"])
        if not artifact.startswith("artifacts/"):
            raise CatalogError(f"{slug}: artifact must live under artifacts/")
        _require_within(
            repository_root / artifact,
            repository_root / "artifacts",
            message=f"{slug}: artifact escapes artifacts/",
        )

        citations = experiment.get("citations")
        if not isinstance(citations, list) or not citations:
            raise CatalogError(f"{slug}: at least one citation is required")
        for citation in citations:
            if not isinstance(citation, dict):
                raise CatalogError(f"{slug}: citations must be objects")
            title = citation.get("title")
            if not isinstance(title, str) or not title.strip():
                raise CatalogError(f"{slug}: citation title is required")
            url = citation.get("url")
            path = citation.get("path")
            has_url = isinstance(url, str) and bool(url.strip())
            has_path = isinstance(path, str) and bool(path.strip())
            if has_url == has_path:
                raise CatalogError(
                    f"{slug}: citation needs exactly one of url or path"
                )
            if has_url and not str(url).startswith("https://"):
                raise CatalogError(f"{slug}: citation URL must use https")

        paths = _declared_paths(experiment)
        for label, relative_path in paths:
            resolved = repository_root / relative_path
            _require_within(
                resolved,
                repository_root,
                message=f"{slug}: {label} path escapes repository",
            )
            if not resolved.is_file():
                raise CatalogError(
                    f"{slug}: missing {label} path {relative_path}"
                )
        experiments.append(experiment)
    return experiments


def capsule_files(
    experiment: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = [
        ("run.py", str(experiment["runner"])),
        ("spec.md", str(experiment["spec"])),
        ("ticket.md", str(experiment["ticket"])),
    ]
    optional_singletons = (
        ("assess.py", "assessor"),
        ("implementation.py", "implementation"),
    )
    for alias, field in optional_singletons:
        value = experiment.get(field)
        if isinstance(value, str):
            files.append((alias, value))
    files.extend(_numbered_aliases("findings", ".md", experiment["findings"]))
    files.extend(
        _numbered_aliases(
            "references", ".md", experiment.get("references", [])
        )
    )
    files.extend(
        _numbered_aliases(
            "supporting-spec",
            ".md",
            experiment.get("supporting_specs", []),
        )
    )
    files.extend(
        _numbered_aliases("test", ".py", experiment.get("tests", []))
    )
    files.extend(
        _numbered_aliases(
            "supporting-script",
            ".py",
            experiment.get("supporting_scripts", []),
        )
    )
    return files


def render_capsule_readme(
    experiment: Mapping[str, Any],
    *,
    evidence_boundary: str,
) -> str:
    status = str(experiment["status"]).replace("_", " ").title()
    lines = [
        f"# {experiment['title']}",
        "",
        f"**Status:** {status}",
        "",
        str(experiment["summary"]),
        "",
        "## Experiment interface",
        "",
    ]
    aliases = dict(capsule_files(experiment))
    interface_rows = [
        ("Frozen specification", "spec.md"),
        ("Conclusion-bearing findings", "findings.md"),
        ("Exact runner", "run.py"),
    ]
    if "assess.py" in aliases:
        interface_rows.append(("Independent assessor", "assess.py"))
    if "implementation.py" in aliases:
        interface_rows.append(("Library implementation", "implementation.py"))
    if "test.py" in aliases:
        interface_rows.append(("Behavioral test", "test.py"))
    interface_rows.append(("Program ticket", "ticket.md"))
    for label, alias in interface_rows:
        lines.append(f"- [{label}]({alias})")

    for alias, _ in capsule_files(experiment):
        if alias.startswith("findings-"):
            lines.append(f"- [Additional retained findings]({alias})")
        elif alias.startswith("supporting-spec-"):
            lines.append(f"- [Supporting specification]({alias})")
        elif alias.startswith("supporting-script-"):
            lines.append(f"- [Supporting script]({alias})")
        elif alias.startswith("test-"):
            lines.append(f"- [Additional behavioral test]({alias})")

    lines.extend(["", "## Primary references", ""])
    reference_aliases = [
        alias
        for alias, _ in capsule_files(experiment)
        if alias.startswith("references")
    ]
    for alias in reference_aliases:
        lines.append(f"- [Pinned primary-source notes]({alias})")
    for citation in experiment["citations"]:
        target = citation.get("url") or _relative_target(
            CAPSULE_ROOT / str(experiment["slug"]),
            REPOSITORY_ROOT / str(citation["path"]),
        )
        lines.append(f"- [{citation['title']}]({target})")

    lines.extend(
        [
            "",
            "## Artifact",
            "",
            f"- Local artifact: `{experiment['artifact']}`",
            (
                "- Fetch after distribution metadata is recorded: "
                f"`python tools/artifacts.py fetch {experiment['slug']}`"
            ),
            "- Published artifact directories are immutable.",
            "- The artifact is intentionally not duplicated into this capsule;",
            "  its manifest and result document bind the evidence identity.",
            "",
            "## Evidence boundary",
            "",
            evidence_boundary,
            "",
            "Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)",
            "for exact environment assumptions and fresh-output rules.",
            "",
        ]
    )
    return "\n".join(lines)


def render_program_readme(
    experiments: Iterable[Mapping[str, Any]],
    *,
    evidence_boundary: str,
) -> str:
    lines = [
        "# JEPA experiment directory",
        "",
        "This directory is the technique-centered entry point for the retained",
        "JEPA program. Each capsule collocates navigable links to the exact",
        "runner, assessor, specification, primary-source notes, findings,",
        "tests, ticket, and artifact identity without changing historical",
        "paths already bound into published artifacts.",
        "",
        f"**Evidence boundary:** {evidence_boundary}",
        "",
        "| Technique | Status | Conclusion |",
        "|---|---|---|",
    ]
    for experiment in experiments:
        title = str(experiment["title"])
        slug = str(experiment["slug"])
        status = str(experiment["status"])
        summary = str(experiment["summary"])
        lines.append(f"| [{title}]({slug}/) | {status} | {summary} |")
    lines.extend(
        [
            "",
            "## Shared program material",
            "",
            "- [Execution conclusion](../../docs/research/jepa-frontier-execution-conclusion-2026.md)",
            "- [Frontier audit](../../docs/research/jepa-frontier-technique-audit-2026.md)",
            "- [Exhaustion refresh](../../docs/research/jepa-frontier-exhaustion-refresh-2026-07-29.md)",
            "- [Evaluation ladder](../../docs/specs/jepa-experiment-ladder-v1.md)",
            "- [Wayfinding map](../../docs/wayfinding/jepa-implementation-program/map.md)",
            "- [Reproduction guide](../../lab/action_dynamics/JEPA_REPRODUCTION.md)",
            "- [Release distribution contract](releases/)",
            "",
            "Run `python tools/sync_experiment_catalog.py --check` after changing",
            "catalog metadata or capsule links.",
            "",
        ]
    )
    return "\n".join(lines)


def synchronize(*, check: bool) -> List[str]:
    catalog = load_catalog()
    experiments = validate_catalog(catalog)
    evidence_boundary = str(catalog["evidence_boundary"])
    problems: List[str] = []

    program_readme = render_program_readme(
        experiments, evidence_boundary=evidence_boundary
    )
    _sync_text(CAPSULE_ROOT / "README.md", program_readme, check, problems)

    expected_directories = set()
    for experiment in experiments:
        capsule = CAPSULE_ROOT / str(experiment["slug"])
        expected_directories.add(capsule)
        if not check:
            capsule.mkdir(parents=True, exist_ok=True)
        elif not capsule.is_dir():
            problems.append(f"missing capsule directory: {capsule}")
            continue
        readme = render_capsule_readme(
            experiment, evidence_boundary=evidence_boundary
        )
        _sync_text(capsule / "README.md", readme, check, problems)
        expected_aliases = {"README.md"}
        for alias, relative_target in capsule_files(experiment):
            expected_aliases.add(alias)
            path = capsule / alias
            target = _relative_target(
                capsule, REPOSITORY_ROOT / relative_target
            )
            _sync_symlink(path, target, check, problems)
        if capsule.is_dir():
            for child in capsule.iterdir():
                if child.name not in expected_aliases:
                    problems.append(f"unmanaged capsule entry: {child}")

    if CAPSULE_ROOT.is_dir():
        for child in CAPSULE_ROOT.iterdir():
            if (
                child.is_dir()
                and child not in expected_directories
                and child.name not in RESERVED_DIRECTORIES
            ):
                problems.append(f"uncataloged capsule directory: {child}")
    return problems


def _declared_paths(experiment: Mapping[str, Any]) -> List[Tuple[str, str]]:
    paths: List[Tuple[str, str]] = []
    for field in (
        "runner",
        "assessor",
        "spec",
        "ticket",
        "implementation",
    ):
        value = experiment.get(field)
        if isinstance(value, str):
            paths.append((field, value))
    for field in (
        "findings",
        "references",
        "tests",
        "supporting_specs",
        "supporting_scripts",
    ):
        values = experiment.get(field, [])
        if not isinstance(values, list):
            raise CatalogError(f"{experiment['slug']}: {field} must be a list")
        for value in values:
            if not isinstance(value, str):
                raise CatalogError(
                    f"{experiment['slug']}: {field} entries must be strings"
                )
            paths.append((field, value))
    for citation in experiment["citations"]:
        value = citation.get("path")
        if isinstance(value, str):
            paths.append(("citation", value))
    return paths


def _numbered_aliases(
    prefix: str, suffix: str, values: Sequence[str]
) -> List[Tuple[str, str]]:
    aliases = []
    for index, value in enumerate(values, start=1):
        alias = f"{prefix}{suffix}" if index == 1 else f"{prefix}-{index}{suffix}"
        aliases.append((alias, value))
    return aliases


def _relative_target(parent: Path, target: Path) -> str:
    return os.path.relpath(target, start=parent)


def _require_within(path: Path, parent: Path, *, message: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise CatalogError(message) from error


def _sync_text(
    path: Path,
    expected: str,
    check: bool,
    problems: List[str],
) -> None:
    if check:
        if not path.is_file() or path.is_symlink():
            problems.append(f"missing generated file: {path}")
        elif path.read_text() != expected:
            problems.append(f"stale generated file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    path.write_text(expected)


def _sync_symlink(
    path: Path,
    expected_target: str,
    check: bool,
    problems: List[str],
) -> None:
    if check:
        if not path.is_symlink():
            problems.append(f"missing generated symlink: {path}")
        elif os.readlink(path) != expected_target:
            problems.append(f"stale generated symlink: {path}")
        elif not path.resolve().is_file():
            problems.append(f"broken generated symlink: {path}")
        return
    if path.is_symlink() and os.readlink(path) == expected_target:
        return
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise CatalogError(f"refusing to replace unmanaged path: {path}")
    path.symlink_to(expected_target)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated capsules without changing the repository",
    )
    options = parser.parse_args(arguments)
    try:
        problems = synchronize(check=options.check)
    except (CatalogError, OSError, json.JSONDecodeError) as error:
        print(f"experiment catalog error: {error}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
