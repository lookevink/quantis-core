#!/usr/bin/env python3
"""Package and restore content-addressed experiment evidence.

GitHub Releases are a distribution layer only. The checked-in catalog selects
authoritative artifact directories, and this tool independently records both
the unpacked tree identity and the uploaded archive identity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPOSITORY_ROOT / "experiments/jepa/catalog.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "dist/artifacts/evidence-jepa-frontier-v1"
DEFAULT_INDEX = (
    REPOSITORY_ROOT
    / "experiments/jepa/releases/evidence-jepa-frontier-v1.json"
)
DEFAULT_REPOSITORY = "lookevink/quantis-core"
DEFAULT_RELEASE_TAG = "evidence-jepa-frontier-v1"
INDEX_NAME = "artifact-index-v1.json"
CHECKSUM_NAME = "SHA256SUMS"
GITHUB_RELEASE_ASSET_LIMIT = 2 * 1024**3
BUFFER_SIZE = 1024 * 1024
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


class ArtifactDistributionError(RuntimeError):
    """Raised when evidence cannot be safely distributed or restored."""


@dataclass(frozen=True)
class TreeIdentity:
    sha256: str
    unpacked_size: int
    file_count: int
    entry_count: int


@dataclass(frozen=True)
class RemoteAssetPlan:
    missing: Sequence[Path]
    starter_asset_ids: Sequence[int]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ArtifactDistributionError("experiment catalog must be an object")
    return value


def selected_experiments(
    catalog: Mapping[str, Any], slugs: Sequence[str]
) -> List[Mapping[str, Any]]:
    raw = catalog.get("experiments")
    if not isinstance(raw, list):
        raise ArtifactDistributionError("catalog experiments must be a list")
    experiments: List[Mapping[str, Any]] = []
    by_slug: Dict[str, Mapping[str, Any]] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise ArtifactDistributionError("catalog experiment must be an object")
        slug = value.get("slug")
        artifact = value.get("artifact")
        if not isinstance(slug, str) or not isinstance(artifact, str):
            raise ArtifactDistributionError(
                "catalog experiment needs slug and artifact"
            )
        if slug in by_slug:
            raise ArtifactDistributionError(f"duplicate experiment slug: {slug}")
        by_slug[slug] = value
    requested = list(slugs) if slugs else list(by_slug)
    for slug in requested:
        try:
            experiments.append(by_slug[slug])
        except KeyError as error:
            raise ArtifactDistributionError(f"unknown experiment slug: {slug}") from error
    return experiments


def artifact_entries(root: Path) -> List[Path]:
    if not root.is_dir():
        raise ArtifactDistributionError(f"artifact directory is missing: {root}")
    entries = [root, *sorted(root.rglob("*"))]
    for path in entries:
        if path.is_symlink():
            raise ArtifactDistributionError(
                f"artifact contains unsupported symlink: {path}"
            )
        if not path.is_file() and not path.is_dir():
            raise ArtifactDistributionError(
                f"artifact contains unsupported file type: {path}"
            )
        if path != root:
            _safe_member_path(path.relative_to(root).as_posix())
    return entries


def normalized_mode(path: Path) -> int:
    if path.is_dir():
        return 0o755
    return 0o755 if path.stat().st_mode & 0o111 else 0o644


def tree_identity(root: Path) -> TreeIdentity:
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    entries = artifact_entries(root)
    for path in entries:
        relative_path = path.relative_to(root)
        relative = "." if relative_path == Path(".") else relative_path.as_posix()
        mode = normalized_mode(path)
        if path.is_dir():
            digest.update(f"dir\0{relative}\0{mode:o}\n".encode())
            continue
        size = path.stat().st_size
        content_hash = file_sha256(path)
        digest.update(
            f"file\0{relative}\0{mode:o}\0{size}\0{content_hash}\n".encode()
        )
        total_size += size
        file_count += 1
    return TreeIdentity(
        sha256=digest.hexdigest(),
        unpacked_size=total_size,
        file_count=file_count,
        entry_count=len(entries),
    )


def archive_name(slug: str, tree_sha256: str) -> str:
    return f"quantis-jepa-{slug}-{tree_sha256[:16]}.tar.gz"


def _validate_commit(value: str, *, field: str) -> None:
    if COMMIT_PATTERN.fullmatch(value) is None:
        raise ArtifactDistributionError(
            f"{field} must be a 40-64 character lowercase hexadecimal commit"
        )


def _resolved_commit(repository_root: Path, value: str, *, field: str) -> str:
    _validate_commit(value, field=field)
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ArtifactDistributionError(f"{field} does not resolve to a commit")
    return result.stdout.strip()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_canonical_selection(
    *,
    catalog: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    release_tag: str,
) -> None:
    if release_tag != DEFAULT_RELEASE_TAG:
        return
    raw = catalog.get("experiments")
    assert isinstance(raw, list)
    catalog_slugs = {
        str(experiment["slug"])
        for experiment in raw
        if isinstance(experiment, dict)
    }
    selected_slugs = {str(experiment["slug"]) for experiment in selected}
    if selected_slugs != catalog_slugs:
        missing = sorted(catalog_slugs - selected_slugs)
        raise ArtifactDistributionError(
            "canonical release requires every catalog experiment; "
            f"missing: {', '.join(missing)}"
        )


def write_deterministic_archive(source: Path, destination: Path, arcname: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ArtifactDistributionError(
            f"refusing to overwrite distribution asset: {destination}"
        )
    with destination.open("xb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w|",
                format=tarfile.PAX_FORMAT,
            ) as bundle:
                paths = [source, *sorted(source.rglob("*"))]
                for path in paths:
                    if path.is_symlink():
                        raise ArtifactDistributionError(
                            f"artifact contains unsupported symlink: {path}"
                        )
                    relative = path.relative_to(source)
                    member_name = (
                        arcname
                        if relative == Path(".")
                        else f"{arcname}/{relative.as_posix()}"
                    )
                    info = bundle.gettarinfo(str(path), arcname=member_name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = normalized_mode(path)
                    info.pax_headers = {}
                    if path.is_file():
                        with path.open("rb") as stream:
                            bundle.addfile(info, stream)
                    elif path.is_dir():
                        bundle.addfile(info)
                    else:
                        raise ArtifactDistributionError(
                            f"artifact contains unsupported file type: {path}"
                        )


def pack(
    *,
    repository_root: Path,
    catalog_path: Path,
    output: Path,
    slugs: Sequence[str],
    release_repository: str,
    release_tag: str,
    source_commit: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    experiments = selected_experiments(catalog, slugs)
    _validate_canonical_selection(
        catalog=catalog,
        selected=experiments,
        release_tag=release_tag,
    )
    source_commit = _resolved_commit(
        repository_root,
        source_commit,
        field="source commit",
    )
    assets: List[Dict[str, Any]] = []
    for experiment in experiments:
        slug = str(experiment["slug"])
        relative_artifact = Path(str(experiment["artifact"]))
        safe_artifact = _safe_member_path(relative_artifact.as_posix())
        if safe_artifact.parts[:1] != ("artifacts",):
            raise ArtifactDistributionError(
                f"{slug}: artifact path must begin with artifacts/"
            )
        source = (repository_root / relative_artifact).resolve()
        _require_within(source, repository_root / "artifacts")
        identity = tree_identity(source)
        if identity.unpacked_size >= GITHUB_RELEASE_ASSET_LIMIT:
            raise ArtifactDistributionError(
                f"{slug}: unpacked bundle is {identity.unpacked_size} bytes and cannot "
                "safely fit the 2 GiB GitHub Release asset limit"
            )
        name = archive_name(slug, identity.sha256)
        asset: Dict[str, Any] = {
            "slug": slug,
            "title": str(experiment.get("title", slug)),
            "artifact_path": safe_artifact.as_posix(),
            "name": name,
            "tree_sha256": identity.sha256,
            "unpacked_size": identity.unpacked_size,
            "file_count": identity.file_count,
            "entry_count": identity.entry_count,
            "media_type": "application/gzip",
        }
        if not dry_run:
            archive = output / name
            write_deterministic_archive(
                source,
                archive,
                safe_artifact.as_posix(),
            )
            size = archive.stat().st_size
            if size >= GITHUB_RELEASE_ASSET_LIMIT:
                archive.unlink()
                raise ArtifactDistributionError(
                    f"{slug}: packed bundle exceeds the 2 GiB GitHub "
                    "Release asset limit"
                )
            asset["archive_size"] = size
            asset["archive_sha256"] = file_sha256(archive)
        assets.append(asset)

    if dry_run:
        for asset in assets:
            print(
                f"{asset['slug']}: {asset['unpacked_size']} bytes, "
                f"{asset['file_count']} files -> {asset['name']}"
            )
        print(f"{len(assets)} artifact bundle(s) selected; no files written")
        return assets

    output.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "kind": "quantis_artifact_release_index",
        "release": {
            "provider": "github-release",
            "repository": release_repository,
            "tag": release_tag,
            "source_commit": source_commit,
        },
        "selection": {
            "program": str(catalog.get("program", "")),
            "slugs": [str(experiment["slug"]) for experiment in experiments],
            "catalog_sha256": _canonical_sha256(catalog),
        },
        "assets": assets,
    }
    (output / INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    (output / CHECKSUM_NAME).write_text(
        "".join(
            f"{asset['archive_sha256']}  {asset['name']}\n"
            for asset in assets
        )
    )
    return assets


def load_index(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ArtifactDistributionError("unsupported artifact release index")
    release = value.get("release")
    assets = value.get("assets")
    if not isinstance(release, dict) or not isinstance(assets, list):
        raise ArtifactDistributionError("artifact index is incomplete")
    return value


def validate_index_contract(
    index: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    require_current_catalog_complete: bool = False,
) -> None:
    release = index.get("release")
    assets = index.get("assets")
    assert isinstance(release, dict)
    assert isinstance(assets, list)
    tag = release.get("tag")
    source_commit = release.get("source_commit")
    if not isinstance(tag, str) or not tag:
        raise ArtifactDistributionError("release tag is required")
    if not isinstance(source_commit, str):
        raise ArtifactDistributionError("release source commit is required")
    _validate_commit(source_commit, field="release source commit")
    selection = index.get("selection")
    if not isinstance(selection, dict):
        raise ArtifactDistributionError("artifact selection is required")
    selected_slugs = selection.get("slugs")
    catalog_hash = selection.get("catalog_sha256")
    if (
        selection.get("program") != catalog.get("program")
        or not isinstance(selected_slugs, list)
        or not all(isinstance(slug, str) for slug in selected_slugs)
        or not isinstance(catalog_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", catalog_hash) is None
    ):
        raise ArtifactDistributionError("artifact selection is invalid")
    if len(selected_slugs) != len(set(selected_slugs)):
        raise ArtifactDistributionError("artifact selection contains duplicate slugs")
    indexed_slugs: List[str] = []
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("slug"), str):
            raise ArtifactDistributionError("indexed asset slug is required")
        indexed_slugs.append(str(asset["slug"]))
    if len(indexed_slugs) != len(set(indexed_slugs)):
        raise ArtifactDistributionError("artifact index contains duplicate slugs")
    if set(indexed_slugs) != set(selected_slugs):
        raise ArtifactDistributionError(
            "artifact assets differ from the pinned release selection"
        )
    selected_catalog = selected_experiments(catalog, indexed_slugs)
    expected_paths = {
        str(experiment["slug"]): str(experiment["artifact"])
        for experiment in selected_catalog
    }
    for asset in assets:
        assert isinstance(asset, dict)
        slug = str(asset["slug"])
        if asset.get("artifact_path") != expected_paths[slug]:
            raise ArtifactDistributionError(
                f"{slug}: artifact path differs from catalog"
            )
    if tag == DEFAULT_RELEASE_TAG and require_current_catalog_complete:
        current_slugs = {
            str(value["slug"]) for value in selected_experiments(catalog, [])
        }
        if set(selected_slugs) != current_slugs:
            raise ArtifactDistributionError(
                "canonical release requires every catalog experiment"
            )
        if catalog_hash != _canonical_sha256(catalog):
            raise ArtifactDistributionError(
                "canonical release catalog identity differs"
            )


def indexed_asset(index: Mapping[str, Any], slug: str) -> Mapping[str, Any]:
    assets = index.get("assets")
    assert isinstance(assets, list)
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("slug") == slug
    ]
    if len(matches) != 1:
        raise ArtifactDistributionError(
            f"artifact index needs exactly one asset for {slug}"
        )
    asset = matches[0]
    required = (
        "artifact_path",
        "name",
        "archive_sha256",
        "archive_size",
        "tree_sha256",
        "unpacked_size",
        "file_count",
        "entry_count",
    )
    if any(field not in asset for field in required):
        raise ArtifactDistributionError(f"{slug}: artifact index entry is incomplete")
    return asset


def _download_asset(
    *,
    index: Mapping[str, Any],
    asset: Mapping[str, Any],
    destination: Path,
) -> None:
    release = index["release"]
    assert isinstance(release, dict)
    repository = release.get("repository")
    tag = release.get("tag")
    if not isinstance(repository, str) or not isinstance(tag, str):
        raise ArtifactDistributionError(
            "artifact index release repository and tag are required"
        )
    name = str(asset["name"])
    url = (
        f"https://github.com/{repository}/releases/download/"
        f"{quote(tag, safe='')}/{quote(name, safe='')}"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "quantis-artifact-fetch/1"},
    )
    with urllib.request.urlopen(request) as response:
        with destination.open("xb") as stream:
            shutil.copyfileobj(response, stream, length=BUFFER_SIZE)


def _verified_archive(
    *,
    index: Mapping[str, Any],
    asset: Mapping[str, Any],
    asset_directory: Optional[Path],
    temporary_directory: Path,
) -> Path:
    name = str(asset["name"])
    if PurePosixPath(name).name != name:
        raise ArtifactDistributionError(f"unsafe asset name: {name}")
    if asset_directory is not None:
        archive = asset_directory / name
        if not archive.is_file():
            raise ArtifactDistributionError(f"release asset is missing: {archive}")
    else:
        archive = temporary_directory / name
        _download_asset(index=index, asset=asset, destination=archive)
    expected_size = int(asset["archive_size"])
    actual_size = archive.stat().st_size
    if actual_size != expected_size:
        raise ArtifactDistributionError(
            f"{name}: archive size differs ({actual_size} != {expected_size})"
        )
    expected_hash = str(asset["archive_sha256"])
    actual_hash = file_sha256(archive)
    if actual_hash != expected_hash:
        raise ArtifactDistributionError(f"{name}: archive SHA-256 differs")
    return archive


def _safe_member_path(member_name: str) -> PurePosixPath:
    path = PurePosixPath(member_name)
    if (
        "\\" in member_name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ArtifactDistributionError(
            f"unsafe archive member path: {member_name}"
        )
    return path


def _extract_archive(
    *,
    archive: Path,
    extraction_root: Path,
    artifact_path: PurePosixPath,
    expected_unpacked_size: int,
    expected_file_count: int,
    expected_entry_count: int,
) -> None:
    if (
        expected_unpacked_size < 0
        or expected_unpacked_size >= GITHUB_RELEASE_ASSET_LIMIT
        or expected_file_count < 0
        or expected_entry_count < 1
    ):
        raise ArtifactDistributionError("unsafe indexed extraction bounds")
    seen: Set[PurePosixPath] = set()
    unpacked_size = 0
    file_count = 0
    entry_count = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            entry_count += 1
            if entry_count > expected_entry_count:
                raise ArtifactDistributionError(
                    "archive exceeds indexed entry count"
                )
            path = _safe_member_path(member.name)
            if path != artifact_path and artifact_path not in path.parents:
                raise ArtifactDistributionError(
                    f"unsafe archive member outside {artifact_path}: {member.name}"
                )
            if path in seen:
                raise ArtifactDistributionError(
                    f"duplicate archive member: {member.name}"
                )
            seen.add(path)
            destination = extraction_root.joinpath(*path.parts)
            _require_within(destination, extraction_root)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=False)
                destination.chmod(0o755)
                continue
            if not member.isfile():
                raise ArtifactDistributionError(
                    f"unsafe archive member type: {member.name}"
                )
            file_count += 1
            unpacked_size += member.size
            if file_count > expected_file_count:
                raise ArtifactDistributionError(
                    "archive exceeds indexed file count"
                )
            if unpacked_size > expected_unpacked_size:
                raise ArtifactDistributionError(
                    "archive exceeds indexed unpacked size"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ArtifactDistributionError(
                    f"archive member has no content: {member.name}"
                )
            with destination.open("xb") as stream:
                shutil.copyfileobj(source, stream, length=BUFFER_SIZE)
            if destination.stat().st_size != member.size:
                raise ArtifactDistributionError(
                    f"archive member size differs: {member.name}"
                )
            destination.chmod(0o755 if member.mode & 0o111 else 0o644)
    if entry_count != expected_entry_count:
        raise ArtifactDistributionError("archive entry count differs")
    if file_count != expected_file_count:
        raise ArtifactDistributionError("archive file count differs")
    if unpacked_size != expected_unpacked_size:
        raise ArtifactDistributionError("archive unpacked size differs")


def fetch(
    *,
    repository_root: Path,
    catalog_path: Path,
    index_path: Path,
    slug: str,
    asset_directory: Optional[Path],
) -> None:
    catalog = load_catalog(catalog_path)
    experiment = selected_experiments(catalog, [slug])[0]
    index = load_index(index_path)
    validate_index_contract(index, catalog)
    asset = indexed_asset(index, slug)
    catalog_path_value = PurePosixPath(str(experiment["artifact"]))
    indexed_path = PurePosixPath(str(asset["artifact_path"]))
    if indexed_path != catalog_path_value:
        raise ArtifactDistributionError(
            f"{slug}: catalog and release artifact paths differ"
        )
    if indexed_path.is_absolute() or indexed_path.parts[:1] != ("artifacts",):
        raise ArtifactDistributionError(f"{slug}: unsafe artifact path")
    destination = repository_root.joinpath(*indexed_path.parts)
    _require_within(destination, repository_root / "artifacts")
    if destination.exists():
        raise ArtifactDistributionError(
            f"artifact destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".quantis-artifact-fetch-",
        dir=destination.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        archive = _verified_archive(
            index=index,
            asset=asset,
            asset_directory=asset_directory,
            temporary_directory=temporary_root,
        )
        extraction_root = temporary_root / "extracted"
        extraction_root.mkdir()
        _extract_archive(
            archive=archive,
            extraction_root=extraction_root,
            artifact_path=indexed_path,
            expected_unpacked_size=int(asset["unpacked_size"]),
            expected_file_count=int(asset["file_count"]),
            expected_entry_count=int(asset["entry_count"]),
        )
        extracted = extraction_root.joinpath(*indexed_path.parts)
        if not extracted.is_dir():
            raise ArtifactDistributionError(
                f"archive did not contain {indexed_path}"
            )
        identity = tree_identity(extracted)
        if identity.sha256 != str(asset["tree_sha256"]):
            raise ArtifactDistributionError(f"{slug}: unpacked tree SHA-256 differs")
        if identity.unpacked_size != int(asset["unpacked_size"]):
            raise ArtifactDistributionError(f"{slug}: unpacked size differs")
        if identity.file_count != int(asset["file_count"]):
            raise ArtifactDistributionError(f"{slug}: unpacked file count differs")
        if identity.entry_count != int(asset["entry_count"]):
            raise ArtifactDistributionError(f"{slug}: unpacked entry count differs")
        extracted.rename(destination)


def verify(
    *,
    repository_root: Path,
    catalog_path: Path,
    index_path: Path,
    slug: str,
    asset_directory: Optional[Path],
) -> None:
    catalog = load_catalog(catalog_path)
    experiment = selected_experiments(catalog, [slug])[0]
    index = load_index(index_path)
    validate_index_contract(index, catalog)
    asset = indexed_asset(index, slug)
    relative_artifact = Path(str(experiment["artifact"]))
    if relative_artifact.as_posix() != str(asset["artifact_path"]):
        raise ArtifactDistributionError(
            f"{slug}: catalog and release artifact paths differ"
        )
    artifact = (repository_root / relative_artifact).resolve()
    _require_within(artifact, repository_root / "artifacts")
    identity = tree_identity(artifact)
    if identity.sha256 != str(asset["tree_sha256"]):
        raise ArtifactDistributionError(f"{slug}: tree SHA-256 differs")
    if identity.unpacked_size != int(asset["unpacked_size"]):
        raise ArtifactDistributionError(f"{slug}: unpacked size differs")
    if identity.file_count != int(asset["file_count"]):
        raise ArtifactDistributionError(f"{slug}: file count differs")
    if identity.entry_count != int(asset["entry_count"]):
        raise ArtifactDistributionError(f"{slug}: entry count differs")
    if asset_directory is not None:
        with tempfile.TemporaryDirectory() as temporary:
            _verified_archive(
                index=index,
                asset=asset,
                asset_directory=asset_directory,
                temporary_directory=Path(temporary),
            )
        print(f"{slug}: archive and tree verified")
    else:
        print(f"{slug}: tree verified")


def _release_assets(
    *,
    index_path: Path,
    index: Mapping[str, Any],
    asset_directory: Path,
) -> List[Path]:
    raw_assets = index["assets"]
    assert isinstance(raw_assets, list)
    assets: List[Path] = []
    checksum_lines: List[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        for raw in raw_assets:
            if not isinstance(raw, dict):
                raise ArtifactDistributionError(
                    "artifact index assets must be objects"
                )
            asset = indexed_asset(index, str(raw.get("slug")))
            archive = _verified_archive(
                index=index,
                asset=asset,
                asset_directory=asset_directory,
                temporary_directory=Path(temporary),
            )
            assets.append(archive)
            checksum_lines.append(
                f"{asset['archive_sha256']}  {asset['name']}\n"
            )
    checksum_path = asset_directory / CHECKSUM_NAME
    if not checksum_path.is_file():
        raise ArtifactDistributionError(
            f"release checksum file is missing: {checksum_path}"
        )
    if checksum_path.read_text() != "".join(checksum_lines):
        raise ArtifactDistributionError("release checksum file differs from index")
    if not index_path.is_file():
        raise ArtifactDistributionError(f"release index is missing: {index_path}")
    return [index_path, checksum_path, *assets]


def _git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ArtifactDistributionError(result.stderr.strip())
    return result.stdout


def _publication_preflight(
    *,
    repository_root: Path,
    index_path: Path,
    index: Mapping[str, Any],
    target: str,
) -> None:
    _validate_commit(target, field="publication target")
    if _git_output(repository_root, "status", "--porcelain").strip():
        raise ArtifactDistributionError(
            "refusing to publish from a dirty working tree"
        )
    head = _git_output(repository_root, "rev-parse", "HEAD").strip()
    if head != target:
        raise ArtifactDistributionError(
            "--target must equal the checked-out HEAD commit"
        )
    release = index["release"]
    assert isinstance(release, dict)
    tag = str(release["tag"])
    if PurePosixPath(tag).name != tag:
        raise ArtifactDistributionError(
            f"release tag cannot name a recorded index safely: {tag}"
        )
    recorded_relative = PurePosixPath(
        f"experiments/jepa/releases/{tag}.json"
    )
    recorded_bytes = _git_output(
        repository_root,
        "show",
        f"HEAD:{recorded_relative}",
    ).encode()
    index_bytes = index_path.read_bytes()
    if index_bytes != recorded_bytes:
        raise ArtifactDistributionError(
            "distribution index differs from the index recorded in HEAD"
        )
    index_commit = _git_output(
        repository_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        str(recorded_relative),
    ).strip()
    if index_commit != target:
        raise ArtifactDistributionError(
            "publication target must be the commit that recorded the index"
        )


def _remote_release(
    *, repository: str, tag: str
) -> Optional[Mapping[str, Any]]:
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/releases?per_page=100",
            "--paginate",
            "--slurp",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ArtifactDistributionError(result.stderr.strip())
    pages = json.loads(result.stdout)
    if not isinstance(pages, list):
        raise ArtifactDistributionError(
            "GitHub returned an invalid release listing"
        )
    for page in pages:
        if not isinstance(page, list):
            raise ArtifactDistributionError(
                "GitHub returned an invalid release listing page"
            )
        for value in page:
            if not isinstance(value, dict):
                raise ArtifactDistributionError(
                    "GitHub returned an invalid release"
                )
            if value.get("tag_name") == tag:
                return value
    return None


def _remote_tag_commit(*, repository: str, tag: str) -> Optional[str]:
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/git/ref/tags/{quote(tag, safe='')}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if "404" in result.stderr or "Not Found" in result.stderr:
            return None
        raise ArtifactDistributionError(result.stderr.strip())
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or not isinstance(value.get("object"), dict):
        raise ArtifactDistributionError("GitHub returned an invalid tag ref")
    target = value["object"]
    for _ in range(8):
        target_type = target.get("type")
        sha = target.get("sha")
        if not isinstance(sha, str):
            raise ArtifactDistributionError("GitHub tag ref has no SHA")
        if target_type == "commit":
            return sha
        if target_type != "tag":
            raise ArtifactDistributionError(
                f"GitHub tag points to unsupported object: {target_type}"
            )
        peeled = subprocess.run(
            ["gh", "api", f"repos/{repository}/git/tags/{sha}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if peeled.returncode != 0:
            raise ArtifactDistributionError(peeled.stderr.strip())
        payload = json.loads(peeled.stdout)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("object"), dict
        ):
            raise ArtifactDistributionError(
                "GitHub returned an invalid annotated tag"
            )
        target = payload["object"]
    raise ArtifactDistributionError("GitHub tag annotation chain is too deep")


def _remote_asset_plan(
    *,
    remote: Mapping[str, Any],
    expected_paths: Sequence[Path],
    target: str,
) -> RemoteAssetPlan:
    if remote.get("draft") is not True:
        raise ArtifactDistributionError(
            "refusing to resume publication into a non-draft release"
        )
    remote_target = remote.get("target_commitish")
    if isinstance(remote_target, str) and remote_target != target:
        raise ArtifactDistributionError(
            "existing draft release targets a different commit"
        )
    expected = {path.name: path for path in expected_paths}
    raw_assets = remote.get("assets")
    if not isinstance(raw_assets, list):
        raise ArtifactDistributionError("GitHub release assets are invalid")
    uploaded: Set[str] = set()
    seen: Set[str] = set()
    starter_asset_ids: List[int] = []
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict):
            raise ArtifactDistributionError("GitHub release asset is invalid")
        name = raw_asset.get("name")
        if not isinstance(name, str) or name not in expected:
            raise ArtifactDistributionError(
                f"existing draft has unexpected asset: {name}"
            )
        if name in seen:
            raise ArtifactDistributionError(
                f"existing draft has duplicate asset: {name}"
            )
        seen.add(name)
        path = expected[name]
        if raw_asset.get("state") == "starter":
            asset_id = raw_asset.get("id")
            if not isinstance(asset_id, int):
                raise ArtifactDistributionError(
                    f"starter release asset has no id: {name}"
                )
            starter_asset_ids.append(asset_id)
            continue
        if int(raw_asset.get("size", -1)) != path.stat().st_size:
            raise ArtifactDistributionError(
                f"existing release asset size differs: {name}"
            )
        digest = raw_asset.get("digest")
        if digest != f"sha256:{file_sha256(path)}":
            raise ArtifactDistributionError(
                f"existing release asset digest differs: {name}"
            )
        uploaded.add(name)
    return RemoteAssetPlan(
        missing=[
            path for name, path in expected.items() if name not in uploaded
        ],
        starter_asset_ids=starter_asset_ids,
    )


def publish(
    *,
    repository_root: Path,
    catalog_path: Path,
    index_path: Path,
    asset_directory: Path,
    execute: bool,
    target: str,
) -> None:
    catalog = load_catalog(catalog_path)
    index = load_index(index_path)
    validate_index_contract(
        index,
        catalog,
        require_current_catalog_complete=True,
    )
    release = index["release"]
    assert isinstance(release, dict)
    repository = release.get("repository")
    tag = release.get("tag")
    if not isinstance(repository, str) or not repository:
        raise ArtifactDistributionError("release repository is required")
    if not isinstance(tag, str) or not tag:
        raise ArtifactDistributionError("release tag is required")
    upload_paths = _release_assets(
        index_path=index_path,
        index=index,
        asset_directory=asset_directory,
    )
    _publication_preflight(
        repository_root=repository_root,
        index_path=index_path,
        index=index,
        target=target,
    )
    source_commit = str(release["source_commit"])
    _resolved_commit(
        repository_root,
        source_commit,
        field="release source commit",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, target],
        cwd=repository_root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ArtifactDistributionError(
            "release source commit is not an ancestor of publication target"
        )
    title = f"Quantis JEPA evidence — {tag}"
    notes = (
        "Content-addressed, conclusion-bearing Quantis JEPA experiment "
        "artifacts. Verify every download against artifact-index-v1.json "
        "and SHA256SUMS."
    )
    create = [
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        repository,
        "--draft",
        "--title",
        title,
        "--notes",
        notes,
    ]
    create.extend(["--target", target])
    remote = _remote_release(repository=repository, tag=tag)
    remote_tag_commit = _remote_tag_commit(repository=repository, tag=tag)
    if remote is None:
        if remote_tag_commit is not None:
            raise ArtifactDistributionError(
                "release tag already exists without a release; refusing "
                "ambiguous --target behavior"
            )
        plan = RemoteAssetPlan(
            missing=upload_paths,
            starter_asset_ids=[],
        )
    else:
        if (
            remote_tag_commit is not None
            and remote_tag_commit != target
        ):
            raise ArtifactDistributionError(
                "existing release tag resolves to a different commit"
            )
        plan = _remote_asset_plan(
            remote=remote,
            expected_paths=upload_paths,
            target=target,
        )
    upload = [
        "gh",
        "release",
        "upload",
        tag,
        *(str(path) for path in plan.missing),
        "--repo",
        repository,
    ]
    if not execute:
        print("DRY RUN; GitHub was not changed")
        if remote is None:
            print("# create the draft; the release does not exist")
            print(shlex.join(create))
        for asset_id in plan.starter_asset_ids:
            print(
                shlex.join(
                    [
                        "gh",
                        "api",
                        "--method",
                        "DELETE",
                        (
                            f"repos/{repository}/releases/assets/"
                            f"{asset_id}"
                        ),
                    ]
                )
            )
        if plan.missing:
            print("# upload only missing assets after remote digest checks")
            print(shlex.join(upload))
        else:
            print("# every expected release asset is already present")
        return
    if remote is None:
        _run_checked(create, cwd=repository_root)
    for asset_id in plan.starter_asset_ids:
        _run_checked(
            [
                "gh",
                "api",
                "--method",
                "DELETE",
                f"repos/{repository}/releases/assets/{asset_id}",
            ],
            cwd=repository_root,
        )
    if not plan.missing:
        print("draft release already contains every verified asset")
        return
    resume_upload = [
        "gh",
        "release",
        "upload",
        tag,
        *(str(path) for path in plan.missing),
        "--repo",
        repository,
    ]
    _run_checked(resume_upload, cwd=repository_root)


def record(
    *,
    repository_root: Path,
    catalog_path: Path,
    index_path: Path,
    asset_directory: Path,
    destination: Path,
) -> None:
    catalog = load_catalog(catalog_path)
    index = load_index(index_path)
    validate_index_contract(
        index,
        catalog,
        require_current_catalog_complete=True,
    )
    release = index["release"]
    assert isinstance(release, dict)
    _resolved_commit(
        repository_root,
        str(release["source_commit"]),
        field="release source commit",
    )
    _release_assets(
        index_path=index_path,
        index=index,
        asset_directory=asset_directory,
    )
    if destination.exists():
        raise ArtifactDistributionError(
            f"refusing to overwrite recorded release index: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(f"recorded verified release metadata: {destination}")


def _run_checked(command: Sequence[str], *, cwd: Path) -> None:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ArtifactDistributionError(
            f"command failed ({result.returncode}): {shlex.join(command)}"
        )


def _require_within(path: Path, parent: Path) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise ArtifactDistributionError(
            f"path escapes artifact root: {path}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distribute content-addressed Quantis evidence"
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser(
        "pack", help="package catalog-selected artifact directories"
    )
    pack_parser.add_argument("--slug", action="append", default=[])
    pack_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    pack_parser.add_argument("--release-repository", default=DEFAULT_REPOSITORY)
    pack_parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    pack_parser.add_argument("--source-commit", required=True)
    pack_parser.add_argument("--dry-run", action="store_true")

    fetch_parser = subparsers.add_parser(
        "fetch", help="download, verify, and restore one artifact"
    )
    fetch_parser.add_argument("slug")
    fetch_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    fetch_parser.add_argument("--asset-directory", type=Path)

    verify_parser = subparsers.add_parser(
        "verify", help="verify a local artifact and optionally its archive"
    )
    verify_parser.add_argument("slug")
    verify_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    verify_parser.add_argument("--asset-directory", type=Path)

    publish_parser = subparsers.add_parser(
        "publish", help="validate and optionally upload a draft GitHub Release"
    )
    publish_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    publish_parser.add_argument("--asset-directory", type=Path, required=True)
    publish_parser.add_argument("--execute", action="store_true")
    publish_parser.add_argument("--target", required=True)

    record_parser = subparsers.add_parser(
        "record", help="record a verified release index in the repository"
    )
    record_parser.add_argument("--index", type=Path, required=True)
    record_parser.add_argument("--asset-directory", type=Path, required=True)
    record_parser.add_argument("--destination", type=Path, default=DEFAULT_INDEX)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    options = parser.parse_args(arguments)
    try:
        if options.command == "pack":
            pack(
                repository_root=options.repository_root,
                catalog_path=options.catalog,
                output=options.output,
                slugs=options.slug,
                release_repository=options.release_repository,
                release_tag=options.release_tag,
                source_commit=options.source_commit,
                dry_run=options.dry_run,
            )
        elif options.command == "fetch":
            fetch(
                repository_root=options.repository_root,
                catalog_path=options.catalog,
                index_path=options.index,
                slug=options.slug,
                asset_directory=options.asset_directory,
            )
        elif options.command == "verify":
            verify(
                repository_root=options.repository_root,
                catalog_path=options.catalog,
                index_path=options.index,
                slug=options.slug,
                asset_directory=options.asset_directory,
            )
        elif options.command == "publish":
            publish(
                repository_root=options.repository_root,
                catalog_path=options.catalog,
                index_path=options.index,
                asset_directory=options.asset_directory,
                execute=options.execute,
                target=options.target,
            )
        elif options.command == "record":
            record(
                repository_root=options.repository_root,
                catalog_path=options.catalog,
                index_path=options.index,
                asset_directory=options.asset_directory,
                destination=options.destination,
            )
        else:
            parser.error(f"unsupported command: {options.command}")
    except (ArtifactDistributionError, OSError, json.JSONDecodeError) as error:
        print(f"artifact distribution error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
