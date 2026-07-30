import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import List, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/artifacts.py"


def _run(
    repository: Path,
    catalog: Path,
    *arguments: str,
    environment: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    values = list(arguments)
    if values and values[0] == "pack" and "--source-commit" not in values:
        source_commit = _git(repository, "rev-parse", "HEAD").stdout.strip()
        values[1:1] = ["--source-commit", source_commit]
    command: List[str] = [
        sys.executable,
        str(TOOL),
        "--repository-root",
        str(repository),
        "--catalog",
        str(catalog),
        *values,
    ]
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=dict(environment) if environment is not None else None,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    artifact = repository / "artifacts/demo"
    artifact.mkdir(parents=True)
    (artifact / "result.json").write_text('{"status": "rejected"}\n')
    (artifact / "nested").mkdir()
    (artifact / "nested/evidence.bin").write_bytes(b"\x00\x01evidence\n")
    catalog = repository / "experiments/jepa/catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "program": "jepa",
                "experiments": [
                    {
                        "slug": "demo",
                        "title": "Demonstration tracer",
                        "artifact": "artifacts/demo",
                    }
                ],
            }
        )
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "Quantis Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Fixture source")
    return repository, catalog


def test_pack_is_deterministic_and_records_content_identities(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_run = _run(
        repository,
        catalog,
        "pack",
        "--slug",
        "demo",
        "--output",
        str(first),
    )
    second_run = _run(
        repository,
        catalog,
        "pack",
        "--slug",
        "demo",
        "--output",
        str(second),
    )

    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    first_index = json.loads((first / "artifact-index-v1.json").read_text())
    second_index = json.loads((second / "artifact-index-v1.json").read_text())
    first_asset = first_index["assets"][0]
    second_asset = second_index["assets"][0]
    assert first_index["release"]["tag"] == "evidence-jepa-frontier-v1"
    assert first_index["release"]["source_commit"] == _git(
        repository, "rev-parse", "HEAD"
    ).stdout.strip()
    assert first_index["selection"]["slugs"] == ["demo"]
    assert first_asset["artifact_path"] == "artifacts/demo"
    assert first_asset["file_count"] == 2
    assert first_asset["entry_count"] == 4
    assert first_asset["unpacked_size"] == 34
    assert len(first_asset["tree_sha256"]) == 64
    assert first_asset["archive_sha256"] == second_asset["archive_sha256"]
    assert (first / first_asset["name"]).read_bytes() == (
        second / second_asset["name"]
    ).read_bytes()
    checksums = (first / "SHA256SUMS").read_text()
    assert (
        f"{first_asset['archive_sha256']}  {first_asset['name']}\n"
        in checksums
    )


def test_fetch_restores_exact_tree_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--slug",
        "demo",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    original = repository / "artifacts/demo"
    expected = (original / "nested/evidence.bin").read_bytes()
    for path in sorted(original.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    original.rmdir()

    fetched = _run(
        repository,
        catalog,
        "fetch",
        "demo",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
    )
    repeated = _run(
        repository,
        catalog,
        "fetch",
        "demo",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
    )

    assert fetched.returncode == 0, fetched.stderr
    assert (original / "nested/evidence.bin").read_bytes() == expected
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr


def test_fetch_rejects_archive_path_traversal(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    output.mkdir()
    archive = output / "malicious.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../../escaped.txt")
        payload = b"escape"
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    index = output / "artifact-index-v1.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release": {
                    "repository": "lookevink/quantis-core",
                    "tag": "evidence-jepa-frontier-v1",
                    "source_commit": _git(
                        repository, "rev-parse", "HEAD"
                    ).stdout.strip(),
                },
                "selection": {
                    "program": "jepa",
                    "slugs": ["demo"],
                    "catalog_sha256": _catalog_sha(catalog),
                },
                "assets": [
                    {
                        "slug": "demo",
                        "artifact_path": "artifacts/demo",
                        "name": archive.name,
                        "archive_sha256": digest,
                        "archive_size": archive.stat().st_size,
                        "tree_sha256": "0" * 64,
                        "unpacked_size": 6,
                        "file_count": 1,
                        "entry_count": 1,
                    }
                ],
            }
        )
    )
    artifact = repository / "artifacts/demo"
    for path in sorted(artifact.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    artifact.rmdir()

    result = _run(
        repository,
        catalog,
        "fetch",
        "demo",
        "--index",
        str(index),
        "--asset-directory",
        str(output),
    )

    assert result.returncode != 0
    assert "unsafe archive member" in result.stderr
    assert not (tmp_path / "escaped.txt").exists()


def test_dry_run_rejects_bundle_that_cannot_fit_a_release_asset(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    oversized = repository / "artifacts/demo/oversized.bin"
    with oversized.open("wb") as stream:
        stream.truncate(2 * 1024**3)

    result = _run(repository, catalog, "pack", "--slug", "demo", "--dry-run")

    assert result.returncode != 0
    assert "2 GiB GitHub Release asset limit" in result.stderr


def test_verify_checks_both_archive_and_restored_tree(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--slug",
        "demo",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr

    verified = _run(
        repository,
        catalog,
        "verify",
        "demo",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
    )
    (repository / "artifacts/demo/result.json").write_text('{"tampered": true}\n')
    tampered = _run(
        repository,
        catalog,
        "verify",
        "demo",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
    )

    assert verified.returncode == 0, verified.stderr
    assert "archive and tree verified" in verified.stdout
    assert tampered.returncode != 0
    assert "tree SHA-256 differs" in tampered.stderr


def test_publish_defaults_to_a_non_mutating_command_plan(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--slug",
        "demo",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    recorded_index = (
        repository
        / "experiments/jepa/releases/evidence-jepa-frontier-v1.json"
    )
    recorded = _run(
        repository,
        catalog,
        "record",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
        "--destination",
        str(recorded_index),
    )
    assert recorded.returncode == 0, recorded.stderr
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Record artifact index")
    target = _git(repository, "rev-parse", "HEAD").stdout.strip()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"""#!{sys.executable}
import json
import sys

if any("/releases?per_page=100" in value for value in sys.argv):
    print(json.dumps([[]]))
    raise SystemExit(0)
print("HTTP 404: Not Found", file=sys.stderr)
raise SystemExit(1)
"""
    )
    fake_gh.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"

    result = _run(
        repository,
        catalog,
        "publish",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
        "--target",
        target,
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN; GitHub was not changed" in result.stdout
    assert "gh release create evidence-jepa-frontier-v1" in result.stdout
    assert "gh release upload evidence-jepa-frontier-v1" in result.stdout


def test_record_writes_only_the_verified_distribution_index(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--slug",
        "demo",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    destination = repository / "experiments/jepa/releases/demo-v1.json"

    recorded = _run(
        repository,
        catalog,
        "record",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
        "--destination",
        str(destination),
    )
    repeated = _run(
        repository,
        catalog,
        "record",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
        "--destination",
        str(destination),
    )

    assert recorded.returncode == 0, recorded.stderr
    assert json.loads(destination.read_text())["assets"][0]["slug"] == "demo"
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr


def test_canonical_release_requires_the_complete_catalog(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    second = repository / "artifacts/second"
    second.mkdir()
    (second / "result.json").write_text("{}\n")
    payload = json.loads(catalog.read_text())
    payload["experiments"].append(
        {
            "slug": "second",
            "title": "Second tracer",
            "artifact": "artifacts/second",
        }
    )
    catalog.write_text(json.dumps(payload))

    result = _run(
        repository,
        catalog,
        "pack",
        "--slug",
        "demo",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "canonical release requires every catalog experiment" in result.stderr


def test_tree_identity_includes_empty_directories(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_run = _run(
        repository,
        catalog,
        "pack",
        "--output",
        str(first),
    )
    assert first_run.returncode == 0, first_run.stderr
    (repository / "artifacts/demo/empty").mkdir()

    second_run = _run(
        repository,
        catalog,
        "pack",
        "--output",
        str(second),
    )

    assert second_run.returncode == 0, second_run.stderr
    first_asset = json.loads(
        (first / "artifact-index-v1.json").read_text()
    )["assets"][0]
    second_asset = json.loads(
        (second / "artifact-index-v1.json").read_text()
    )["assets"][0]
    assert first_asset["tree_sha256"] != second_asset["tree_sha256"]
    assert first_asset["name"] != second_asset["name"]
    assert second_asset["entry_count"] == first_asset["entry_count"] + 1


def test_fetch_enforces_indexed_size_before_writing_member(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    output.mkdir()
    archive = output / "oversized.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        root = tarfile.TarInfo("artifacts/demo")
        root.type = tarfile.DIRTYPE
        bundle.addfile(root)
        member = tarfile.TarInfo("artifacts/demo/large.bin")
        payload = b"larger than indexed"
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    index = output / "artifact-index-v1.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release": {
                    "repository": "lookevink/quantis-core",
                    "tag": "evidence-jepa-frontier-v1",
                    "source_commit": _git(
                        repository, "rev-parse", "HEAD"
                    ).stdout.strip(),
                },
                "selection": {
                    "program": "jepa",
                    "slugs": ["demo"],
                    "catalog_sha256": _catalog_sha(catalog),
                },
                "assets": [
                    {
                        "slug": "demo",
                        "artifact_path": "artifacts/demo",
                        "name": archive.name,
                        "archive_sha256": hashlib.sha256(
                            archive.read_bytes()
                        ).hexdigest(),
                        "archive_size": archive.stat().st_size,
                        "tree_sha256": "0" * 64,
                        "unpacked_size": 1,
                        "file_count": 1,
                        "entry_count": 2,
                    }
                ],
            }
        )
    )
    _remove_artifact(repository / "artifacts/demo")

    result = _run(
        repository,
        catalog,
        "fetch",
        "demo",
        "--index",
        str(index),
        "--asset-directory",
        str(output),
    )

    assert result.returncode != 0
    assert "exceeds indexed unpacked size" in result.stderr
    assert not (repository / "artifacts/demo").exists()


def test_publish_plan_rejects_index_that_differs_from_head(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    recorded_index = (
        repository
        / "experiments/jepa/releases/evidence-jepa-frontier-v1.json"
    )
    recorded = _run(
        repository,
        catalog,
        "record",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
        "--destination",
        str(recorded_index),
    )
    assert recorded.returncode == 0, recorded.stderr
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Record artifact index")
    target = _git(repository, "rev-parse", "HEAD").stdout.strip()
    changed = json.loads((output / "artifact-index-v1.json").read_text())
    changed["release"]["repository"] = "someone/else"
    (output / "artifact-index-v1.json").write_text(json.dumps(changed))

    result = _run(
        repository,
        catalog,
        "publish",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
        "--target",
        target,
    )

    assert result.returncode != 0
    assert "differs from the index recorded in HEAD" in result.stderr


def test_publish_resumes_a_matching_partial_draft(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    index_path = output / "artifact-index-v1.json"
    recorded_index = (
        repository
        / "experiments/jepa/releases/evidence-jepa-frontier-v1.json"
    )
    recorded = _run(
        repository,
        catalog,
        "record",
        "--index",
        str(index_path),
        "--asset-directory",
        str(output),
        "--destination",
        str(recorded_index),
    )
    assert recorded.returncode == 0, recorded.stderr
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Record artifact index")
    target = _git(repository, "rev-parse", "HEAD").stdout.strip()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "gh-log.jsonl"
    remote = {
        "tag_name": "evidence-jepa-frontier-v1",
        "draft": True,
        "target_commitish": target,
        "assets": [
            {
                "name": index_path.name,
                "id": 41,
                "state": "uploaded",
                "size": index_path.stat().st_size,
                "digest": (
                    "sha256:"
                    + hashlib.sha256(index_path.read_bytes()).hexdigest()
                ),
            },
            {
                "name": json.loads(index_path.read_text())["assets"][0][
                    "name"
                ],
                "id": 42,
                "state": "starter",
                "size": 0,
                "digest": None,
            },
        ],
    }
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

with Path(os.environ["GH_TEST_LOG"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "api" and "--method" not in sys.argv:
    if "/git/ref/tags/" in sys.argv[-1]:
        print("HTTP 404: Not Found", file=sys.stderr)
        raise SystemExit(1)
    else:
        print({json.dumps(json.dumps([[remote]]))})
raise SystemExit(0)
"""
    )
    fake_gh.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["GH_TEST_LOG"] = str(log)

    result = _run(
        repository,
        catalog,
        "publish",
        "--index",
        str(index_path),
        "--asset-directory",
        str(output),
        "--target",
        target,
        "--execute",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][0] == "api"
    assert calls[0][1].endswith("/releases?per_page=100")
    assert calls[0][-2:] == ["--paginate", "--slurp"]
    assert not any(call[:2] == ["release", "create"] for call in calls)
    assert any(
        call[:3] == ["api", "--method", "DELETE"] for call in calls
    )
    upload = next(call for call in calls if call[:2] == ["release", "upload"])
    assert str(index_path) not in upload
    assert str(output / "SHA256SUMS") in upload
    assert str(
        output / json.loads(index_path.read_text())["assets"][0]["name"]
    ) in upload


def test_pack_rejects_a_path_fetch_would_reject(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    payload = json.loads(catalog.read_text())
    payload["experiments"][0]["artifact"] = "artifacts/demo/../demo"
    catalog.write_text(json.dumps(payload))

    result = _run(repository, catalog, "pack", "--dry-run")

    assert result.returncode != 0
    assert "unsafe archive member path" in result.stderr


def test_pack_rejects_delimiter_like_member_name(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    (repository / "artifacts/demo/literal\\0separator").write_text("unsafe\n")

    result = _run(repository, catalog, "pack", "--dry-run")

    assert result.returncode != 0
    assert "unsafe archive member path" in result.stderr


def test_publish_rejects_a_preexisting_release_tag(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    index_path = output / "artifact-index-v1.json"
    recorded_index = (
        repository
        / "experiments/jepa/releases/evidence-jepa-frontier-v1.json"
    )
    recorded = _run(
        repository,
        catalog,
        "record",
        "--index",
        str(index_path),
        "--asset-directory",
        str(output),
        "--destination",
        str(recorded_index),
    )
    assert recorded.returncode == 0, recorded.stderr
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Record artifact index")
    target = _git(repository, "rev-parse", "HEAD").stdout.strip()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"""#!{sys.executable}
import json
import sys

if any("/releases?per_page=100" in value for value in sys.argv):
    print(json.dumps([[]]))
    raise SystemExit(0)
if "/git/ref/tags/" in sys.argv[-1]:
    print(json.dumps({{"object": {{"type": "commit", "sha": {json.dumps(target)}}}}}))
    raise SystemExit(0)
print("HTTP 404: Not Found", file=sys.stderr)
raise SystemExit(1)
"""
    )
    fake_gh.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"

    result = _run(
        repository,
        catalog,
        "publish",
        "--index",
        str(index_path),
        "--asset-directory",
        str(output),
        "--target",
        target,
        environment=environment,
    )

    assert result.returncode != 0
    assert "release tag already exists without a release" in result.stderr


def test_pack_rejects_a_nonexistent_source_commit(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)

    result = _run(
        repository,
        catalog,
        "pack",
        "--source-commit",
        "f" * 40,
        "--dry-run",
    )

    assert result.returncode != 0
    assert "source commit does not resolve" in result.stderr


def test_record_rejects_catalog_path_substitution(tmp_path: Path) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    index_path = output / "artifact-index-v1.json"
    index = json.loads(index_path.read_text())
    index["assets"][0]["artifact_path"] = "artifacts/substituted"
    index_path.write_text(json.dumps(index))

    result = _run(
        repository,
        catalog,
        "record",
        "--index",
        str(index_path),
        "--asset-directory",
        str(output),
        "--destination",
        str(tmp_path / "recorded.json"),
    )

    assert result.returncode != 0
    assert "artifact path differs from catalog" in result.stderr


def test_recorded_v1_remains_fetchable_after_catalog_growth(
    tmp_path: Path,
) -> None:
    repository, catalog = _fixture(tmp_path)
    output = tmp_path / "distribution"
    packed = _run(
        repository,
        catalog,
        "pack",
        "--output",
        str(output),
    )
    assert packed.returncode == 0, packed.stderr
    payload = json.loads(catalog.read_text())
    payload["experiments"].append(
        {
            "slug": "future",
            "title": "Future tracer",
            "artifact": "artifacts/future",
        }
    )
    catalog.write_text(json.dumps(payload))
    future = repository / "artifacts/future"
    future.mkdir()
    (future / "result.json").write_text("{}\n")
    _remove_artifact(repository / "artifacts/demo")

    result = _run(
        repository,
        catalog,
        "fetch",
        "demo",
        "--index",
        str(output / "artifact-index-v1.json"),
        "--asset-directory",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert (repository / "artifacts/demo/result.json").is_file()


def _git(
    repository: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    )


def _catalog_sha(catalog: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            json.loads(catalog.read_text()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _remove_artifact(artifact: Path) -> None:
    for path in sorted(artifact.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    artifact.rmdir()
