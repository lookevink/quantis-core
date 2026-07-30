import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _catalog_module() -> ModuleType:
    path = ROOT / "tools/sync_experiment_catalog.py"
    spec = importlib.util.spec_from_file_location(
        "sync_experiment_catalog", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jepa_catalog_has_complete_generated_capsules() -> None:
    module = _catalog_module()

    catalog = module.load_catalog()
    experiments = module.validate_catalog(catalog)

    assert len(experiments) == 24
    assert {experiment["status"] for experiment in experiments} == {
        "active",
        "rejected",
    }
    assert module.synchronize(check=True) == []


def test_jepa_catalog_rejects_duplicate_slugs() -> None:
    module = _catalog_module()
    catalog = module.load_catalog()
    duplicate = copy.deepcopy(catalog["experiments"][0])
    catalog["experiments"].append(duplicate)

    with pytest.raises(module.CatalogError, match="duplicate experiment slug"):
        module.validate_catalog(catalog)


def test_jepa_catalog_requires_a_citation() -> None:
    module = _catalog_module()
    catalog = module.load_catalog()
    catalog["experiments"][0]["citations"] = []

    with pytest.raises(module.CatalogError, match="citation"):
        module.validate_catalog(catalog)


def test_jepa_catalog_requires_conclusion_bearing_findings() -> None:
    module = _catalog_module()
    catalog = module.load_catalog()
    catalog["experiments"][0]["findings"] = []

    with pytest.raises(module.CatalogError, match="findings"):
        module.validate_catalog(catalog)


def test_jepa_catalog_paths_cannot_escape_repository() -> None:
    module = _catalog_module()
    catalog = module.load_catalog()
    catalog["experiments"][0]["runner"] = "../outside.py"

    with pytest.raises(module.CatalogError, match="escapes repository"):
        module.validate_catalog(catalog)


def test_jepa_catalog_renders_supporting_artifact_identities() -> None:
    module = _catalog_module()
    catalog = module.load_catalog()
    experiment = catalog["experiments"][0]
    experiment["supporting_artifacts"] = [
        "artifacts/action-dynamics/supporting-v1"
    ]

    validated = module.validate_catalog(catalog)
    rendered = module.render_capsule_readme(
        validated[0],
        evidence_boundary=catalog["evidence_boundary"],
    )

    assert (
        "- Supporting artifact: "
        "`artifacts/action-dynamics/supporting-v1`"
    ) in rendered
    experiment["supporting_artifacts"] = ["artifacts/../outside"]
    with pytest.raises(module.CatalogError, match="escapes artifacts"):
        module.validate_catalog(catalog)


def test_repository_markdown_links_resolve() -> None:
    path = ROOT / "tools/check_markdown_links.py"
    spec = importlib.util.spec_from_file_location("check_markdown_links", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.check_markdown_links(module.markdown_files()) == []
