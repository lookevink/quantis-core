PYTHON ?= .venv/bin/python
ARTIFACT_DISTRIBUTION ?= dist/artifacts/evidence-jepa-frontier-v1
ARTIFACT_INDEX ?= experiments/jepa/releases/evidence-jepa-frontier-v1.json
SOURCE_COMMIT ?= $(shell git rev-parse HEAD)
TARGET_COMMIT ?= $(shell git rev-parse HEAD)

.PHONY: catalog docs-check test maintenance-test maintenance-typecheck check full-check artifacts-plan artifacts-pack artifacts-record artifacts-fetch artifacts-verify artifacts-publish-plan

catalog:
	$(PYTHON) tools/sync_experiment_catalog.py --check

docs-check:
	$(PYTHON) tools/check_markdown_links.py

test:
	$(PYTHON) -m pytest

maintenance-test:
	$(PYTHON) -m pytest tests/test_experiment_catalog.py tests/test_artifact_distribution.py

maintenance-typecheck:
	$(PYTHON) -m mypy --strict tools/sync_experiment_catalog.py tools/check_markdown_links.py tools/artifacts.py tests/test_experiment_catalog.py tests/test_artifact_distribution.py

check: catalog docs-check maintenance-typecheck maintenance-test

full-check: check test

artifacts-plan:
	$(PYTHON) tools/artifacts.py pack --source-commit $(SOURCE_COMMIT) --dry-run

artifacts-pack:
	$(PYTHON) tools/artifacts.py pack --source-commit $(SOURCE_COMMIT) --output $(ARTIFACT_DISTRIBUTION)

artifacts-record:
	$(PYTHON) tools/artifacts.py record --index $(ARTIFACT_DISTRIBUTION)/artifact-index-v1.json --asset-directory $(ARTIFACT_DISTRIBUTION) --destination $(ARTIFACT_INDEX)

artifacts-fetch:
	@if [ -z "$(TECHNIQUE)" ]; then echo "TECHNIQUE=<catalog slug> is required"; exit 2; fi
	$(PYTHON) tools/artifacts.py fetch $(TECHNIQUE) --index $(ARTIFACT_INDEX)

artifacts-verify:
	@if [ -z "$(TECHNIQUE)" ]; then echo "TECHNIQUE=<catalog slug> is required"; exit 2; fi
	$(PYTHON) tools/artifacts.py verify $(TECHNIQUE) --index $(ARTIFACT_INDEX)

artifacts-publish-plan:
	$(PYTHON) tools/artifacts.py publish --index $(ARTIFACT_DISTRIBUTION)/artifact-index-v1.json --asset-directory $(ARTIFACT_DISTRIBUTION) --target $(TARGET_COMMIT)
