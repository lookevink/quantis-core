PYTHON ?= .venv/bin/python

.PHONY: catalog docs-check test catalog-test maintenance-typecheck check full-check

catalog:
	$(PYTHON) tools/sync_experiment_catalog.py --check

docs-check:
	$(PYTHON) tools/check_markdown_links.py

test:
	$(PYTHON) -m pytest

catalog-test:
	$(PYTHON) -m pytest tests/test_experiment_catalog.py

maintenance-typecheck:
	$(PYTHON) -m mypy --strict tools/sync_experiment_catalog.py tools/check_markdown_links.py tests/test_experiment_catalog.py

check: catalog docs-check maintenance-typecheck catalog-test

full-check: check test
